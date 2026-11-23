"""Offline 3D from every recorded camera a session's calibration holds, and the pose-consistency check.

session_offline_3d is what intersession processing calls once the session's calibration has
resolved, whether or not the session's left-right 3D could be read. It matches the cameras
the session recorded to the calibration's by serial (select_session_cameras): a camera whose
body was swapped, one the calibration left 2D only, and one it does not hold are excluded,
each with its reason. So are a calibrated camera with no 2D pose file or one that cannot be
read, and one recorded at a frame rate the others were not. The 2D pose of the rest, each
camera's <session prefix>_<camera>_raw2D_live.h5 (ProjectInfo.get_intersession_pose_path) in
the pixels inference saw, is triangulated with the calibration's anipose TOML at each
camera's recorded geometry (store.load_preset_set):

- the session is read and processed in blocks of BLOCK_FRAMES frames, so memory does not
  grow with its length;
- every point is undistorted to normalised coordinates with its camera's K and distortion,
  then triangulated by N-view DLT with P = [R|t] (triangulate_dlt), from the views whose
  likelihood reaches the confidence cutoff; fewer than 2 such views give NaN;
- points3d.h5 holds the points in the calibration's motor frame, in mm; reprojections.h5
  holds each camera's projection of them and its distance from that camera's own point.
  Both are chunked and compressed, and written under a temporary name and renamed into
  place, so a failed run leaves no partial file and an earlier result as it was.

The pose-consistency check (pose_consistency, spec section 8(b)) measures each camera
against the point triangulated from the other cameras (leave-one-out), which needs 3 or
more cameras. A camera whose median residual is over the limit is "possibly moved". A moved
camera also pulls the points the others are measured against, so attribute_moved_cameras
flags the one with the highest median and measures the rest again without it, until none
is over its limit. With exactly 2 cameras the residual is shared, the sum of the two
cameras' reprojection errors, which is about how far the pair is from agreeing; it cannot
say which camera moved, so it reports "pair inconsistent". The residuals are kept as
histograms (ResidualStats), so the median and p95 are read to within 0.6 % of their value.
The settings are in <library>/settings.yaml under checks.pose_consistency, global with an
optional per_camera override, and are only read here.

Nothing is written inside the calibration entry: a geometry no stored file has is derived
into the library's .derived cache, or into the scratch folder the caller gives.
"""

import json
import math
import os
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd

try:
    import tomllib
except ImportError:  # Python 3.10, as on the rigs: the backport, which the installer puts in the environment
    import tomli as tomllib

from autotrainer.core import ProjectInfo
from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.multires_calibration.geometry import PresetGeometry
from autotrainer.inference.multires_calibration.library import derived_root, read_settings
from autotrainer.inference.multires_calibration.store import load_index, load_preset_set

logger = get_verbose_logger(__name__)

POINTS3D_FILE = "points3d.h5"
REPROJECTIONS_FILE = "reprojections.h5"
REPORT_FILE = "offline3d_report.json"
MIN_VIEWS = 2
NOT_IN_CALIBRATION = "not in the session's calibration"
NO_POSE_FILE = "no 2D pose file"

STATUS_OK = "ok"
STATUS_MOVED = "possibly moved"
STATUS_PAIR = "pair inconsistent"
STATUS_TOO_LITTLE = "not enough data"

# SpinCam's default geometry, used for a parameter a camera's configuration leaves out. The
# same values as tools/acquisition/model/calibration_geometry.SPINCAM_DEFAULT_GEOMETRY, which
# the app reads live geometry with (live_inference_geometry); the inference package cannot
# import the app, so tests/offline_3d_test.py checks the two agree where both are importable.
SPINCAM_DEFAULT_GEOMETRY = dict(hbin=4, width=256, height=256, offsetx=52, offsety=6)

# Frames read and processed at a time: 20 000 frames of 6 cameras and 16 keypoints hold about
# 30 MB of 2D points, whatever the session's length.
BLOCK_FRAMES = 20_000
# Frames per HDF5 chunk of the outputs.
_H5_CHUNK_FRAMES = 1_000
# Points triangulated per batch: the DLT matrices of a batch take 64 * cameras bytes each.
_CHUNK = 50_000

# The residual histograms: confidence in hundredths (rows) by residual (columns). The residual
# bins are [0, 1e-4) px, 400 per decade from 1e-4 to 1e4 px, and [1e4, inf).
_CONFIDENCE_STEPS = 100
_RESIDUAL_EDGES = np.concatenate([[0.0], np.logspace(-4.0, 4.0, 8 * 400 + 1), [np.inf]])


# --- the anipose TOML ---------------------------------------------------------------------

@dataclass(frozen=True)
class TomlCamera:
    name: str
    size: Tuple[int, int]  # (width, height)
    matrix: np.ndarray  # 3x3
    distortions: np.ndarray  # 5
    rotation: np.ndarray  # Rodrigues vector, motor frame -> camera
    translation: np.ndarray  # mm

    @property
    def projection(self) -> np.ndarray:
        """[R|t], the 3x4 projection to normalised coordinates."""
        return np.hstack([cv2.Rodrigues(self.rotation)[0], self.translation.reshape(3, 1)])


def load_anipose_toml(path: Path) -> Tuple[Optional[str], List[TomlCamera]]:
    """(the calibration id in [metadata], the cameras cam_0, cam_1, ... in table order)."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    cameras = []
    i = 0
    while f"cam_{i}" in data:
        table = data[f"cam_{i}"]
        cameras.append(TomlCamera(
            name=str(table["name"]),
            size=tuple(int(v) for v in table["size"]),
            matrix=np.asarray(table["matrix"], float).reshape(3, 3),
            distortions=np.asarray(table["distortions"], float).ravel(),
            rotation=np.asarray(table["rotation"], float).ravel(),
            translation=np.asarray(table["translation"], float).ravel(),
        ))
        i += 1
    calibration_id = (data.get("metadata") or {}).get("calibration_id")
    return (None if calibration_id is None else str(calibration_id)), cameras


# --- triangulation ------------------------------------------------------------------------

def undistort_normalised(points: np.ndarray, camera: TomlCamera) -> np.ndarray:
    """(..., 2) pixels to normalised coordinates, undistorted; NaN stays NaN."""
    points = np.asarray(points, float)
    flat = points.reshape(-1, 2)
    finite = np.isfinite(flat).all(axis=1)
    out = np.full_like(flat, np.nan)
    if finite.any():
        # The iterative form, run to convergence: the default 5 iterations leave
        # hundredths of a pixel near the image edge at the rig's distortion.
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
        out[finite] = cv2.undistortPointsIter(
            flat[finite].reshape(-1, 1, 2), camera.matrix, camera.distortions, None, None, criteria
        ).reshape(-1, 2)
    return out.reshape(points.shape)


def triangulate_dlt(points: np.ndarray, projections: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """N-view DLT: (n_cams, n_frames, n_kp, 2) points and (n_cams, 3, 4) projections -> (n_frames, n_kp, 3).

    A view counts where `mask` (n_cams, n_frames, n_kp) is true and its point is finite. Each
    view gives the two rows x * P[2] - P[0] and y * P[2] - P[1]; a view that does not count
    gives zero rows, which leave the solution unchanged. The point is the right singular
    vector of the smallest singular value, dehomogenised. NaN where fewer than 2 views count.
    """
    points = np.asarray(points, float)
    projections = np.asarray(projections, float)
    n_cams, n_frames, n_kp, _ = points.shape
    if projections.shape != (n_cams, 3, 4):
        raise ValueError(f"expected {n_cams} projections of shape (3, 4), got {projections.shape}")
    valid = (np.asarray(mask, bool) & np.isfinite(points).all(axis=-1)).reshape(n_cams, -1)
    flat = np.where(valid[..., None], points.reshape(n_cams, -1, 2), 0.0)
    out = np.full((flat.shape[1], 3), np.nan)
    for start in range(0, flat.shape[1], _CHUNK):
        x = flat[:, start:start + _CHUNK]  # (C, N, 2)
        w = valid[:, start:start + _CHUNK]
        enough = w.sum(axis=0) >= MIN_VIEWS
        if not enough.any():
            continue
        rows = x[..., :, None] * projections[:, None, 2:3, :] - projections[:, None, :2, :]  # (C, N, 2, 4)
        rows = rows * w[..., None, None]
        A = rows.transpose(1, 0, 2, 3).reshape(x.shape[1], 2 * n_cams, 4)[enough]
        homogeneous = np.linalg.svd(A)[2][:, -1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = homogeneous[:, :3] / homogeneous[:, 3:]
        block = out[start:start + _CHUNK]
        block[enough] = xyz
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out.reshape(n_frames, n_kp, 3)


def project(points3d: np.ndarray, camera: TomlCamera) -> np.ndarray:
    """(..., 3) motor-frame points to (..., 2) pixels of `camera`, with its distortion; NaN stays NaN."""
    flat = np.asarray(points3d, float).reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    out = np.full((flat.shape[0], 2), np.nan)
    if finite.any():
        out[finite] = cv2.projectPoints(
            flat[finite], camera.rotation, camera.translation, camera.matrix, camera.distortions
        )[0].reshape(-1, 2)
    return out.reshape(*np.shape(points3d)[:-1], 2)


# --- the 2D pose files --------------------------------------------------------------------

class PoseFile:
    """A DLC-style pose file, read in row blocks: (bodypart, x / y / likelihood) columns, and
    frame_idx when live pose wrote it.

    Opening it reads the whole file once, a block at a time, for its frame indices and body
    parts; raises when it cannot be read as a pose file.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        with pd.HDFStore(self.path, mode="r") as store:
            keys = store.keys()
            if len(keys) != 1:
                raise ValueError(f"{self.path}: expected one table, found {len(keys)}")
            self._key = keys[0]
            self.node_names: Optional[List[str]] = None
            has_frame_idx, frame_idx, self.n_rows = False, [], 0
            while True:
                df = store.select(self._key, start=self.n_rows, stop=self.n_rows + BLOCK_FRAMES)
                if self.node_names is None:
                    self.node_names, has_frame_idx = self._columns(df)
                if has_frame_idx and len(df):
                    frame_idx.append(np.asarray(df["frame_idx"]).reshape(len(df), -1)[:, 0].astype(np.int64))
                self.n_rows += len(df)
                if len(df) < BLOCK_FRAMES:
                    break
        # (n_rows,) recording-relative frame index, None without the column.
        self.frame_idx = np.concatenate(frame_idx + [np.zeros(0, np.int64)]) if has_frame_idx else None

    def _columns(self, df: pd.DataFrame) -> Tuple[List[str], bool]:
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError(f"{self.path}: expected (bodypart, coordinate) columns")
        parts = [p for p in dict.fromkeys(df.columns.get_level_values(0)) if p != "frame_idx"]
        if not parts:
            raise ValueError(f"{self.path}: no body parts")
        return parts, "frame_idx" in df.columns.get_level_values(0)

    def read(self, start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
        """Rows [start, stop): (xy (n, parts, 2), likelihood (n, parts)), parts in node_names' order."""
        with pd.HDFStore(self.path, mode="r") as store:
            df = store.select(self._key, start=start, stop=stop)

        def column(coordinate: str) -> np.ndarray:
            return np.stack([df[(p, coordinate)].to_numpy(float) for p in self.node_names], axis=1)

        return np.stack([column("x"), column("y")], axis=-1), column("likelihood")


@dataclass(frozen=True)
class _Block:
    frames: np.ndarray  # (n,) frame index of each row
    xy: np.ndarray  # (cams, n, kp, 2) pixels, NaN where the camera has no point
    likelihood: np.ndarray  # (cams, n, kp)
    mask: np.ndarray  # (cams, n, kp) the views used: likelihood at the cutoff and a finite point
    normalised: np.ndarray  # (cams, n, kp, 2) undistorted normalised coordinates


class _SessionPoses:
    """The cameras' pose files aligned frame by frame, read a block of frames at a time.

    Frames are matched by frame_idx when every file has it, over the union of indices: a frame
    a camera did not pose is NaN there. Without it, by row. The parts are those every camera
    has, in the first camera's order.
    """

    def __init__(self, cameras: Sequence[TomlCamera], sources: Sequence[PoseFile], min_confidence: float):
        self.cameras = list(cameras)
        self.min_confidence = min_confidence
        self.sources = list(sources)
        first = self.sources[0]
        self.node_names = [p for p in first.node_names if all(p in s.node_names for s in self.sources)]
        if not self.node_names:
            raise ValueError("the cameras' pose files have no body part in common")
        self._columns = [[s.node_names.index(p) for p in self.node_names] for s in self.sources]
        if all(s.frame_idx is not None for s in self.sources):
            self.frames = np.unique(np.concatenate([s.frame_idx for s in self.sources]))
            self._rows = [np.searchsorted(self.frames, s.frame_idx) for s in self.sources]
        else:
            self.frames = np.arange(max(s.n_rows for s in self.sources), dtype=np.int64)
            self._rows = [np.arange(s.n_rows) for s in self.sources]

    def blocks(self) -> Iterator[_Block]:
        n_cams, n_kp, size = len(self.sources), len(self.node_names), BLOCK_FRAMES
        for start in range(0, len(self.frames), size):
            stop = min(start + size, len(self.frames))
            xy = np.full((n_cams, stop - start, n_kp, 2), np.nan)
            likelihood = np.zeros((n_cams, stop - start, n_kp))
            for c, (source, rows, columns) in enumerate(zip(self.sources, self._rows, self._columns)):
                # The file rows whose frames fall in this block: one contiguous read when the
                # file's frame indices rise, as live pose writes them.
                selected = np.flatnonzero((rows >= start) & (rows < stop))
                if not selected.size:
                    continue
                first = int(selected.min())
                block_xy, block_likelihood = source.read(first, int(selected.max()) + 1)
                at = rows[selected] - start
                xy[c, at] = block_xy[selected - first][:, columns]
                likelihood[c, at] = np.nan_to_num(block_likelihood[selected - first][:, columns], nan=0.0)
            mask = (likelihood >= self.min_confidence) & np.isfinite(xy).all(axis=-1)
            normalised = np.stack([undistort_normalised(xy[i], c) for i, c in enumerate(self.cameras)])
            yield _Block(self.frames[start:stop], xy, likelihood, mask, normalised)


# --- residuals ----------------------------------------------------------------------------

def _confidence_bin(confidence: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(np.asarray(confidence, float) * _CONFIDENCE_STEPS + 1e-9), 0, _CONFIDENCE_STEPS).astype(
        np.int64)


@dataclass
class ResidualStats:
    """One camera's judged observations, as histograms, so their size does not grow with the session.

    counts holds the observations by confidence (in hundredths) and residual bin
    (_RESIDUAL_EDGES); frame_counts the frames with a judged observation, by the highest
    confidence among them. leave_one_out is True when the residual is against the point the
    other cameras give; False with exactly 2 cameras, where it is the pair's shared residual.
    """

    camera: str
    leave_one_out: bool
    counts: np.ndarray = None
    frame_counts: np.ndarray = None

    def __post_init__(self):
        if self.counts is None:
            self.counts = np.zeros((_CONFIDENCE_STEPS + 1, _RESIDUAL_EDGES.size - 1), np.int64)
        if self.frame_counts is None:
            self.frame_counts = np.zeros(_CONFIDENCE_STEPS + 1, np.int64)

    def add(self, residual: np.ndarray, confidence: np.ndarray, judged: np.ndarray) -> None:
        """Add a block of (frames, kp) residuals in px, their confidences, and which are judged."""
        n_bins = _RESIDUAL_EDGES.size - 1
        rows = _confidence_bin(confidence[judged])
        columns = np.clip(np.searchsorted(_RESIDUAL_EDGES, residual[judged], side="right") - 1, 0, n_bins - 1)
        self.counts += np.bincount(rows * n_bins + columns, minlength=self.counts.size).reshape(self.counts.shape)
        seen = judged.any(axis=1)
        best = np.where(judged, confidence, -np.inf).max(axis=1)[seen]
        self.frame_counts += np.bincount(_confidence_bin(best), minlength=self.frame_counts.size)

    @staticmethod
    def _from(min_confidence: float) -> int:
        return max(0, int(math.ceil(min_confidence * _CONFIDENCE_STEPS - 1e-9)))

    def n_obs(self, min_confidence: float = 0.0) -> int:
        """Observations at or above `min_confidence` (to the hundredth)."""
        return int(self.counts[self._from(min_confidence):].sum())

    def n_frames(self, min_confidence: float = 0.0) -> int:
        """Frames with an observation at or above `min_confidence` (to the hundredth)."""
        return int(self.frame_counts[self._from(min_confidence):].sum())

    def quantile(self, q: float, min_confidence: float = 0.0) -> Optional[float]:
        """The q-th quantile (0 to 1) of the residuals at or above `min_confidence`, within
        its histogram bin; None without any."""
        hist = self.counts[self._from(min_confidence):].sum(axis=0)
        total = int(hist.sum())
        if not total:
            return None
        cumulative = np.cumsum(hist)
        target = q * total
        i = min(int(np.searchsorted(cumulative, target, side="left")), hist.size - 1)
        low, high = _RESIDUAL_EDGES[i], _RESIDUAL_EDGES[i + 1]
        if not np.isfinite(high):
            return float(low)
        below = cumulative[i] - hist[i]
        return float(low + (target - below) / hist[i] * (high - low)) if hist[i] else float(low)


def _residuals(block: _Block, cameras: Sequence[TomlCamera], projections: np.ndarray,
               use: Sequence[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{camera index: (residual, confidence, judged)} of the cameras `use` on one block.

    Only the cameras in `use` take part. With 2 of them, the residual is their shared one:
    the sum of their reprojection errors from the point they give together. With more, each
    camera's leave-one-out residual against the point the others give.
    """
    in_use = np.zeros(len(cameras), bool)
    in_use[list(use)] = True
    mask = block.mask & in_use[:, None, None]
    out = {}
    if len(use) == MIN_VIEWS:
        points3d = triangulate_dlt(block.normalised, projections, mask)
        errors = []
        for i in use:
            error = np.linalg.norm(project(points3d, cameras[i]) - block.xy[i], axis=-1)
            error[~mask[i]] = np.nan
            errors.append(error)
        shared = errors[0] + errors[1]
        confidence = np.minimum(*(block.likelihood[i] for i in use))
        for i in use:
            out[i] = (shared, confidence, np.isfinite(shared))
        return out
    for i in use:
        others = mask.copy()
        others[i] = False
        point = triangulate_dlt(block.normalised, projections, others)
        residual = np.linalg.norm(project(point, cameras[i]) - block.xy[i], axis=-1)
        out[i] = (residual, block.likelihood[i], mask[i] & np.isfinite(residual))
    return out


def _measure(poses: _SessionPoses, projections: np.ndarray, use: Sequence[int]) -> Dict[str, ResidualStats]:
    """The residuals of the cameras `use`, triangulated from them alone, over the whole session."""
    cameras = poses.cameras
    stats = {cameras[i].name: ResidualStats(cameras[i].name, leave_one_out=len(use) > MIN_VIEWS) for i in use}
    for block in poses.blocks():
        for i, (residual, confidence, judged) in _residuals(block, cameras, projections, use).items():
            stats[cameras[i].name].add(residual, confidence, judged)
    return stats


# --- export -------------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsistencyVerdict:
    camera: str
    status: str  # STATUS_OK, STATUS_MOVED, STATUS_PAIR or STATUS_TOO_LITTLE
    median_px: Optional[float]
    p95_px: Optional[float]
    n_obs: int
    median_limit_px: Optional[float] = None
    p95_limit_px: Optional[float] = None

    def describe(self) -> str:
        """In plain words, e.g. "possibly moved: camera4, median residual 12.0 px over 5000 observations (limit 6.0)"."""
        if self.status == STATUS_TOO_LITTLE:
            return f"{self.status}: {self.camera}, {self.n_obs} observations"
        text = f"{self.status}: {self.camera}, median residual {self.median_px:.1f} px over {self.n_obs} observations"
        if self.status != STATUS_OK:
            text += f" (limit {self.median_limit_px:.1f}"
            if self.p95_limit_px is not None:
                text += f"; p95 {self.p95_px:.1f} px, limit {self.p95_limit_px:.1f}"
            text += ")"
        return text

    def to_plain(self) -> dict:
        return {"status": self.status, "medianPx": self.median_px, "p95Px": self.p95_px, "nObs": self.n_obs,
                "medianLimitPx": self.median_limit_px, "p95LimitPx": self.p95_limit_px}


@dataclass(frozen=True)
class Offline3DResult:
    points3d_path: Path
    reprojections_path: Path
    residuals: Dict[str, ResidualStats]
    # Filled by session_offline_3d: recorded name -> why it was not triangulated, and the check.
    excluded: Dict[str, str] = field(default_factory=dict)
    consistency: Dict[str, ConsistencyVerdict] = field(default_factory=dict)
    # The residuals of the cameras left when those named are taken out, measured again from
    # the pose files (attribute_moved_cameras).
    recompute: Optional[Callable[[Sequence[str]], Dict[str, ResidualStats]]] = field(
        default=None, repr=False, compare=False)


DEFAULT_MIN_CONFIDENCE = 0.5


def export_session_3d(
    session_dir: Path,
    calibration_toml: Path,
    camera_files: Mapping[str, Path],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> Offline3DResult:
    """Triangulate the cameras in `camera_files` (TOML camera name -> its pose file) into `session_dir`.

    A view is used where its likelihood is at least `min_confidence`. Writes points3d.h5
    (points3d (frames, kp, 3) in mm, node_names, camera_names, frame_idx; attrs
    calibration_id and units "mm") and reprojections.h5 (reprojections (cams, frames, kp, 2),
    reprojection_error (cams, frames, kp), NaN where the camera's own point was not used,
    node_names, camera_names, frame_idx). Cameras are in the TOML's order. The session is
    processed BLOCK_FRAMES frames at a time, into chunked, gzip-compressed datasets that grow
    along the frame axis; both files are written under a temporary name, then renamed.
    """
    return _export(session_dir, calibration_toml, {name: PoseFile(Path(p)) for name, p in camera_files.items()},
                   min_confidence=min_confidence)


def _export(session_dir: Path, calibration_toml: Path, sources: Mapping[str, PoseFile], *,
            min_confidence: float) -> Offline3DResult:
    session_dir = Path(session_dir)
    calibration_id, toml_cameras = load_anipose_toml(calibration_toml)
    in_toml = {c.name for c in toml_cameras}
    unknown = [name for name in sources if name not in in_toml]
    if unknown:
        raise ValueError(f"{calibration_toml} has no camera {', '.join(unknown)}")
    cameras = [c for c in toml_cameras if c.name in sources]
    if len(cameras) < MIN_VIEWS:
        raise ValueError(f"offline 3D needs at least {MIN_VIEWS} cameras, got {len(cameras)}")
    poses = _SessionPoses(cameras, [sources[c.name] for c in cameras], min_confidence)
    projections = np.stack([c.projection for c in cameras])
    names = [c.name for c in cameras]
    n_cams, n_kp = len(cameras), len(poses.node_names)
    use = range(n_cams)
    residuals = {c.name: ResidualStats(c.name, leave_one_out=n_cams > MIN_VIEWS) for c in cameras}

    points3d_path = session_dir / POINTS3D_FILE
    reprojections_path = session_dir / REPROJECTIONS_FILE
    temps = [path.with_name(f"{path.name}.{os.getpid()}.tmp") for path in (points3d_path, reprojections_path)]
    used = total = 0
    try:
        with h5py.File(temps[0], "w") as points_fh, h5py.File(temps[1], "w") as reprojections_fh:
            points_ds = _growing(points_fh, "points3d", (0, n_kp, 3), axis=0)
            reprojections_ds = _growing(reprojections_fh, "reprojections", (n_cams, 0, n_kp, 2), axis=1)
            error_ds = _growing(reprojections_fh, "reprojection_error", (n_cams, 0, n_kp), axis=1)
            frame_ds = [_growing(fh, "frame_idx", (0,), axis=0, dtype=np.int64)
                        for fh in (points_fh, reprojections_fh)]
            for fh, units in ((points_fh, "mm"), (reprojections_fh, "px")):
                _write_names(fh, poses.node_names, names)
                fh.attrs["calibration_id"] = calibration_id or ""
                fh.attrs["units"] = units
            for block in poses.blocks():
                points3d = triangulate_dlt(block.normalised, projections, block.mask)
                reprojections = np.stack([project(points3d, c) for c in cameras])
                error = np.linalg.norm(reprojections - block.xy, axis=-1)
                error[~block.mask] = np.nan
                _append(points_ds, points3d, axis=0)
                _append(reprojections_ds, reprojections, axis=1)
                _append(error_ds, error, axis=1)
                for ds in frame_ds:
                    _append(ds, np.asarray(block.frames, np.int64), axis=0)
                for i, (residual, confidence, judged) in _residuals(block, cameras, projections, use).items():
                    residuals[cameras[i].name].add(residual, confidence, judged)
                used += int(np.isfinite(points3d).all(axis=-1).sum())
                total += points3d.shape[0] * points3d.shape[1]
        os.replace(temps[0], points3d_path)
        os.replace(temps[1], reprojections_path)
    except BaseException:
        for temp in temps:
            with suppress(OSError):
                temp.unlink()
        raise
    logger.info("offline 3D of %s from %s: %d of %d points triangulated, into %s",
                session_dir, ", ".join(names), used, total, points3d_path)

    def recompute(left_out: Sequence[str]) -> Dict[str, ResidualStats]:
        return _measure(poses, projections, [i for i, c in enumerate(cameras) if c.name not in set(left_out)])

    return Offline3DResult(points3d_path, reprojections_path, residuals, recompute=recompute)


def _growing(fh: h5py.File, name: str, shape: Tuple[int, ...], *, axis: int, dtype=float) -> h5py.Dataset:
    """A chunked, gzip-compressed dataset that `_append` grows along `axis`, the frame axis."""
    maxshape = tuple(None if i == axis else n for i, n in enumerate(shape))
    chunks = tuple(_H5_CHUNK_FRAMES if i == axis else (1 if i < axis else n) for i, n in enumerate(shape))
    return fh.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype,
                             compression="gzip", shuffle=True)


def _append(ds: h5py.Dataset, data: np.ndarray, *, axis: int) -> None:
    n = ds.shape[axis]
    ds.resize(n + data.shape[axis], axis=axis)
    index = [slice(None)] * ds.ndim
    index[axis] = slice(n, n + data.shape[axis])
    ds[tuple(index)] = data


def _write_names(fh: h5py.File, node_names: Sequence[str], camera_names: Sequence[str]) -> None:
    text = h5py.string_dtype()
    fh.create_dataset("node_names", data=np.array(list(node_names), dtype=object), dtype=text)
    fh.create_dataset("camera_names", data=np.array(list(camera_names), dtype=object), dtype=text)


# --- the pose-consistency check -----------------------------------------------------------

@dataclass(frozen=True)
class ConsistencyLimits:
    median_px: float = 6.0
    p95_px: Optional[float] = None
    min_frames: int = 300
    min_observations: int = 1000
    min_confidence: float = 0.5


@dataclass(frozen=True)
class PoseConsistencySettings:
    enabled: bool = False
    limits: ConsistencyLimits = ConsistencyLimits()
    per_camera: Mapping[str, ConsistencyLimits] = field(default_factory=dict)

    def for_camera(self, name: str) -> ConsistencyLimits:
        return self.per_camera.get(name, self.limits)


_LIMIT_TYPES = {"median_px": float, "p95_px": float, "min_frames": int, "min_observations": int,
                "min_confidence": float}


def _limits(raw: Any, base: ConsistencyLimits, where: str) -> ConsistencyLimits:
    if not isinstance(raw, Mapping):
        if raw is not None:
            logger.warning("ignoring %s: expected a mapping, got %r", where, raw)
        return base
    values = {}
    for key, kind in _LIMIT_TYPES.items():
        if key not in raw:
            continue
        value = raw[key]
        if value is None and key == "p95_px":
            values[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            logger.warning("ignoring %s.%s = %r: expected a number of 0 or more", where, key, value)
            continue
        if kind is int and value != int(value):
            logger.warning("ignoring %s.%s = %r: expected a whole number", where, key, value)
            continue
        values[key] = kind(value)
    return replace(base, **values)


def pose_consistency_settings(raw: Optional[Mapping]) -> PoseConsistencySettings:
    """checks.pose_consistency as settings; a missing or wrong value takes its default, with a warning."""
    where = "checks.pose_consistency"
    if raw is None:
        return PoseConsistencySettings()
    if not isinstance(raw, Mapping):
        logger.warning("ignoring %s: expected a mapping, got %r", where, raw)
        return PoseConsistencySettings()
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        logger.warning("ignoring %s.enabled = %r: expected true or false", where, enabled)
        enabled = False
    limits = _limits(raw, ConsistencyLimits(), where)
    per_camera = {}
    overrides = raw.get("per_camera") or {}
    if not isinstance(overrides, Mapping):
        logger.warning("ignoring %s.per_camera: expected a mapping, got %r", where, overrides)
        overrides = {}
    for name, override in overrides.items():
        per_camera[str(name)] = _limits(override, limits, f"{where}.per_camera.{name}")
    return PoseConsistencySettings(enabled=enabled, limits=limits, per_camera=per_camera)


def read_pose_consistency_settings(library_dir: Path) -> PoseConsistencySettings:
    """The check's settings from <library_dir>/settings.yaml (library.read_settings); never written here."""
    checks = read_settings(library_dir).get("checks")
    return pose_consistency_settings(checks.get("pose_consistency") if isinstance(checks, Mapping) else None)


def pose_consistency(
    residuals: Mapping[str, ResidualStats], settings: Any
) -> Dict[str, ConsistencyVerdict]:
    """Each camera's verdict on its residuals, with its limits from `settings`, in one round.

    `settings` is a PoseConsistencySettings or the raw checks.pose_consistency mapping. Only
    observations at or above the camera's min_confidence count. None of them, too few of
    them, or too few frames is "not enough data"; otherwise a median (or, when set, p95)
    residual over its limit is "possibly moved" for a leave-one-out residual and "pair
    inconsistent" for the shared residual of 2 cameras, and "ok" otherwise. The verdict is
    computed whether or not the check is enabled; the caller decides whether to run it.
    attribute_moved_cameras repeats it without a camera found moved.
    """
    if not isinstance(settings, PoseConsistencySettings):
        settings = pose_consistency_settings(settings)
    verdicts = {}
    for name, stats in residuals.items():
        limits = settings.for_camera(name)
        n_obs = stats.n_obs(limits.min_confidence)
        n_frames = stats.n_frames(limits.min_confidence)
        median = stats.quantile(0.5, limits.min_confidence)
        p95 = stats.quantile(0.95, limits.min_confidence)
        if n_obs == 0 or n_obs < limits.min_observations or n_frames < limits.min_frames:
            status = STATUS_TOO_LITTLE
        elif median > limits.median_px or (limits.p95_px is not None and p95 > limits.p95_px):
            status = STATUS_MOVED if stats.leave_one_out else STATUS_PAIR
        else:
            status = STATUS_OK
        verdicts[name] = ConsistencyVerdict(name, status, median, p95, n_obs, limits.median_px, limits.p95_px)
    return verdicts


def attribute_moved_cameras(result: Offline3DResult, settings: Any) -> Dict[str, ConsistencyVerdict]:
    """The check's verdicts, with a moved camera told apart from the cameras it misleads.

    With 3 or more cameras, each camera's leave-one-out residual is judged (pose_consistency).
    While any camera still measured is over its limit, the one with the highest median is
    flagged "possibly moved" and the others are measured again, triangulated without every
    flagged camera (result.recompute). That stops when none is over, or when fewer than 3
    cameras are left: those 2 are then judged on their shared residual, "pair inconsistent"
    when it is over the limit, else "ok". Without result.recompute, or with 2 cameras, this
    is pose_consistency.
    """
    if not isinstance(settings, PoseConsistencySettings):
        settings = pose_consistency_settings(settings)
    verdicts = pose_consistency(result.residuals, settings)
    remaining = list(verdicts)
    while result.recompute is not None and len(remaining) > MIN_VIEWS:
        over = [verdicts[name] for name in remaining if verdicts[name].status == STATUS_MOVED]
        if not over:
            break
        worst = max(over, key=lambda v: v.median_px)
        remaining.remove(worst.camera)
        flagged = [name for name in verdicts if name not in remaining]
        logger.info("pose consistency: %s has the highest median of the cameras over their limit; measuring %s"
                    " again without %s", worst.camera, ", ".join(remaining), ", ".join(flagged))
        verdicts.update(pose_consistency(result.recompute(flagged), settings))
    return verdicts


# --- a session ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionCameras:
    used: Dict[str, str]  # recorded name -> the calibration's name for it
    excluded: Dict[str, str]  # recorded name -> why it is not triangulated


def select_session_cameras(calibration_cameras: Sequence[Mapping], recorded: Mapping[str, Mapping]) -> SessionCameras:
    """Which recorded cameras are which calibrated cameras, matched by serial.

    `calibration_cameras` are the index's ({name, serial, calibrated, reason}); `recorded`
    maps a recorded camera's name to its hardware.configured.cameras entry (serial,
    inCalibration, uncalibratedReason). A camera is its serial, so a renamed camera keeps
    its calibration; a camera without a serial on either side is matched by name. A recorded
    camera whose serial differs from the calibration's camera of the same name is "swapped".
    A match the calibration did not calibrate takes its reason, and one the session recorded
    as not in the calibration (inCalibration false) is excluded with the recorded reason.
    """
    by_serial = {str(c["serial"]): c for c in calibration_cameras if c.get("serial")}
    by_name = {str(c["name"]): c for c in calibration_cameras}
    used, excluded = {}, {}
    for name, entry in recorded.items():
        entry = entry or {}
        serial = entry.get("serial")
        serial = None if serial in (None, "") else str(serial)
        same_name = by_name.get(name)
        match = by_serial.get(serial) if serial is not None else None
        if match is None and same_name is not None:
            if same_name.get("serial") and serial is not None:
                excluded[name] = f"swapped: calibrated serial {same_name['serial']}, now {serial}"
                continue
            match = same_name
        if match is None:
            excluded[name] = NOT_IN_CALIBRATION
        elif not match.get("calibrated", True):
            excluded[name] = str(match.get("reason") or "not calibrated")
        elif entry.get("inCalibration") is False:
            excluded[name] = str(entry.get("uncalibratedReason") or NOT_IN_CALIBRATION)
        elif match["name"] in used.values():
            excluded[name] = f"another recorded camera is the calibration's {match['name']}"
        else:
            used[name] = str(match["name"])
    return SessionCameras(used, excluded)


def recorded_geometry(params: Mapping[str, Any]) -> PresetGeometry:
    """The pixel space a camera's pose is in: its configured base capture (SpinCam's defaults for a missing key).

    The same reading as the app's live_inference_geometry. Raises ValueError for unequal
    binning or a value that is not a whole number.
    """
    def whole(keys, default):
        for key in keys:
            value = params.get(key)
            if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none"}):
                continue
            number = float(value)
            if not number.is_integer():
                raise ValueError(f"{key}={value!r} is not a whole number")
            return int(number)
        return default

    defaults = SPINCAM_DEFAULT_GEOMETRY
    hbin = whole(("hbin",), defaults["hbin"])
    vbin = whole(("vbin",), defaults["hbin"])
    if hbin != vbin:
        raise ValueError(f"the camera bins differently across and down: hbin={hbin} vbin={vbin}")
    return PresetGeometry(
        binning=hbin,
        width=whole(("width",), defaults["width"]),
        height=whole(("height",), defaults["height"]),
        offset_x=whole(("offset_x", "offsetx"), defaults["offsetx"]),
        offset_y=whole(("offset_y", "offsety"), defaults["offsety"]),
    )


def configured_params(metadata: Mapping) -> Dict[str, Mapping]:
    """Each recorded camera's configuration params, by name, from the session metadata's configuration."""
    cameras = ((metadata.get("configuration") or {}).get("cameras")) or []
    return {str(c["name"]): (c.get("params") or {}) for c in cameras if isinstance(c, Mapping) and "name" in c}


def recorded_frame_rate(entry: Any, params: Any) -> Optional[float]:
    """The frame rate a camera recorded at, in Hz: its hardware.configured.cameras entry's
    frameRateHz or fps, else its configuration params' fps; None when neither gives one."""
    for source, keys in ((entry, ("frameRateHz", "fps")), (params, ("fps",))):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool):
                continue
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(rate) and rate > 0:
                return rate
    return None


def session_offline_3d(
    project: ProjectInfo, folder: Path, metadata: Mapping, *, library_dir: Path,
    derived_dir: Optional[Path] = None,
) -> Optional[Offline3DResult]:
    """Offline 3D of the session of `project` with the format-3 calibration in `folder`.

    `metadata` is the session's; `library_dir` is the library whose settings.yaml holds the
    check's settings. A geometry the entry has no file for is derived into `derived_dir`,
    else the library's .derived cache. Each camera's 2D pose is its
    project.get_intersession_pose_path(camera, suffix="_live"). Writes points3d.h5,
    reprojections.h5 and offline3d_report.json into the session folder; the report names
    the cameras used, those excluded and why, and the check's verdicts (null when it is
    off). With fewer than 2 cameras to triangulate the session is 2D only: the report says
    why, and None is returned.
    """
    session_dir, folder = Path(project.get_session_path()[0]), Path(folder)
    index = load_index(folder)
    configured = ((metadata.get("hardware") or {}).get("configured") or {}).get("cameras")
    if not isinstance(configured, Mapping):
        # Recorded before the metadata listed its cameras: the calibration's, matched by name.
        configured = {c["name"]: {} for c in index.cameras}
    selected = select_session_cameras(index.cameras, configured)
    excluded = dict(selected.excluded)
    params = configured_params(metadata)

    sources = {}
    for name in selected.used:
        path = Path(project.get_intersession_pose_path(name, suffix="_live"))
        if not path.is_file():
            excluded[name] = NO_POSE_FILE
            continue
        try:
            sources[name] = PoseFile(path)
        except Exception as error:
            excluded[name] = f"2D pose file unreadable: {str(error) or type(error).__name__}"

    # One frame rate: frames of cameras recorded at different rates do not pair by index.
    rates = {name: recorded_frame_rate(configured.get(name), params.get(name)) for name in sources}
    known = [rate for rate in rates.values() if rate is not None]
    if known:
        common = Counter(known).most_common(1)[0][0]
        for name, rate in rates.items():
            if rate is not None and rate != common:
                excluded[name] = f"different frame rate ({rate:g} Hz vs {common:g} Hz)"
                del sources[name]

    live = next((s["cameras"] for s in index.preset_sets if s["name"] == "live"), {})
    geometries = {}
    for name in sources:
        calibration_name = selected.used[name]
        try:
            if name in params:
                geometries[name] = recorded_geometry(params[name])
            elif calibration_name in live:
                g = live[calibration_name]
                geometries[name] = PresetGeometry(g["binning"], g["width"], g["height"], g["offset_x"], g["offset_y"])
                logger.info("session %s records no configuration for %s; reading it at the calibration's"
                            " live geometry %s", session_dir, name, geometries[name].name)
            else:
                excluded[name] = "its recorded camera geometry is unknown"
        except ValueError as error:
            excluded[name] = f"its recorded camera geometry cannot be read: {error}"
    used = {name: selected.used[name] for name in geometries}
    report = {"calibrationId": index.id, "calibration": str(folder), "cameras": used, "excluded": excluded,
              "twoDOnlyReason": None, "poseConsistency": None}
    for name, reason in excluded.items():
        logger.warning("offline 3D of %s leaves out %s: %s", session_dir, name, reason)
    if len(used) < MIN_VIEWS:
        reason = f"fewer than {MIN_VIEWS} calibrated cameras: {', '.join(used) or 'none'}"
        if excluded:
            reason += " (" + "; ".join(f"{name}: {why}" for name, why in excluded.items()) + ")"
        report["twoDOnlyReason"] = reason
        logger.warning("session %s is 2D only: %s", session_dir, reason)
        _write_report(session_dir, report)
        return None
    loaded = load_preset_set(folder, {used[name]: g for name, g in geometries.items()},
                             derived_root=derived_root(library_dir) if derived_dir is None else Path(derived_dir))
    settings = _per_camera_by_calibration_name(read_pose_consistency_settings(library_dir), used)
    result = _export(session_dir, loaded.toml_path, {used[name]: sources[name] for name in used},
                     min_confidence=settings.limits.min_confidence)
    verdicts = {}
    if settings.enabled:
        verdicts = attribute_moved_cameras(result, settings)
        for verdict in verdicts.values():
            (logger.info if verdict.status in (STATUS_OK, STATUS_TOO_LITTLE) else logger.warning)(
                "session %s: %s", session_dir, verdict.describe())
        report["poseConsistency"] = {name: v.to_plain() for name, v in verdicts.items()}
    _write_report(session_dir, report)
    return replace(result, excluded=excluded, consistency=verdicts)


def _per_camera_by_calibration_name(settings: PoseConsistencySettings,
                                    used: Mapping[str, str]) -> PoseConsistencySettings:
    """`settings` with its per_camera overrides moved from the configured camera names they
    are given under to the calibration's names the residuals carry, through `used` (recorded
    name -> calibration name, matched by serial), as the reference-image check takes them. A
    key that names no camera used is dropped, with a note in the log."""
    per_camera = {calibration: settings.per_camera[recorded] for recorded, calibration in used.items()
                  if recorded in settings.per_camera}
    for name in set(settings.per_camera) - set(used):
        logger.info("checks.pose_consistency.per_camera.%s names no camera of this session; not used", name)
    return replace(settings, per_camera=per_camera)


def _write_report(session_dir: Path, report: dict) -> None:
    path = Path(session_dir) / REPORT_FILE
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
