"""Offline 3D from every recorded camera a session's calibration holds, and the pose-consistency check.

session_offline_3d is what intersession processing calls once the session's calibration has
resolved. It matches the cameras the session recorded to the calibration's by serial
(select_session_cameras): a camera whose body was swapped, one the calibration left 2D only,
and one it does not hold are excluded, each with its reason. The 2D pose of the rest, each
camera's *_<camera>_raw2D_live.h5 in the pixels inference saw, is triangulated with the
calibration's anipose TOML at each camera's recorded geometry (store.load_preset_set):

- every point is undistorted to normalised coordinates with its camera's K and distortion,
  then triangulated by N-view DLT with P = [R|t] (triangulate_dlt), from the views whose
  likelihood reaches the confidence cutoff; fewer than 2 such views give NaN;
- points3d.h5 holds the points in the calibration's motor frame, in mm; reprojections.h5
  holds each camera's projection of them and its distance from that camera's own point.

The pose-consistency check (pose_consistency, spec section 8(b)) measures each camera
against the point triangulated from the other cameras (leave-one-out), which needs 3 or
more cameras. A camera whose median residual is over the limit is "possibly moved". With
exactly 2 cameras the residual is shared, the sum of the two cameras' reprojection errors,
which is about how far the pair is from agreeing; it cannot say which camera moved, so it
reports "pair inconsistent". The settings are in <library>/settings.yaml under
checks.pose_consistency, global with an optional per_camera override, and are only read here.

Nothing is written inside the calibration entry: a geometry no stored file has is derived
into the library's .derived cache.
"""

import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd

try:
    import tomllib
except ImportError:  # Python 3.10, as on the rigs: the backport, which the installer puts in the environment
    import tomli as tomllib

from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.multires_calibration.geometry import PresetGeometry
from autotrainer.inference.multires_calibration.library import derived_root, read_settings
from autotrainer.inference.multires_calibration.store import load_index, load_preset_set

logger = get_verbose_logger(__name__)

POINTS3D_FILE = "points3d.h5"
REPROJECTIONS_FILE = "reprojections.h5"
REPORT_FILE = "offline3d_report.json"
POSE_FILE_SUFFIX = "_raw2D_live.h5"
MIN_VIEWS = 2
NOT_IN_CALIBRATION = "not in the session's calibration"

STATUS_OK = "ok"
STATUS_MOVED = "possibly moved"
STATUS_PAIR = "pair inconsistent"
STATUS_TOO_LITTLE = "not enough data"

# SpinCam's default geometry, used for a parameter a camera's configuration leaves out. The
# same values as tools/acquisition/model/calibration_geometry.SPINCAM_DEFAULT_GEOMETRY, which
# the app reads live geometry with (live_inference_geometry); the inference package cannot
# import the app, so tests/offline_3d_test.py checks the two agree where both are importable.
SPINCAM_DEFAULT_GEOMETRY = dict(hbin=4, width=256, height=256, offsetx=52, offsety=6)

# Points triangulated per batch: the DLT matrices of a batch take 64 * cameras bytes each.
_CHUNK = 50_000


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

@dataclass(frozen=True)
class PoseTrack:
    frame_idx: Optional[np.ndarray]  # (n,) recording-relative frame index, None without the column
    node_names: List[str]
    xy: np.ndarray  # (n, kp, 2)
    likelihood: np.ndarray  # (n, kp)


def read_pose_file(path: Path) -> PoseTrack:
    """A DLC-style pose file: (bodypart, x / y / likelihood) columns, and frame_idx when live pose wrote it."""
    df = pd.read_hdf(path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"{path}: expected (bodypart, coordinate) columns")
    parts = [p for p in dict.fromkeys(df.columns.get_level_values(0)) if p != "frame_idx"]
    if not parts:
        raise ValueError(f"{path}: no body parts")

    def column(coordinate: str) -> np.ndarray:
        return np.stack([df[(p, coordinate)].to_numpy(float) for p in parts], axis=1)

    frame_idx = None
    if "frame_idx" in df.columns.get_level_values(0):
        frame_idx = np.asarray(df["frame_idx"]).reshape(len(df), -1)[:, 0].astype(np.int64)
    return PoseTrack(frame_idx, parts, np.stack([column("x"), column("y")], axis=-1), column("likelihood"))


def _aligned(tracks: Mapping[str, PoseTrack]) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """(frame_idx, node_names, xy (cams, frames, kp, 2), likelihood (cams, frames, kp)) over all cameras.

    Frames are matched by frame_idx when every file has it, over the union of indices: a frame
    a camera did not pose is NaN there. Without it, by row. The parts are those every camera
    has, in the first camera's order.
    """
    first = next(iter(tracks.values()))
    names = [p for p in first.node_names if all(p in t.node_names for t in tracks.values())]
    if not names:
        raise ValueError("the cameras' pose files have no body part in common")
    if all(t.frame_idx is not None for t in tracks.values()):
        frames = np.unique(np.concatenate([t.frame_idx for t in tracks.values()]))
        rows = {cam: np.searchsorted(frames, t.frame_idx) for cam, t in tracks.items()}
    else:
        frames = np.arange(max(len(t.xy) for t in tracks.values()))
        rows = {cam: np.arange(len(t.xy)) for cam, t in tracks.items()}
    xy = np.full((len(tracks), len(frames), len(names), 2), np.nan)
    likelihood = np.zeros((len(tracks), len(frames), len(names)))
    for c, (cam, track) in enumerate(tracks.items()):
        columns = [track.node_names.index(p) for p in names]
        xy[c, rows[cam]] = track.xy[:, columns]
        likelihood[c, rows[cam]] = np.nan_to_num(track.likelihood[:, columns], nan=0.0)
    return frames, names, xy, likelihood


# --- export -------------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualStats:
    """One camera's judged observations: the residual in px, the confidence and the frame of each.

    leave_one_out is True when the residual is against the point the other cameras give; False
    with exactly 2 cameras, where it is the pair's shared residual.
    """

    camera: str
    residual_px: np.ndarray
    confidence: np.ndarray
    frame: np.ndarray
    leave_one_out: bool


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
    node_names, camera_names, frame_idx). Cameras are in the TOML's order.
    """
    session_dir = Path(session_dir)
    calibration_id, toml_cameras = load_anipose_toml(calibration_toml)
    in_toml = {c.name for c in toml_cameras}
    unknown = [name for name in camera_files if name not in in_toml]
    if unknown:
        raise ValueError(f"{calibration_toml} has no camera {', '.join(unknown)}")
    cameras = [c for c in toml_cameras if c.name in camera_files]
    if len(cameras) < MIN_VIEWS:
        raise ValueError(f"offline 3D needs at least {MIN_VIEWS} cameras, got {len(cameras)}")
    tracks = {c.name: read_pose_file(Path(camera_files[c.name])) for c in cameras}
    frames, node_names, xy, likelihood = _aligned(tracks)
    mask = (likelihood >= min_confidence) & np.isfinite(xy).all(axis=-1)
    normalised = np.stack([undistort_normalised(xy[i], c) for i, c in enumerate(cameras)])
    projections = np.stack([c.projection for c in cameras])

    points3d = triangulate_dlt(normalised, projections, mask)
    reprojections = np.stack([project(points3d, c) for c in cameras])
    error = np.linalg.norm(reprojections - xy, axis=-1)
    error[~mask] = np.nan

    residuals = {}
    if len(cameras) == MIN_VIEWS:
        shared = error[0] + error[1]
        judged = np.isfinite(shared)
        confidence = np.minimum(likelihood[0], likelihood[1])
        for c in cameras:
            residuals[c.name] = _stats(c.name, shared, confidence, frames, judged, leave_one_out=False)
    else:
        for i, c in enumerate(cameras):
            others = mask.copy()
            others[i] = False
            residual = np.linalg.norm(project(triangulate_dlt(normalised, projections, others), c) - xy[i], axis=-1)
            judged = mask[i] & np.isfinite(residual)
            residuals[c.name] = _stats(c.name, residual, likelihood[i], frames, judged, leave_one_out=True)

    names = [c.name for c in cameras]
    points3d_path = session_dir / POINTS3D_FILE
    reprojections_path = session_dir / REPROJECTIONS_FILE
    with h5py.File(points3d_path, "w") as fh:
        fh.create_dataset("points3d", data=points3d)
        _write_names(fh, node_names, names, frames)
        fh.attrs["calibration_id"] = calibration_id or ""
        fh.attrs["units"] = "mm"
    with h5py.File(reprojections_path, "w") as fh:
        fh.create_dataset("reprojections", data=reprojections)
        fh.create_dataset("reprojection_error", data=error)
        _write_names(fh, node_names, names, frames)
        fh.attrs["calibration_id"] = calibration_id or ""
        fh.attrs["units"] = "px"
    used = int(np.isfinite(points3d).all(axis=-1).sum())
    logger.info("offline 3D of %s from %s: %d of %d points triangulated, into %s",
                session_dir, ", ".join(names), used, points3d.shape[0] * points3d.shape[1], points3d_path)
    return Offline3DResult(points3d_path, reprojections_path, residuals)


def _stats(camera, residual, confidence, frames, judged, *, leave_one_out) -> ResidualStats:
    frame_of = np.broadcast_to(frames[:, None], judged.shape)
    return ResidualStats(camera, residual[judged], confidence[judged], frame_of[judged], leave_one_out)


def _write_names(fh: h5py.File, node_names: Sequence[str], camera_names: Sequence[str], frames: np.ndarray) -> None:
    text = h5py.string_dtype()
    fh.create_dataset("node_names", data=np.array(list(node_names), dtype=object), dtype=text)
    fh.create_dataset("camera_names", data=np.array(list(camera_names), dtype=object), dtype=text)
    fh.create_dataset("frame_idx", data=np.asarray(frames, np.int64))


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
    """Each camera's verdict on its residuals, with its limits from `settings`.

    `settings` is a PoseConsistencySettings or the raw checks.pose_consistency mapping. Only
    observations at or above the camera's min_confidence count. Too few of them, or too few
    frames, is "not enough data"; otherwise a median (or, when set, p95) residual over its
    limit is "possibly moved" for a leave-one-out residual and "pair inconsistent" for the
    shared residual of 2 cameras, and "ok" otherwise. The verdict is computed whether or not
    the check is enabled; the caller decides whether to run it.
    """
    if not isinstance(settings, PoseConsistencySettings):
        settings = pose_consistency_settings(settings)
    verdicts = {}
    for name, stats in residuals.items():
        limits = settings.for_camera(name)
        keep = stats.confidence >= limits.min_confidence
        values = stats.residual_px[keep]
        n_obs = int(values.size)
        n_frames = int(np.unique(stats.frame[keep]).size)
        median = float(np.median(values)) if n_obs else None
        p95 = float(np.percentile(values, 95)) if n_obs else None
        if n_obs < limits.min_observations or n_frames < limits.min_frames:
            status = STATUS_TOO_LITTLE
        elif median > limits.median_px or (limits.p95_px is not None and p95 > limits.p95_px):
            status = STATUS_MOVED if stats.leave_one_out else STATUS_PAIR
        else:
            status = STATUS_OK
        verdicts[name] = ConsistencyVerdict(name, status, median, p95, n_obs, limits.median_px, limits.p95_px)
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


def find_pose_file(session_dir: Path, camera: str) -> Optional[Path]:
    """The camera's *_<camera>_raw2D_live.h5 in `session_dir`, None when there is none or more than one."""
    found = sorted(Path(session_dir).glob(f"*_{camera}{POSE_FILE_SUFFIX}"))
    if len(found) > 1:
        logger.warning("more than one pose file for %s in %s: %s", camera, session_dir, ", ".join(map(str, found)))
        return None
    return found[0] if found else None


def session_offline_3d(
    session_dir: Path, folder: Path, metadata: Mapping, *, library_dir: Path
) -> Optional[Offline3DResult]:
    """Offline 3D of the session in `session_dir` with the format-3 calibration in `folder`.

    `metadata` is the session's; `library_dir` is the library whose settings.yaml holds the
    check's settings and whose .derived cache takes any geometry the entry has no file for.
    Writes points3d.h5, reprojections.h5 and offline3d_report.json, which names the cameras
    used, those excluded and why, and the check's verdicts (null when it is off). With
    fewer than 2 cameras to triangulate the session is 2D only: the report says why, and
    None is returned.
    """
    session_dir, folder = Path(session_dir), Path(folder)
    index = load_index(folder)
    configured = ((metadata.get("hardware") or {}).get("configured") or {}).get("cameras")
    if not isinstance(configured, Mapping):
        # Recorded before the metadata listed its cameras: the calibration's, matched by name.
        configured = {c["name"]: {} for c in index.cameras}
    recorded = {name: entry for name, entry in configured.items() if find_pose_file(session_dir, name) is not None}
    selected = select_session_cameras(index.cameras, recorded)
    excluded = dict(selected.excluded)
    params = configured_params(metadata)
    live = next((s["cameras"] for s in index.preset_sets if s["name"] == "live"), {})
    geometries = {}
    for name, calibration_name in selected.used.items():
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
                             derived_root=derived_root(library_dir))
    settings = read_pose_consistency_settings(library_dir)
    result = export_session_3d(
        session_dir, loaded.toml_path, {used[name]: find_pose_file(session_dir, name) for name in used},
        min_confidence=settings.limits.min_confidence,
    )
    verdicts = {}
    if settings.enabled:
        verdicts = pose_consistency(result.residuals, settings)
        for verdict in verdicts.values():
            (logger.info if verdict.status in (STATUS_OK, STATUS_TOO_LITTLE) else logger.warning)(
                "session %s: %s", session_dir, verdict.describe())
        report["poseConsistency"] = {name: v.to_plain() for name, v in verdicts.items()}
    _write_report(session_dir, report)
    return replace(result, excluded=excluded, consistency=verdicts)


def _write_report(session_dir: Path, report: dict) -> None:
    path = Path(session_dir) / REPORT_FILE
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
