"""Find the paused waypoints in a calibration capture and detect the board at each.

The capture records the board while the stage steps through its move list, stopping at
every waypoint for a settle and a dwell. Each camera's video is split into stationary
segments by frame-to-frame motion, and segment i is matched to distinct waypoint i (a
no-op move leaves the board still, so it adds no segment). If a camera's segment count
differs from the number of distinct waypoints, frames cannot be matched to positions
safely, and WaypointAlignmentError is raised instead of guessing.

The cameras are hardware-synchronised, so the same dwell must start at the same frame in
every camera; segments that drift apart also raise WaypointAlignmentError.

In every segment the board is detected in a few frames at its middle, and the per-corner
median of those detections is the waypoint's observation. The 6x8 board looks the same
turned 180 degrees, so the detector may start from either end. Each camera's detections
are put in the order of its first detection (normalize_corner_order with a reference), so
the order is consistent across that camera's waypoints. Two cameras can still label the
board from opposite ends; fit.reconcile_corner_order settles that before the fit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.calibration_FLIR import (
    adjust_gamma,
    average_chessboard_distance,
    imadjust_like_matlab,
    refine_corners,
    stretchlim_like_matlab,
)
from autotrainer.inference.multires_calibration.fit import BoardSpec, WaypointObservation

logger = get_verbose_logger(__name__)

_REFINEMENTS = ("subpix", "legacy")
# find_stationary_segments' defaults, which extract_observations also uses.
_MIN_FRAMES = 8
_SCALE = 8
_SUBPIX_WINDOW = (5, 5)
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
# The board cannot move while the stage is paused, so detections in one segment that
# disagree by more than this are not trusted.
_MAX_CORNER_STD_PX = 1.0
# The threshold takes p20 of the motion as the noise level, but never below this, in grey
# levels at the downscaled size. A codec repeats a still frame exactly except at its key
# frames, so p20 can be 0 (on the synthetic capture 148 of 245 m_t were exactly 0 in mp4v)
# and sqrt(p20 * p95) would then be 0, leaving no frame stationary. Key frames are also
# where a still, noisy scene jumps: mp4v's every 12th frame moved 0.2 grey levels when
# sensor-like noise (sigma 1) moved the others 0.01, so a near-zero p20 would split
# dwells there. With the floor, tau >= sqrt(0.05 * p95), 0.5 at a p95 of 5.
_NOISE_FLOOR = 0.05
# A video whose 95th-percentile motion is within this factor of its noise level never moved.
_STATIC_RATIO = 2.0
# The cameras are hardware-synchronised, so a dwell starts at the same frame in each; this
# allows for the threshold catching the slow end of a move a frame or two apart.
_MAX_START_OFFSET_FRAMES = 3
# The gamma the legacy calibration passed to create_corner_matrix (the former
# handle_3d_calibration.process_capture). refine_corners' fixed Canny thresholds were tuned
# on the image after that step.
_LEGACY_GAMMA = 2.0

_NOT_FOUND = "not found"
_INCONSISTENT = "inconsistent"


class WaypointAlignmentError(RuntimeError):
    """A camera's stationary segments do not match the distinct waypoints one to one."""


@dataclass
class ExtractionResult:
    # One per distinct waypoint, in order; a camera is absent where it was rejected.
    observations: List[WaypointObservation]
    segments: Dict[str, List[Tuple[int, int]]]  # camera -> half-open frame ranges, one per distinct waypoint
    rejected: List[Tuple[int, str, str]]  # (waypoint index, camera, "not found" or "inconsistent")
    # camera -> median over its kept waypoints of the median per-corner std, in sensor px
    corner_scatter_px: Dict[str, float]
    # camera -> p20, p95 and tau of the motion m_t, moving_fraction (the fraction of frames
    # in no segment) and n_frames, so a real capture's segmentation can be checked
    segmentation_stats: Dict[str, Dict[str, float]]


def distinct_waypoints(motor_positions: Sequence[np.ndarray], tol: float = 1e-6) -> List[int]:
    """Indices of the waypoints left after collapsing consecutive positions equal within `tol`.

    The first of a run of equal positions is kept: the stage did not move after it, so the
    run is one stationary segment in the video. A return to an earlier, non-adjacent
    position is a real move and is kept.
    """
    kept, previous = [], None
    for i, position in enumerate(motor_positions):
        position = np.asarray(position, float).reshape(-1)
        if previous is None or np.max(np.abs(position - previous)) > tol:
            kept.append(i)
        previous = position
    return kept


def find_stationary_segments(
    video: Path, *, min_frames: int = _MIN_FRAMES, scale: int = _SCALE
) -> List[Tuple[int, int]]:
    """Half-open frame ranges, in order, over which the scene in `video` holds still.

    Every frame is downscaled by `scale` (INTER_AREA) to float grey, and its motion is
    m_t = mean |f_t - f_(t-1)|. Frames with m_t below tau = sqrt(p20(m) * p95(m)) are
    stationary, with p20 floored at _NOISE_FLOOR, and a segment is a maximal stationary
    run of at least `min_frames`. A video that never moved is one segment.
    """
    return _segment_video(video, min_frames=min_frames, scale=scale)[0]


def _segment_video(video: Path, *, min_frames: int, scale: int) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """find_stationary_segments' segments, and the statistics ExtractionResult.segmentation_stats records."""
    motion, previous = [], None
    for gray in _grey_frames(video):
        h, w = gray.shape
        small = cv2.resize(gray, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float64)
        if previous is not None:
            motion.append(float(np.mean(np.abs(small - previous))))
        previous = small
    n = len(motion) + 1 if previous is not None else 0
    segments = _segments_from_motion(np.asarray(motion), min_frames=min_frames) if n else []
    p20, p95, tau, _ = _motion_levels(np.asarray(motion)) if len(motion) else (np.nan,) * 4
    still = sum(stop - start for start, stop in segments)
    stats = {
        "p20": p20,
        "p95": p95,
        "tau": tau,
        "moving_fraction": 1.0 - still / n if n else float("nan"),
        "n_frames": float(n),
    }
    return segments, stats


def _motion_levels(motion: np.ndarray) -> Tuple[float, float, float, bool]:
    """(p20, p95, tau, static) of m_t: tau = sqrt(max(p20, _NOISE_FLOOR) * p95)."""
    p20, p95 = np.percentile(motion, [20, 95])
    noise = max(float(p20), _NOISE_FLOOR)
    # Nothing moved when the motion is all noise; the geometric mean would then split the
    # video at random.
    return float(p20), float(p95), float(np.sqrt(noise * p95)), bool(p95 <= _STATIC_RATIO * noise)


def _segments_from_motion(motion: np.ndarray, *, min_frames: int) -> List[Tuple[int, int]]:
    """Stationary runs from m_t, t = 1 .. n-1, of an n-frame video."""
    n = len(motion) + 1
    if n == 1:
        return [(0, 1)] if min_frames <= 1 else []
    _, _, tau, static = _motion_levels(motion)
    if static:
        return [(0, n)] if n >= min_frames else []
    still = np.empty(n, bool)
    still[1:] = motion < tau
    # m_0 is undefined; frame 0 is stationary when frame 1 is.
    still[0] = still[1]

    segments, start = [], None
    for t in range(n + 1):
        if t < n and still[t]:
            if start is None:
                start = t
        elif start is not None:
            # A run cut short by the start or end of the recording is kept only if it is long enough.
            if t - start >= min_frames:
                segments.append((start, t))
            start = None
    return segments


def normalize_corner_order(
    corners: np.ndarray, board: BoardSpec, reference: Optional[np.ndarray] = None
) -> np.ndarray:
    """The corners as (rows*cols, 2) float64, in one of the board's two 180-degree orders.

    Without a reference, the order starts from the end of the board nearer the image
    origin (smaller x + y). That is ambiguous when the diagonal from the first corner to
    the last points near the image's (1, -1) direction: there, perspective alone can tip
    x + y either way from one waypoint to the next. With `reference`, an already-normalised
    (rows*cols, 2) detection from the same camera, the corners are reversed when their
    first-to-last diagonal points against the reference's. The board only translates, so
    the diagonal keeps its direction across the capture and the order stays consistent.
    """
    n = board.rows * board.cols
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(points) != n:
        raise ValueError(f"expected {n} corners, got {len(points)}")
    grid = points.reshape(board.rows, board.cols, 2)
    if reference is None:
        flip = grid[0, 0].sum() > grid[-1, -1].sum()
    else:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape != (n, 2):
            raise ValueError(f"expected a ({n}, 2) reference, got shape {ref.shape}")
        flip = float(np.dot(grid[-1, -1] - grid[0, 0], ref[-1] - ref[0])) < 0.0
    if flip:
        points = points[::-1]
    return points.copy()


def detect_board(gray: np.ndarray, board: BoardSpec, *, refine: str = "subpix") -> Optional[np.ndarray]:
    """The board's inner corners in `gray` as (rows*cols, 2) float64 px in normal order, or None.

    The detector runs on a contrast-stretched copy, as the legacy calibration does.
    "subpix" refines on the original grey, whose values the stretch has not bent.
    "legacy" refines on the image the legacy calibration refines on: the stretched copy after
    adjust_gamma(gamma=2), for which refine_corners' fixed Canny thresholds were tuned.
    """
    _check_refine(refine)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    stretched = _detection_image(gray)
    if stretched is None:
        return None
    found, corners = cv2.findChessboardCorners(stretched, (board.cols, board.rows))
    if not found:
        return None
    if refine == "subpix":
        corners = cv2.cornerSubPix(gray, corners, _SUBPIX_WINDOW, (-1, -1), _SUBPIX_CRITERIA)
        return normalize_corner_order(corners, board)
    legacy_image = adjust_gamma(stretched, gamma=_LEGACY_GAMMA)
    window = int(round(1.5 * average_chessboard_distance(corners, (board.cols, board.rows))))
    try:
        refined = [
            np.asarray(refine_corners(legacy_image, c, window), float).ravel() for c in corners.reshape(-1, 2)
        ]
    except (cv2.error, np.linalg.LinAlgError) as err:
        # The legacy line fit fails when a corner's window holds too few edge points or two
        # parallel lines. A board with an unrefinable corner is treated as not found.
        logger.debug("legacy corner refinement failed: %s", err)
        return None
    return normalize_corner_order(np.array(refined), board)


def extract_observations(
    videos: Mapping[str, Path],
    motor_positions: Sequence[np.ndarray],
    board: BoardSpec,
    *,
    frames_per_waypoint: int = 5,
    refine: str = "subpix",
) -> ExtractionResult:
    """One observation per distinct waypoint from each camera's capture video.

    The board is detected in the `frames_per_waypoint` frames at the middle of each
    segment, and a camera's corners for the waypoint are the per-corner median over the
    frames in which it was found, all in the order of the camera's first detection.
    Raises WaypointAlignmentError, naming the camera, when a camera's stationary segments
    do not number exactly the distinct waypoints, and naming both cameras when a segment
    starts more than 3 frames apart in two of them. A board found in none of the frames,
    or whose detections disagree by more than 1 px within the segment, only drops that
    camera from that waypoint and is listed in `rejected`.
    """
    _check_refine(refine)
    if frames_per_waypoint < 1:
        raise ValueError(f"frames_per_waypoint must be at least 1, got {frames_per_waypoint}")
    distinct = distinct_waypoints(motor_positions)

    segments: Dict[str, List[Tuple[int, int]]] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for cam, video in videos.items():
        segments[cam], stats[cam] = _segment_video(Path(video), min_frames=_MIN_FRAMES, scale=_SCALE)
        if len(segments[cam]) != len(distinct):
            raise WaypointAlignmentError(
                f"{cam}: expected {len(distinct)} stationary segments, found {len(segments[cam])}"
            )
    _check_segments_agree(segments)

    corners: List[Dict[str, np.ndarray]] = [{} for _ in distinct]
    rejected: List[Tuple[int, str, str]] = []
    scatter: Dict[str, float] = {}
    for cam, video in videos.items():
        windows = [_middle_frames(segment, frames_per_waypoint) for segment in segments[cam]]
        detections = _detect_in_windows(Path(video), windows, board, refine)
        kept_scatter = []
        reference = None
        for i, (index, found) in enumerate(zip(distinct, detections)):
            if not found:
                rejected.append((index, cam, _NOT_FOUND))
                continue
            if reference is None:
                reference = found[0]
            # (frames, corners, 2), every frame in the order of the camera's first detection
            stack = np.stack([normalize_corner_order(c, board, reference=reference) for c in found])
            # Per corner, the RMS distance of its detections from their mean.
            spread = np.sqrt(((stack - stack.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))
            if spread.max() > _MAX_CORNER_STD_PX:
                rejected.append((index, cam, _INCONSISTENT))
                continue
            corners[i][cam] = np.median(stack, axis=0)
            kept_scatter.append(float(np.median(spread)))
        scatter[cam] = float(np.median(kept_scatter)) if kept_scatter else float("nan")

    observations = [
        WaypointObservation(index=index, motor_mm=np.asarray(motor_positions[index], float).copy(), corners=c)
        for index, c in zip(distinct, corners)
    ]
    for index, cam, reason in rejected:
        logger.info("waypoint %d: %s camera rejected (%s)", index, cam, reason)
    return ExtractionResult(
        observations=observations,
        segments=segments,
        rejected=rejected,
        corner_scatter_px=scatter,
        segmentation_stats=stats,
    )


def _check_segments_agree(segments: Mapping[str, List[Tuple[int, int]]]) -> None:
    """Raise WaypointAlignmentError when a segment starts more than _MAX_START_OFFSET_FRAMES
    apart in two cameras. The counts already match."""
    cams = list(segments)
    for cam in cams[1:]:
        for i, ((first, _), (other, _)) in enumerate(zip(segments[cams[0]], segments[cam])):
            if abs(first - other) > _MAX_START_OFFSET_FRAMES:
                raise WaypointAlignmentError(
                    f"segment {i}: {cams[0]} starts at frame {first} and {cam} at frame {other};"
                    f" synchronised cameras must agree within {_MAX_START_OFFSET_FRAMES} frames"
                )


def _check_refine(refine: str) -> None:
    if refine not in _REFINEMENTS:
        raise ValueError(f"refine must be one of {_REFINEMENTS}, got {refine!r}")


def _detection_image(gray: np.ndarray) -> Optional[np.ndarray]:
    """The legacy contrast stretch (create_corner_matrix) without its final gamma step, or
    None for a frame with no contrast to stretch, which cannot hold a board."""
    g = gray.astype(np.float64) / 255.0
    low_in, high_in = np.percentile(g, [5, 95])
    if high_in <= low_in:
        return None
    low_out, high_out = stretchlim_like_matlab(g, tol=(0.01, 0.99))
    # Some pixels are above low_in and some below high_in, so 0 < mean < 1 and gamma is finite.
    gamma = (np.log(np.mean(g)) / np.log(0.5)) * 1.5
    adjusted = imadjust_like_matlab(g, (low_in, high_in), (low_out, high_out), gamma)
    return np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)


def _middle_frames(segment: Tuple[int, int], count: int) -> range:
    """The `count` frames centred in the half-open `segment`, or all of it if it is shorter."""
    start, stop = segment
    count = min(count, stop - start)
    first = start + (stop - start - count) // 2
    return range(first, first + count)


def _detect_in_windows(video: Path, windows: List[range], board: BoardSpec, refine: str) -> List[List[np.ndarray]]:
    """Per window, the board detections in its frames; frames without a board are left out.

    The video is read through once rather than seeking, so frame numbers match the
    sequential read that found the segments.
    """
    owner = {t: i for i, window in enumerate(windows) for t in window}
    found: List[List[np.ndarray]] = [[] for _ in windows]
    last = max(owner, default=-1)
    for t, gray in enumerate(_grey_frames(video)):
        if t > last:
            break
        i = owner.get(t)
        if i is not None:
            corners = detect_board(gray, board, refine=refine)
            if corners is not None:
                found[i].append(corners)
    return found


def _grey_frames(video: Path) -> Iterator[np.ndarray]:
    """Every decodable frame of `video` in order, as uint8 grey."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"could not open the video {video}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        capture.release()
