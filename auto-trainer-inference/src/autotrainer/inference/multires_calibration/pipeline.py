"""Calibrate every camera preset from one paused-waypoint capture.

A capture folder holds what the capture wrote:

    waypoints.yaml          the WaypointLog: the commanded motor position of every waypoint,
                            whether the motor reported arriving there, and the times
    source_videos/<camera>.mp4  each camera at full sensor resolution, bin 1, while the
                            stage paused at each waypoint in turn. The cameras are the
                            log's `cameras`, 2 to 6 of them; a log without them is left and right
    source_videos/<camera>_timestamps.txt  each camera's timestamps file, whose column 0 is
                            every frame's wall-clock time (a capture made before 2026-11-20
                            has none)

calibrate_capture turns it into a calibration folder (store.py):

1. The board is detected at each paused waypoint (extract.py) and tagged with the waypoint's
   COMMANDED motor position. The waypoint's frames are those of its logged dwell when every
   camera has a timestamps file, and its stationary segment by motion otherwise. The fit measures the motor scale against the board's squares,
   so a measured position would hide exactly the error the fit is there to find.
2. Each camera is held to the per-camera rule (per_camera_rule): enough views, spread over
   the stage, enough of them shared with another calibrated camera, and connected to the
   others through shared views. A camera that fails it is not calibrated, so it is 2D only:
   the report names it and the reason, first, and the rest are fitted without it. Fewer
   than 2 calibrated cameras refuse the calibration (CalibrationRefused).
3. Every calibrated camera is put in one 180-degree board order
   (fit.reconcile_corner_order); the report notes a camera that had to be reversed.
4. The calibrated cameras are fitted jointly, once, in sensor pixels (fit.fit_motor_frame),
   and every preset's intrinsics are mapped from that fit (geometry.map_intrinsics) when
   the store writes it: an anipose TOML for every calibrated camera in sensor pixels and
   for each preset set, plus the left-right extras where left and right share a geometry.
5. Each binned preset is cross-checked: the frames the detection used are pooled the way the
   camera bins them (INTER_AREA by the binning) and cropped to the preset's ROI, the board is
   detected there and the preset is fitted directly in its own pixels, starting from the
   mapped sensor fit. The report records the mean offset of the preset's corners from the
   sensor corners mapped into it, which checks the mapping (graded), and the largest
   difference between the direct fit's K and the mapped one, which is limited by fit noise
   (informational). A bin-1 preset is a crop of the frames the sensor fit used, with the same
   detections shifted, so it would only check the arithmetic of the shift and is not
   cross-checked. A preset whose direct fit fails gets an undetermined (inf) K difference
   and a note, which warns; its corner offset is graded all the same, and the calibration
   is not aborted.
6. The report compares with the previous calibration when that is one of these folders, at
   format 2 or 3, on the cameras both calibrated; a legacy folder has no motor-frame fit to
   compare with.
7. The calibration is written with the coverage images (one per capture camera, calibrated
   or not) and waypoints.yaml, both before its index (store.write_calibration's
   extra_files). The source videos are not copied: the
   index's capture.capture_folder names the capture folder, absolute, where they stay.

Errors are not caught here: WaypointAlignmentError from extraction, CalibrationRefused
from the per-camera rule, RuntimeError and ValueError from the sensor fit and
FileExistsError for an existing output folder reach the caller, which reports them to the
operator. None of them leaves anything written.
"""

import dataclasses
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from autotrainer.core.logging import get_verbose_logger
# The cross-check must detect in exactly the frames extraction used, so it takes extraction's
# own frame window (_middle_frames) and sequential reader (_grey_frames) rather than copies.
from autotrainer.inference.multires_calibration.extract import (
    _MIN_FRAMES,
    _grey_frames,
    _middle_frames,
    FROM_DWELLS,
    MAX_CAMERAS,
    MIN_CAMERAS,
    detect_board,
    extract_observations,
    normalize_corner_order,
)
from autotrainer.inference.multires_calibration.fit import (
    BoardSpec,
    CaptureCamera,
    MotorFrameFit,
    WaypointObservation,
    fit_motor_frame,
    reconcile_corner_order,
)
from autotrainer.inference.multires_calibration.geometry import (
    PresetGeometry,
    PresetSet,
    map_intrinsics,
    presets_for_cameras,
    sensor_to_preset_points,
)
from autotrainer.inference.multires_calibration.report import (
    CameraVerdict,
    CaptureSummary,
    PresetCrossCheck,
    PreviousCalibration,
    QualityReport,
    build_report,
    write_coverage_images,
)
from autotrainer.inference.multires_calibration.store import (
    CalibrationIndex,
    is_multires_calibration,
    load_sensor_fit,
    write_calibration,
)

logger = get_verbose_logger(__name__)

WAYPOINT_LOG = "waypoints.yaml"
SOURCE_VIDEOS = "source_videos"
# source_videos/<camera><TIMESTAMPS_SUFFIX>: the camera's timestamps file as the recorder
# wrote it, "frame_time, fps, camera stamp, perf_counter, frame id" per frame. frame_time is
# time.time() at the first frame plus frame-id steps at the configured fps, the clock
# WaypointLog.times uses.
TIMESTAMPS_SUFFIX = "_timestamps.txt"

# The capture settings recorded when waypoints.yaml has none: the settings the capture sets
# (gain 1.0 dB, camera gamma off, 4500 us exposure, 40 fps).
_DEFAULT_CAPTURE_SETTINGS = {"gain_db": 1.0, "gamma": None, "exposure_us": 4500, "fps": 40}
# Frames detected per waypoint, in extraction and again in the cross-check, so both see the
# same frames.
_FRAMES_PER_WAYPOINT = 5
# The WaypointLog fields that hold one entry per waypoint, beside `motor`.
_PER_WAYPOINT_FIELDS = ("reported", "verified", "dcs_moves", "times")
# A waypoint whose reported position is further than this from its commanded one, on any
# axis, was not where the fit puts the board (mm). The christielab10 motor check
# (2026-10-26) measured arrivals within 0.007 mm, and a stage stopped at its travel limit
# 1.72 mm short.
UNREACHED_TOLERANCE_MM = 0.1
# The per-camera rule (per_camera_rule): a camera needs MIN_VIEWS kept board views, and
# MIN_COVISIBLE of them shared with another calibrated camera. The motor frame makes shared
# views unnecessary to the fit itself; they are required to expose stage geometry the model
# leaves out (axis skew).
MIN_VIEWS = 12
MIN_COVISIBLE = 8
# Commanded positions that differ by no more than this on an axis (mm) do not span it.
_SPAN_TOLERANCE_MM = 1e-6


class CalibrationRefused(RuntimeError):
    """Fewer than 2 cameras meet the per-camera rule, so there is nothing to calibrate.
    `verdicts` is every camera's CameraVerdict, and the message names each failed camera
    and its reason."""

    def __init__(self, verdicts: Mapping[str, CameraVerdict]):
        self.verdicts = dict(verdicts)
        calibrated = [v.name for v in self.verdicts.values() if v.calibrated]
        reasons = "; ".join(f"{v.name}: {v.reason}" for v in self.verdicts.values() if not v.calibrated)
        super().__init__(
            f"the calibration needs at least {MIN_CAMERAS} calibrated cameras and has"
            f" {len(calibrated)}{' (' + ', '.join(calibrated) + ')' if calibrated else ''}: {reasons}"
        )


def _default_cameras() -> List[CaptureCamera]:
    """The cameras of a log that names none: every capture before 2026-11-20 was left and right."""
    return [CaptureCamera("left", None, 0), CaptureCamera("right", None, 1)]


@dataclass
class WaypointLog:
    """What the capture logged at each waypoint, in the order the stage visited them."""

    motor: List[Tuple[float, float, float]]  # commanded motor coordinates
    reported: List[Optional[Tuple[float, float, float]]]  # the motor's last_position; None if it reported none
    # True when the motor reported the commanded position (within 0.05 mm, in time). That is
    # the firmware's word, not a traced arrival: last_position echoes the target about 0.1 s
    # after the command (christielab10 motor check, 2026-10-26).
    verified: List[bool]
    dcs_moves: List[Tuple[str, float]]  # (axis, DCS value) of the move to the waypoint
    times: List[Tuple[float, float]]  # (arrive, leave), time.time()
    # gain_db, gamma, exposure_us and fps as the capture set them; empty when not logged
    capture_settings: dict = field(default_factory=dict)
    # The capture's cameras, in order; 2 to 6 (calibrate_capture checks)
    cameras: List[CaptureCamera] = field(default_factory=_default_cameras)


def capture_cameras(log: WaypointLog) -> List[str]:
    """The names of the log's cameras, in order."""
    return [camera.name for camera in log.cameras]


def write_waypoint_log(path: Path, log: WaypointLog) -> None:
    """Write `log` as plain YAML. Raises ValueError when its lists do not all have one entry per waypoint."""
    _check_waypoint_log(log)
    data = {
        "motor": [[float(v) for v in p] for p in log.motor],
        "reported": [None if p is None else [float(v) for v in p] for p in log.reported],
        "verified": [bool(v) for v in log.verified],
        "dcs_moves": [[str(axis), float(value)] for axis, value in log.dcs_moves],
        "times": [[float(arrive), float(leave)] for arrive, leave in log.times],
        "capture_settings": _yaml_safe(dict(log.capture_settings)),
        "cameras": [
            {"name": str(c.name), "serial": None if c.serial is None else str(c.serial),
             "camera_id": None if c.camera_id is None else int(c.camera_id)}
            for c in log.cameras
        ],
    }
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None), encoding="utf-8")


def load_waypoint_log(path: Path) -> WaypointLog:
    """The log in `path`. Raises ValueError when a field is missing or garbled, or when its
    lists do not all have one entry per waypoint."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    try:
        log = WaypointLog(
            motor=[_triple(p) for p in data["motor"]],
            reported=[None if p is None else _triple(p) for p in data["reported"]],
            verified=[bool(v) for v in data["verified"]],
            dcs_moves=[(str(axis), float(value)) for axis, value in data["dcs_moves"]],
            times=[(float(arrive), float(leave)) for arrive, leave in data["times"]],
            capture_settings=dict(data.get("capture_settings") or {}),
            cameras=_default_cameras() if data.get("cameras") is None else [
                CaptureCamera(
                    name=str(c["name"]),
                    serial=None if c.get("serial") is None else str(c["serial"]),
                    camera_id=None if c.get("camera_id") is None else int(c["camera_id"]),
                )
                for c in data["cameras"]
            ],
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{path} is not a waypoint log: {type(error).__name__}: {error}") from error
    _check_waypoint_log(log)
    return log


def load_frame_times(capture_dir: Path, cameras: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
    """camera -> column 0 of source_videos/<camera>_timestamps.txt, every frame's wall-clock
    time, when every camera in `cameras` has the file; None when none has, as before 2026-11-20.

    Raises FileNotFoundError when only some have it: the cameras would be segmented two ways.
    """
    paths = {cam: Path(capture_dir, SOURCE_VIDEOS, f"{cam}{TIMESTAMPS_SUFFIX}") for cam in cameras}
    present = [cam for cam, path in paths.items() if path.is_file()]
    if not present:
        return None
    if len(present) != len(paths):
        missing = ", ".join(str(path) for cam, path in paths.items() if cam not in present)
        raise FileNotFoundError(f"the capture has no timestamps file {missing}")
    return {cam: np.loadtxt(path, delimiter=",", usecols=0, ndmin=1) for cam, path in paths.items()}


def per_camera_rule(observations: Sequence[WaypointObservation], cameras: Sequence[str]) -> Dict[str, CameraVerdict]:
    """Each camera's verdict, in the order of `cameras`. A camera is calibrated when

    1. it kept at least MIN_VIEWS board views ("too few board views: {n} of 12 needed");
    2. their commanded positions vary on each of x, y and z ("board views do not span all 3
       stage axes");
    3. at least MIN_COVISIBLE of its views were also seen by another calibrated camera ("too
       few views shared with another camera: {n} of 8 needed"). This is repeated until
       nothing changes, so a camera whose shared views are only with cameras the rule
       dropped is dropped too;
    4. it is in the largest group of calibrated cameras connected through shared views ("not
       connected to the other cameras through shared views"). Two cameras are connected
       when they saw the board at a common waypoint. Of equal groups, the one holding the
       camera listed first wins.

    The first rule a camera fails gives its reason. Dropping a camera from the shared views
    (rule 3) can only lower the others' counts, so the cameras left are the largest set in
    which every camera meets it; rule 4 then changes no count, since no shared view crosses
    between groups.
    """
    cameras = list(cameras)
    reasons: Dict[str, str] = {}
    for cam in cameras:
        motor = np.array([np.asarray(o.motor_mm, float).reshape(3) for o in observations if cam in o.corners])
        if len(motor) < MIN_VIEWS:
            reasons[cam] = f"too few board views: {len(motor)} of {MIN_VIEWS} needed"
        elif not np.all(np.ptp(motor, axis=0) > _SPAN_TOLERANCE_MM):
            reasons[cam] = "board views do not span all 3 stage axes"

    def seen_by_calibrated(o: WaypointObservation) -> List[str]:
        return [cam for cam in cameras if cam in o.corners and cam not in reasons]

    while True:
        shared = {cam: 0 for cam in cameras if cam not in reasons}
        for o in observations:
            seen = seen_by_calibrated(o)
            if len(seen) > 1:
                for cam in seen:
                    shared[cam] += 1
        dropped = {
            cam: f"too few views shared with another camera: {n} of {MIN_COVISIBLE} needed"
            for cam, n in shared.items()
            if n < MIN_COVISIBLE
        }
        if not dropped:
            break
        reasons.update(dropped)

    # Connected groups, by merging the groups of the cameras that share each waypoint.
    group = {cam: {cam} for cam in cameras if cam not in reasons}
    for o in observations:
        seen = seen_by_calibrated(o)
        for cam in seen[1:]:
            if group[cam] is not group[seen[0]]:
                merged = group[seen[0]] | group[cam]
                for member in merged:
                    group[member] = merged
    if group:
        # max keeps the first of equal groups, and `group` is in camera order.
        largest = max(group.values(), key=len)
        for cam in group:
            if cam not in largest:
                reasons[cam] = "not connected to the other cameras through shared views"
    return {cam: CameraVerdict(cam, cam not in reasons, reasons.get(cam, "")) for cam in cameras}


def unreached_waypoints(log: WaypointLog, tolerance: float = UNREACHED_TOLERANCE_MM) -> List[int]:
    """The waypoints whose reported position is more than `tolerance` mm from the commanded
    one on some axis. A waypoint with no finite reported position carries no evidence
    either way and is not counted."""
    unreached = []
    for k, (motor, reported) in enumerate(zip(log.motor, log.reported)):
        if reported is None or not all(math.isfinite(v) for v in reported):
            continue
        if max(abs(r - m) for r, m in zip(reported, motor)) > tolerance:
            unreached.append(k)
    return unreached


def calibrate_capture(
    capture_dir: Path,
    *,
    output_dir: Path,
    presets: Sequence[PresetGeometry],
    board: BoardSpec = BoardSpec(),
    previous_dir: Optional[Path] = None,
    refine: str = "subpix",
    preset_sets: Optional[Sequence[PresetSet]] = None,
    live_geometries: Optional[Mapping[str, PresetGeometry]] = None,
) -> Tuple[CalibrationIndex, QualityReport]:
    """Calibrate every preset in `presets` from the capture in `capture_dir` and write the
    calibration to `output_dir`, which must not exist yet.

    The base preset, whose sensor ROI the report grades coverage and the principal point
    against, is the one with the largest binning: the live inference preset. Every preset
    from presets_from_base covers the same sensor window. `presets` are cross-checked.
    `preset_sets` (geometry.presets_for_cameras) are the per-camera geometries written;
    without them each preset is a set of every calibrated camera at that geometry.
    `live_geometries` (camera -> its live geometry) gives the sets instead, built after the
    fit from the calibrated cameras only (calibrated_preset_sets), so a camera left 2D only
    neither duplicates a set nor costs the others a binning; it must name every camera.
    `previous_dir` is the calibration this one replaces, if any. `refine` is
    extract.detect_board's corner refinement.

    Only the cameras that meet per_camera_rule are fitted; the report's `cameras` gives
    every camera's verdict. Raises CalibrationRefused, writing nothing, when fewer than 2 do.
    """
    capture_dir, output_dir = Path(capture_dir), Path(output_dir)
    # Refused before minutes of extraction and fitting, not after.
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; a calibration folder is never overwritten")
    presets = list(presets)
    if not presets:
        raise ValueError("no presets to calibrate")
    log = load_waypoint_log(capture_dir / WAYPOINT_LOG)
    cameras = tuple(capture_cameras(log))
    if not MIN_CAMERAS <= len(cameras) <= MAX_CAMERAS:
        raise ValueError(f"a capture has {MIN_CAMERAS} to {MAX_CAMERAS} cameras, the log lists {len(cameras)}")
    if len(set(cameras)) != len(cameras):
        raise ValueError(f"the log names a camera twice: {', '.join(cameras)}")
    videos = {cam: capture_dir / SOURCE_VIDEOS / f"{cam}.mp4" for cam in cameras}
    for video in videos.values():
        if not video.is_file():
            raise FileNotFoundError(f"the capture has no video {video}")
    image_size = _video_size(videos)
    for g in presets:
        _check_inside_sensor(g, image_size)
    for preset_set in preset_sets or ():
        for g in preset_set.cameras.values():
            _check_inside_sensor(g, image_size)
    for g in (live_geometries or {}).values():
        _check_inside_sensor(g, image_size)

    motor = [np.asarray(p, float) for p in log.motor]
    frame_times = load_frame_times(capture_dir, cameras)
    extraction = extract_observations(
        videos, motor, board, frames_per_waypoint=_FRAMES_PER_WAYPOINT, refine=refine,
        frame_times=frame_times, dwells=None if frame_times is None else log.times,
    )
    verdicts = per_camera_rule(extraction.observations, cameras)
    calibrated = tuple(cam for cam in cameras if verdicts[cam].calibrated)
    for v in verdicts.values():
        if not v.calibrated:
            logger.warning("calibration capture: %s is not calibrated (2D only): %s", v.name, v.reason)
    if len(calibrated) < MIN_CAMERAS:
        raise CalibrationRefused(verdicts)
    if live_geometries is not None:
        if preset_sets is not None:
            raise ValueError("give preset_sets or live_geometries, not both")
        preset_sets = calibrated_preset_sets(live_geometries, calibrated)
    # The fit, the corner order and the cross-check see only the calibrated cameras. Every
    # waypoint stays, even one no calibrated camera saw, so observations[i] is still
    # extraction's segments[cam][i].
    kept = [
        WaypointObservation(
            index=o.index,
            motor_mm=o.motor_mm,
            corners={cam: c for cam, c in o.corners.items() if cam in calibrated},
        )
        for o in extraction.observations
    ]
    observations, reversed_order = reconcile_corner_order(kept, board, image_size, calibrated)
    logger.info("calibration capture: %d distinct waypoints; fitting %d of %d cameras",
                len(observations), len(calibrated), len(cameras))
    fit = fit_motor_frame(observations, board, image_size, calibrated)

    notes: List[str] = []
    cross_checks = _cross_checks(
        {cam: videos[cam] for cam in calibrated}, extraction.segments, observations, fit, presets, board, refine, notes
    )
    for cam, flipped in reversed_order.items():
        if flipped:
            notes.append(
                f"{cam}: board corner order reversed to match the other camera's"
                " (the board looks the same turned 180 degrees)"
            )

    # The fit tags each view with its commanded position, so a waypoint the motor reported
    # elsewhere puts its view in the wrong place; the report fails it.
    unreached = unreached_waypoints(log)
    for k in unreached:
        off = max(abs(r - m) for r, m in zip(log.reported[k], log.motor[k]))
        notes.append(
            f"waypoint {k}: the motor reported {_format_xyz(log.reported[k])}, {off:.2f} mm from the"
            f" commanded {_format_xyz(log.motor[k])}"
        )
    # How many frames extraction had at each waypoint, against what it needs: the margin the
    # settle and the dwell leave, which only a real capture measures.
    still = _still_frames(extraction.segments, extraction.observations)
    from_dwells = extraction.segments_from == FROM_DWELLS
    if still:
        notes.append(
            ("frames per logged dwell: " if from_dwells else "still frames per waypoint: ")
            + "; ".join(f"{cam} min {s['min']} (waypoint {s['min_waypoint']}), median {s['median']:g}"
                        for cam, s in still.items())
            + f" (segments shorter than {_MIN_FRAMES} frames are "
            + ("refused)" if from_dwells else "not found)")
        )

    base = max(presets, key=lambda g: g.binning)
    rejections: Dict[str, Dict[str, int]] = {cam: {} for cam in cameras}
    for _, cam, reason in extraction.rejected:
        rejections[cam][reason] = rejections[cam].get(reason, 0) + 1
    summary = CaptureSummary(
        waypoints_total=len(log.motor),
        waypoints_verified=sum(log.verified),
        waypoints_unreached=len(unreached),
        views_kept={cam: sum(cam in o.corners for o in extraction.observations) for cam in cameras},
        rejections=rejections,
    )
    previous = _previous(previous_dir, notes)
    report = build_report(fit, observations, board, base.sensor_roi(), summary, cross_checks, previous,
                          presets=presets, cameras=verdicts)
    report.notes.extend(notes)

    capture = {
        **_DEFAULT_CAPTURE_SETTINGS,
        **log.capture_settings,
        "waypoints_total": len(log.motor),
        "waypoints_distinct": len(observations),
        "waypoints_verified": sum(log.verified),
        "waypoints_unreached": len(unreached),
        "corner_refinement": refine,
        "segmentation_stats": extraction.segmentation_stats,
        "segments_from": extraction.segments_from,
        "still_frames": still,
        "corner_order_reversed": reversed_order,
        "corner_scatter_px": extraction.corner_scatter_px,
        # Where the source videos stay: they are not copied into the calibration, which the
        # install copies again and every backup keeps (two 1440x1080 videos a month).
        "capture_folder": str(capture_dir.resolve()),
    }
    with tempfile.TemporaryDirectory(prefix="calibration_coverage_") as tmp:
        # Every camera's, from extraction: a 2D-only camera's shows what it did see. The
        # hulls do not depend on the corner order.
        coverage = write_coverage_images(extraction.observations, image_size, base.sensor_roi(), Path(tmp), cameras)
        extra_files = {path.name: path for path in coverage}
        extra_files[WAYPOINT_LOG] = capture_dir / WAYPOINT_LOG
        index = write_calibration(
            output_dir,
            fit=fit,
            presets=presets,
            preset_sets=preset_sets,
            cameras=log.cameras,
            board=board,
            report=report,
            capture=_yaml_safe(capture),
            extra_files=extra_files,
        )
    logger.info("calibration %s written to %s: %s", index.id, output_dir, report.verdict)
    return index, report


def calibrated_preset_sets(live: Mapping[str, PresetGeometry], calibrated: Sequence[str]) -> List[PresetSet]:
    """presets_for_cameras over the `calibrated` cameras' live geometries only.

    Built from all the capture's cameras, a set would be dropped for a binning only a 2D-only
    camera's window misses, and two sets could hold the same geometries once that camera is
    left out. Raises ValueError when `live` has no geometry for a calibrated camera.
    """
    missing = [cam for cam in calibrated if cam not in live]
    if missing:
        raise ValueError(f"no live geometry for {', '.join(missing)}")
    return presets_for_cameras({cam: live[cam] for cam in calibrated})


def _cross_checks(
    videos: Mapping[str, Path],
    segments: Mapping[str, List[Tuple[int, int]]],
    observations: Sequence[WaypointObservation],
    fit: MotorFrameFit,
    presets: Sequence[PresetGeometry],
    board: BoardSpec,
    refine: str,
    notes: List[str],
) -> List[PresetCrossCheck]:
    """A PresetCrossCheck per binned preset, in the order of `presets`; a failed fit adds a note.

    `videos` holds the calibrated cameras, the fit's. observations[i] is the waypoint of every
    camera's segments[cam][i], as extraction returned them. A camera is detected only at the
    waypoints where it was kept, in the same frames extraction used there, and its corners are the per-corner median over the frames
    that found the board. Each detection is put in the order of the sensor detection mapped
    into the preset, so the direct fit labels the board as the sensor fit did.
    """
    binned = [g for g in presets if g.binning > 1]
    # preset name -> per waypoint -> camera -> the detections in its frames
    found: Dict[str, List[Dict[str, List[np.ndarray]]]] = {g.name: [{} for _ in observations] for g in binned}
    for cam, video in videos.items():
        windows = {
            i: _middle_frames(segments[cam][i], _FRAMES_PER_WAYPOINT)
            for i, o in enumerate(observations)
            if cam in o.corners
        }
        owner = {t: i for i, window in windows.items() for t in window}
        last = max(owner, default=-1)
        # One sequential read per camera for every preset, as extraction reads it, so frame
        # numbers match and no frame is held longer than its detections take.
        for t, gray in enumerate(_grey_frames(video)):
            if t > last:
                break
            i = owner.get(t)
            if i is None:
                continue
            for g in binned:
                corners = detect_board(_preset_frame(gray, g), board, refine=refine)
                if corners is not None:
                    reference = sensor_to_preset_points(observations[i].corners[cam], g)
                    ordered = normalize_corner_order(corners, board, reference=reference)
                    found[g.name][i].setdefault(cam, []).append(ordered)

    checks = []
    for g in binned:
        preset_observations = [
            WaypointObservation(
                index=o.index,
                motor_mm=np.array(o.motor_mm, dtype=float),
                corners={cam: np.median(np.stack(c), axis=0) for cam, c in found[g.name][i].items()},
            )
            for i, o in enumerate(observations)
        ]
        offset = _corner_offset(preset_observations, observations, g)
        try:
            # Started from the sensor fit mapped to the preset, not from the planar start,
            # which is degenerate in focal length: from it, the 512 preset's fit on the
            # synthetic capture ran out of evaluations. The start does not move the optimum:
            # from the mapped K, from one 1 % and 2 px off, and from the edge-convention
            # mapping, the fit reached the same K to 1e-4 px.
            direct = fit_motor_frame(
                preset_observations, board, (g.width, g.height), tuple(videos), initial=_mapped_fit(fit, g)
            )
        except (RuntimeError, ValueError) as error:
            logger.warning("cross-check %s: the direct fit failed: %s", g.name, error)
            notes.append(f"cross-check {g.name}: the direct fit in the preset's pixels failed: {error}")
            checks.append(PresetCrossCheck(geometry=g, max_intrinsic_delta_px=math.inf, corner_offset_px=offset))
            continue
        delta = max(
            _intrinsic_delta(direct.cameras[cam].K, map_intrinsics(fit.cameras[cam].K, g)) for cam in videos
        )
        checks.append(PresetCrossCheck(geometry=g, max_intrinsic_delta_px=delta, corner_offset_px=offset))
    return checks


def _still_frames(
    segments: Mapping[str, List[Tuple[int, int]]], observations: Sequence[WaypointObservation]
) -> Dict[str, Dict[str, float]]:
    """Per camera, the length in frames of its stationary segments: the shortest ("min"), the
    waypoint it belongs to ("min_waypoint", as the waypoint log numbers it) and the median.

    segments[cam][i] is distinct waypoint observations[i], as extraction returned them.
    """
    still = {}
    for cam, ranges in segments.items():
        lengths = [stop - start for start, stop in ranges]
        if not lengths:
            continue
        shortest = int(np.argmin(lengths))
        still[cam] = {
            "min": int(lengths[shortest]),
            "median": float(np.median(lengths)),
            "min_waypoint": int(observations[shortest].index),
        }
    return still


def _corner_offset(
    preset_observations: Sequence[WaypointObservation],
    observations: Sequence[WaypointObservation],
    g: PresetGeometry,
) -> float:
    """|mean of (corner detected in preset g's frames - the sensor detection mapped into g)|
    over every corner of every camera and waypoint the preset found, in preset px; NaN when
    it found none.

    It tests the mapping where the K delta cannot: a convention error shifts every corner
    alike, so the mean exposes it, while detection noise averages away over thousands of
    corners (sem about 0.0007 px on the synthetic capture).
    """
    differences = [
        p.corners[cam] - sensor_to_preset_points(o.corners[cam], g)
        for p, o in zip(preset_observations, observations)
        for cam in p.corners
    ]
    if not differences:
        return math.nan
    return float(np.linalg.norm(np.concatenate(differences).mean(axis=0)))


def _mapped_fit(fit: MotorFrameFit, g: PresetGeometry) -> MotorFrameFit:
    """`fit` with every camera's K in preset `g`'s pixels. Binning and cropping change only K:
    the distortion, the poses, the board rotation and the motor scale stay as they are."""
    cameras = {cam: dataclasses.replace(m, K=map_intrinsics(m.K, g)) for cam, m in fit.cameras.items()}
    return dataclasses.replace(fit, cameras=cameras, image_size=(g.width, g.height))


def _preset_frame(gray: np.ndarray, g: PresetGeometry) -> np.ndarray:
    """A full-sensor frame as preset `g` records it: binned by area averaging, then cropped to the ROI."""
    height, width = gray.shape
    pooled = cv2.resize(gray, (width // g.binning, height // g.binning), interpolation=cv2.INTER_AREA)
    return pooled[g.offset_y:g.offset_y + g.height, g.offset_x:g.offset_x + g.width]


def _intrinsic_delta(K_direct: np.ndarray, K_mapped: np.ndarray) -> float:
    """max |dfx|, |dfy|, |dcx|, |dcy|, in preset px."""
    a, b = np.asarray(K_direct, float), np.asarray(K_mapped, float)
    return float(max(abs(a[0, 0] - b[0, 0]), abs(a[1, 1] - b[1, 1]), abs(a[0, 2] - b[0, 2]), abs(a[1, 2] - b[1, 2])))


def _previous(previous_dir: Optional[Path], notes: List[str]) -> Optional[PreviousCalibration]:
    """The calibration to compare with. A folder without a format-2 or format-3 index is legacy.
    The report compares the cameras the two have in common (report._change_metrics), so a
    format-2 (left and right) previous serves a calibration of more cameras.

    A format-2 or 3 folder whose fit cannot be read is compared with nothing, and a note says
    why: it is only the reference for the change metrics, and a damaged previous folder
    must not stop the rig from being calibrated again.
    """
    if previous_dir is None:
        return None
    previous_dir = Path(previous_dir)
    if not is_multires_calibration(previous_dir):
        return PreviousCalibration(legacy=True, fit=None)
    try:
        return PreviousCalibration(legacy=False, fit=load_sensor_fit(previous_dir))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        logger.warning("the previous calibration in %s cannot be compared: %s", previous_dir, error)
        notes.append(
            f"the previous calibration in {previous_dir} could not be read"
            f" ({type(error).__name__}: {error}), so the change since it is not reported"
        )
        return PreviousCalibration(legacy=False, fit=None)


def _video_size(videos: Mapping[str, Path]) -> Tuple[int, int]:
    """(width, height) shared by every video; raises ValueError when they differ."""
    sizes = {}
    for cam, video in videos.items():
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise OSError(f"could not open the video {video}")
        try:
            sizes[cam] = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            capture.release()
    if len(set(sizes.values())) != 1:
        raise ValueError(f"the capture videos differ in size: {sizes}")
    return next(iter(sizes.values()))


def _check_inside_sensor(g: PresetGeometry, image_size: Tuple[int, int]) -> None:
    """The capture is the full sensor at bin 1, so every preset's ROI must lie inside it."""
    x0, y0, x1, y1 = g.sensor_roi()
    width, height = image_size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"preset {g.name} at ROI offset ({g.offset_x}, {g.offset_y}) covers sensor px"
            f" ({x0:g}, {y0:g})-({x1:g}, {y1:g}), outside the {width}x{height} capture"
        )


def _check_waypoint_log(log: WaypointLog) -> None:
    n = len(log.motor)
    for name in _PER_WAYPOINT_FIELDS:
        count = len(getattr(log, name))
        if count != n:
            raise ValueError(f"the waypoint log has {count} {name} entries for {n} waypoints")


def _triple(values) -> Tuple[float, float, float]:
    x, y, z = (float(v) for v in values)
    return x, y, z


def _format_xyz(values) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in values) + ")"


def _yaml_safe(value):
    """`value` as the plain types yaml.safe_dump writes: numpy values become Python ones and
    tuples become lists. NaN (a corner scatter with no kept waypoint, the statistics of an
    empty video) becomes None, which reads back as "not determined" rather than as a float
    that compares unequal to itself."""
    if isinstance(value, np.ndarray):
        return _yaml_safe(value.tolist())
    if isinstance(value, np.generic):
        return _yaml_safe(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return value
