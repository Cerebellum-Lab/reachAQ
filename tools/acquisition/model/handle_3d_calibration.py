import math
import shutil
import time
import dataclasses
from typing import Dict, Any, Optional, List, Mapping, Sequence, Tuple
from pathlib import Path

from autotrainer.core import Offset3DTuple, get_verbose_logger, ProjectInterval
from autotrainer.inference import calibration_FLIR
from autotrainer.pyside.content_widget import InvokeMethod
from autotrainer.video import VideoRecordMode
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from autotrainer.inference.multires_calibration.pipeline import (
    SOURCE_VIDEOS,
    TIMESTAMPS_SUFFIX,
    WAYPOINT_LOG,
    WaypointLog,
    calibrate_capture,
    write_waypoint_log,
)
from autotrainer.inference.multires_calibration.report import QualityReport
from autotrainer.inference.multires_calibration.store import CalibrationIndex

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.calibration_geometry import live_inference_geometry
from tools.acquisition.model.video_capture_model import VideoCaptureModel

logger = get_verbose_logger(__name__)

x, y, z = 'xyz'

start = (
    (x, -9.4),
    (y, 10.0),
    (z, -35.0),
)

moves = [
    (z, -30.0),
    (x, -5.0),
    (y, 15.0),
    (z, -35.0),
    (x, -1.0),
    (z, -27.0),
    (x, -9.0),
    (y, 20.0),
    (z, -35.0),
    (x, 0.0),
    (z, -25.0),
    (y, 25.0),
    (x, -9.0),
    (z, -35.0),
    (z, -30.0),
    (x, 5.0),
    (z, -25.0),
    (y, 30.0),
    (x, -7.0),
    (z, -35.0),
    (z, -27.0),
    (y, 35.0),
    (x, 10.0),
    (z, -22.0),
    (x, -5.0),
    # move through center:
    (z, -27.0),
    (x, -3.0),
    (y, 30.0),
    (z, -30.0),
    (x, -5.0),
    (y, 25.0),
    (z, -30.0),
    (x, -9.0),
    (y, 20.0),
    (z, -35.0),
    (y, 15.0),
]


# start = [
#     (x, -11.7),
#     (y, 38.5),
#     (z, -1.9),
# ]
#
#
# moves = [
#     (x, 0),
#     (z, -4),
#     (x, -11.7),
#     (z, -1.9),
# ]


default_params = dict(
    vbin=1,
    hbin=1,
    width=1440,
    height=1080,
    exposure=4500,
    fps=40,
    # primary=cam is left,
    offsetx=0,
    offsety=0,
    # 1 dB, and the camera's gamma off. The URL carries gamma=None, which SpinCam reads as
    # None (ast.literal_eval) and so disables gamma. Both must be set: left out, SpinCam
    # cannot open the camera at all, since VideoManager.parse_params fills in the class's
    # non-string default (0.7 for gamma), which SpinCam.set_property's ast.literal_eval
    # rejects ("Could not create camera").
    gain=1.0,
    gamma=None,
)

AXES = "xyz"

# The stage's lower travel limit on every axis, in motor units (mm). HardwareModel._set_axis
# clamps SET targets below it (hardware_model.py, "limited move to 0"); _move_axis sends MOVE
# targets as they are, but the firmware stops there: in the 2026-10-26 motor check on
# christielab10 a MOVE to x -1.722 left the stage at 0.000.
MOTOR_LOWER_LIMIT = 0.0

# The settle after each move's ack, before the arrival check and the dwell (ruling R52):
#
#     settle = max(SETTLE_FLOOR_S, distance / v_max + v_max / a_max + SETTLE_MARGIN_S)
#
# with v_max (mm/s) and a_max (mm/s^2) the moving axis's, from the motor configurations
# HardwareModel wrote at connect (~/Autotrainer/motor_config.yaml), and SETTLE_FALLBACK_S when
# the axis has none. The 2026-10-26 motor check on christielab10 showed last_position is the
# firmware echoing the target about 0.1 s after the command, so the arrival check cannot see
# the stage still moving; the ack came 0.20-0.30 s after the command for 2.7-5 mm moves, which
# fits "ack at motion end" but was not shown for long moves. The settle therefore covers the
# whole move as if the ack meant only "received": distance / v_max + v_max / a_max is a
# trapezoidal profile's duration, an upper bound for a move too short to reach v_max, and the
# margin is for the stage coming to rest. Without it a 17 mm move at 20 mm/s would still be
# moving through most of the 0.5 s dwell, leaving fewer still frames than extraction needs.
SETTLE_MARGIN_S = 0.25
SETTLE_FLOOR_S = 0.25
SETTLE_FALLBACK_S = 1.0


def capture_settings_of(params: Mapping[str, Any]) -> Dict[str, Any]:
    """The capture settings the waypoint log records, from the camera params the capture sets.

    Only the params present are recorded: the log never claims a value the capture did not
    set. (gamma=None present is recorded; it turns the camera's gamma off.) A capture that
    left gain or gamma out would not run, though: SpinCam fails to open a camera without
    them (see default_params).
    """
    names = {"gain_db": "gain", "gamma": "gamma", "exposure_us": "exposure", "fps": "fps"}
    return {setting: params[param] for setting, param in names.items() if param in params}


def _dcs_axis_to_motor(dcs: DiamondTriangleOffsetConfig, axis: str, value: float) -> float:
    # Exactly as the moves were converted before waypoints (and as set_dcs_x/y/z in
    # hardware_model.py convert): the DCS value on its own axis, zero on the other two, and
    # that axis of the result. diamond_to_motor is per axis, a translation and the fixed
    # flips (1, -1, -1), so the zeros never reach the axis read back.
    i = AXES.index(axis)
    dcs_xyz = [0, 0, 0]
    dcs_xyz[i] = value
    return float(dcs.diamond_to_motor(Offset3DTuple(*dcs_xyz))[i])


def plan_waypoints(
    start: Sequence[Tuple[str, float]],
    moves: Sequence[Tuple[str, float]],
    dcs: DiamondTriangleOffsetConfig,
) -> List[Tuple[str, float, Tuple[float, float, float]]]:
    """(axis, DCS value, commanded motor xyz after the move) for the start and each move.

    Waypoint 0 is the start position, which the start's moves, one per axis, reach
    together; its (axis, DCS value) is the last of them. Each move then changes one axis of
    the waypoint before, so every motor xyz is the whole position the stage is commanded
    to, in motor units (mm). Raises ValueError when the start does not set every axis.

    A motor coordinate below MOTOR_LOWER_LIMIT is commanded at the limit, where the stage
    stops anyway, so the commands and the waypoint log hold where the stage really is; the
    calibration fits on those positions. The clamped waypoints are logged. A move that
    clamping leaves where the stage already was stays in the plan and is logged: the log
    keeps one entry per move, and calibrate_capture collapses a run of equal positions into
    the one stationary segment it is, as it does the no-op move already in `moves`.
    """
    if {axis for axis, _ in start} != set(AXES):
        raise ValueError(f"the start {tuple(start)} must set every axis of {AXES}")
    target = [math.nan] * 3  # where the DCS puts the stage, before clamping
    for axis, value in start:
        target[AXES.index(axis)] = _dcs_axis_to_motor(dcs, axis, value)
    last_axis, last_value = start[-1]
    steps = [(last_axis, last_value), *moves]
    plan, targets = [], []
    for k, (axis, value) in enumerate(steps):
        if k > 0:
            target[AXES.index(axis)] = _dcs_axis_to_motor(dcs, axis, value)
        targets.append(tuple(target))
        plan.append((axis, float(value), tuple(max(t, MOTOR_LOWER_LIMIT) for t in target)))
    _log_clamping(plan, targets)
    return plan


def _log_clamping(plan, targets) -> None:
    clamped = [
        f"waypoint {k}: {axis} {t:.3f} -> {MOTOR_LOWER_LIMIT:.3f}"
        for k, target in enumerate(targets)
        for axis, t in zip(AXES, target)
        if t < MOTOR_LOWER_LIMIT
    ]
    if clamped:
        n_clamped = sum(1 for target in targets if min(target) < MOTOR_LOWER_LIMIT)
        logger.warning(
            "%d of %d waypoints are below the stage's travel limit %.1f and are commanded at it: %s",
            n_clamped, len(plan), MOTOR_LOWER_LIMIT, "; ".join(clamped))
    for k in range(1, len(plan)):
        if plan[k][2] == plan[k - 1][2] and targets[k] != targets[k - 1]:
            logger.warning(
                "waypoint %d (%s %s) is clamped where waypoint %d left it, so the stage does not move;"
                " the calibration takes the two as one waypoint", k, plan[k][0], plan[k][1], k - 1)


def _reported_position(hard) -> Optional[Tuple[float, float, float]]:
    # NaN until the motor has reported the axis (after connect), so a partial report is none.
    position = hard.last_position
    if position is None or not all(math.isfinite(v) for v in position):
        return None
    return tuple(float(v) for v in position)


def wait_until_arrived(
    hard,
    target_xyz: Sequence[float],
    *,
    tolerance: float = 0.05,
    timeout: float = 2.0,
    poll: float = 0.01,
    sleep=time.sleep,
    now=time.monotonic,
) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
    """Poll hard.last_position until every axis is within `tolerance` of `target_xyz`.

    Returns (arrived, the last reported position). The position is None while the motor
    has reported none on some axis, which is never arrived. After `timeout` seconds without
    arriving it returns False.
    """
    t_start = now()
    while True:
        reported = _reported_position(hard)
        if reported is not None and all(abs(r - t) <= tolerance for r, t in zip(reported, target_xyz)):
            return True, reported
        if now() - t_start >= timeout:
            return False, reported
        sleep(poll)


def axis_limits(motor_configurations) -> Dict[str, Optional[Tuple[float, float]]]:
    """Per axis, (maximum velocity mm/s, maximum acceleration mm/s^2) of its stepper in
    `motor_configurations` (HardwareModel.motor_configurations); None for an axis whose values
    are missing or not positive and finite, and for every axis when there are none."""
    limits = {}
    for axis in AXES:
        try:
            _, config = getattr(motor_configurations, f"{axis}_config")
            values = (float(config.maximum_velocity), float(config.maximum_acceleration))
        except (AttributeError, TypeError, ValueError):
            limits[axis] = None
            continue
        limits[axis] = values if all(math.isfinite(v) and v > 0 for v in values) else None
    return limits


def settle_time(distance: float, limits: Optional[Tuple[float, float]]) -> float:
    """The settle after a move of `distance` mm on an axis with `limits` (axis_limits); see SETTLE_MARGIN_S."""
    if limits is None:
        return SETTLE_FALLBACK_S
    v_max, a_max = limits
    return max(SETTLE_FLOOR_S, abs(distance) / v_max + v_max / a_max + SETTLE_MARGIN_S)


def _move_settle(
    limits: Mapping[str, Optional[Tuple[float, float]]],
    axes: Sequence[str],
    target: Sequence[float],
    origin: Optional[Sequence[float]],
) -> float:
    """The settle for moving `axes` from `origin` to `target`: the slowest axis's, or the
    fallback when where the stage started is unknown."""
    if origin is None:
        return SETTLE_FALLBACK_S
    return max(
        settle_time(target[AXES.index(axis)] - origin[AXES.index(axis)], limits[axis]) for axis in axes
    )


def _command(hard, axis: str, motor_value: float) -> None:
    move = {"x": hard.move_x, "y": hard.move_y, "z": hard.move_z}[axis]
    token = move(motor_value)
    if token is None:
        raise RuntimeError(f"the move of {axis} to {motor_value} was not queued")
    hard.wait_pending_command_acked(token)


def run_waypoints(
    hard,
    plan: Sequence[Tuple[str, float, Tuple[float, float, float]]],
    *,
    settle: Optional[float] = None,
    dwell: float = 0.5,
    sleep=time.sleep,
    now=time.time,
    clock=time.monotonic,
    tolerance: float = 0.05,
    arrival_timeout: float = 2.0,
    capture_settings: Optional[Mapping[str, Any]] = None,
) -> WaypointLog:
    """Visit each waypoint of `plan` (plan_waypoints) in turn and log it.

    At each waypoint: command the move and wait for its ack, settle, check the motor's
    reported position (wait_until_arrived), then dwell, so the video holds the board still
    there long enough for calibrate_capture to find it. Waypoint 0 commands every axis of
    the start. A waypoint the motor does not confirm is logged unverified and the capture
    goes on; the calibration report counts them.

    The settle is per move, from the distance and hard.motor_configurations (settle_time);
    waypoint 0's starts from the position the motor last reported, and is the fallback when
    there is none. A `settle` given in seconds replaces it for every move, for tests.

    `now` stamps arrive and leave (time.time(), as WaypointLog.times records). `clock`
    times the arrival wait, so a step of the wall clock cannot stretch or cut it short; a
    test may pass one fake for both. `capture_settings` are those of default_params unless
    given.
    """
    if capture_settings is None:
        capture_settings = capture_settings_of(default_params)
    limits = axis_limits(getattr(hard, "motor_configurations", None))
    if settle is None:
        logger.info("settling each move at the motor's configured speed, (v_max mm/s, a_max mm/s^2): %s",
                    ", ".join(f"{axis} {limits[axis]}" for axis in AXES))
    log = WaypointLog(motor=[], reported=[], verified=[], dcs_moves=[], times=[],
                      capture_settings=dict(capture_settings))
    origin = _reported_position(hard)  # where prepare() left the stage, if the motor said
    for i, (axis, value, motor) in enumerate(plan):
        moved = AXES if i == 0 else (axis,)
        for move_axis in moved:
            _command(hard, move_axis, motor[AXES.index(move_axis)])
        wait = _move_settle(limits, moved, motor, origin) if settle is None else settle
        logger.debug("waypoint %d: %s to %s, settle %.3f s", i, "".join(moved), motor, wait)
        sleep(wait)
        origin = motor  # the next move starts where this one was commanded to
        arrived, reported = wait_until_arrived(
            hard, motor, tolerance=tolerance, timeout=arrival_timeout, sleep=sleep, now=clock)
        arrive = now()
        if not arrived:
            logger.warning("waypoint %d: the motor reported %s, not within %s of the commanded %s",
                           i, reported, tolerance, motor)
        sleep(dwell)
        leave = now()
        log.motor.append(tuple(motor))
        log.reported.append(reported)
        log.verified.append(arrived)
        log.dcs_moves.append((axis, value))
        log.times.append((arrive, leave))
    return log


def wait_cams_capture_status(cameras, capture_status: CaptureProcessStatus, timeout: float = 3):
    """Wait until every camera reports `capture_status`, all within `timeout` seconds.

    Raises RuntimeError naming the first camera that did not.
    """
    p_timeout = time.perf_counter() + timeout
    for cam in cameras:
        if not cam.wait_for_capture_status(capture_status, timeout=p_timeout - time.perf_counter()):
            raise RuntimeError(f"{cam.name}: no {capture_status.name} within {timeout} s")
        logger.info("%s: got %s", cam.name, capture_status)


def capture_presets(live_params: Mapping[str, Any]) -> List[PresetGeometry]:
    """Every preset the capture calibrates: the presets covering the live camera geometry.

    `live_params` are the left camera's configured params, read before the capture replaces
    them with its own bin-1 full-frame params, which are no base for the binned presets.
    """
    return presets_from_base(live_inference_geometry(live_params))


def calibrate_waypoint_capture(
    capture_dir: Path,
    log: WaypointLog,
    *,
    session_dir: Path,
    presets: Sequence[PresetGeometry],
    previous_dir: Optional[Path],
) -> Tuple[Path, CalibrationIndex, QualityReport]:
    """Write `log` as the capture's waypoints.yaml beside its source_videos, then calibrate
    every preset into <session_dir>/<DEFAULT_3D_CALIB_DIR_NAME>.new.

    Returns (that folder, the calibration's index, its quality report). The folder is
    named here only, and its installer takes it from the return value.
    calibrate_capture's errors reach the caller.
    """
    write_waypoint_log(Path(capture_dir, WAYPOINT_LOG), log)
    output_dir = Path(session_dir, f"{DEFAULT_3D_CALIB_DIR_NAME}.new")
    index, report = calibrate_capture(
        Path(capture_dir),
        output_dir=output_dir,
        presets=presets,
        previous_dir=previous_dir,
    )
    return output_dir, index, report


def make_3d_calib(
    app_model: AppModel,
    cam_params: Optional[Dict[str, Any]] = None,
    record_mode: VideoRecordMode = VideoRecordMode.TRIGGER,
) -> Tuple[Path, Path, CalibrationIndex, QualityReport]:
    """Capture the board paused at every waypoint, then calibrate every preset from it.

    The cameras record in TRIGGER mode, as a session's do. On christielab10 (2026-11-19)
    START_CONTINUOUS never reached RECORDING ("left: no RECORDING within 5 s", three runs),
    while TRIGGER recorded every waypoint.

    Returns (the capture folder, the folder the new calibration was written to, its index,
    its quality report). That folder is <session>/<DEFAULT_3D_CALIB_DIR_NAME>.new; installing
    it is left to the caller.
    """
    if cam_params is None:
        cam_params = default_params
    # The DCS as loaded: a new 3D calibration leaves it STALE, held off its consumers until
    # DCS is redone, and the next 3D calibration must still be possible then. The waypoints
    # only need to keep the board in view; the fit uses the commanded motor positions,
    # whichever DCS placed them.
    diamond_triangle_cfg = app_model.behavior.algorithm.loaded_diamond_triangle_config
    if diamond_triangle_cfg is None or not diamond_triangle_cfg.fully_valid:
        raise RuntimeError(f"Please first calibrate diamond-triangle by starting the acquisition")
    plan = plan_waypoints(start, moves, diamond_triangle_cfg)
    left = app_model.left_camera
    right = app_model.right_camera
    # Read before prepare() replaces the camera configurations with the capture's own.
    presets = capture_presets(left.active_config.params)
    cameras: List[VideoCaptureModel] = []
    # put primary first:
    for camera in (left, right):
        if camera.is_primary:
            cameras.insert(0, camera)
        else:
            cameras.append(camera)
    cams_before_cfg = tuple(camera.save_configuration() for camera in cameras)
    hard = app_model.hardware
    #
    project = app_model.make_project_info()
    project.calculate_next_session_index()
    #
    sess_path = project.get_session_path()
    #
    square_size = 4  # in millimeters
    row_ct = 6
    col_ct = 8
    # If the frames were sampled at a higher resolution than will be used for
    # normal acquisition, this is the x-fold oversampling factor
    oversample = 4
    src_dir = Path(
        calibration_FLIR.make_new_calibration(square_size, row_ct, col_ct, oversample, sess_path.location))

    def prepare():
        logger.notice("Preparing ..")
        #
        logger.info("Connecting to HW ..")
        hard.connect(app_model.message_handler.input_queue)
        token = hard.send_home()
        hard.wait_pending_command_acked(token)
        #
        # Waypoint 0 is reached before recording starts, so the video opens on it rather
        # than on the move from home, which would add a stationary segment no waypoint has.
        logger.verbose("Setting start position")
        for axis, motor_value in zip(AXES, plan[0][2]):
            _command(hard, axis, motor_value)
        #
        for cam, cfg in zip(cameras, cams_before_cfg):
            params = cam_params.copy()
            params["primary"] = cfg.params.get("primary", "off")
            logger.info("Preparing cam %s with params=%s", cam.name, params)
            new_cfg = dataclasses.replace(
                cfg,
                record_mode=record_mode,
                record_prebuffer_duration=0,
                is_enabled=True,
                is_record_enabled=True,
                params=params,
            )
            cam.load_configuration(new_cfg)
            cam.project = project

        #

        for cam in cameras:
            if cam.is_primary:
                logger.info("%s: prepare capture ..", cam.name)
                if not cam.on_prepare_capture():
                    raise RuntimeError(f"{cam.name}.on_prepare_capture() failed")
                cam.wait_for_capture_status(CaptureProcessStatus.RUNNING, timeout=5)

        for cam in cameras:
            if not cam.is_primary:
                logger.info("%s: prepare capture ..", cam.name)
                if not cam.on_prepare_capture():
                    raise RuntimeError(f"{cam.name}.on_prepare_capture() failed")

        wait_cams_capture_status(cameras, CaptureProcessStatus.RUNNING, timeout=5)

        for cam in cameras:
            if not cam.is_primary:
                logger.info("%s: capture start ..", cam.name)
                cam.on_capture_start()
        for cam in cameras:
            if cam.is_primary:
                logger.info("%s: capture start ..", cam.name)
                cam.on_capture_start()

        if record_mode == VideoRecordMode.TRIGGER:
            logger.info("Triggering recording")
            for cam in cameras:
                cam.on_trigger_recording(True)

        logger.notice("Waiting RECORDING on cams")
        wait_cams_capture_status(cameras, CaptureProcessStatus.RECORDING, 5)

    def run() -> WaypointLog:
        logger.notice("Running 3d calib ..")
        logger.info("Now executing calib moves, pausing at each waypoint ..")

        waypoint_log = run_waypoints(hard, plan, capture_settings=capture_settings_of(cam_params))

        logger.success("executed %s moves; the motor reported %d of %d waypoints at their commanded position",
                       len(moves), sum(waypoint_log.verified), len(waypoint_log.verified))

        if record_mode == VideoRecordMode.TRIGGER:
            logger.info("Requesting cameras stop recording")
            for cam in reversed(cameras):
                cam.on_trigger_recording(False)
            wait_cams_capture_status(cameras, CaptureProcessStatus.RUNNING, 15)
        return waypoint_log

    failed = None
    log = None
    try:
        prepare()
        log = run()
    except Exception as err:
        logger.exception("Could not prepare and run 3d calib: %s", err)
        failed = err
    finally:
        for camera in reversed(cameras):
            logger.info("%s: notify end cam", camera.name)
            camera.on_capture_notify_end()
        for camera in reversed(cameras):
            logger.info("%s: stopping cam", camera.name)
            camera.on_capture_stop()
        hard.disconnect()
        try:
            wait_cams_capture_status(cameras, CaptureProcessStatus.TERMINATED, 5)
        except RuntimeError as err:
            # The configurations are restored all the same, but videos that may still be
            # being written are not calibrated from.
            logger.error("Cameras did not stop: %s", err)
            if failed is None:
                failed = err
        logger.verbose("Resetting cams to previous config")
        for camera, cam_cfg in zip(cameras, cams_before_cfg):
            camera.load_configuration(cam_cfg)

    if failed is not None:
        raise failed

    for camera in cameras:
        video_path, timestamps_path, *_ = project.get_video_path(
            camera.name,
            allow_overwrite=True,
            interval=ProjectInterval.HOUR if record_mode == VideoRecordMode.CONTINUOUS else ProjectInterval.NONE
        )
        vp = Path(video_path)
        target = src_dir.joinpath(f"{SOURCE_VIDEOS}/{camera.name}.mp4")
        logger.verbose("%s -> %s", vp.as_posix(), target.as_posix())
        vp.rename(target)
        # Extraction takes each waypoint's frames by the times its dwell was logged, from
        # this file; a copy, so the session keeps its own.
        ts_target = src_dir.joinpath(SOURCE_VIDEOS, f"{camera.name}{TIMESTAMPS_SUFFIX}")
        logger.verbose("%s -> %s", timestamps_path, ts_target.as_posix())
        shutil.copyfile(timestamps_path, ts_target)

    logger.notice("Calibrating every preset from the capture in %s", src_dir)
    output_dir, index, report = calibrate_waypoint_capture(
        src_dir,
        log,
        session_dir=Path(sess_path.location),
        presets=presets,
        previous_dir=app_model.active_calibration_dir,
    )

    logger.success("Successfully calibrated the capture: %s, verdict %s", index.id, report.verdict)

    return src_dir, output_dir, index, report
