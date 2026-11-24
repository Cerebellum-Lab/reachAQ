import math
import shutil
import time
import uuid
import dataclasses
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List, Mapping, Sequence, Tuple
from pathlib import Path
from types import SimpleNamespace

import cv2
import yaml

from reachaq.core import Offset3DTuple, get_verbose_logger, ProjectInterval
from reachaq.inference import calibration_FLIR
from reachaq.pyside.content_widget import InvokeMethod
from reachaq.video import VideoRecordMode
from reachaq.core.capture import CaptureProcessStatus
from reachaq.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from reachaq.device import MotorConfigurationFile, stepper_travel_refusal
from reachaq.inference.multires_calibration.fit import BoardSpec, CaptureCamera
from reachaq.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from reachaq.inference.multires_calibration.library import add_entry, entry_path, read_settings
from reachaq.inference.multires_calibration.pipeline import (
    MAX_CAMERAS,
    MIN_CAMERAS,
    REFERENCE_DIR,
    SOURCE_VIDEOS,
    TIMESTAMPS_SUFFIX,
    WAYPOINT_LOG,
    WaypointLog,
    calibrate_capture,
    load_frame_times,
    write_waypoint_log,
)
from reachaq.inference.multires_calibration.report import QualityReport
from reachaq.inference.multires_calibration.scout import (
    DEFAULT_COUNTS,
    MAX_PATH,
    MIN_VIEWS,
    SCOUT_MOVE_LIST_FILE,
    Bounds,
    ScoutAborted,
    ScoutResult,
    Travel,
    bounds_from_plan,
    choose_move_list,
    configured_board,
    configured_bounds,
    diagnose,
    load_move_list,
    merge_advice,
    save_move_list,
    scout_grid,
    selection_shortfalls,
    scout_views_from_capture,
    video_sizes,
)
from reachaq.inference.multires_calibration.store import INDEX_FILE, CalibrationIndex, load_index

from reachaq.app.model.app_model import AppModel
from reachaq.app.model.calibration_geometry import camera_serial, live_inference_geometry
from reachaq.app.model.video_capture_model import VideoCaptureModel

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
# clamps SET targets below it (hardware_model.py, "limited move to 0"). Without a configured
# travel, _move_axis sends MOVE targets as they are, and the firmware stops there: in the
# 2026-10-26 motor check on christielab10 a MOVE to x -1.722 left the stage at 0.000. With
# one (min_pos / max_pos in motor_config.yaml), a MOVE outside it is refused.
MOTOR_LOWER_LIMIT = 0.0

# The settle after each move's ack, before the arrival check and the dwell (ruling R52):
#
#     settle = max(SETTLE_FLOOR_S, distance / v_max + v_max / a_max + SETTLE_MARGIN_S)
#
# with v_max (mm/s) and a_max (mm/s^2) the moving axis's, from the motor configurations
# HardwareModel wrote at connect (~/reachAQ/motor_config.yaml), and SETTLE_FALLBACK_S when
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


def motor_travel() -> Travel:
    """Per axis x, y, z, the stage's (min_pos, max_pos) travel in mm from the motor config
    file HardwareModel loads at connect (MotorConfigurationFile.DEFAULT_LOCATION, parsed as
    load_default_motor_config parses it), None on a side it does not set and on every side
    without a file. Read from the file, so a scout before any connect has it too. A travel
    the file gets wrong raises ValueError naming the axis, as the connect would."""
    location = MotorConfigurationFile.DEFAULT_LOCATION.expanduser()
    if not location.exists():
        return ((None, None),) * 3
    configurations = MotorConfigurationFile.from_file(location)
    return tuple(
        (config.minimum_position, config.maximum_position)
        for _, config in (configurations.x_config, configurations.y_config, configurations.z_config)
    )


def check_plan_in_travel(
    plan: Sequence[Tuple[str, float, Tuple[float, float, float]]], travel: Travel, which: str
) -> None:
    """Raise RuntimeError when a motor position of `plan` lies outside `travel` (motor_travel),
    checked as HardwareModel checks a move. Called before anything is homed or recorded, so a
    list the stage cannot follow is refused up front, not at its first bad waypoint mid-capture."""
    outside = []
    for k, (_, _, motor) in enumerate(plan):
        for axis, value, (minimum, maximum) in zip(AXES, motor, travel):
            limits = SimpleNamespace(minimum_position=minimum, maximum_position=maximum)
            if stepper_travel_refusal(axis, value, limits) is not None:
                low = -math.inf if minimum is None else minimum
                high = math.inf if maximum is None else maximum
                outside.append(f"waypoint {k} {axis} {value:.3f} mm (travel {low:g}-{high:g} mm)")
    if outside:
        more = f" and {len(outside) - 3} more" if len(outside) > 3 else ""
        name = "hand-written" if which == HANDWRITTEN else which
        raise RuntimeError(f"the {name} move list leaves the configured travel: "
                           f"{'; '.join(outside[:3])}{more}. Nothing was moved")


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
    cameras: Optional[Sequence[CaptureCamera]] = None,
    strict: bool = False,
) -> WaypointLog:
    """Visit each waypoint of `plan` (plan_waypoints) in turn and log it.

    At each waypoint: command the move and wait for its ack, settle, check the motor's
    reported position (wait_until_arrived), then dwell, so the video holds the board still
    there long enough for calibrate_capture to find it. Waypoint 0 commands every axis of
    the start. A waypoint the motor does not confirm is logged unverified and the capture
    goes on; the calibration report counts them.

    `strict` (the scout) stops at once instead, raising ScoutAborted and sending no further
    command: on a command that is not queued or acknowledged (a CAN error raises there), a
    motor driver fault after any command or arrival, or a waypoint the motor does not
    confirm. The stage is left where it stopped.

    The settle is per move, from the distance and hard.motor_configurations (settle_time);
    waypoint 0's starts from the position the motor last reported, and is the fallback when
    there is none. A `settle` given in seconds replaces it for every move, for tests.

    `now` stamps arrive and leave (time.time(), as WaypointLog.times records). `clock`
    times the arrival wait, so a step of the wall clock cannot stretch or cut it short; a
    test may pass one fake for both. `capture_settings` are those of default_params unless
    given. `cameras` are the capture's cameras (capture_camera_list), left and right unless
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
    if cameras is not None:
        log.cameras = list(cameras)
    origin = _reported_position(hard)  # where prepare() left the stage, if the motor said
    for i, (axis, value, motor) in enumerate(plan):
        moved = AXES if i == 0 else (axis,)
        if strict:
            _abort_on_fault(hard, f"waypoint {i}")  # a fault latched since the last check: no command
        for move_axis in moved:
            if not strict:
                _command(hard, move_axis, motor[AXES.index(move_axis)])
                continue
            try:
                _command(hard, move_axis, motor[AXES.index(move_axis)])
            except Exception as err:
                raise ScoutAborted(f"waypoint {i}: the move of {move_axis} to {motor[AXES.index(move_axis)]:.3f}"
                                   f" failed: {err}") from err
            _abort_on_fault(hard, f"waypoint {i}")
        wait = _move_settle(limits, moved, motor, origin) if settle is None else settle
        logger.debug("waypoint %d: %s to %s, settle %.3f s", i, "".join(moved), motor, wait)
        sleep(wait)
        origin = motor  # the next move starts where this one was commanded to
        arrived, reported = wait_until_arrived(
            hard, motor, tolerance=tolerance, timeout=arrival_timeout, sleep=sleep, now=clock)
        arrive = now()
        if strict:
            if not arrived:
                raise ScoutAborted(f"waypoint {i}: the motor reported {reported}, not the commanded {motor}")
            _abort_on_fault(hard, f"waypoint {i}")
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


SCOUT = "scout"
HANDWRITTEN = "handwritten"
# The scout's list replaces the hand-written one only with at least this many chosen points.
MIN_SCOUT_POINTS = MIN_VIEWS
# The scout's pause at each grid point: about 12 frames at 40 fps, of which one is searched.
SCOUT_DWELL_S = 0.3
SCOUT_DIR_NAME = "scout"


class ScoutCancelled(RuntimeError):
    """The operator chose Cancel at the scout's advice; nothing was captured."""


def _abort_on_fault(hard, where: str) -> None:
    """Raise ScoutAborted when the motor reports a driver fault on any axis."""
    faulted = tuple(getattr(hard, "pellet_axes_with_driver_fault", ()) or ())
    if faulted:
        raise ScoutAborted(f"{where}: motor driver fault on {', '.join(faulted)}")


def scout_plan(
    points: Sequence[Tuple[float, float, float]], dcs: DiamondTriangleOffsetConfig
) -> List[Tuple[str, float, Tuple[float, float, float]]]:
    """A scout move list as plan_waypoints gives one: (axis, DCS value, motor xyz).

    `points` are motor positions, each one axis from the one before (order_path); waypoint
    0's axis is z, as the hand-written start's last move is. Raises ValueError when two
    points in a row differ in other than one axis.
    """
    plan = []
    for k, point in enumerate(points):
        point = tuple(float(v) for v in point)
        if k == 0:
            axis = AXES[-1]
        else:
            changed = [a for j, a in enumerate(AXES) if point[j] != plan[-1][2][j]]
            if len(changed) != 1:
                raise ValueError(f"scout point {k} {point} is not one axis from {plan[-1][2]}")
            axis = changed[0]
        i = AXES.index(axis)
        value = float(dcs.motor_to_diamond(Offset3DTuple(*point))[i])
        plan.append((axis, value, point))
    return plan


def choose_plan(
    move_list: Optional[str], dcs: DiamondTriangleOffsetConfig, scout: Optional[ScoutResult]
) -> Tuple[List[Tuple[str, float, Tuple[float, float, float]]], str]:
    """(the plan, which list it is): the scout's when `move_list` is "scout" and the scout
    chose MIN_SCOUT_POINTS or more points, otherwise the hand-written list."""
    if move_list == SCOUT and scout is not None and len(scout.selection.points) >= MIN_SCOUT_POINTS:
        return scout_plan(scout.points, dcs), SCOUT
    if move_list == SCOUT:
        chosen = 0 if scout is None else len(scout.selection.points)
        logger.warning("the scout chose %d points, fewer than %d: using the hand-written move list",
                       chosen, MIN_SCOUT_POINTS)
    return plan_waypoints(start, moves, dcs), HANDWRITTEN


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
    """The presets the capture's report grades against: those covering one camera's live geometry.

    `live_params` are the primary camera's configured params, read before the capture
    replaces them with its own bin-1 full-frame params, which are no base for the binned presets.
    """
    return presets_from_base(live_inference_geometry(live_params))


def capture_live_geometries(cameras: Sequence[VideoCaptureModel]) -> Dict[str, PresetGeometry]:
    """camera name -> its own live geometry, from its configured params; the calibration's
    preset sets are made from these for the cameras it calibrates (presets_for_cameras)."""
    return {camera.name: live_inference_geometry(camera.active_config.params) for camera in cameras}


def capture_camera_list(cameras: Sequence[VideoCaptureModel]) -> List[CaptureCamera]:
    """The waypoint log's cameras: each one's name, its serial (camera_serial: the configured
    host of a Spinnaker camera, else None) and its camera id."""
    return [
        CaptureCamera(camera.name, camera_serial(camera.active_config),
                      None if camera.camera_id is None else int(camera.camera_id))
        for camera in cameras
    ]


def calibrate_waypoint_capture(
    capture_dir: Path,
    log: WaypointLog,
    *,
    session_dir: Path,
    presets: Sequence[PresetGeometry],
    previous_dir: Optional[Path],
    live_geometries: Optional[Mapping[str, PresetGeometry]] = None,
    board: Optional[BoardSpec] = None,
) -> Tuple[Path, CalibrationIndex, QualityReport]:
    """Write `log` as the capture's waypoints.yaml beside its source_videos, then calibrate
    every preset into <session_dir>/<DEFAULT_3D_CALIB_DIR_NAME>.new. With `live_geometries`
    (capture_live_geometries), the preset sets are each calibrated camera's own. `board` is
    the calibration board (scout.configured_board); None leaves calibrate_capture's default,
    the checkerboard.

    Returns (that folder, the calibration's index, its quality report). The folder is
    named here only, and its installer takes it from the return value.
    calibrate_capture's errors reach the caller.
    """
    write_waypoint_log(Path(capture_dir, WAYPOINT_LOG), log)
    output_dir = Path(session_dir, f"{DEFAULT_3D_CALIB_DIR_NAME}.new")
    extra = {} if live_geometries is None else {"live_geometries": dict(live_geometries)}
    if board is not None:
        extra["board"] = board
    index, report = calibrate_capture(
        Path(capture_dir),
        output_dir=output_dir,
        presets=presets,
        previous_dir=previous_dir,
        **extra,
    )
    return output_dir, index, report


def _prepare_capture(app_model, hard, cameras, cams_before_cfg, cam_params, record_mode, project, start_xyz,
                     *, strict: bool = False) -> None:
    """Connect and home the stage, go to `start_xyz`, give the cameras the capture's
    configuration and start them recording: the one camera lifecycle the capture and the
    scout share (_record_plan).

    `strict` (the scout) checks for a motor driver fault before homing, after it, and after
    each start-position command, raising ScoutAborted before the next command.
    """
    logger.notice("Preparing ..")
    #
    logger.info("Connecting to HW ..")
    hard.connect(app_model.message_handler.input_queue)
    if strict:
        _abort_on_fault(hard, "before homing")
    token = hard.send_home()
    hard.wait_pending_command_acked(token)
    if strict:
        _abort_on_fault(hard, "homing")
    #
    # Waypoint 0 is reached before recording starts, so the video opens on it rather
    # than on the move from home, which would add a stationary segment no waypoint has.
    logger.verbose("Setting start position")
    for axis, motor_value in zip(AXES, start_xyz):
        _command(hard, axis, motor_value)
        if strict:
            _abort_on_fault(hard, f"the start move of {axis}")
    #
    _start_cameras(cameras, cams_before_cfg, [cam_params] * len(cameras), record_mode, project)


def _start_cameras(cameras, cams_before_cfg, params_per_camera, record_mode, project) -> None:
    """Give each camera its params in `params_per_camera` (its own primary role kept) and
    start them all recording, primary first: the camera half of _prepare_capture, which the
    reference capture uses alone, with no stage."""
    for cam, cfg, cam_params in zip(cameras, cams_before_cfg, params_per_camera):
        params = dict(cam_params)
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


def _stop_recording(cameras, record_mode) -> None:
    if record_mode == VideoRecordMode.TRIGGER:
        logger.info("Requesting cameras stop recording")
        for cam in reversed(cameras):
            cam.on_trigger_recording(False)
        wait_cams_capture_status(cameras, CaptureProcessStatus.RUNNING, 15)


def _teardown_capture(cameras, hard, cams_before_cfg, failed: Optional[BaseException]) -> Optional[BaseException]:
    """Stop the cameras, disconnect the stage (none to disconnect when `hard` is None) and
    restore the cameras' configurations, on every path; returns `failed`, or the cameras'
    failure to stop when nothing failed before."""
    for camera in reversed(cameras):
        logger.info("%s: notify end cam", camera.name)
        camera.on_capture_notify_end()
    for camera in reversed(cameras):
        logger.info("%s: stopping cam", camera.name)
        camera.on_capture_stop()
    if hard is not None:
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
    return failed


def _collect_videos(project, cameras, record_mode, dest_dir: Path) -> None:
    """Move each camera's video to <dest_dir>/source_videos/<camera>.mp4 and copy its
    timestamps file beside it; extraction takes the waypoints' frames by those times."""
    Path(dest_dir, SOURCE_VIDEOS).mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        video_path, timestamps_path, *_ = project.get_video_path(
            camera.name,
            allow_overwrite=True,
            interval=ProjectInterval.HOUR if record_mode == VideoRecordMode.CONTINUOUS else ProjectInterval.NONE
        )
        vp = Path(video_path)
        target = Path(dest_dir, SOURCE_VIDEOS, f"{camera.name}.mp4")
        logger.verbose("%s -> %s", vp.as_posix(), target.as_posix())
        vp.rename(target)
        # A copy, so the session keeps its own.
        ts_target = Path(dest_dir, SOURCE_VIDEOS, f"{camera.name}{TIMESTAMPS_SUFFIX}")
        logger.verbose("%s -> %s", timestamps_path, ts_target.as_posix())
        shutil.copyfile(timestamps_path, ts_target)


def _record_plan(
    app_model, cameras, cams_before_cfg, cam_params, record_mode, project, plan, capture_list, dest_dir: Path,
    *, label: str = "3d calib", **waypoint_options,
) -> WaypointLog:
    """Record `cameras` while the stage visits `plan`, then put the videos in `dest_dir`.

    The capture and the scout both record through here: _prepare_capture, run_waypoints
    (given `waypoint_options`), the recording stopped, and _teardown_capture on every path.
    The first error is raised after the teardown, and then no video is moved.
    """
    hard = app_model.hardware
    failed = None
    log = None
    try:
        _prepare_capture(app_model, hard, cameras, cams_before_cfg, cam_params, record_mode, project, plan[0][2],
                         strict=bool(waypoint_options.get("strict", False)))
        logger.notice("Running %s ..", label)
        logger.info("Now executing the moves, pausing at each waypoint ..")
        log = run_waypoints(hard, plan, capture_settings=capture_settings_of(cam_params), cameras=capture_list,
                            **waypoint_options)
        logger.success("executed %s moves; the motor reported %d of %d waypoints at their commanded position",
                       len(plan) - 1, sum(log.verified), len(log.verified))
        _stop_recording(cameras, record_mode)
    except Exception as err:
        logger.exception("Could not prepare and run %s: %s", label, err)
        failed = err
    finally:
        failed = _teardown_capture(cameras, hard, cams_before_cfg, failed)

    if failed is not None:
        raise failed
    _collect_videos(project, cameras, record_mode, dest_dir)
    return log


def run_scout(
    app_model: AppModel,
    cameras: Sequence[VideoCaptureModel],
    plan_targets: Sequence[Sequence[float]],
    *,
    project,
    session_dir: Path,
    cams_before_cfg,
    capture_list: Sequence[CaptureCamera],
    cam_params: Optional[Dict[str, Any]] = None,
    record_mode: VideoRecordMode = VideoRecordMode.TRIGGER,
    configured: Optional[Bounds] = None,
    counts: Sequence[int] = DEFAULT_COUNTS,
    board: BoardSpec = BoardSpec(),
    max_path: int = MAX_PATH,
    travel: Optional[Travel] = None,
    **timing,
) -> ScoutResult:
    """A short capture over a coarse grid, then the move list every camera sees.

    The grid is scout_grid inside bounds_from_plan(plan_targets, configured, travel): per
    axis the stage's configured travel (`travel`, motor_travel() when not given), else the
    box the hand-written plan's targets span; never wider than scoutBounds (`configured`)
    either, nor ever outside the travel. It is recorded as the capture is
    (_record_plan, TRIGGER), each point one axis from the one before and paused
    SCOUT_DWELL_S, with run_waypoints strict: a motor fault, a failed command or a point
    the motor does not confirm raises ScoutAborted at once, sending nothing more. The
    videos and timestamps go to <session>/scout/source_videos.

    Each point's frames are taken by its logged dwell and the board looked for in the
    middle one (scout_views_from_capture). The move list is choose_move_list's, and the
    advice diagnose's. `timing` (sleep, now, clock) goes to run_waypoints, for tests.
    """
    if cam_params is None:
        cam_params = default_params
    dcs = app_model.behavior.algorithm.loaded_diamond_triangle_config
    bounds = bounds_from_plan(plan_targets, configured, motor_travel() if travel is None else travel)
    grid = scout_grid(bounds, counts)
    plan = scout_plan(grid, dcs)
    names = [camera.name for camera in cameras]
    logger.notice("Scouting %d points in %s - %s with %s", len(grid), bounds.lo, bounds.hi, ", ".join(names))
    scout_dir = Path(session_dir, SCOUT_DIR_NAME)
    log = _record_plan(app_model, cameras, cams_before_cfg, cam_params, record_mode, project, plan, capture_list,
                       scout_dir, label="scout", dwell=SCOUT_DWELL_S, strict=True, **timing)
    videos = {name: Path(scout_dir, SOURCE_VIDEOS, f"{name}.mp4") for name in names}
    frame_times = load_frame_times(scout_dir, names)
    if frame_times is None:
        raise RuntimeError(f"the scout capture in {scout_dir} has no timestamps files")
    views = scout_views_from_capture(videos, frame_times, log.motor, log.times, board)
    sizes = video_sizes(videos)
    selection, path = choose_move_list(grid, views, names, image_sizes=sizes, max_path=max_path)
    diagnosis = diagnose(views, names, sizes)
    logger.info("scout chose %d points (a path of %d waypoints); views per camera %s; cameras joined: %s",
                len(selection.points), len(path), selection.per_camera, selection.connected)
    for camera, advice in diagnosis.items():
        logger.warning("%s: %s", camera, advice)
    return ScoutResult(points=path, selection=selection, diagnosis=diagnosis, views=views)


def saved_scout_list(root: Optional[Path], identities: Sequence[str], bounds: Bounds):
    """The scout list saved in the library `root` for these cameras, or None when there is
    none, it has fewer than MIN_SCOUT_POINTS points, a point lies outside `bounds` (today's
    scout box: the configured travel, or the hand-written box the DCS may have moved) or two
    in a row differ in other than one axis."""
    if root is None:
        return None
    points = load_move_list(Path(root, SCOUT_MOVE_LIST_FILE), identities)
    if points is None:
        return None
    if len(points) < MIN_SCOUT_POINTS:
        logger.warning("the saved scout move list has %d points, fewer than %d; scouting again",
                       len(points), MIN_SCOUT_POINTS)
        return None
    if not all(lo <= v <= hi for p in points for v, lo, hi in zip(p, bounds.lo, bounds.hi)):
        logger.warning("the saved scout move list leaves today's scout box %s - %s; scouting again",
                       bounds.lo, bounds.hi)
        return None
    if not all(sum(a != b for a, b in zip(p, q)) == 1 for p, q in zip(points, points[1:])):
        logger.warning("the saved scout move list has a move of more than one axis; scouting again")
        return None
    return points


# The reference clip: long enough for every camera to record frames after RECORDING, short
# enough that nothing in view changes. Its middle frame is the reference.
REFERENCE_CLIP_S = 1.0
# Where capture_reference_entry builds the copy before the library takes it in.
REFERENCE_ENTRY_DIR_NAME = "reference_entry"


def middle_frame(video: Path):
    """The middle frame of `video` as a grayscale array, or None when it has no frame."""
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        return None
    frame = frames[len(frames) // 2]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def fit_to_live_geometry(frame, params: Mapping[str, Any]):
    """`frame` area-averaged to the live geometry of a camera configured with `params`, as the
    display and inference queues receive it (frame_fit.QueueFit), when the camera recorded at a
    capture-binning preset k times larger; otherwise as it is."""
    try:
        geometry = live_inference_geometry(params)
    except ValueError:
        return frame
    rows, cols = frame.shape[:2]
    target = (geometry.height, geometry.width)
    if (rows, cols) == target or rows % target[0] or cols % target[1] or rows // target[0] != cols // target[1]:
        return frame
    return cv2.resize(frame, (target[1], target[0]), interpolation=cv2.INTER_AREA)


def capture_reference_images(
    cameras: Sequence[VideoCaptureModel],
    cams_before_cfg,
    project,
    dest_dir: Path,
    *,
    names: Optional[Mapping[str, str]] = None,
    record_mode: VideoRecordMode = VideoRecordMode.TRIGGER,
    duration: Optional[float] = None,
    sleep=time.sleep,
) -> Dict[str, Path]:
    """Record a short clip with `cameras` at each one's live geometry and keep its middle frame.

    Each camera records with its own configuration in `cams_before_cfg` (the one a session
    records with), in `record_mode`, for `duration` seconds (REFERENCE_CLIP_S unless given). Nothing moves: the stage is
    neither connected nor commanded. The camera lifecycle is the capture's (_start_cameras,
    _stop_recording, _teardown_capture on every path). The clips go to
    <dest_dir>/reference/source_videos, and each middle frame to
    <dest_dir>/reference/<name>.png, named by `names` (camera -> its calibration name) or
    the camera's own name, at the live geometry the camera panel shows at Record
    (fit_to_live_geometry). Returns camera name -> that file. Raises the first error after
    the teardown, and RuntimeError for a clip with no frame.
    """
    clip_dir = Path(dest_dir, REFERENCE_DIR)
    duration = REFERENCE_CLIP_S if duration is None else duration
    params = [dict(cfg.params) for cfg in cams_before_cfg]
    failed = None
    try:
        logger.notice("Recording the reference images of %s ..", ", ".join(camera.name for camera in cameras))
        _start_cameras(cameras, cams_before_cfg, params, record_mode, project)
        sleep(duration)
        _stop_recording(cameras, record_mode)
    except Exception as err:
        logger.exception("Could not record the reference images: %s", err)
        failed = err
    finally:
        failed = _teardown_capture(cameras, None, cams_before_cfg, failed)
    if failed is not None:
        raise failed
    _collect_videos(project, cameras, record_mode, clip_dir)
    written = {}
    for camera, cfg in zip(cameras, cams_before_cfg):
        frame = middle_frame(Path(clip_dir, SOURCE_VIDEOS, f"{camera.name}.mp4"))
        if frame is None:
            raise RuntimeError(f"{camera.name}: the reference clip has no frame")
        frame = fit_to_live_geometry(frame, cfg.params)
        path = Path(clip_dir, f"{(names or {}).get(camera.name, camera.name)}.png")
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"{camera.name}: the reference image could not be written to {path}")
        logger.info("%s: reference image %dx%d written to %s", camera.name, frame.shape[1], frame.shape[0], path)
        written[camera.name] = path
    return written


QUALITY_REPORT_JSON = "quality_report.json"
QUALITY_REPORT_TEXT = "quality_report.txt"


def copy_with_references(source: Path, references: Mapping[str, Path], built: Path, *, now: datetime) -> str:
    """Build in `built` a copy of the calibration in `source` that holds `references`
    (calibration camera name -> PNG) as reference/<name>.png, under a new id, and return it.

    The fit and every file but the old references are copied as they are. The index gets
    the new id, parent_id (the source's id) and references_captured (`now`); its created
    stays the fit's, so the calibration is as old as its fit. The anipose TOMLs name the
    calibration by id in their metadata, so the old id is replaced by the new one there,
    and in quality_report.json and quality_report.txt, whose notes also get the line
    "references added to <the source's id, 8 characters>", so the copy's report says what it is.
    Raises ValueError for a source that is not at format 2 or 3 (a legacy import).
    """
    source, built = Path(source), Path(built)
    source_id = load_index(source).id
    new_id = uuid.uuid4().hex
    shutil.copytree(source, built, ignore=lambda folder, names: [REFERENCE_DIR] if Path(folder) == source else [])
    data = yaml.safe_load(Path(built, INDEX_FILE).read_text(encoding="utf-8"))
    data["id"] = new_id
    data["parent_id"] = source_id
    data["references_captured"] = now.replace(microsecond=0).isoformat()
    for toml in built.rglob("*.toml"):
        toml.write_text(toml.read_text(encoding="utf-8").replace(source_id, new_id), encoding="utf-8")
    report_json = Path(built, QUALITY_REPORT_JSON)
    if report_json.is_file():
        report = QualityReport.from_json(report_json.read_text(encoding="utf-8").replace(source_id, new_id))
        report = dataclasses.replace(report, notes=[*report.notes, f"references added to {source_id[:8]}"])
        report_json.write_text(report.to_json(), encoding="utf-8")
        Path(built, QUALITY_REPORT_TEXT).write_text(report.to_text() + "\n", encoding="utf-8")
    Path(built, REFERENCE_DIR).mkdir()
    for name, png in references.items():
        shutil.copyfile(png, Path(built, REFERENCE_DIR, f"{name}.png"))
    # Last, as store.write_calibration does: the index is what makes the folder a calibration.
    Path(built, INDEX_FILE).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return new_id


def capture_reference_entry(
    app_model,
    root: Path,
    source_id: str,
    cameras: Sequence[VideoCaptureModel],
    *,
    names: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
    record_only: Sequence[str] = (),
    **capture_options,
) -> str:
    """Capture reference images now for library entry `source_id` and add them as a new entry.

    Entries never change, so the references go into a copy of the entry that inherits its
    fit (copy_with_references), added to the library `root`, which is not made active; the
    copy holds the references just captured and none of the entry's old ones.
    `cameras` are the configured cameras matched to the entry's; `names` maps each to its
    name in the entry. The cameras named in `record_only` record with the others but keep
    no reference: the configured primary, which triggers them, when the entry did not
    calibrate it. The clip and the copy are built in a new session folder. Returns the
    new entry's id. Raises ValueError for a legacy entry, before any camera starts.
    """
    source = entry_path(root, source_id)
    load_index(source)
    cameras = [camera for camera in cameras if camera.is_primary] + [c for c in cameras if not c.is_primary]
    cams_before_cfg = tuple(camera.save_configuration() for camera in cameras)
    project = app_model.make_project_info()
    project.calculate_next_session_index()
    session = Path(project.get_session_path().location)
    written = capture_reference_images(cameras, cams_before_cfg, project, session, names=names, **capture_options)
    references = {(names or {}).get(camera, camera): path for camera, path in written.items()
                  if camera not in record_only}
    built = Path(session, REFERENCE_ENTRY_DIR_NAME)
    copy_with_references(source, references, built, now=now or datetime.now())
    calibration_id = add_entry(root, built)
    logger.success("reference images of %s added as calibration %s, from %s",
                   ", ".join(references), calibration_id, source_id)
    return calibration_id


def make_3d_calib(
    app_model: AppModel,
    cameras: Sequence[VideoCaptureModel],
    cam_params: Optional[Dict[str, Any]] = None,
    record_mode: VideoRecordMode = VideoRecordMode.TRIGGER,
    move_list: Optional[str] = None,
    confirm_scout: Optional[Callable[[Mapping[str, str]], bool]] = None,
    capture_references: bool = False,
    confirm_references: Optional[Callable[[], bool]] = None,
    rescout: bool = False,
) -> Tuple[Path, Path, CalibrationIndex, QualityReport]:
    """Capture the board paused at every waypoint with `cameras`, then calibrate them from it.

    `cameras` are 2 to 6 distinct cameras; the primary is put first, for the trigger order.
    Each camera's own live geometry gives its presets (capture_live_geometries), and the
    waypoint log records each one's name, serial and camera id (capture_camera_list).

    `move_list` is "handwritten" (the default, `moves`) or "scout". With "scout", the list
    saved in the calibration library for the same cameras (by serial, by name without one)
    is used when it lies inside the hand-written list's box; otherwise the scout runs
    (run_scout), then choose_plan: the scout's list with 12 or more chosen points, else the
    hand-written list. When a camera fails the rule over the scout's grid (diagnose) or in
    the scout list to be used (selection_shortfalls: fewer than 12 views, fewer than 8
    shared, or not joined to the others), `confirm_scout(advice by camera)` is asked first:
    False raises ScoutCancelled and nothing is captured; True goes on, and that camera may
    end up 2D only. A scout list is saved for next time only when no camera has advice and
    the list joins every camera. With `rescout` the saved list is ignored for this run (not
    deleted) and the scout runs.

    With `capture_references`, after the capture the operator is asked to remove the board
    (`confirm_references()`; False skips it) and each camera records a short still clip at
    its live geometry (capture_reference_images), whose middle frame goes into the capture's
    reference/ folder and from there into the calibration. A reference capture that fails
    is logged and the calibration goes on without references.

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
    if move_list not in (None, HANDWRITTEN, SCOUT):
        raise ValueError(f"unknown move list {move_list!r}; it is {SCOUT!r} or {HANDWRITTEN!r}")
    given = list(cameras)
    names = [camera.name for camera in given]
    if not MIN_CAMERAS <= len(given) <= MAX_CAMERAS:
        raise ValueError(f"a 3D calibration takes {MIN_CAMERAS} to {MAX_CAMERAS} cameras, not {len(given)}")
    if len(set(names)) != len(names):
        raise ValueError(f"a camera is given twice: {', '.join(names)}")
    plan = plan_waypoints(start, moves, diamond_triangle_cfg)
    # The stage travel (motor_config.yaml): a list that leaves it is refused before anything moves.
    travel = motor_travel()
    if move_list != SCOUT:
        check_plan_in_travel(plan, travel, HANDWRITTEN)
    # put primary first:
    cameras = [camera for camera in given if camera.is_primary] + [camera for camera in given if not camera.is_primary]
    # Read before prepare() replaces the camera configurations with the capture's own.
    presets = capture_presets(cameras[0].active_config.params)
    live_geometries = capture_live_geometries(cameras)
    capture_list = capture_camera_list(cameras)
    cams_before_cfg = tuple(camera.save_configuration() for camera in cameras)
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

    # The board from the library settings (default: the 4 mm checkerboard), read before any
    # motion so a malformed setting stops the calibration before it starts.
    root = getattr(app_model, "calibration_library_root", None)
    board = configured_board({} if root is None else read_settings(root))
    if board.kind != BoardSpec().kind:
        logger.notice("Calibration board: %s, %d x %d inner corners, %g mm squares, %g mm %s markers",
                      board.kind, board.rows, board.cols, board.square_mm, board.marker_mm, board.dictionary)

    if move_list == SCOUT:
        identities = [camera.serial or camera.name for camera in capture_list]
        configured = None if root is None else configured_bounds(read_settings(root))
        targets = [motor for *_, motor in plan]
        saved = None if rescout else saved_scout_list(root, identities, bounds_from_plan(targets, configured, travel))
        if rescout:
            logger.notice("Scouting again, as asked: any saved scout move list is not used this time")
        if saved is not None:
            logger.notice("Using the scout move list saved for these cameras: %d waypoints", len(saved))
            plan = scout_plan(saved, diamond_triangle_cfg)
            check_plan_in_travel(plan, travel, SCOUT)
        else:
            scout = run_scout(app_model, cameras, targets, project=project, session_dir=Path(sess_path.location),
                              cams_before_cfg=cams_before_cfg, capture_list=capture_list, cam_params=cam_params,
                              record_mode=record_mode, configured=configured, travel=travel, board=board)
            plan, used = choose_plan(SCOUT, diamond_triangle_cfg, scout)
            check_plan_in_travel(plan, travel, used)  # a hand-written fallback is first checked here
            # The rule is checked on the list that will be used, too: the 60-waypoint cap can
            # leave a camera short, and shared views can join the cameras in separate groups.
            shortfalls = {} if used != SCOUT else selection_shortfalls(
                scout.selection, [camera.name for camera in cameras])
            advice = merge_advice(scout.diagnosis, shortfalls)
            for camera, text in shortfalls.items():
                logger.warning("%s: %s", camera, text)
            if advice and confirm_scout is not None and not confirm_scout(advice):
                raise ScoutCancelled("cancelled at the scout's advice; nothing was captured")
            if used == SCOUT and root is not None and not advice and scout.selection.connected:
                save_move_list(Path(root, SCOUT_MOVE_LIST_FILE), scout.points, identities, datetime.now())

    log = _record_plan(app_model, cameras, cams_before_cfg, cam_params, record_mode, project, plan, capture_list,
                       src_dir)

    if capture_references:
        if confirm_references is None or confirm_references():
            try:
                capture_reference_images(cameras, cams_before_cfg, project, src_dir, record_mode=record_mode)
            except Exception as err:
                logger.error("No reference images: %s. The calibration goes on without them.", err)
        else:
            logger.notice("Reference images skipped")

    logger.notice("Calibrating every preset from the capture in %s", src_dir)
    output_dir, index, report = calibrate_waypoint_capture(
        src_dir,
        log,
        session_dir=Path(sess_path.location),
        presets=presets,
        previous_dir=app_model.active_calibration_dir,
        live_geometries=live_geometries,
        board=board,
    )

    logger.success("Successfully calibrated the capture: %s, verdict %s", index.id, report.verdict)

    return src_dir, output_dir, index, report
