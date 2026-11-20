"""The 3D calibration capture pauses at every motor waypoint and verifies it got there.

The stage visits the start position and 36 single-axis DCS moves. At each waypoint the
capture commands the move, waits for the motor's ack, settles for as long as the move
takes at the axis's configured speed plus 0.25 s (1.0 s when the speed is not known),
checks the motor's reported position against the commanded one, and dwells 0.5 s, so
every waypoint is a still stretch of video that calibrate_capture can tag with its
commanded position.

Nothing here touches hardware: the motor, the cameras and the clock are fakes.
"""

import contextlib
import logging
import math
import urllib.parse
import uuid
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import yaml

from autotrainer.core import CameraConfiguration, Offset3DTuple, SystemCommandKind
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.device import MotorConfigurationFile
from autotrainer.inference.multires_calibration.extract import distinct_waypoints
from autotrainer.inference.multires_calibration.geometry import PresetGeometry
from autotrainer.inference.multires_calibration.pipeline import WAYPOINT_LOG, load_waypoint_log, write_waypoint_log
from autotrainer.video.video_record import VideoRecordMode
from tools.acquisition.model import handle_3d_calibration as h3d
from tools.acquisition.model import hardware_model as hardware_model_module
from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.handle_3d_calibration import (
    calibrate_waypoint_capture,
    capture_presets,
    default_params,
    make_3d_calib,
    moves,
    plan_waypoints,
    run_waypoints,
    start,
    wait_cams_capture_status,
    wait_until_arrived,
)

NAN = float("nan")
# The live 256 preset's camera configuration: bin 4, 256x256 at ROI offset (52, 6).
LIVE_PARAMS = dict(hbin=4, vbin=4, width=256, height=256, offsetx=52, offsety=6, exposure=140, fps=150)
SETTINGS = {"gain_db": 1.0, "gamma": None, "exposure_us": 4500, "fps": 40}


def _dcs(used=(10, 40, 30), offset=(0, 0, 0), **overrides) -> DiamondTriangleOffsetConfig:
    # The default places every waypoint at a positive motor position, so nothing is clamped.
    values = dict(
        used_position=Offset3DTuple(*used),
        measured_offset=Offset3DTuple(*offset),
        diamond_coord=Offset3DTuple(0, 0, 0),
        raw_diamond_coord=Offset3DTuple(0, 0, 0),
        version=DiamondTriangleOffsetConfig.current_config_version,
    )
    values.update(overrides)
    return DiamondTriangleOffsetConfig(**values)


def _todays_commands(dcs):
    """(axis, motor value) as the make_3d_calib before this change sent them through its m_x/m_y/m_z."""
    d_to_m = dcs.diamond_to_motor
    m = {
        "x": lambda v: d_to_m(Offset3DTuple(v, 0, 0)).x,
        "y": lambda v: d_to_m(Offset3DTuple(0, v, 0)).y,
        "z": lambda v: d_to_m(Offset3DTuple(0, 0, v)).z,
    }
    return [(axis, m[axis](value)) for axis, value in (*start, *moves)]


class _Clock:
    """A clock that moves only when slept on."""

    def __init__(self, events=None, t0=1_000.0):
        self.t = t0
        self.events = [] if events is None else events

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.events.append(("sleep", seconds))
        self.t += seconds


class _Motor:
    """A motor that reports a commanded position `travel` seconds after the command.

    A command whose (axis, value) is in `stop_short` ends that much short of it, and the
    axis stays there until it is commanded again. After connect nothing is reported (NaN).
    """

    def __init__(self, clock, *, travel=0.3, stop_short=None):
        self._clock = clock
        self._travel = travel
        self._stop_short = stop_short or {}
        # per axis: (position before the command, where the command ends, time of the command)
        self._axes = {axis: (NAN, NAN, -math.inf) for axis in "xyz"}
        self.commands = []

    def _position(self, axis):
        before, end, t_cmd = self._axes[axis]
        return end if self._clock.now() >= t_cmd + self._travel else before

    def _move(self, axis, value, absolute):
        assert absolute, "the capture commands absolute positions"
        end = value - self._stop_short.get((axis, value), 0.0)
        self._axes[axis] = (self._position(axis), end, self._clock.now())
        token = uuid.uuid4()
        self.commands.append((axis, value))
        self._clock.events.append(("move", axis, value, token))
        return token

    def move_x(self, value, *, absolute=True):
        return self._move("x", value, absolute)

    def move_y(self, value, *, absolute=True):
        return self._move("y", value, absolute)

    def move_z(self, value, *, absolute=True):
        return self._move("z", value, absolute)

    def wait_pending_command_acked(self, token, *, timeout=3, raise_on_timeout=True):
        self._clock.events.append(("ack", token))

    @property
    def last_position(self):
        return Offset3DTuple(*(self._position(axis) for axis in "xyz"))


class _Reports:
    """A motor whose reported positions are read from a list; the last one repeats."""

    def __init__(self, *positions):
        self._positions = positions
        self.reads = 0

    @property
    def last_position(self):
        position = self._positions[min(self.reads, len(self._positions) - 1)]
        self.reads += 1
        return position


def _todays_waypoints(dcs):
    """The motor position after the start and after each move, as today's commands left it, unclamped."""
    position, waypoints = {}, []
    for i, (axis, value) in enumerate(_todays_commands(dcs)):
        position[axis] = value
        if i >= len(start) - 1:
            waypoints.append(tuple(position[a] for a in "xyz"))
    return waypoints


def _no_ops(positions):
    """The waypoints that leave the stage where the one before left it."""
    return [i for i in range(1, len(positions)) if positions[i] == positions[i - 1]]


# christielab10's DCS (2026-10-26 motor check), rounded as the motor check logged it. Its DCS
# x -9.4 and -9.0 land at motor x -1.72 and -1.32, below the stage's travel.
RIG_DCS = dict(used=(0.48, 19.01, 5.98), offset=(-7.20, 16.12, -8.19))
RIG_CLAMPED = [0, 1, 7, 8, 9, 13, 14, 15, 33, 34, 35, 36]


@pytest.mark.parametrize("used, offset", [((10, 40, 30), (0, 0, 0)), ((12.0, 40.5, 6.5), (1.7, -4.0, -8.4))])
def test_plan_matches_todays_moves(used, offset):
    dcs = _dcs(used, offset)
    expected = _todays_waypoints(dcs)
    # these DCS put every waypoint inside the stage's travel, so nothing is clamped
    assert min(v for position in expected for v in position) > 0

    plan = plan_waypoints(start, moves, dcs)

    assert len(plan) == 37
    # Waypoint 0 is where the three start moves leave the stage, and each move then
    # changes one axis of the previous waypoint.
    assert [motor for _, _, motor in plan] == expected
    assert [(axis, value) for axis, value, _ in plan] == [start[-1], *moves]
    assert all(isinstance(v, float) for _, value, motor in plan for v in (value, *motor))


def test_plan_motor_values_by_hand():
    # used (10, 40, 30), offset 0: motor x = 10 + x, but y and z are flipped: 40 - y, 30 - z.
    plan = plan_waypoints(start, moves, _dcs())

    assert plan[0][0:2] == ("z", -35.0)
    assert plan[0][2] == pytest.approx((0.6, 30.0, 65.0))  # start (-9.4, 10, -35)
    assert plan[1] == ("z", -30.0, pytest.approx((0.6, 30.0, 60.0)))
    assert plan[2] == ("x", -5.0, pytest.approx((5.0, 30.0, 60.0)))
    assert plan[3] == ("y", 15.0, pytest.approx((5.0, 25.0, 60.0)))
    assert plan[-1] == ("y", 15.0, pytest.approx((1.0, 25.0, 65.0)))  # x -9, y 15, z -35


def test_a_rig_like_dcs_plans_no_motor_position_below_the_stage_travel(caplog):
    dcs = _dcs(**RIG_DCS)
    unclamped = _todays_waypoints(dcs)
    below = [i for i, position in enumerate(unclamped) if min(position) < 0]
    assert below == RIG_CLAMPED  # all of them on x, at DCS x -9.4 and -9.0
    assert all(position[0] < 0 and min(position[1:]) > 0 for position in (unclamped[i] for i in below))

    with caplog.at_level("WARNING", logger=h3d.logger.name):
        plan = plan_waypoints(start, moves, dcs)

    motors = [motor for _, _, motor in plan]
    assert min(v for position in motors for v in position) >= 0
    # Commanded at the limit, where the stage stops anyway; every other coordinate as today.
    assert motors == [tuple(max(v, 0.0) for v in position) for position in unclamped]
    assert [i for i, position in enumerate(motors) if position != unclamped[i]] == RIG_CLAMPED
    assert all(motors[i][0] == 0.0 for i in RIG_CLAMPED)
    # The DCS moves stay as planned; only where the stage can go changed.
    assert [(axis, value) for axis, value, _ in plan] == [start[-1], *moves]

    # Every clamped waypoint is named in the warning, with its axis and the position it lost.
    [warning] = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert "12 of 37 waypoints" in warning
    for i in RIG_CLAMPED:
        assert f"waypoint {i}: x {unclamped[i][0]:.3f} -> 0.000" in warning


def test_clamping_on_this_rig_leaves_every_move_moving():
    plan = plan_waypoints(start, moves, _dcs(**RIG_DCS))
    motors = [motor for _, _, motor in plan]

    # Only today's own no-op (z -30 again at waypoint 32): no clamped move lands where the
    # stage already was, so the capture has the same 36 distinct waypoints as before.
    assert _no_ops(motors) == _no_ops(_todays_waypoints(_dcs(**RIG_DCS))) == [32]
    assert distinct_waypoints(motors) == [i for i in range(37) if i != 32]


def test_a_move_that_clamping_leaves_motionless_is_one_dwell_with_the_waypoint_before(caplog):
    # motor x = 4 + DCS x: x -9.4, -9, -7 and -5 all land below 0. The move to x -5 right
    # after the start (waypoint 2), and the move to x -9 after x -5 (waypoint 33), then
    # command the stage where it already is.
    dcs = _dcs(used=(4, 40, 30))
    assert _no_ops(_todays_waypoints(dcs)) == [32]

    with caplog.at_level("WARNING", logger=h3d.logger.name):
        plan = plan_waypoints(start, moves, dcs)

    motors = [motor for _, _, motor in plan]
    assert len(plan) == 37  # kept: the log still has one entry per move
    assert _no_ops(motors) == [2, 32, 33]
    # calibrate_capture collapses each run of equal commanded positions into the one
    # stationary segment the stage really made, as it does today's no-op.
    assert distinct_waypoints(motors) == [i for i in range(37) if i not in (2, 32, 33)]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("waypoint 2" in w and "where waypoint 1 left it" in w for w in warnings), warnings
    assert any("waypoint 33" in w and "where waypoint 32 left it" in w for w in warnings), warnings
    assert not any("waypoint 32" in w and "where waypoint 31 left it" in w for w in warnings), warnings


def test_plan_needs_a_start_that_sets_every_axis():
    with pytest.raises(ValueError, match="start"):
        plan_waypoints(start[:2], moves, _dcs())


def test_wait_until_arrived_true_and_false():
    target = (1.0, 2.0, 3.0)

    clock = _Clock()
    converging = _Reports(
        Offset3DTuple(NAN, NAN, NAN), Offset3DTuple(0.5, 2.0, 3.0), Offset3DTuple(1.04, 1.96, 3.04))
    arrived, reported = wait_until_arrived(converging, target, sleep=clock.sleep, now=clock.now)
    assert arrived is True
    assert reported == pytest.approx((1.04, 1.96, 3.04))
    assert clock.events == [("sleep", 0.01), ("sleep", 0.01)]  # polled until the third report

    clock = _Clock()
    one_axis_off = _Reports(Offset3DTuple(1.0, 2.0, 3.06))
    arrived, reported = wait_until_arrived(one_axis_off, target, sleep=clock.sleep, now=clock.now)
    assert arrived is False
    assert reported == pytest.approx((1.0, 2.0, 3.06))
    # it gave up at the timeout, not before and not much after
    assert 2.0 <= clock.t - 1_000.0 <= 2.0 + 0.01 + 1e-9

    for silent in (_Reports(Offset3DTuple(NAN, NAN, NAN)), _Reports(None), _Reports(Offset3DTuple(1.0, 2.0, NAN))):
        clock = _Clock()
        assert wait_until_arrived(silent, target, timeout=0.05, sleep=clock.sleep, now=clock.now) == (False, None)


def test_capture_loop_logs_dwell_and_verification(tmp_path):
    plan = plan_waypoints(start, moves, _dcs())
    clock = _Clock()
    # Waypoint 23 is DCS x 10.0, motor x 20.0. The motor stops 0.2 short there and stays
    # short through waypoint 24 (a z move) until x is commanded again at waypoint 25.
    motor = _Motor(clock, travel=0.3, stop_short={("x", 20.0): 0.2})
    assert plan[23][0:2] == ("x", 10.0) and plan[23][2][0] == pytest.approx(20.0)

    # A settle fixed at 0.25 s, shorter than the 0.3 s travel, so arrival must be polled for.
    log = run_waypoints(motor, plan, settle=0.25, sleep=clock.sleep, now=clock.now, clock=clock.now)

    assert len(log.motor) == 37
    assert log.motor == [motor_xyz for _, _, motor_xyz in plan]
    assert log.dcs_moves == [(axis, value) for axis, value, _ in plan]
    assert log.verified == [i not in (23, 24) for i in range(37)]
    for i, (reported, commanded) in enumerate(zip(log.reported, log.motor)):
        short = (0.2, 0, 0) if i in (23, 24) else (0, 0, 0)
        assert reported == pytest.approx(tuple(c - s for c, s in zip(commanded, short)))

    # Waypoint 0 commands the whole start, x then y then z; every later waypoint its one axis.
    assert motor.commands[:3] == [(axis, v) for axis, v in zip("xyz", plan[0][2])]
    assert motor.commands[3:] == [(axis, motor_xyz["xyz".index(axis)]) for axis, _, motor_xyz in plan[1:]]

    # (arrive, leave) per waypoint, in time order, each dwell 0.5 s.
    flat = [t for pair in log.times for t in pair]
    assert flat == sorted(flat) and len(set(flat)) == len(flat)
    assert all(leave - arrive == pytest.approx(0.5) for arrive, leave in log.times)

    # Each waypoint, in order: its move(s), each acked before anything else happens, the
    # 0.25 s settle, arrival polls of 0.01 s, then the 0.5 s dwell before the next move.
    chunks, current = [], None
    for event in clock.events:
        if event[0] == "move" and (current is None or current[-1][0] == "sleep"):
            current = []
            chunks.append(current)
        current.append(event)
    assert len(chunks) == 37
    # Today's list holds one no-op move (z -30 again at waypoint 32): the stage is already
    # there, so its arrival needs no poll.
    no_ops = [i for i in range(1, 37) if plan[i][2] == plan[i - 1][2]]
    assert no_ops == [32]
    for i, chunk in enumerate(chunks):
        n_moves = 3 if i == 0 else 1
        head, sleeps = chunk[:2 * n_moves], [e[1] for e in chunk[2 * n_moves:]]
        assert [e[0] for e in head] == ["move", "ack"] * n_moves
        assert all(head[k + 1][1] == head[k][3] for k in range(0, len(head), 2))
        assert all(e[0] == "sleep" for e in chunk[2 * n_moves:])
        assert sleeps[0] == 0.25 and sleeps[-1] == 0.5
        assert set(sleeps[1:-1]) <= {0.01}
        # travel 0.3 s outlasts the settle, so arrival was polled for, not assumed
        polled = 2.0 if i in (23, 24) else 0.0 if i in no_ops else 0.05
        assert sum(sleeps[1:-1]) == pytest.approx(polled, abs=0.011)

    assert log.capture_settings == SETTINGS


def test_waypoint_log_writes_all_four_capture_settings(tmp_path):
    plan = plan_waypoints(start, moves, _dcs())
    clock = _Clock()
    log = run_waypoints(_Motor(clock, travel=0.0), plan, sleep=clock.sleep, now=clock.now, clock=clock.now)

    write_waypoint_log(tmp_path / WAYPOINT_LOG, log)

    raw = yaml.safe_load((tmp_path / WAYPOINT_LOG).read_text(encoding="utf-8"))
    # all four keys, gamma among them as an explicit null: a missing key would be filled
    # in with the default silently, so the log must say what the capture set.
    assert raw["capture_settings"] == SETTINGS
    assert set(raw["capture_settings"]) == {"gain_db", "gamma", "exposure_us", "fps"}
    assert load_waypoint_log(tmp_path / WAYPOINT_LOG) == log


def test_capture_settings_follow_the_camera_params():
    clock = _Clock()
    params = dict(default_params, gain=2.5, gamma=0.8, exposure=3000, fps=30)
    log = run_waypoints(_Motor(clock, travel=0.0), plan_waypoints(start, moves, _dcs()),
                        sleep=clock.sleep, now=clock.now, clock=clock.now,
                        capture_settings=h3d.capture_settings_of(params))
    assert log.capture_settings == {"gain_db": 2.5, "gamma": 0.8, "exposure_us": 3000, "fps": 30}


def test_capture_settings_record_only_the_params_the_capture_sets():
    # The log never claims a value for a param the capture did not set (one without gain or
    # gamma would not open SpinCam at all). default_params sets gamma to None explicitly,
    # which is recorded.
    without_gamma = {k: v for k, v in default_params.items() if k != "gamma"}
    assert h3d.capture_settings_of(without_gamma) == {"gain_db": 1.0, "exposure_us": 4500, "fps": 40}
    assert h3d.capture_settings_of(default_params) == SETTINGS
    assert h3d.capture_settings_of({}) == {}


def test_the_arrival_timeout_runs_on_clock_not_on_now():
    # `now` stamps arrive and leave on the wall clock, which can step; `clock` times the
    # arrival wait. Here `now` never advances and `clock` does: a waypoint the motor never
    # reaches must still give up after arrival_timeout on `clock`.
    plan = plan_waypoints(start, moves, _dcs())[:2]
    clock = _Clock()
    polls = []

    def sleep(seconds):
        if len(polls) > 10_000:
            raise AssertionError("the arrival wait never timed out")
        polls.append(seconds)
        clock.sleep(seconds)

    def frozen_wall_clock():
        return 5_000.0

    motor = _Motor(clock, travel=0.0, stop_short={("z", plan[1][2][2]): 0.2})

    log = run_waypoints(motor, plan, settle=0.25, sleep=sleep, now=frozen_wall_clock, clock=clock.now)

    assert log.verified == [True, False]
    assert log.times == [(5_000.0, 5_000.0), (5_000.0, 5_000.0)]
    # waypoint 1's sleeps: its settle, its arrival polls, then its dwell
    waypoint_1 = polls[polls.index(0.5) + 1:]
    assert waypoint_1[0] == 0.25 and waypoint_1[-1] == 0.5
    assert sum(waypoint_1[1:-1]) == pytest.approx(2.0, abs=0.011)


def test_a_move_that_is_not_queued_stops_the_capture():
    clock = _Clock()
    motor = _Motor(clock)
    motor.move_y = lambda value, *, absolute=True: None
    with pytest.raises(RuntimeError, match="not queued"):
        run_waypoints(motor, plan_waypoints(start, moves, _dcs()),
                      sleep=clock.sleep, now=clock.now, clock=clock.now)


# The settle after each move's ack (ruling R52). The 2026-10-26 motor check showed that
# last_position echoes the target about 0.1 s after the command, so the arrival check cannot
# see the stage still moving: the settle alone has to cover the move, as if the ack meant
# only "received".

def _motor_configurations(**axes):
    """Motor configurations as HardwareModel keeps them from connect; axis -> (max_vel mm/s, max_acc mm/s^2)."""
    pellet = {axis: {"max_vel": v, "max_acc": a} for axis, (v, a) in axes.items()}
    return MotorConfigurationFile.from_yaml_dict({"pellet": pellet}, source=None)


SPEEDS = {"x": (20.0, 600.0), "y": (40.0, 1000.0), "z": (10.0, 300.0)}


def _settles(events):
    """Each waypoint's settle: the first sleep after its moves' acks."""
    return [events[k + 1][1] for k in range(len(events) - 1)
            if events[k][0] == "ack" and events[k + 1][0] == "sleep"]


def test_a_17_mm_move_settles_for_its_travel_at_the_configured_speed():
    # At the device package's 20 mm/s a 17 mm move, the longest in the list, takes 0.85 s.
    assert h3d.settle_time(17.0, (20.0, 600.0)) == pytest.approx(17.0 / 20.0 + 20.0 / 600.0 + 0.25)
    assert h3d.settle_time(17.0, (120.0, 600.0)) == pytest.approx(17.0 / 120.0 + 120.0 / 600.0 + 0.25)


def test_a_settle_without_a_known_speed_is_1_s():
    assert h3d.settle_time(17.0, None) == 1.0
    assert h3d.settle_time(0.0, None) == 1.0
    # No configuration yet, or none usable on an axis, gives that axis no limits.
    assert h3d.axis_limits(None) == {"x": None, "y": None, "z": None}
    configs = _motor_configurations(x=(20.0, 600.0), y=(0.0, 600.0), z=(math.nan, 600.0))
    assert h3d.axis_limits(configs) == {"x": (20.0, 600.0), "y": None, "z": None}


def test_a_short_move_settles_at_least_0_25_s():
    assert h3d.settle_time(0.0, (120.0, 1e12)) == pytest.approx(0.25)
    for distance in (0.0, 0.01, 0.5, 2.7):
        for limits in ((20.0, 1000.0), (120.0, 600.0), (61.0, 244.0)):
            assert h3d.settle_time(distance, limits) >= 0.25


def test_each_move_settles_for_its_own_axis_and_distance(caplog):
    plan = plan_waypoints(start, moves, _dcs())
    clock = _Clock()
    motor = _Motor(clock, travel=0.0)
    motor.motor_configurations = _motor_configurations(**SPEEDS)

    with caplog.at_level(logging.DEBUG, logger=h3d.logger.name):
        run_waypoints(motor, plan, sleep=clock.sleep, now=clock.now, clock=clock.now)

    settles = _settles(clock.events)
    assert len(settles) == len(plan)
    # The motor had reported no position before waypoint 0 (NaN after connect), so that
    # move's distance is unknown.
    assert settles[0] == 1.0
    for i in range(1, len(plan)):
        axis = plan[i][0]
        k = "xyz".index(axis)
        distance = abs(plan[i][2][k] - plan[i - 1][2][k])
        assert settles[i] == pytest.approx(h3d.settle_time(distance, SPEEDS[axis])), i
    # Waypoint 23 is the 17 mm x move, DCS -7 to 10.
    assert plan[23][0] == "x" and plan[23][2][0] - plan[22][2][0] == pytest.approx(17.0)
    assert settles[23] == pytest.approx(17.0 / 20.0 + 20.0 / 600.0 + 0.25)
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG and "settle" in r.getMessage()]
    assert len(logged) == len(plan)


def test_waypoint_0_settles_for_its_slowest_axis_from_where_the_motor_reported_it():
    plan = plan_waypoints(start, moves, _dcs())[:1]
    clock = _Clock()
    motor = _Motor(clock, travel=0.0)
    motor.motor_configurations = _motor_configurations(**SPEEDS)
    # Where prepare() left the stage, near the start: x 4 mm and y 1 mm away, z on it.
    target = plan[0][2]
    for move, value in zip((motor.move_x, motor.move_y, motor.move_z), (target[0] - 4.0, target[1] + 1.0, target[2])):
        move(value)
    before = len(clock.events)

    run_waypoints(motor, plan, sleep=clock.sleep, now=clock.now, clock=clock.now)

    (settle,) = _settles(clock.events[before:])
    expected = [h3d.settle_time(4.0, SPEEDS["x"]), h3d.settle_time(1.0, SPEEDS["y"]), h3d.settle_time(0.0, SPEEDS["z"])]
    assert settle == pytest.approx(max(expected))


def test_without_motor_configurations_every_move_settles_1_s():
    plan = plan_waypoints(start, moves, _dcs())
    clock = _Clock()
    run_waypoints(_Motor(clock, travel=0.0), plan, sleep=clock.sleep, now=clock.now, clock=clock.now)
    assert _settles(clock.events) == [1.0] * len(plan)


class _Subscribers:
    """An event that handlers are added to with +=."""

    def __iadd__(self, handler):
        return self


def test_the_hardware_model_keeps_the_motor_configurations_it_writes_at_connect(hardware_model, monkeypatch):
    # The capture times its settles from these, so they must be what connect sent the motors
    # (WRITE_MOTOR_CONFIGURATION, from ~/Autotrainer/motor_config.yaml), not a package default.
    written = _motor_configurations(x=(120.0, 600.0), y=(120.0, 600.0), z=(120.0, 600.0))
    assert hardware_model.motor_configurations is None, "nothing is written before a connect"

    class Device:
        default_command_ack_timeout_duration = 0.1
        default_board_status_timeout_delay = 1.0

        def __init__(self, **kwargs):
            self.property_changed = _Subscribers()
            self.latency_observer = None
            self.device_interface = None

    class Connection:
        def __init__(self, can_device, cmd_queue, **kwargs):
            self.device = SimpleNamespace(connected=True)

        def request_connect(self):
            pass

        def wait_connected(self, *, timeout):
            pass

        @contextlib.contextmanager
        def await_acknowledge(self, tokens, timeout):
            yield

        def send_message(self, kind, data=None, context=None):
            if kind is SystemCommandKind.REQUEST_VERSION:
                hardware_model._firmware_version_received.set()

        def load_default_motor_config(self):
            return written

        def load_default_move_config(self):
            pass

    class Homing(Exception):
        """Connect got past the motor configuration to homing, where this test stops it."""

    def home(send_acknowledged_command):
        raise Homing

    monkeypatch.setattr(hardware_model_module, "CanDevice", Device)
    monkeypatch.setattr(hardware_model_module, "DeviceConnection", Connection)
    monkeypatch.setattr(HardwareModel, "pellet_commands_allowed", property(lambda self: True))
    monkeypatch.setattr(HardwareModel, "_home_and_detach_pellet", staticmethod(home))
    with pytest.raises(Homing):
        hardware_model.connect(Queue())
    assert hardware_model.motor_configurations is written


def test_capture_params_set_gain_1_db_and_turn_camera_gamma_off():
    assert default_params["gain"] == 1.0
    assert "gamma" in default_params and default_params["gamma"] is None
    # spinnaker_cam imports PySpin when it loads; PySpin is on the rig.
    pytest.importorskip("PySpin")
    from autotrainer.video.camera.camera_base import CameraBase
    from autotrainer.video.camera.spinnaker_cam import SpinCam
    from autotrainer.video.video_manager import VideoManager

    # The camera URL as VideoCaptureModel.load_configuration builds it from the params, read
    # back the way the capture process reads it. SpinCam's own default gamma is 0.7, so a
    # gamma dropped on the way would read back as 0.7, not None.
    url = "spinnaker://12345678?" + urllib.parse.urlencode(dict(default_params, primary="on"))
    _, parsed = VideoManager.parse_params(url)
    camera = SpinCam.__new__(SpinCam)
    CameraBase.__init__(camera, "left")
    camera._camera = None  # nothing for __del__ to release
    for name in ("gain", "gamma"):
        assert camera.set_property(name, parsed[name])
    assert camera._gain == 1.0
    assert camera._gamma is None  # _apply_settings then sets GammaEnable off


class _Cam:
    def __init__(self, name, reaches):
        self.name = name
        self._reaches = reaches

    def wait_for_capture_status(self, status, *, timeout):
        return self._reaches


def test_a_camera_that_misses_the_capture_status_raises():
    with pytest.raises(RuntimeError, match=r"right.*RECORDING"):
        wait_cams_capture_status([_Cam("left", True), _Cam("right", False)], CaptureProcessStatus.RECORDING, 5)
    wait_cams_capture_status([_Cam("left", True), _Cam("right", True)], CaptureProcessStatus.RUNNING, 5)


def test_presets_come_from_the_live_camera_configuration():
    assert capture_presets(LIVE_PARAMS) == [
        PresetGeometry(4, 256, 256, 52, 6),
        PresetGeometry(2, 512, 512, 104, 12),
        PresetGeometry(1, 1024, 1024, 208, 24),
    ]
    # The capture's own bin-1 full-frame params are no base for the presets: they must be
    # taken from the configuration before the capture replaces it.
    with pytest.raises(ValueError):
        capture_presets(default_params)


def test_the_capture_is_calibrated_into_the_session_beside_its_sources(tmp_path, monkeypatch):
    capture_dir = tmp_path / "session" / DEFAULT_3D_CALIB_DIR_NAME
    capture_dir.mkdir(parents=True)
    clock = _Clock()
    log = run_waypoints(_Motor(clock, travel=0.0), plan_waypoints(start, moves, _dcs()),
                        sleep=clock.sleep, now=clock.now, clock=clock.now)
    presets = capture_presets(LIVE_PARAMS)
    previous = tmp_path / "Autotrainer" / DEFAULT_3D_CALIB_DIR_NAME
    calls = []

    def fake_calibrate_capture(capture, **kwargs):
        # the log must be in the capture folder before calibrate_capture reads it
        calls.append((Path(capture), kwargs, load_waypoint_log(Path(capture) / WAYPOINT_LOG)))
        return "index", "report"

    monkeypatch.setattr(h3d, "calibrate_capture", fake_calibrate_capture)

    result = calibrate_waypoint_capture(
        capture_dir, log, session_dir=tmp_path / "session", presets=presets, previous_dir=previous)

    assert result == (tmp_path / "session" / f"{DEFAULT_3D_CALIB_DIR_NAME}.new", "index", "report")
    [(capture, kwargs, written)] = calls
    assert capture == capture_dir
    assert kwargs == dict(
        output_dir=tmp_path / "session" / f"{DEFAULT_3D_CALIB_DIR_NAME}.new",
        presets=presets,
        previous_dir=previous,
    )
    assert written == log


class _Stop(Exception):
    """Raised where make_3d_calib starts the session: everything before it ran."""


def _app_model(*, effective, loaded):
    def make_project_info():
        raise _Stop

    def camera(name, primary):
        return SimpleNamespace(name=name, is_primary=primary, active_config=SimpleNamespace(params=dict(LIVE_PARAMS)),
                               save_configuration=lambda: SimpleNamespace(params=dict(LIVE_PARAMS)))

    return SimpleNamespace(
        behavior=SimpleNamespace(algorithm=SimpleNamespace(
            diamond_triangle_config=effective, loaded_diamond_triangle_config=loaded)),
        left_camera=camera("left", True),
        right_camera=camera("right", False),
        hardware=SimpleNamespace(),  # any motor call would raise AttributeError
        make_project_info=make_project_info,
        active_calibration_dir=None,
    )


def test_calibration_uses_the_loaded_dcs_while_stale_and_refuses_without_one():
    # STALE: consumers see no DCS, but the capture still converts with the loaded one.
    with pytest.raises(_Stop):
        make_3d_calib(_app_model(effective=None, loaded=_dcs()))
    for missing in (None, _dcs(version=1), _dcs(used=(NAN, 20, 30))):
        with pytest.raises(RuntimeError, match="calibrate diamond-triangle"):
            make_3d_calib(_app_model(effective=missing, loaded=missing))


class _CaptureCamera:
    """A camera that records the calls make_3d_calib makes on it; `stops` is whether it
    reaches TERMINATED when stopped."""

    def __init__(self, name, primary, calls, *, stops):
        self.name = name
        self.is_primary = primary
        self.saved = CameraConfiguration(name=name, params=dict(LIVE_PARAMS, primary="on" if primary else "off"))
        self.active_config = self.saved
        self.project = None
        self._calls = calls
        self._stops = stops

    def _call(self, *what):
        self._calls.append((self.name, *what))

    def save_configuration(self):
        return self.saved

    def load_configuration(self, cfg):
        self._call("load_configuration", cfg)

    def on_prepare_capture(self):
        self._call("on_prepare_capture")
        return True

    def wait_for_capture_status(self, status, *, timeout):
        self._call("wait_for_capture_status", status)
        return self._stops or status is not CaptureProcessStatus.TERMINATED

    def on_capture_start(self):
        self._call("on_capture_start")

    def on_trigger_recording(self, on):
        self._call("on_trigger_recording", on)

    def on_capture_notify_end(self):
        self._call("on_capture_notify_end")

    def on_capture_stop(self):
        self._call("on_capture_stop")


class _CaptureMotor:
    def __init__(self, calls):
        self._calls = calls

    def connect(self, queue):
        self._calls.append(("motor", "connect"))

    def send_home(self):
        return uuid.uuid4()

    def move_x(self, value, *, absolute=True):
        return uuid.uuid4()

    move_y = move_z = move_x

    def wait_pending_command_acked(self, token, *, timeout=3, raise_on_timeout=True):
        pass

    def disconnect(self):
        self._calls.append(("motor", "disconnect"))


def _capture_app_model(tmp_path, calls, *, stops):
    session = tmp_path / "session"
    session.mkdir()
    project = SimpleNamespace(
        calculate_next_session_index=lambda: None,
        get_session_path=lambda: SimpleNamespace(location=str(session), prefix="20261023_000000"),
        get_video_path=lambda name, **kwargs: (str(session / f"{name}_recording.mp4"),),
    )
    return SimpleNamespace(
        behavior=SimpleNamespace(algorithm=SimpleNamespace(diamond_triangle_config=_dcs(),
                                                           loaded_diamond_triangle_config=_dcs())),
        left_camera=_CaptureCamera("left", True, calls, stops=stops),
        right_camera=_CaptureCamera("right", False, calls, stops=stops),
        hardware=_CaptureMotor(calls),
        message_handler=SimpleNamespace(input_queue=None),
        make_project_info=lambda: project,
        active_calibration_dir=None,
    )


def _assert_cameras_stopped_and_restored(app_model, calls):
    for camera in (app_model.left_camera, app_model.right_camera):
        mine = [call[1:] for call in calls if call[0] == camera.name]
        assert ("on_capture_stop",) in mine
        # the capture loaded its own configuration; the last one loaded is the operator's again
        loads = [call for call in mine if call[0] == "load_configuration"]
        assert len(loads) == 2 and loads[0][1] is not camera.saved and loads[-1][1] is camera.saved
        assert mine[-1] == loads[-1]
        assert mine.index(("on_capture_stop",)) < len(mine) - 1
    assert ("motor", "disconnect") in calls


@pytest.mark.parametrize("stops", [True, False], ids=["cameras_stop", "cameras_do_not_stop"])
def test_a_failed_capture_stops_and_restores_the_cameras_and_raises_its_own_error(tmp_path, monkeypatch, stops):
    calls = []
    app_model = _capture_app_model(tmp_path, calls, stops=stops)
    fault = RuntimeError("motor fault mid-capture")

    def failing_run_waypoints(*args, **kwargs):
        raise fault

    calibrated = []
    monkeypatch.setattr(h3d, "run_waypoints", failing_run_waypoints)
    monkeypatch.setattr(h3d, "calibrate_waypoint_capture", lambda *a, **k: calibrated.append(a))

    with pytest.raises(RuntimeError) as raised:
        make_3d_calib(app_model)

    # the capture's own error, also when the cameras then fail to stop as well
    assert raised.value is fault
    _assert_cameras_stopped_and_restored(app_model, calls)
    terminated = [call for call in calls if call[1:] == ("wait_for_capture_status", CaptureProcessStatus.TERMINATED)]
    assert terminated  # the stop was waited for, and failed in the second case
    assert calibrated == []


def test_the_capture_triggers_recording_like_a_session(tmp_path, monkeypatch):
    # START_CONTINUOUS never reached RECORDING on christielab10 (2026-11-19); the capture
    # records as a session does: TRIGGER on after capture start, before the RECORDING wait,
    # and off after the waypoints.
    calls = []
    app_model = _capture_app_model(tmp_path, calls, stops=True)
    monkeypatch.setattr(h3d, "run_waypoints", lambda *a, **k: SimpleNamespace(verified=[True]))
    monkeypatch.setattr(h3d, "calibrate_waypoint_capture", lambda *a, **k: (None, SimpleNamespace(id="cal"), SimpleNamespace(verdict="pass")))
    for name in ("left", "right"):  # what the cameras would have recorded
        (tmp_path / "session" / f"{name}_recording.mp4").write_bytes(b"")

    make_3d_calib(app_model)

    for name in ("left", "right"):
        mine = [call[1:] for call in calls if call[0] == name]
        loaded = [call[1] for call in mine if call[0] == "load_configuration"][0]
        assert loaded.record_mode == VideoRecordMode.TRIGGER
        start = mine.index(("on_capture_start",))
        on = mine.index(("on_trigger_recording", True))
        recording = mine.index(("wait_for_capture_status", CaptureProcessStatus.RECORDING))
        off = mine.index(("on_trigger_recording", False))
        assert start < on < recording < off


def test_cameras_that_do_not_stop_fail_a_capture_that_ran(tmp_path, monkeypatch):
    calls = []
    app_model = _capture_app_model(tmp_path, calls, stops=False)
    calibrated = []
    monkeypatch.setattr(h3d, "run_waypoints", lambda *a, **k: SimpleNamespace(verified=[True]))
    monkeypatch.setattr(h3d, "calibrate_waypoint_capture", lambda *a, **k: calibrated.append(a))

    with pytest.raises(RuntimeError, match="TERMINATED"):
        make_3d_calib(app_model)

    _assert_cameras_stopped_and_restored(app_model, calls)
    assert calibrated == []  # videos that may still be being written are not calibrated from


def test_a_capture_that_ran_returns_where_its_calibration_was_written(tmp_path, monkeypatch):
    calls = []
    app_model = _capture_app_model(tmp_path, calls, stops=True)
    session = tmp_path / "session"
    for camera in ("left", "right"):
        (session / f"{camera}_recording.mp4").touch()
    log = SimpleNamespace(verified=[True])
    index, report = SimpleNamespace(id="abc123"), SimpleNamespace(verdict="pass")
    calibrated = []

    def fake_calibrate_waypoint_capture(capture_dir, waypoint_log, **kwargs):
        calibrated.append((Path(capture_dir), waypoint_log, kwargs))
        return session / "written-here", index, report

    monkeypatch.setattr(h3d, "run_waypoints", lambda *a, **k: log)
    monkeypatch.setattr(h3d, "calibrate_waypoint_capture", fake_calibrate_waypoint_capture)

    result = make_3d_calib(app_model)

    capture_dir = session / DEFAULT_3D_CALIB_DIR_NAME
    assert result == (capture_dir, session / "written-here", index, report)
    [(calibrated_dir, calibrated_log, kwargs)] = calibrated
    assert (calibrated_dir, calibrated_log, kwargs["session_dir"]) == (capture_dir, log, session)
    assert sorted(p.name for p in (capture_dir / "source_videos").iterdir()) == ["left.mp4", "right.mp4"]
