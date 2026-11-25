"""Per-move ack deadlines, checked against acks measured on christielab10, not against the formula.

Pellet firmware 2.3.1 acknowledges a move on its 100 ms status tick once the
axis is done, with a raised-cosine profile; a home runs to the switch at
home_vel; SEND_TO_FIXED_XYZ moves the axes one per state, a state per tick.
A move whose ack is late beyond its deadline has stalled or run away: ESTOP.
The tables below are real acks; none of them may trip. Limits on that rig:
max_vel 120 mm/s, max_acc 600 mm/s^2, home_vel 30 mm/s.
"""

import logging
import math
import threading
import time

import pytest

from reachaq.core import Motor, Offset3DTuple
from reachaq.core.message import SystemCommandKind, SystemStatusMessageKind
from reachaq.device import StepperConfig, Target
from reachaq.device import can_device as can_device_module
from reachaq.device.can_interface import _STEPPER_MAX_POS
from reachaq.device.device_interface import Acknowledge
from reachaq.device.motion_deadline import READER_SILENT_BACKSTOP_S, profile_time

from can_device_refusal_test import _Board, _LOST

_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)
_AXIS = dict(zip("XYZ", _XYZ))

# christielab10, ~/reachAQ-calib-iso2/data/20261124/christielab10/<log>: (log, axis, from mm,
# to mm, ack s), the MOVE with the least headroom at each distance (rounded mm).
MEASURED_MOVES = [
    ("000", "X", 1.0, 1.0, 0.007), ("002", "Y", 0.0, 1.0, 0.197), ("004", "Z", 2.0, 4.0, 0.240),
    ("004", "Z", 8.0, 5.0, 0.288), ("003", "Z", 8.0, 12.0, 0.251), ("003", "Z", 0.0, 5.0, 0.297),
    ("003", "Z", 15.0, 22.0, 0.352), ("004", "Z", 15.5, 8.0, 0.353), ("003", "Z", 15.0, 5.0, 0.448),
    ("001", "X", 12.333, 1.0, 0.475), ("003", "X", 0.0, 12.3, 0.403), ("003", "Z", 25.0, 12.0, 0.497),
    ("004", "Z", 1.0, 15.5, 0.523), ("002", "Z", 0.0, 15.5, 0.500), ("000", "Z", 18.0, 35.0, 0.556),
    ("003", "X", 23.7, 6.0, 0.522), ("004", "X", 35.0, 12.33, 0.620), ("003", "Y", 30.0, 6.0, 0.640),
    ("001", "Z", 30.0, 1.0, 0.653), ("001", "Z", 0.0, 30.0, 0.596), ("000", "Z", 35.0, 1.0, 0.547),
    ("004", "X", 0.0, 35.0, 0.699),
]
# Same logs: the longest homes (from mm where the host knew it, else NaN; ack s).
MEASURED_HOMES = [
    ("001", "Y", 35.0, 1.240), ("003", "Y", math.nan, 1.229), ("000", "Y", 35.0, 1.207),
    ("002", "Y", 35.0, 1.206), ("005", "Y", math.nan, 1.203), ("000", "Z", 35.0, 1.102),
]
# Every SEND_TO_FIXED_XYZ in ~/Documents/rawdatalocal: (log, commanded xyz, send xyz, ack s).
# The trial send after a load, 2026-09-24 and 2026-10-02, and the z sends of 2026-11-24.
_TRIAL = ((25.0, 0.0, 10.0), (0.0, 20.0, 0.0))
MEASURED_FIXED_XYZ = [
    ("20260924_001", *_TRIAL, 1.854), ("20260924_001", *_TRIAL, 1.810),
    ("20260924_520", *_TRIAL, 1.858), ("20260924_520", *_TRIAL, 1.826),
    ("20260924_520", *_TRIAL, 1.816), ("20260924_520", *_TRIAL, 1.800),
    ("20261002_006", *_TRIAL, 1.812), ("20261002_006", *_TRIAL, 1.829),
    ("20261002_006", *_TRIAL, 1.856), ("20261002_006", *_TRIAL, 1.858),
    ("20261002_006", *_TRIAL, 1.857), ("20261002_006", *_TRIAL, 1.856),
    ("20261002_006", *_TRIAL, 1.849), ("20261002_006", *_TRIAL, 1.804),
    ("20261002_006", (0.0, 20.0, 0.0), (0.0, 20.0, 0.0), 0.277),
    ("20261124_005", (0.0, 0.0, 0.0), (1.0, 34.0, 0.0), 1.019),
    ("20261124_005", (1.0, 34.0, 0.0), (1.0, 34.0, 6.0), 0.541),
    ("20261124_005", (1.0, 34.0, 6.0), (1.0, 34.0, 15.0), 0.619),
    ("20261124_005", (1.0, 34.0, 15.0), (1.0, 34.0, 28.5), 0.660),
    ("20261124_005", (1.0, 34.0, 28.5), (1.0, 34.0, 30.0), 0.468),
]


def _rig_config():
    config = StepperConfig()
    config.maximum_velocity, config.maximum_acceleration, config.homing_velocity = 120.0, 600.0, 30.0
    return config


@pytest.fixture
def device():
    from reachaq.device import CanDevice, DeviceApi

    device = CanDevice(api=DeviceApi(), force_emulation=True, required_targets=(Target.PELLET_DEVICE,))
    for motor in _XYZ:
        device._motor_configs[motor] = _rig_config()
    return device


# -- the firmware's profile -------------------------------------------------------

def test_the_profile_is_the_firmwares_raised_cosine():
    # motor_math.c:58-79: each ramp takes 2v/a. At 120/600 that needs 48 mm, more than the
    # whole travel: every pellet move is a "short" one, 2 * sqrt(2d/a).
    assert profile_time(14.5, 120.0, 600.0) == pytest.approx(2.0 * math.sqrt(2.0 * 14.5 / 600.0))
    assert profile_time(60.0, 120.0, 600.0) == pytest.approx(60.0 / 120.0 + 2.0 * 120.0 / 600.0)
    assert profile_time(-14.5, 120.0, 600.0) == profile_time(14.5, 120.0, 600.0)


@pytest.mark.parametrize("v, a", [(0.0, 600.0), (120.0, 0.0), (math.nan, 600.0), (120.0, math.inf)])
def test_a_profile_without_usable_limits_is_unknown(v, a):
    assert profile_time(10.0, v, a) is None


# -- no measured ack would have tripped -----------------------------------------------

@pytest.mark.parametrize("log, axis, start, end, ack_s", MEASURED_MOVES)
def test_no_measured_move_would_have_tripped(device, log, axis, start, end, ack_s):
    motor = _AXIS[axis]
    device._commanded_pos[motor] = start

    deadline, _ = device._move_deadline(motor, end)

    assert deadline > ack_s + 0.3, f"{log}: {axis} {start} -> {end} acked after {ack_s} s"


@pytest.mark.parametrize("log, axis, start, ack_s", MEASURED_HOMES)
def test_no_measured_home_would_have_tripped(device, log, axis, start, ack_s):
    motor = _AXIS[axis]
    device._commanded_pos[motor] = start

    deadline, end = device._move_deadline(motor, 0.0, home=True)

    assert end == 0.0
    assert deadline > ack_s + 0.45, f"{log}: home {axis} acked after {ack_s} s"


@pytest.mark.parametrize("log, start, send, ack_s", MEASURED_FIXED_XYZ)
def test_no_measured_send_to_fixed_xyz_would_have_tripped(device, log, start, send, ack_s):
    device._commanded_pos.update(dict(zip(_XYZ, start)))
    device._last_send_pos = Offset3DTuple(*send)

    deadline, ends = device._fixed_xyz_deadline()

    assert deadline > ack_s + 0.6, f"{log}: {start} -> {send} acked after {ack_s} s"
    assert ends == dict(zip(_XYZ, send))


def test_a_home_is_given_the_full_travel_whatever_the_logical_position(device):
    # Review I-B: after lost steps the carriage can be far from where the host thinks.
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0
    near, _ = device._move_deadline(Motor.PELLET_Z_MOTOR, 0.0, home=True)
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = math.nan
    unknown, _ = device._move_deadline(Motor.PELLET_Z_MOTOR, 0.0, home=True)

    assert near == unknown
    assert 1.7 < near < 1.8  # 36.55 mm at 30 mm/s, no ramp: 1.74 s


def test_a_save_as_fixed_step_does_not_move_the_commanded_position(device):
    # Review M-A: it only stores the send position.
    device._commanded_pos[Motor.PELLET_X_MOTOR] = 3.0

    _, ends = device._compound_step_deadline({"x": 20.0, "save_as_fixed": True})

    assert ends == {}
    assert device._commanded_pos[Motor.PELLET_X_MOTOR] == 3.0


# -- the board, end to end ------------------------------------------------------------

@pytest.fixture
def board():
    b = _Board()
    for motor in _XYZ:
        b.device._motor_configs[motor] = _rig_config()
    b.start()
    try:
        yield b
    finally:
        b.close()


def _sent(board, moves, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not moves and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)


def test_the_measured_trial_send_acked_after_1_86_s_does_not_trip(board):
    # Review C-1: commanded (25, 0, 10), send (0, 20, 0), the board acks at 1.86 s.
    device, interface = board.device, board.interface
    device._commanded_pos.update(dict(zip(_XYZ, _TRIAL[0])))
    device._last_send_pos = Offset3DTuple(*_TRIAL[1])

    def slow_send(*args, **kwargs):
        uuid = interface.next_uuid()
        threading.Timer(1.86, lambda: device._handle_ack(Acknowledge(uuid=uuid))).start()
        return True

    board._replace("fixed_position", slow_send)
    token = board.send(SystemCommandKind.SEND_PELLET)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token, timeout=4.0) is not None
    assert interface.estop_frames == []


def test_a_normal_move_acknowledged_inside_its_deadline_does_not_trip(board):
    # The 15:55:20 move, 15.5 -> 30 mm, was acked after 0.438 s.
    device, interface = board.device, board.interface
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 15.5

    def slow_move(*args, **kwargs):
        uuid = interface.next_uuid()
        threading.Timer(0.44, lambda: device._handle_ack(Acknowledge(uuid=uuid))).start()
        return True

    board._replace("move_motor_z", slow_move)
    token = board.send(SystemCommandKind.MOVE_Z, 30.0)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert interface.estop_frames == []


def _fake_clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(can_device_module, "get_perf_now", lambda: now[0])
    return now


def test_the_1555_runaway_is_stopped_at_its_deadline_once_the_reader_is_heard(board, monkeypatch):
    # 15:55:53.48: MOVE_Z 1.0 -> 15.5 at 120/600, never acknowledged; z free-ran.
    now = _fake_clock(monkeypatch)
    device = board.device
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0
    moves = board.answer("move_motor_z", _LOST)
    token = board.send(SystemCommandKind.MOVE_Z, 15.5)
    _sent(board, moves)
    armed = device._boards_pending_ctx[Target.PELLET_DEVICE].ack_timeout_used
    assert 0.85 < armed < 0.92

    now[0] += armed + 0.02
    time.sleep(0.3)
    assert board.interface.estop_frames == [], "no frame read since the deadline: not yet"

    device.notify_data([])  # the reader delivers a status frame: it is alive
    refusal = board.refusal_of(token)

    assert board.interface.estop_frames == [True]
    assert refusal.estop_reason == f"MOVE_Z was not acknowledged within {armed:.2f} s"


def test_a_silent_reader_still_stops_the_move_one_second_past_its_deadline(board, monkeypatch):
    now = _fake_clock(monkeypatch)
    device = board.device
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0
    moves = board.answer("move_motor_z", _LOST)
    token = board.send(SystemCommandKind.MOVE_Z, 15.5)
    _sent(board, moves)
    armed = device._boards_pending_ctx[Target.PELLET_DEVICE].ack_timeout_used

    now[0] += armed + READER_SILENT_BACKSTOP_S - 0.05
    time.sleep(0.3)
    assert board.interface.estop_frames == []

    now[0] += 0.1
    board.refusal_of(token)

    assert board.interface.estop_frames == [True]


def test_a_deadline_expiry_and_an_ack_are_logged_in_one_parseable_line(board, monkeypatch, caplog):
    device, interface = board.device, board.interface
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 15.5
    with caplog.at_level(logging.INFO, logger="reachaq.device.can_device"):
        token = board.send(SystemCommandKind.MOVE_Z, 30.0)
        assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    acked = [r for r in caplog.records if r.getMessage().startswith("pellet-ack-timing outcome=acked")]
    assert len(acked) == 1 and acked[0].levelno == logging.INFO
    fields = dict(item.split("=") for item in acked[0].getMessage().split()[1:])
    assert fields["command"] == "MOVE_Z" and fields["axes"] == "Z"
    assert fields["distance_mm"] == "14.500"
    assert float(fields["profile_s"]) == pytest.approx(profile_time(14.5, 120.0, 600.0), abs=1e-3)
    assert float(fields["deadline_s"]) > float(fields["profile_s"])
    assert float(fields["elapsed_s"]) >= 0.0

    caplog.clear()
    now = _fake_clock(monkeypatch)
    moves = board.answer("move_motor_z", _LOST)
    with caplog.at_level(logging.INFO, logger="reachaq.device.can_device"):
        token = board.send(SystemCommandKind.MOVE_Z, 1.0)
        _sent(board, moves)
        now[0] += 3.0
        device.notify_data([])
        board.refusal_of(token)
    expired = [r for r in caplog.records if r.getMessage().startswith("pellet-ack-timing outcome=deadline_expired")]
    assert len(expired) == 1 and expired[0].levelno == logging.ERROR
    fields = dict(item.split("=") for item in expired[0].getMessage().split()[1:])
    assert fields["distance_mm"] == "29.000" and float(fields["elapsed_s"]) == pytest.approx(3.0, abs=0.01)


# -- the switch ------------------------------------------------------------------------

def test_with_profile_deadlines_off_a_move_waits_the_fixed_timeout_and_still_stops(board):
    device = board.device
    device.profile_ack_deadlines = False
    device.default_command_ack_timeout_duration = 0.3
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0
    moves = board.answer("move_motor_z", _LOST)

    token = board.send(SystemCommandKind.MOVE_Z, 15.5)
    _sent(board, moves)
    assert device._boards_pending_ctx[Target.PELLET_DEVICE].ack_timeout_used == 0.3
    refusal = board.refusal_of(token)

    assert board.interface.estop_frames == [True]
    assert refusal.estop_reason == "MOVE_Z was not acknowledged within 0.30 s"


def test_with_profile_deadlines_on_a_move_waits_its_profile_deadline(board):
    device = board.device
    assert device.profile_ack_deadlines is True
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0
    moves = board.answer("move_motor_z", _LOST)
    board.send(SystemCommandKind.MOVE_Z, 15.5)
    _sent(board, moves)

    assert 0.85 < device._boards_pending_ctx[Target.PELLET_DEVICE].ack_timeout_used < 0.92


@pytest.mark.parametrize("pellet, expected", [
    ({}, True),
    ({"profile_ack_deadlines": True}, True),
    ({"profile_ack_deadlines": False}, False),
], ids=["missing", "true", "false"])
def test_the_switch_is_read_from_motor_config_yaml_and_missing_means_on(pellet, expected):
    from reachaq.device import MotorConfigurationFile

    assert MotorConfigurationFile.from_yaml_dict({"pellet": pellet}, source=None).profile_ack_deadlines is expected


def test_a_switch_that_is_not_a_boolean_is_refused():
    from reachaq.device import MotorConfigurationFile

    with pytest.raises(ValueError, match="profile_ack_deadlines"):
        MotorConfigurationFile.from_yaml_dict({"pellet": {"profile_ack_deadlines": "no"}}, source=None)


# -- review minors -----------------------------------------------------------------------

def test_an_estop_during_a_held_busy_set_keeps_the_set_and_sends_it(board):
    # Review M-D: the held SET moves nothing; it stays queued and goes out on the latched board.
    device = board.device
    device.busy_set_retry_delay_s = 5.0
    sets = board.answer("set_motor_x", -16)
    token = board.send(SystemCommandKind.SET_X, 1.0, token="held-set")
    _sent(board, sets)
    assert device._boards_pending_ctx[Target.PELLET_DEVICE].busy_set_entry is not None

    device.estop("test stop")

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert len(sets) == 2
    assert not board.wait_for(SystemStatusMessageKind.COMMAND_REFUSED, token, timeout=0.0)


def test_an_unhandled_command_does_not_leave_motion_in_flight(board):
    # Review M-E: a token whose command is never executed must not stay "in flight".
    device = board.device
    device._command_handlers.pop(SystemCommandKind.MOVE_LOAD_SERVO)

    board.send(SystemCommandKind.MOVE_LOAD_SERVO, 35.0, token="unhandled")
    deadline = time.monotonic() + 2.0
    while "unhandled" in device._motion_tokens and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not device.motion_possibly_in_flight
