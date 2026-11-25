"""Per-move ack deadlines: a stepper frame must be acknowledged within its own motion profile.

Pellet firmware 2.3.1 acknowledges a move when it is done, so the ack of a move
of known length is due a known time after it was sent. A stalled or runaway
move (the 2026-11-24 15:55 z runaway on christielab10) is then stopped within a
fraction of a second, not after the fixed 3 s ack timeout. Measured on
christielab10 on 2026-11-24 (sessions 000-006, 683 moves, 60 homes at
max_vel 120 mm/s, max_acc 600 mm/s^2, home_vel 30 mm/s): the ack came at most
0.24 s after the computed profile time, and at most 1.68 times it.
"""

import math
import threading
import time

import pytest

from reachaq.core import Motor
from reachaq.core.message import SystemCommandKind, SystemStatusMessageKind
from reachaq.device import StepperConfig, Target
from reachaq.device import can_device as can_device_module
from reachaq.device.can_interface import _STEPPER_MAX_POS
from reachaq.device.device_interface import Acknowledge
from reachaq.device.motion_deadline import (
    DEADLINE_FACTOR,
    DEADLINE_FLOOR_S,
    DEADLINE_MARGIN_S,
    ack_deadline,
    profile_time,
)

from can_device_refusal_test import _Board, _LOST

_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)


def _rig_config(**overrides):
    # christielab10's motor_config.yaml, 2026-11-24: max_vel 120, max_acc 600, home_vel 30.
    config = StepperConfig()
    config.maximum_velocity = overrides.get("v", 120.0)
    config.maximum_acceleration = overrides.get("a", 600.0)
    config.homing_velocity = overrides.get("home_v", 30.0)
    return config


# -- the maths -----------------------------------------------------------------

def test_a_long_move_has_a_trapezoid_profile():
    # 30 mm at 120 mm/s, 600 mm/s^2: 0.2 s ramp each way (12 mm each), 6 mm at speed.
    assert profile_time(30.0, 120.0, 600.0) == pytest.approx(30.0 / 120.0 + 120.0 / 600.0)


def test_a_short_move_has_a_triangle_profile():
    # 14.5 mm never reaches 120 mm/s (that needs 24 mm): t = 2 * sqrt(d / a).
    assert profile_time(14.5, 120.0, 600.0) == pytest.approx(2.0 * math.sqrt(14.5 / 600.0))
    assert profile_time(-14.5, 120.0, 600.0) == profile_time(14.5, 120.0, 600.0)


def test_the_deadline_is_the_profile_times_the_factor_plus_the_margin():
    profile = profile_time(14.5, 120.0, 600.0)

    assert ack_deadline(profile) == pytest.approx(profile * DEADLINE_FACTOR + DEADLINE_MARGIN_S)
    assert ack_deadline(profile) < 1.0


def test_a_short_move_gets_the_floor():
    assert ack_deadline(profile_time(0.0, 120.0, 600.0)) == DEADLINE_FLOOR_S
    assert ack_deadline(profile_time(0.5, 120.0, 600.0)) == DEADLINE_FLOOR_S


@pytest.mark.parametrize("v, a", [(0.0, 600.0), (120.0, 0.0), (math.nan, 600.0), (120.0, math.inf)])
def test_a_profile_without_usable_limits_is_unknown(v, a):
    assert profile_time(10.0, v, a) is None
    assert ack_deadline(None) is None


def test_the_constants_cover_every_ack_measured_on_christielab10():
    # The worst cases of 2026-11-24 (ack delay, profile): the largest excess over the
    # profile, the largest ratio, a zero-length move, and a 35 mm home at 30 mm/s.
    for delay, profile in ((0.640, 0.400), (0.552, 0.337), (0.104, 0.0), (1.240, profile_time(35.0, 30.0, 600.0))):
        assert ack_deadline(profile) > delay + 0.1, (delay, profile)


# -- what CanDevice computes per frame -------------------------------------------

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


def test_a_move_from_a_known_position_is_due_after_its_own_profile(board):
    device = board.device
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0

    deadline, end = device._move_deadline(Motor.PELLET_Z_MOTOR, 15.5)

    assert end == 15.5
    assert deadline == pytest.approx(ack_deadline(profile_time(14.5, 120.0, 600.0)))


def test_a_move_from_an_unknown_position_is_given_the_full_travel(board):
    deadline, _end = board.device._move_deadline(Motor.PELLET_Z_MOTOR, 15.5)

    assert deadline == pytest.approx(ack_deadline(profile_time(_STEPPER_MAX_POS, 120.0, 600.0)))


def test_a_home_is_due_after_the_distance_to_the_switch_at_the_homing_speed(board):
    device = board.device
    device._commanded_pos[Motor.PELLET_Y_MOTOR] = 20.0

    deadline, end = device._move_deadline(Motor.PELLET_Y_MOTOR, 0.0, home=True)

    assert end == 0.0
    assert deadline == pytest.approx(ack_deadline(profile_time(20.0, 30.0, 600.0)))
    unknown, _ = device._move_deadline(Motor.PELLET_X_MOTOR, 0.0, home=True)
    assert unknown == pytest.approx(ack_deadline(profile_time(_STEPPER_MAX_POS, 30.0, 600.0)))


def test_send_to_fixed_xyz_sums_the_axes_moved_one_after_another(board):
    device = board.device
    device._commanded_pos.update({Motor.PELLET_X_MOTOR: 0.0, Motor.PELLET_Y_MOTOR: 0.0,
                                  Motor.PELLET_Z_MOTOR: 0.0})
    device._last_send_pos = device._last_send_pos.replace(x=10.0, y=20.0, z=5.0)

    deadline, ends = device._fixed_xyz_deadline()

    total = sum(profile_time(d, 120.0, 600.0) for d in (10.0, 20.0, 5.0))
    assert deadline == pytest.approx(ack_deadline(total))
    assert ends == {Motor.PELLET_X_MOTOR: 10.0, Motor.PELLET_Y_MOTOR: 20.0, Motor.PELLET_Z_MOTOR: 5.0}


def test_an_explicit_uuid_ack_timeout_still_wins(board):
    device = board.device
    device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 7.0
    moves = board.answer("move_motor_z", _LOST)
    board.send(SystemCommandKind.MOVE_Z, 15.5)
    deadline = time.monotonic() + 2.0
    while not moves and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)

    assert device._boards_pending_ctx[Target.PELLET_DEVICE].ack_timeout_used == 7.0


def test_a_normal_move_acknowledged_inside_its_deadline_does_not_trip(board):
    # 1 -> 15.5 mm: deadline about 0.79 s; the board acks after 0.45 s, as it did at 15:55:20.
    device = board.device
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0
    interface = board.interface

    def slow_move(*args, **kwargs):
        uuid = interface.next_uuid()
        threading.Timer(0.45, lambda: device._handle_ack(Acknowledge(uuid=uuid))).start()
        return True

    board._replace("move_motor_z", slow_move)
    token = board.send(SystemCommandKind.MOVE_Z, 15.5)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert interface.estop_frames == []


def test_the_1555_runaway_is_stopped_within_its_deadline_not_after_3_s(board, monkeypatch):
    # 15:55:53.48, christielab10: MOVE_Z 1.0 -> 15.5 at 120/600, never acknowledged. The
    # axis free-ran into the obstruction. Fake clock: the deadline runs on it alone.
    now = [1000.0]
    monkeypatch.setattr(can_device_module, "get_perf_now", lambda: now[0])
    device = board.device
    device._commanded_pos[Motor.PELLET_Z_MOTOR] = 1.0
    expected = ack_deadline(profile_time(14.5, 120.0, 600.0))
    assert expected < 0.8
    moves = board.answer("move_motor_z", _LOST)
    token = board.send(SystemCommandKind.MOVE_Z, 15.5)
    deadline = time.monotonic() + 2.0
    while not moves and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.1)
    assert device._boards_pending_ctx[Target.PELLET_DEVICE].ack_timeout_used == pytest.approx(expected)

    now[0] += expected - 0.02
    time.sleep(0.3)
    assert board.interface.estop_frames == [], "not before its deadline"

    now[0] += 0.04
    refusal = board.refusal_of(token)

    assert board.interface.estop_frames == [True]
    assert refusal.estop_reason == f"MOVE_Z was not acknowledged within {expected:.2f} s"
