"""The host stops the pellet stage with ESTOP when a stepper may be running on.

Pellet firmware 2.3.1 (v1.2 boards) has no motion timeout and no travel
clamp, and a limit-switch stop mid-move can leave the step timer free-running:
the axis keeps pulsing, and every later move is answered -EBUSY. ESTOP is the
one command that stops it (queue purged, timer off). It is one broadcast frame
with no uuid and no ack; the board then refuses motion with -EPERM until a
zero-payload ESTOP releases it, after which every axis needs a home.

So: a stepper frame whose ack times out is stopped, never re-sent; an -EBUSY
on a stepper command is stopped; the ESTOP is written at once, never queued
behind the command awaiting its ack, and never re-sent; what it stopped ends
with a refusal carrying its reason; the release goes out in order, before the
home queued after it.
"""

import logging
import threading
import time

import pytest

from autotrainer.core import Motor
from autotrainer.core.message import SystemCommandKind, SystemStatusMessageKind
from autotrainer.device import CommandRefusal, RefusedCommandClass, Target
from autotrainer.device.device_interface import Acknowledge

from can_device_refusal_test import _Board, _LOST

_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)


class _EstopBoard(_Board):
    """_Board that also records every ESTOP the device reports, and each frame in order."""

    def __init__(self):
        super().__init__()
        self.reported = []
        self.device._estop_callback = lambda reason, axes, sent: self.reported.append((reason, axes, sent))
        self.frames = []
        real_estop = self.interface.estop

        def estop(asserted=True):
            self.frames.append(("estop", bool(asserted)))
            return real_estop(asserted)

        self.interface.estop = estop

    def log_frames(self, name):
        """Record each call of interface.<name> in self.frames, in order with the ESTOPs."""
        real = getattr(self.interface, name)

        def call(*args, **kwargs):
            self.frames.append((name, args))
            return real(*args, **kwargs)

        self._replace(name, call)


@pytest.fixture
def board():
    b = _EstopBoard()
    b.start()
    try:
        yield b
    finally:
        b.close()


# -- ack timeouts -------------------------------------------------------------

@pytest.mark.parametrize("kind, data, sender", [
    (SystemCommandKind.MOVE_X, 1.0, "move_motor_x"),
    (SystemCommandKind.MOVE_Z, 15.5, "move_motor_z"),
    (SystemCommandKind.SEND_FIXED_XYZ, None, "fixed_position"),
], ids=["move-x", "move-z", "send-to-fixed-xyz"])
def test_a_lost_stepper_ack_stops_the_stage_and_is_never_resent(board, kind, data, sender):
    board.device.default_command_ack_timeout_duration = 0.2
    calls = board.answer(sender, _LOST)

    token = board.send(kind, data)
    refusal = board.refusal_of(token)
    time.sleep(0.5)  # two more timeouts: nothing is re-sent, no second ESTOP

    assert len(calls) == 1, "a stepper frame whose ack is lost must never be re-sent"
    assert board.interface.estop_frames == [True]
    assert refusal.stopped_by_estop
    assert "not acknowledged within 0.2 s" in refusal.estop_reason
    assert refusal.needs_home == _XYZ
    assert refusal.message.startswith("Pellet stage stopped: ")
    assert "Home before continuing" in refusal.message
    assert not board.acked(token)
    assert [axes for _reason, axes, _sent in board.reported] == [_XYZ]
    assert not board.device.motion_possibly_in_flight


def test_a_lost_home_ack_in_the_home_sequence_stops_it_and_sends_no_further_home(board):
    board.device.default_command_ack_timeout_duration = 0.2
    homes = board.answer("stepper_home", _LOST)

    token = board.send(SystemCommandKind.SEND_HOME)
    refusal = board.refusal_of(token)
    time.sleep(0.5)

    assert homes == [(Motor.PELLET_Y_MOTOR,)], "neither re-sent nor followed by the Z and X homes"
    assert board.interface.estop_frames == [True]
    assert refusal.command is SystemCommandKind.SEND_HOME
    assert refusal.command_class is RefusedCommandClass.STEPPER_HOME


def test_a_lost_send_to_fixed_xyz_ack_in_a_pellet_send_stops_the_stage(board):
    board.device.default_command_ack_timeout_duration = 0.2
    sends = board.answer("fixed_position", _LOST)

    token = board.send(SystemCommandKind.SEND_PELLET)
    refusal = board.refusal_of(token)
    time.sleep(0.5)

    assert len(sends) == 1
    assert board.interface.estop_frames == [True]
    assert refusal.command_class is RefusedCommandClass.FIXED_XYZ


def test_a_lost_ack_of_any_frame_of_a_stepper_sequence_stops_the_stage(board):
    # The load's delay frame: the sequence drives the steppers, so a lost ack in it stops them.
    board.device.default_command_ack_timeout_duration = 0.2
    delays = board.answer("delay", _LOST)

    token = board.send(SystemCommandKind.LOAD_PELLET)
    refusal = board.refusal_of(token, timeout=5.0)
    time.sleep(0.4)

    assert len(delays) == 1
    assert board.interface.estop_frames == [True]
    assert refusal.stopped_by_estop


def test_a_lost_set_ack_is_still_resent_and_stops_nothing(board):
    # SET_X/Y/Z only store the send position: re-sending one moves nothing.
    board.device.default_command_ack_timeout_duration = 0.2
    sets = board.answer("set_motor_y", _LOST)

    token = board.send(SystemCommandKind.SET_Y, 2.0)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert len(sets) == 2
    assert board.interface.estop_frames == []


def test_a_lost_stimulus_ack_is_still_skipped_without_a_stop(board):
    lost = board.answer("pulse_digital_output", _LOST)

    token = board.send(SystemCommandKind.PULSE_DIGITAL_OUTPUT, (3, 1000))

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert len(lost) == 1
    assert board.interface.estop_frames == []


# -- refusals -----------------------------------------------------------------

@pytest.mark.parametrize("kind, data, sender", [
    (SystemCommandKind.MOVE_Z, 15.5, "move_motor_z"),
    (SystemCommandKind.SEND_FIXED_XYZ, None, "fixed_position"),
], ids=["move-z", "send-to-fixed-xyz"])
def test_ebusy_on_a_stepper_command_stops_the_stage(board, kind, data, sender):
    # "Still moving": the axis whose timer free-runs answers -EBUSY to every move.
    board.answer(sender, -16)

    refusal = board.refusal_of(board.send(kind, data))
    deadline = time.monotonic() + 2.0
    while not board.interface.estop_frames and time.monotonic() < deadline:
        time.sleep(0.01)

    assert refusal.error == -16 and not refusal.stopped_by_estop
    assert board.interface.estop_frames == [True]
    assert "EBUSY" in board.reported[0][0]


def test_ebusy_on_a_resent_set_is_still_moving_and_stops_the_stage(board):
    board.device.default_command_ack_timeout_duration = 0.2
    board.answer("set_motor_x", _LOST, -16)

    refusal = board.refusal_of(board.send(SystemCommandKind.SET_X, 1.0))

    assert refusal.resent and refusal.outcome_unknown
    assert board.interface.estop_frames == [True]


@pytest.mark.parametrize("error", [-1, -5, -116])
def test_a_refusal_that_already_stopped_the_axis_sends_no_estop(board, error):
    board.answer("move_motor_x", error)

    board.refusal_of(board.send(SystemCommandKind.MOVE_X, 1.0))
    time.sleep(0.2)

    assert board.interface.estop_frames == []


def test_ebusy_on_a_servo_move_sends_no_estop(board):
    board.answer("move_servo_motor", -16)

    board.refusal_of(board.send(SystemCommandKind.MOVE_LOAD_SERVO, 35.0))
    time.sleep(0.2)

    assert board.interface.estop_frames == []


# -- the frame itself ---------------------------------------------------------

def test_estop_is_written_at_once_and_never_queued_behind_the_pending_move(board):
    board.device.default_command_ack_timeout_duration = 30.0  # the move would wait its ack for 30 s
    moves = board.answer("move_motor_z", _LOST)
    token = board.send(SystemCommandKind.MOVE_Z, 15.5, token="moving")
    deadline = time.monotonic() + 2.0
    while not moves and time.monotonic() < deadline:
        time.sleep(0.005)
    assert board.device.motion_possibly_in_flight

    sent = board.device.estop("test stop")

    # Written by the call itself, on this thread, while the move still awaits its ack.
    assert sent is True
    assert board.interface.estop_frames == [True]
    refusal = board.refusal_of(token)
    assert refusal.estop_reason == "test stop"
    time.sleep(0.3)
    assert len(moves) == 1, "the stopped move is never re-sent"
    assert board.interface.estop_frames == [True], "the ESTOP is never re-sent"
    assert not board.device.motion_possibly_in_flight
    ctx = board.device._boards_pending_ctx[Target.PELLET_DEVICE]
    assert ctx.is_available() and ctx.uuid is None


def test_estop_drops_the_queued_motion_commands_and_keeps_the_rest(board):
    board.device.default_command_ack_timeout_duration = 30.0
    board.answer("move_motor_z", _LOST)
    y_moves = board.spy("move_motor_y")
    pending = board.send(SystemCommandKind.MOVE_Z, 15.5, token="pending")
    queued_move = board.send(SystemCommandKind.MOVE_Y, 3.0, token="queued-move")
    queued_load = board.send(SystemCommandKind.LOAD_PELLET, None, token="queued-load")
    queued_set = board.send(SystemCommandKind.SET_X, 2.0, token="queued-set")
    time.sleep(0.1)

    board.device.estop("test stop")

    for token in (pending, queued_move, queued_load):
        assert board.refusal_of(token).estop_reason == "test stop"
    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, queued_set) is not None
    assert y_moves == []


def test_the_release_goes_out_after_the_stop_and_before_the_home_queued_after_it(board):
    board.log_frames("stepper_home")
    board.device.estop("test stop")
    board.device.release_estop()
    token = board.send(SystemCommandKind.SEND_HOME)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert board.frames[:3] == [
        ("estop", True), ("estop", False), ("stepper_home", (Motor.PELLET_Y_MOTOR,)),
    ]


def test_an_estop_on_a_dead_bus_is_logged_and_swallowed(board, caplog):
    def dead(asserted=True):
        raise OSError(100, "Network is down")

    board.interface.estop = dead
    with caplog.at_level(logging.ERROR, logger="autotrainer.device.can_device"):
        sent = board.device.estop("test stop")

    assert sent is False
    assert board.reported == [("test stop", _XYZ, False)]
    assert any("NOT SENT" in r.getMessage() and "test stop" in r.getMessage() for r in caplog.records)


def test_every_estop_is_logged_at_error_with_its_reason(board, caplog):
    with caplog.at_level(logging.ERROR, logger="autotrainer.device.can_device"):
        board.device.estop("operator STOP")

    assert any(r.levelno == logging.ERROR and "Pellet ESTOP sent: operator STOP" in r.getMessage()
               for r in caplog.records)


# -- motion possibly in flight --------------------------------------------------

def test_motion_in_flight_is_set_by_a_move_and_cleared_by_its_ack(board):
    released = threading.Event()

    def held_move(*args, **kwargs):
        # Take the uuid now and ack it only when the test says so.
        uuid = board.interface.next_uuid()

        def ack_later():
            released.wait(2)
            board.device._handle_ack(Acknowledge(uuid=uuid))

        threading.Thread(target=ack_later, daemon=True).start()
        return True

    board._replace("move_motor_x", held_move)
    token = board.send(SystemCommandKind.MOVE_X, 1.0)
    deadline = time.monotonic() + 2.0
    while not board.device.motion_possibly_in_flight and time.monotonic() < deadline:
        time.sleep(0.005)
    assert board.device.motion_possibly_in_flight

    released.set()
    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert not board.device.motion_possibly_in_flight


def test_a_set_awaiting_its_ack_is_not_motion_in_flight(board):
    board.device.default_command_ack_timeout_duration = 30.0
    sets = board.answer("set_motor_x", _LOST)
    board.send(SystemCommandKind.SET_X, 1.0)
    deadline = time.monotonic() + 2.0
    while not sets and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)

    assert board.device._boards_pending_ctx[Target.PELLET_DEVICE].uuid is not None
    assert not board.device.motion_possibly_in_flight


# -- the refusal that ends a stopped command -----------------------------------

def test_a_stopped_command_needs_every_axis_homed_and_says_why():
    refusal = CommandRefusal(
        token="t", error=-1, command=SystemCommandKind.MOVE_Z, step="MOVE_Z",
        command_class=RefusedCommandClass.STEPPER_MOVE, motors=(Motor.PELLET_Z_MOTOR,),
        estop_reason="MOVE_Z was not acknowledged within 3 s",
    )

    assert refusal.stopped_by_estop
    assert refusal.needs_home == _XYZ
    assert not refusal.outcome_unknown
    assert refusal.message.startswith(
        "Pellet stage stopped: MOVE_Z was not acknowledged within 3 s. Home before continuing.")


@pytest.mark.parametrize("command_class, error, expected", [
    (RefusedCommandClass.STEPPER_MOVE, -16, True),
    (RefusedCommandClass.STEPPER_HOME, -16, True),
    (RefusedCommandClass.FIXED_XYZ, -16, True),
    (RefusedCommandClass.SERVO_MOVE, -16, False),
    (RefusedCommandClass.PULSE, -16, False),
    (RefusedCommandClass.STEPPER_MOVE, -1, False),
    (RefusedCommandClass.STEPPER_MOVE, -5, False),
])
def test_only_ebusy_on_a_stepper_command_may_leave_an_axis_moving(command_class, error, expected):
    refusal = CommandRefusal(token="t", error=error, command=None, step="s", command_class=command_class)

    assert refusal.motion_may_continue is expected
