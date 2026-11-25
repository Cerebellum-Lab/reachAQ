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
from autotrainer.device import CanDevice, CommandRefusal, RefusedCommandClass, Target
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
    # Per-move ack deadlines (motion_deadline): christielab10's limits, every axis homed and
    # the send position known, so a move's deadline is well inside the 3 s these tests wait.
    from autotrainer.core import Offset3DTuple
    from autotrainer.device import StepperConfig

    for motor in _XYZ:
        config = StepperConfig()
        config.maximum_velocity, config.maximum_acceleration, config.homing_velocity = 120.0, 600.0, 30.0
        b.device._motor_configs[motor] = config
        b.device._commanded_pos[motor] = 0.0
    b.device._last_send_pos = Offset3DTuple(1.0, 1.0, 1.0)
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
    assert "not acknowledged within" in refusal.estop_reason
    assert board.device._boards_pending_ctx[Target.PELLET_DEVICE].ack_timeout_used is not None
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


def test_a_lost_delay_ack_inside_a_load_is_resent_without_an_estop(board):
    # Only stepper-motion frames stop the stage; a delay frame moves nothing (Ben, 2026-11-24).
    board.device.default_command_ack_timeout_duration = 0.2
    delays = board.answer("delay", _LOST, 0, 0, 0)

    token = board.send(SystemCommandKind.LOAD_PELLET)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token, timeout=5.0) is not None
    assert len(delays) == 4, "the lost delay is re-sent, then the load's other two run"
    assert board.interface.estop_frames == []


def test_a_lost_release_servo_ack_inside_a_send_is_resent_without_an_estop(board):
    board.device.default_command_ack_timeout_duration = 0.2
    releases = board.answer("release_pellet", _LOST, 0)

    token = board.send(SystemCommandKind.SEND_PELLET, {"pre_reveal_stimulus": (200, 1000, 2)})

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token, timeout=5.0) is not None
    assert len(releases) == 2
    assert board.interface.estop_frames == []


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


@pytest.mark.parametrize("kind, data, sender", [
    (SystemCommandKind.SEND_HOME, None, "stepper_home"),
    (SystemCommandKind.LOAD_PELLET, None, "move_motor_x"),
    (SystemCommandKind.SEND_PELLET, None, "fixed_position"),
], ids=["home", "load-sequence", "send-sequence-fixed-xyz"])
def test_ebusy_on_a_home_or_a_sequence_stops_the_stage(board, kind, data, sender):
    board.answer(sender, -16)

    board.refusal_of(board.send(kind, data))
    deadline = time.monotonic() + 2.0
    while not board.interface.estop_frames and time.monotonic() < deadline:
        time.sleep(0.01)

    assert board.interface.estop_frames == [True]


def test_a_busy_set_sends_no_estop_and_is_retried_until_the_axis_is_idle(board, caplog):
    # A SET moves nothing: -EBUSY on it is logged and the SET sent again a little later.
    board.device.busy_set_retry_delay_s = 0.05
    sets = board.answer("set_motor_x", -16, -16)
    y_moves = board.spy("move_motor_y")

    with caplog.at_level(logging.WARNING, logger="autotrainer.device.can_device"):
        token = board.send(SystemCommandKind.SET_X, 1.0, token="busy-set")
        after = board.send(SystemCommandKind.MOVE_Y, 2.0, token="after")
        assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None

    assert len(sets) == 3, "two busy answers, then the SET goes through"
    assert board.interface.estop_frames == []
    assert not board.wait_for(SystemStatusMessageKind.COMMAND_REFUSED, token, timeout=0.0)
    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, after) is not None
    assert len(y_moves) == 1, "the move queued behind the SET waits for it, then goes"
    assert any("busy" in r.getMessage() and "SET_X" in r.getMessage() for r in caplog.records)


def test_a_set_still_busy_after_its_retries_is_refused_without_an_estop(board):
    board.device.busy_set_retry_delay_s = 0.05
    board.device.busy_set_retry_count = 2
    sets = board.answer("set_motor_x", -16, -16, -16)

    refusal = board.refusal_of(board.send(SystemCommandKind.SET_X, 1.0))
    time.sleep(0.2)

    assert refusal.error == -16
    assert len(sets) == 3
    assert board.interface.estop_frames == []


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
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0  # explicit: stays pending
    moves = board.answer("move_motor_z", _LOST)
    token = board.send(SystemCommandKind.MOVE_Z, 15.5, token="moving")
    deadline = time.monotonic() + 2.0
    while not moves and time.monotonic() < deadline:
        time.sleep(0.005)
    assert board.device.motion_possibly_in_flight

    threads = []
    real = board.interface.estop
    board.interface.estop = lambda asserted=True: (threads.append(threading.current_thread()), real(asserted))[1]
    sent = board.device.estop("test stop")

    # Written by the call itself, on this thread, while the move still awaits its ack.
    assert sent is True
    assert board.interface.estop_frames == [True]
    assert threads == [threading.current_thread()], "written on the calling thread, not by the loop"
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
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0  # explicit: stays pending
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
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0  # explicit: stays pending
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


@pytest.mark.parametrize("command", ["SET_X", "SET_Y", "SET_Z"])
def test_ebusy_on_a_set_is_not_motion(command):
    refusal = CommandRefusal(token="t", error=-16, command=getattr(SystemCommandKind, command),
                             step=command, command_class=RefusedCommandClass.STEPPER_MOVE)

    assert refusal.motion_may_continue is False


# -- every trigger stops all axes ------------------------------------------------
# Pellet firmware 2.4.0's STOP can name axes; a stop naming only some leaves the others
# finishing their moves. Every host failure path sends the board-wide form: the ESTOP
# command with no axis selection.

def test_the_estop_frame_has_no_axis_selection():
    from autotrainer.device import socketcan_jerrycan as jc

    sent = []
    bus = jc.SocketCanJerryCAN.__new__(jc.SocketCanJerryCAN)
    bus.SendMessage = lambda msg, dst_id: sent.append(jc.encode_frame(msg, dst_id)) or 0

    assert bus.EStop(True) == 0
    assert bus.EStop(False) == 0

    (assert_id, assert_payload), (release_id, release_payload) = sent
    assert assert_id == jc.encode_can_id(jc.JerryCANCmdType.ESTOP, 0x1F), "broadcast, to every board"
    # The ESTOP payload is one byte, assert or release, with no axis field; then uuid 0 (none).
    assert assert_payload == b"\x01\x00" and release_payload == b"\x00\x00"
    assert release_id == assert_id


def _lost_move(board):
    board.device.default_command_ack_timeout_duration = 0.2
    board.answer("move_motor_z", _LOST)
    return board.send(SystemCommandKind.MOVE_Z, 15.5)


def _lost_fixed_xyz(board):
    board.device.default_command_ack_timeout_duration = 0.2
    board.answer("fixed_position", _LOST)
    return board.send(SystemCommandKind.SEND_FIXED_XYZ)


def _lost_sequence_frame(board):
    board.device.default_command_ack_timeout_duration = 0.2
    board.answer("stepper_home", _LOST)
    return board.send(SystemCommandKind.SEND_HOME)


def _ebusy(board):
    board.answer("move_motor_z", -16)
    return board.send(SystemCommandKind.MOVE_Z, 15.5)


def _ebusy_home(board):
    board.answer("stepper_home", -16)
    return board.send(SystemCommandKind.SEND_HOME)


def _public(board):
    board.device.estop("operator STOP")
    return None


@pytest.mark.parametrize("trigger", [
    _lost_move, _lost_fixed_xyz, _lost_sequence_frame, _ebusy, _ebusy_home, _public,
], ids=["move-ack-timeout", "fixed-xyz-ack-timeout", "sequence-ack-timeout", "ebusy",
        "ebusy-home", "estop-call"])
def test_every_device_trigger_sends_the_all_axes_estop(board, trigger):
    calls = []
    real = board.interface.estop

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    board.interface.estop = spy
    token = trigger(board)
    if token is not None:
        board.refusal_of(token)
    deadline = time.monotonic() + 2.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)

    assert calls == [((True,), {})], "one ESTOP, asserted, naming no axis"
    assert [axes for _reason, axes, _sent in board.reported] == [_XYZ]


# -- review b2d30887: crashes, teardown races, ordering ----------------------------

def _wait(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def _pending_move(board):
    board.device.default_command_ack_timeout_duration = 30.0
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0  # explicit: stays pending
    moves = board.answer("move_motor_z", _LOST)
    board.send(SystemCommandKind.MOVE_Z, 15.5, token="moving")
    assert _wait(lambda: moves)
    return moves


def test_a_device_disconnect_during_a_move_sends_an_estop_first(board):
    # I-1: CanDevice.disconnect is the device-level choke point: the reader's finally,
    # HardwareModel's teardown and connect()'s own disconnect all pass through it.
    _pending_move(board)

    board.device.disconnect()

    assert board.interface.estop_frames == [True]
    assert "disconnected during a move" in board.reported[0][0]


def test_a_reader_crash_during_a_move_sends_an_estop():
    # I-1: the transport worker's finally disconnects the device; nothing else would stop it.
    from queue import Queue
    from autotrainer.device import DeviceApi, DeviceConnection

    device = CanDevice(api=DeviceApi(), force_emulation=True, required_targets=(Target.PELLET_DEVICE,))
    interface = device.device_interface
    connection = DeviceConnection(device, message_queue=Queue())
    connection.request_connect()
    try:
        assert _wait(lambda: device.connected)
        device.default_command_ack_timeout_duration = 30.0
        def lost_move(*args, **kwargs):
            interface.next_uuid()
            return True

        interface.move_motor_z = lost_move
        device._init_handlers()
        device.notify_message(SystemCommandKind.MOVE_Z, 15.5, context="moving")
        assert _wait(lambda: device.motion_possibly_in_flight)

        def crash(*args, **kwargs):
            raise RuntimeError("reader decode failure")

        interface.read = crash

        assert _wait(lambda: interface.estop_frames == [True]), interface.estop_frames
    finally:
        connection.request_disconnect()
        connection.join()


def test_a_command_loop_crash_during_a_move_sends_an_estop(board):
    # I-2: a crash ends as CanFailureKind.COMMAND, which starts no teardown.
    _pending_move(board)

    board.device.notify_message("not a command kind", None, context="crash")

    assert _wait(lambda: board.interface.estop_frames == [True])
    assert "command loop failed" in board.reported[0][0]


def test_a_disconnect_between_the_frames_of_a_send_sends_an_estop(board):
    # I-3: during the pre-reveal delay no stepper frame awaits its ack, but the send is not done.
    board.device.default_command_ack_timeout_duration = 30.0
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0  # explicit: stays pending
    delays = board.answer("delay", _LOST)
    board.send(SystemCommandKind.SEND_PELLET, {"pre_reveal_stimulus": (200, 1000, 2)}, token="send")
    assert _wait(lambda: delays)
    ctx = board.device._boards_pending_ctx[Target.PELLET_DEVICE]
    assert not ctx.stepper_motion

    assert board.device.motion_possibly_in_flight, "a motion command is still pending"
    board.device.disconnect()

    assert board.interface.estop_frames == [True]


def test_a_queued_motion_command_is_motion_possibly_in_flight(board):
    # I-3: queued behind a pending SET, the move has not been sent yet.
    board.device.default_command_ack_timeout_duration = 30.0
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0  # explicit: stays pending
    sets = board.answer("set_motor_x", _LOST)
    board.send(SystemCommandKind.SET_X, 1.0, token="set")
    assert _wait(lambda: sets)
    assert not board.device.motion_possibly_in_flight

    board.send(SystemCommandKind.MOVE_Y, 2.0, token="queued")

    assert board.device.motion_possibly_in_flight


def test_an_idle_device_disconnect_sends_no_estop(board):
    token = board.send(SystemCommandKind.MOVE_Y, 2.0)
    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert _wait(lambda: not board.device.motion_possibly_in_flight)

    board.device.disconnect()

    assert board.interface.estop_frames == []


def test_the_ebusy_estop_is_written_before_the_refusal_reaches_its_waiters(board):
    # M-1: the frame first, then the listeners.
    seen = []
    real = board.device.api.message_callback

    def callback(kind, data):
        if kind == SystemStatusMessageKind.COMMAND_REFUSED:
            seen.append(list(board.interface.estop_frames))
        real(kind, data)

    board.device.api.message_callback = callback
    board.answer("move_motor_z", -16)

    board.refusal_of(board.send(SystemCommandKind.MOVE_Z, 15.5))

    assert seen == [[True]]


def test_estop_queues_its_clean_up_before_it_notifies_listeners():
    # M-2: so the stopped move ends "stopped", not refused by the board's -EPERM ack.
    from autotrainer.device import DeviceApi
    from autotrainer.device.can_device import _estop_abort

    device = CanDevice(api=DeviceApi(), force_emulation=True, required_targets=(Target.PELLET_DEVICE,))
    seen = []
    device._estop_callback = lambda reason, axes, sent: seen.append(
        [item[0] for item in list(device._commands_queue.queue)])

    device.estop("test stop")

    assert seen == [[_estop_abort]]


def test_motion_is_in_flight_before_the_move_frame_is_written(board):
    # M-4: a concurrent teardown must see the move from the moment it is on its way.
    seen = []
    real = board.interface.move_motor_x

    def move(*args, **kwargs):
        seen.append(board.device.motion_possibly_in_flight)
        return real(*args, **kwargs)

    board._replace("move_motor_x", move)
    token = board.send(SystemCommandKind.MOVE_X, 1.0)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert seen == [True]


def test_a_failed_estop_write_is_tried_once_more(board):
    # M-8: ESTOP latches and is idempotent; one immediate retry of a write error.
    calls = []
    real = board.interface.estop

    def flaky(asserted=True):
        calls.append(asserted)
        if len(calls) == 1:
            raise OSError(105, "No buffer space available")
        return real(asserted)

    board.interface.estop = flaky

    assert board.device.estop("test stop") is True
    assert calls == [True, True]
    assert board.reported == [("test stop", _XYZ, True)]
