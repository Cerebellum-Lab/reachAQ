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

from reachaq.core import Motor
from reachaq.core.message import SystemCommandKind, SystemStatusMessageKind
from reachaq.device import CanDevice, CommandRefusal, RefusedCommandClass, Target
from reachaq.device.device_interface import Acknowledge

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
    from reachaq.core import Offset3DTuple
    from reachaq.device import StepperConfig

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

    with caplog.at_level(logging.WARNING, logger="reachaq.device.can_device"):
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
    with caplog.at_level(logging.ERROR, logger="reachaq.device.can_device"):
        sent = board.device.estop("test stop")

    assert sent is False
    assert board.reported == [("test stop", _XYZ, False)]
    assert any("NOT SENT" in r.getMessage() and "test stop" in r.getMessage() for r in caplog.records)


def test_every_estop_is_logged_at_error_with_its_reason(board, caplog):
    with caplog.at_level(logging.ERROR, logger="reachaq.device.can_device"):
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
    from reachaq.device import socketcan_jerrycan as jc

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
    from reachaq.device import DeviceApi, DeviceConnection

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
    from reachaq.device import DeviceApi
    from reachaq.device.can_device import _estop_abort

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


# -- pellet 2.4.0 (capability bit 9): STEPPER_STOP instead of ESTOP ------------------
# The same entry point (CanDevice.estop / _estop_in_loop) sends an all-axes STEPPER_STOP
# when the board advertises bit 9, and ESTOP on a 2.3.1 board, which the tests above use.

from reachaq.device import StepperFault  # noqa: E402

_BIT_9 = 1 << 9


class _SafetyBoard(_EstopBoard):
    """_EstopBoard with capability bit 9 whose reader also hands on STEPPER_FAULT frames,
    in the order the board sent them, before the acks that follow."""

    def __init__(self, capabilities=0x217):
        super().__init__()
        self.interface.board_capabilities = capabilities
        real_stop = self.interface.stepper_stop

        def stepper_stop(motor_mask=0):
            self.frames.append(("stop", int(motor_mask)))
            return real_stop(motor_mask)

        self.interface.stepper_stop = stepper_stop

    def _deliver(self):
        while not self._stop.is_set():
            for message in self.interface.read(64):
                if isinstance(message, Acknowledge):
                    self.device._handle_ack(message)
                elif isinstance(message, StepperFault):
                    self.device.notify_data([message])
            time.sleep(0.001)

    def answer_with_faults(self, name, error, *faults):
        """interface.<name> answers once: these STEPPER_FAULT frames (reason, motor), with
        its uuid, then its ack with `error`, in that order as the board sends them."""
        calls = []

        def call(*args, **kwargs):
            calls.append(args)
            uuid = self.interface.next_uuid()
            for reason, motor in faults:
                self.interface._messages.append(_fault(reason, motor, uuid))
            if error is not None:
                self.interface._messages.append(Acknowledge(uuid=uuid, error=error))
            return True

        self._replace(name, call)
        return calls


def _fault(reason, motor, uuid, *, flags=0x02, issued=571, planned=1142):
    return StepperFault(target=Target.PELLET_DEVICE, motor=motor, reason=reason, flags=flags,
                        pulses_issued=issued, pulses_planned=planned, believed_position_turns=2.5,
                        believed_position_mm=6.09, pulses_per_turn=192.0, uuid=uuid)


@pytest.fixture
def safety_board():
    b = _SafetyBoard()
    b.start()
    from reachaq.core import Offset3DTuple
    from reachaq.device import StepperConfig

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


def _stops(board):
    return [frame for frame in board.frames if frame[0] in ("stop", "estop")]


@pytest.mark.parametrize("trigger", [
    _lost_move, _lost_fixed_xyz, _lost_sequence_frame, _ebusy, _ebusy_home, _public,
], ids=["move-ack-timeout", "fixed-xyz-ack-timeout", "sequence-ack-timeout", "ebusy",
        "ebusy-home", "estop-call"])
def test_on_a_2_4_0_board_every_trigger_sends_one_all_axes_stop_and_no_estop(safety_board, trigger):
    board = safety_board
    token = trigger(board)
    if token is not None:
        refusal = board.refusal_of(token)
        assert refusal.motion_safety
    assert _wait(lambda: board.interface.stop_frames)
    time.sleep(0.3)  # no second stop, no re-send

    assert board.interface.stop_frames == [0], "one STOP, motor_mask 0: every stepper"
    assert board.interface.estop_frames == []
    assert board.device.last_stop_frame == "STOP"
    assert [axes for _reason, axes, _sent in board.reported] == [_XYZ]
    # The STOP's own ACK (uuid 0) was taken as such, not left for a command.
    assert _wait(lambda: board.device._stop_acks_outstanding == 0)


def test_a_2_3_1_board_still_gets_estop_and_never_stop(board):
    _lost_move(board)
    assert _wait(lambda: board.interface.estop_frames)

    assert board.interface.estop_frames == [True]
    assert board.interface.stop_frames == []
    assert board.device.last_stop_frame == "ESTOP"
    assert not board.device.stepper_motion_safety


def test_a_stop_that_cannot_be_written_falls_back_to_estop(safety_board):
    def dead(motor_mask=0):
        raise OSError(100, "Network is down")

    safety_board.interface.stepper_stop = dead

    assert safety_board.device.estop("test stop") is True

    assert safety_board.interface.estop_frames == [True]
    assert safety_board.device.last_stop_frame == "ESTOP"


def test_the_release_before_a_home_is_still_estop_release_on_2_4_0(safety_board):
    # Harmless on a board that is not latched, needed on one a fallback ESTOP latched.
    safety_board.device.estop("test stop")
    safety_board.device.release_estop()

    assert _wait(lambda: safety_board.interface.estop_frames == [False])
    assert safety_board.interface.stop_frames == [0]


def test_the_late_ecanceled_of_the_stopped_move_is_recognised(safety_board, caplog):
    board = safety_board
    board.device.default_command_ack_timeout_duration = 30.0
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0
    moves = board.answer("move_motor_z", _LOST)
    token = board.send(SystemCommandKind.MOVE_Z, 15.5, token="moving")
    assert _wait(lambda: moves)
    uuid = board.interface.uuid()

    with caplog.at_level(logging.INFO, logger="reachaq.device.can_device"):
        board.device.estop("operator STOP")
        refusal = board.refusal_of(token)
        board.interface._messages.append(Acknowledge(uuid=uuid, error=-140))
        assert _wait(lambda: "ended a stopped command" in caplog.text)

    assert refusal.stopped_by_estop
    assert f"uuid={uuid}) with ECANCELED" in caplog.text


# -- the board's own faults (STEPPER_FAULT and the new ACK codes) ----------------------

@pytest.mark.parametrize("error, name", [(-62, "ETIME"), (-14, "EFAULT"), (-5, "EIO")])
def test_on_2_4_0_a_board_fault_ack_stops_every_axis_and_is_blocking(safety_board, error, name):
    board = safety_board
    board.answer("move_motor_z", error)

    refusal = board.refusal_of(board.send(SystemCommandKind.MOVE_Z, 15.5))
    assert _wait(lambda: board.interface.stop_frames)

    assert refusal.error == error and refusal.error_name == name
    assert refusal.blocking_fault and not refusal.stopped_by_estop
    assert board.interface.stop_frames == [0]
    assert name in board.reported[0][0]
    assert Motor.PELLET_Z_MOTOR in refusal.needs_home


@pytest.mark.parametrize("error", [-62, -14, -5])
def test_on_2_3_1_the_same_codes_stop_nothing(board, error):
    board.answer("move_motor_z", error)

    refusal = board.refusal_of(board.send(SystemCommandKind.MOVE_Z, 15.5))
    time.sleep(0.2)

    assert not refusal.blocking_fault
    assert board.interface.estop_frames == []


def test_erange_is_refused_without_a_stop_and_without_needs_home(safety_board):
    board = safety_board
    board.answer("move_motor_z", -34)

    refusal = board.refusal_of(board.send(SystemCommandKind.MOVE_Z, 15.5))
    time.sleep(0.2)

    assert refusal.error_name == "ERANGE"
    assert refusal.needs_home == ()
    assert "past axis Z's travel" in refusal.message
    assert board.interface.stop_frames == [] and board.interface.estop_frames == []


@pytest.mark.parametrize("reason", [1, 2, 3, 4, 5])
def test_a_fault_frame_before_its_ack_stops_the_stage_once(safety_board, reason):
    board = safety_board
    board.answer_with_faults("move_motor_z", -14, (reason, Motor.PELLET_Z_MOTOR))

    token = board.send(SystemCommandKind.MOVE_Z, 15.5)
    refusal = board.refusal_of(token)
    time.sleep(0.3)

    assert board.interface.stop_frames == [0], "the fault stops; its -EFAULT ack does not stop again"
    assert refusal.stopped_by_estop, "the STOP the fault sent ended the move"
    assert "the pellet board reported a fault: axis Z" in refusal.estop_reason
    faults = [d for k, d in board.messages if k == SystemStatusMessageKind.PELLET_STEPPER_FAULT]
    assert [f.reason for f in faults] == [reason]
    assert len(board.reported) == 1


def test_a_fault_frame_after_its_ack_does_not_stop_the_stage_again(safety_board):
    # H6: either order. The -EFAULT ack stops; the late fault frame is only reported.
    board = safety_board
    calls = []

    def call(*args, **kwargs):
        calls.append(args)
        uuid = board.interface.next_uuid()
        board.interface._messages.append(Acknowledge(uuid=uuid, error=-14))

        def late():
            time.sleep(0.1)
            board.interface._messages.append(_fault(2, Motor.PELLET_Z_MOTOR, uuid))

        threading.Thread(target=late, daemon=True).start()
        return True

    board._replace("move_motor_z", call)
    refusal = board.refusal_of(board.send(SystemCommandKind.MOVE_Z, 15.5))
    assert _wait(lambda: any(k == SystemStatusMessageKind.PELLET_STEPPER_FAULT for k, _ in board.messages))
    time.sleep(0.1)

    assert refusal.error_name == "EFAULT"
    assert board.interface.stop_frames == [0]
    assert len(board.reported) == 1


def test_a_fixed_xyz_refused_whole_for_needs_home_prompts_a_home_and_stops_nothing(safety_board):
    board = safety_board
    board.answer_with_faults("fixed_position", -16, (6, Motor.PELLET_X_MOTOR), (6, Motor.PELLET_Z_MOTOR))

    refusal = board.refusal_of(board.send(SystemCommandKind.SEND_FIXED_XYZ))
    time.sleep(0.3)

    assert refusal.needs_home_axes == (Motor.PELLET_X_MOTOR, Motor.PELLET_Z_MOTOR)
    assert refusal.needs_home == (Motor.PELLET_X_MOTOR, Motor.PELLET_Z_MOTOR)
    assert not refusal.motion_may_continue
    assert "Pellet axis X/Z needs a home" in refusal.message and "send it again" in refusal.message
    assert board.interface.stop_frames == [] and board.interface.estop_frames == []
    assert board.reported == []


def test_a_bare_ebusy_on_fixed_xyz_without_a_needs_home_frame_still_stops(safety_board):
    # The prompt is keyed on the NEEDS_HOME frame, not on -EBUSY: a bare one may be a move.
    board = safety_board
    board.answer("fixed_position", -16)

    refusal = board.refusal_of(board.send(SystemCommandKind.SEND_FIXED_XYZ))
    assert _wait(lambda: board.interface.stop_frames)

    assert refusal.needs_home_axes == ()
    assert refusal.motion_may_continue
    assert refusal.needs_home == _XYZ


def test_an_idle_2_4_0_teardown_sends_a_quiet_stop(safety_board):
    token = safety_board.send(SystemCommandKind.MOVE_Y, 2.0)
    assert safety_board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert _wait(lambda: not safety_board.device.motion_possibly_in_flight)

    safety_board.device.disconnect()

    assert safety_board.interface.stop_frames == [0], "H1: a STOP before CAN closes"
    assert safety_board.interface.estop_frames == []
    assert safety_board.reported == [], "nothing was moving: no stop is reported"


def test_a_2_4_0_disconnect_during_a_move_sends_a_reported_stop(safety_board):
    _pending_move(safety_board)

    safety_board.device.disconnect()

    assert safety_board.interface.stop_frames == [0]
    assert "disconnected during a move" in safety_board.reported[0][0]


# -- CommandRefusal: the 2.4.0 codes ----------------------------------------------------

@pytest.mark.parametrize("error, name", [
    (-140, "ECANCELED"), (-62, "ETIME"), (-14, "EFAULT"), (-34, "ERANGE"), (-5, "EIO"), (-16, "EBUSY"),
])
def test_the_2_4_0_ack_codes_have_their_names(error, name):
    from reachaq.device import board_error_name

    assert board_error_name(error) == name


@pytest.mark.parametrize("error, motion_safety, expected", [
    (-62, True, True), (-14, True, True), (-5, True, True),
    (-62, False, False), (-14, False, False), (-5, False, False),
    (-140, True, False), (-34, True, False), (-16, True, False), (-1, True, False),
])
def test_only_the_board_fault_codes_on_a_2_4_0_board_are_blocking(error, motion_safety, expected):
    refusal = CommandRefusal(token="t", error=error, command=SystemCommandKind.MOVE_Z, step="MOVE_Z",
                             command_class=RefusedCommandClass.STEPPER_MOVE, motors=(Motor.PELLET_Z_MOTOR,),
                             motion_safety=motion_safety)

    assert refusal.blocking_fault is expected


def test_a_blocking_code_on_a_servo_or_a_set_is_not_a_board_stepper_fault():
    servo = CommandRefusal(token="t", error=-5, command=None, step="s",
                           command_class=RefusedCommandClass.SERVO_MOVE, motion_safety=True)
    a_set = CommandRefusal(token="t", error=-62, command=SystemCommandKind.SET_X, step="SET_X",
                           command_class=RefusedCommandClass.STEPPER_MOVE, motion_safety=True)

    assert not servo.blocking_fault and not a_set.blocking_fault


@pytest.mark.parametrize("error, command_class, expected", [
    (-140, RefusedCommandClass.STEPPER_MOVE, (Motor.PELLET_Z_MOTOR,)),
    (-62, RefusedCommandClass.STEPPER_MOVE, (Motor.PELLET_Z_MOTOR,)),
    (-14, RefusedCommandClass.STEPPER_HOME, (Motor.PELLET_Z_MOTOR,)),
    (-14, RefusedCommandClass.FIXED_XYZ, _XYZ),
    (-34, RefusedCommandClass.STEPPER_MOVE, ()),
])
def test_what_the_2_4_0_codes_leave_needing_a_home(error, command_class, expected):
    refusal = CommandRefusal(token="t", error=error, command=SystemCommandKind.MOVE_Z, step="s",
                             command_class=command_class, motors=(Motor.PELLET_Z_MOTOR,), motion_safety=True)

    assert refusal.needs_home == expected


# -- review I1, m1, m4 -------------------------------------------------------------------

def test_a_2_4_0_board_rebooted_into_2_3_1_is_stopped_with_estop(safety_board):
    # The interface forgets bit 9 on the reboot (socketcan_backend_test); the next stop is
    # ESTOP, both while the board has not answered and once it says 0x04.
    board = safety_board
    board.interface.board_capabilities = None  # the reboot was seen
    board.device.estop("MOVE_Z was not acknowledged within 0.9 s")
    board.interface.board_capabilities = 0x04  # it answered: 2.3.1
    _lost_move(board)
    assert _wait(lambda: len(board.interface.estop_frames) == 2)

    assert board.interface.stop_frames == []
    assert board.interface.estop_frames == [True, True]
    assert board.device.last_stop_frame == "ESTOP"


def test_a_failed_teardown_stop_says_no_estop_follows(safety_board, caplog):
    safety_board.interface.stepper_stop = lambda motor_mask=0: False

    with caplog.at_level(logging.WARNING, logger="reachaq.device.can_device"):
        safety_board.device.disconnect()

    assert safety_board.interface.estop_frames == []
    assert "STOP NOT SENT" in caplog.text and "trying ESTOP" not in caplog.text


def test_a_reissued_uuid_forgets_its_last_stop(safety_board):
    # m4: the 8-bit uuid wraps; a new command on an old uuid is not "already stopped".
    board = safety_board
    calls = board.answer("move_motor_z", _LOST)
    board.device.default_command_ack_timeout_duration = 30.0
    board.device._motor_configs[Motor.PELLET_Z_MOTOR].uuid_ack_timeout = 30.0
    uuid = board.interface.uuid() % 255 + 1  # the uuid the move will get
    board.device._remember_stopped_uuid(uuid)
    board.device._needs_home_faults[uuid] = (0.0, (Motor.PELLET_X_MOTOR,))

    board.send(SystemCommandKind.MOVE_Z, 15.5, token="moving")
    assert _wait(lambda: calls)
    assert _wait(lambda: board.device._boards_pending_ctx[Target.PELLET_DEVICE].uuid == uuid)

    assert not board.device._was_stopped(uuid)
    assert uuid not in board.device._needs_home_faults
