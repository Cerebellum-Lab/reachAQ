"""A board that refuses a command says so in its ACK, and the host must act on it.

Pellet firmware 2.3.1 refuses or interrupts motion with an errno in the ACK's
error field: -EPERM (ESTOP), -EIO (driver fault, drivers not proven), -EBUSY
(axis moving or needing a home), -ETIMEDOUT (enable off after host silence).
-EAGAIN means the axis is already there and is success. These are picolibc
numbers, not the host's errno values.

A refusal must stop the command's compound sequence, must never be
acknowledged to the caller as a success, and must reach every waiter as a
COMMAND_REFUSED message.
"""

import errno
import queue
import threading
import time

import pytest

from autotrainer.core import Motor, Offset3DTuple
from autotrainer.core.message import (
    MessageHandler,
    SystemCommandKind,
    SystemStatusMessageKind,
)
from autotrainer.core.message.message_handler import TERMINATE
from autotrainer.device import (
    CanDevice,
    CommandRefusal,
    DeviceApi,
    DeviceConnection,
    DigitalOutputs,
    MotorSteps,
    RefusedCommandClass,
    Target,
    board_error_name,
)
from autotrainer.device.can_device import _uuid_ack, default_move_retract
from autotrainer.device.device_interface import Acknowledge

_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)
_LOST = None  # an answer that takes a uuid and never acks it


class _Board:
    """A connected emulated pellet board with scripted acks.

    The command handler runs on its own thread, and a pump thread hands the
    emulator's acks back as DeviceConnection's reader would. Only acks are
    delivered, so the stepper status the emulator reports cannot overwrite
    state a test sets.
    """

    def __init__(self):
        self.messages = []
        self._changed = threading.Condition()
        self.device = CanDevice(
            api=DeviceApi(message_callback=self._on_message),
            force_emulation=True,
            required_targets=(Target.PELLET_DEVICE,),
        )
        self.interface = self.device.device_interface
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._deliver, daemon=True)

    def _on_message(self, kind, data):
        with self._changed:
            self.messages.append((kind, data))
            self._changed.notify_all()

    def _deliver(self):
        while not self._stop.is_set():
            for message in self.interface.read(64):
                if isinstance(message, Acknowledge):
                    self.device._handle_ack(message)
            time.sleep(0.001)

    def start(self):
        self.interface.open()
        self._reader.start()
        self.device.connect()
        time.sleep(0.05)  # the handler thread is then waiting on its queue

    def close(self):
        self._stop.set()
        self._reader.join(1)
        self.device.disconnect()
        self.interface.close()

    def answer(self, name, *answers):
        """Make interface.<name> answer each call from the script, in turn.

        An int is acked with that error at once; _LOST takes a uuid and is
        never acked. Calls past the script run the real method. Returns the
        list of calls made.
        """
        real = getattr(self.interface, name)
        calls = []

        def call(*args, **kwargs):
            calls.append(args)
            if len(calls) > len(answers):
                return real(*args, **kwargs)
            uuid = self.interface.next_uuid()
            error = answers[len(calls) - 1]
            if error is not _LOST:
                self.interface._messages.append(Acknowledge(uuid=uuid, error=error))
            return True

        self._replace(name, call)
        return calls

    def spy(self, name):
        real = getattr(self.interface, name)
        calls = []

        def call(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        self._replace(name, call)
        return calls

    def _replace(self, name, method):
        setattr(self.interface, name, method)
        # The command handlers bind interface methods when they are built.
        self.device._init_handlers()

    def send(self, kind, data=None, token="refusal-test"):
        self.device.notify_message(kind, data, context=token)
        return token

    def wait_for(self, kind, token, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                for k, d in self.messages:
                    if k == kind and d[0] == token:
                        return d
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._changed.wait(remaining)

    def refusal_of(self, token, timeout=3.0) -> CommandRefusal:
        data = self.wait_for(SystemStatusMessageKind.COMMAND_REFUSED, token, timeout)
        assert data is not None, "the refusal never reached the api"
        tok, _perf_c, refusal = data
        assert tok == token
        return refusal

    def acked(self, token):
        return any(
            k == SystemStatusMessageKind.ACKNOWLEDGE and d[0] == token
            for k, d in list(self.messages)
        )


@pytest.fixture
def board():
    b = _Board()
    b.start()
    try:
        yield b
    finally:
        b.close()


def test_handle_ack_queues_the_boards_error_with_the_uuid():
    device = CanDevice(
        api=DeviceApi(), force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )

    device._handle_ack(Acknowledge(uuid=7, error=-5))

    kind, data, ctx = device._commands_queue.get_nowait()
    assert kind is _uuid_ack
    assert ctx is None
    uuid, _perf_c, error = data
    assert (uuid, error) == (7, -5)


@pytest.mark.parametrize("error", [-1, -5, -16, -116, -22])
def test_a_refused_step_stops_the_compound_sequence_and_sends_no_further_step(board, error):
    # default_move_retract: send_z_rel 0, send_x_rel 0, send_y_rel -15, each an
    # absolute move from the last send position.
    assert [list(step) for step in default_move_retract().steps] == [
        ["send_z_rel"], ["send_x_rel"], ["send_y_rel"],
    ]
    board.device._last_send_pos = Offset3DTuple(0.0, 0.0, 0.0)
    z_calls = board.answer("move_motor_z", error)
    x_calls = board.spy("move_motor_x")
    y_calls = board.spy("move_motor_y")

    token = board.send(SystemCommandKind.SEND_RETRACT)
    board.refusal_of(token)
    time.sleep(0.2)  # longer than the loop's 50 ms step wait

    assert len(z_calls) == 1
    assert x_calls == [] and y_calls == []


def test_a_refused_step_is_reported_refused_and_never_acknowledged_to_its_caller(board):
    board.device._last_send_pos = Offset3DTuple(0.0, 0.0, 0.0)
    board.answer("move_motor_z", -5)

    token = board.send(SystemCommandKind.SEND_RETRACT)
    refusal = board.refusal_of(token)
    time.sleep(0.2)

    assert not board.acked(token)
    assert refusal.token == token
    assert refusal.error == -5
    assert refusal.error_name == "EIO"
    assert refusal.command is SystemCommandKind.SEND_RETRACT
    assert refusal.command_class is RefusedCommandClass.STEPPER_MOVE
    assert refusal.motors == (Motor.PELLET_Z_MOTOR,)


@pytest.mark.parametrize("step, method", [
    ({"x": 1.0}, "move_motor_x"),
    ({"y": 1.0}, "move_motor_y"),
    ({"z": 1.0}, "move_motor_z"),
    ({"predefined": "send"}, "fixed_position"),
], ids=["x", "y", "z", "predefined-send"])
def test_eagain_ack_lets_the_compound_sequence_advance(board, step, method):
    board.device._move_retract = MotorSteps("probe", [step, {"home": Motor.PELLET_X_MOTOR}])
    board.answer(method, -11)
    homes = board.spy("stepper_home")

    token = board.send(SystemCommandKind.SEND_RETRACT)

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, token) is not None
    assert homes == [(Motor.PELLET_X_MOTOR,)]
    assert board.wait_for(SystemStatusMessageKind.COMMAND_REFUSED, token, timeout=0) is None


def test_a_refusal_leaves_the_board_context_available_for_the_next_command(board):
    board.answer("move_motor_x", -16)

    refused = board.send(SystemCommandKind.MOVE_X, 1.0, token="refused")
    board.refusal_of(refused)
    after = board.send(SystemCommandKind.MOVE_Y, 2.0, token="after")

    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, after) is not None
    ctx = board.device._boards_pending_ctx[Target.PELLET_DEVICE]
    assert ctx.is_available()
    assert ctx.kind is None and ctx.repeated_command_count == 0


def test_a_refusal_clears_the_ack_timeout_engaged_flag_like_an_ack(board):
    board.device.default_command_ack_timeout_duration = 0.3
    board.answer("move_motor_x", _LOST, -5)
    engaged = []
    board.device.property_changed += (
        lambda name, value, old:
        engaged.append(value) if name == board.device.UUID_ACK_TIMEOUT_ENGAGED else None)

    token = board.send(SystemCommandKind.MOVE_X, 1.0)
    board.refusal_of(token)

    assert engaged == [True, False]
    assert not board.device._boards_pending_ctx[Target.PELLET_DEVICE].uuid_ack_timeout_engaged


def test_a_timeout_still_resends_and_a_refusal_never_does(board):
    board.device.default_command_ack_timeout_duration = 0.2
    refused_calls = board.answer("move_motor_x", -5)
    lost_calls = board.answer("move_motor_y", _LOST)

    refused = board.send(SystemCommandKind.MOVE_X, 1.0, token="refused")
    board.refusal_of(refused)
    time.sleep(0.5)  # past the 0.2 s timeout: a refusal is never re-sent
    assert len(refused_calls) == 1

    lost = board.send(SystemCommandKind.MOVE_Y, 1.0, token="lost")
    assert board.wait_for(SystemStatusMessageKind.ACKNOWLEDGE, lost) is not None
    assert len(lost_calls) == 2, "a lost ack is still re-sent once it times out"


def test_ebusy_on_a_first_send_stepper_command_marks_the_axis_needs_home(board):
    board.answer("move_motor_x", -16)

    refusal = board.refusal_of(board.send(SystemCommandKind.MOVE_X, 1.0))

    assert refusal.resent is False
    assert refusal.needs_home == (Motor.PELLET_X_MOTOR,)
    assert not refusal.outcome_unknown
    assert "needs homing" in refusal.message


def test_ebusy_on_a_resent_stepper_command_is_not_reported_as_needs_home(board):
    """The resend of a move still running is refused at once with -EBUSY.

    That is "still moving or unknown", not "needs homing": the axis is
    finishing a normal move.
    """
    board.device.default_command_ack_timeout_duration = 0.3
    board.answer("move_motor_x", _LOST, -16)

    refusal = board.refusal_of(board.send(SystemCommandKind.MOVE_X, 1.0))

    assert refusal.resent is True
    assert refusal.needs_home == ()
    assert refusal.outcome_unknown
    assert "unknown" in refusal.message


@pytest.mark.parametrize("error", [-1, -16])
def test_eperm_and_ebusy_on_a_stim_pulse_mark_no_axis_needs_home(board, error):
    board.answer("pulse_digital_output", error)

    refusal = board.refusal_of(board.send(
        SystemCommandKind.PULSE_DIGITAL_OUTPUT,
        (int(DigitalOutputs.STIMULUS_3.value), 1000),
    ))

    assert refusal.command_class is RefusedCommandClass.PULSE
    assert refusal.needs_home == ()
    assert "E-stop" not in refusal.message
    assert "homing" not in refusal.message


def test_ebusy_on_a_servo_move_is_still_moving_not_needs_home(board):
    board.answer("move_servo_motor", -16)

    refusal = board.refusal_of(board.send(SystemCommandKind.MOVE_COVER_SERVO, 40))

    assert refusal.command_class is RefusedCommandClass.SERVO_MOVE
    assert refusal.motors == (Motor.PELLET_COVER_SERVO,)
    assert refusal.needs_home == ()
    assert "moving" in refusal.message


def test_a_refused_pre_reveal_pulse_stops_the_send_before_the_reveal(board):
    """A refused stimulus stops the trial's SEND like a refused move."""
    board.answer("pulse_digital_output", -16)
    releases = board.spy("release_pellet")
    fixed = board.spy("fixed_position")

    token = board.send(SystemCommandKind.SEND_PELLET, {"pre_reveal_stimulus": (200, 1000, 2)})
    refusal = board.refusal_of(token)
    time.sleep(0.4)  # past the 200 ms pre-reveal delay

    assert refusal.command is SystemCommandKind.SEND_PELLET
    assert refusal.command_class is RefusedCommandClass.PULSE
    assert releases == [] and fixed == []
    assert not board.acked(token)


def test_board_error_numbers_are_compared_as_numbers_not_host_errno():
    # picolibc's ETIMEDOUT is 116; the host's is 110 on Linux, 10060 on Windows.
    assert errno.ETIMEDOUT != 116
    assert board_error_name(-116) == "ETIMEDOUT"
    assert board_error_name(-errno.ETIMEDOUT) != "ETIMEDOUT"
    assert [board_error_name(e) for e in (-1, -5, -11, -16)] == [
        "EPERM", "EIO", "EAGAIN", "EBUSY",
    ]
    assert board_error_name(-22) == "board error -22"


def _refusal(command_class, error, *, motors=(), resent=False):
    return CommandRefusal(
        token="t", error=error, command=SystemCommandKind.SEND_PELLET,
        step="step", command_class=command_class, motors=motors, resent=resent,
    )


def test_ebusy_on_a_first_send_fixed_xyz_marks_every_axis_it_could_have_been():
    """FIXED_XYZ reports one error for three axes and does not say which."""
    refusal = _refusal(RefusedCommandClass.FIXED_XYZ, -16, motors=_XYZ)

    assert refusal.needs_home == _XYZ


@pytest.mark.parametrize("command_class, motors", [
    (RefusedCommandClass.STEPPER_MOVE, (Motor.PELLET_Y_MOTOR,)),
    (RefusedCommandClass.STEPPER_HOME, (Motor.PELLET_Y_MOTOR,)),
    (RefusedCommandClass.FIXED_XYZ, _XYZ),
    (RefusedCommandClass.SERVO_MOVE, (Motor.PELLET_LOAD_SERVO,)),
])
def test_eperm_on_any_motion_marks_every_axis_and_names_the_estop(command_class, motors):
    # ESTOP marks every stepper axis as needing a home on the board.
    refusal = _refusal(command_class, -1, motors=motors)

    assert refusal.needs_home == _XYZ
    assert "E-stop" in refusal.message


@pytest.mark.parametrize("error", [-5, -116])
def test_eio_and_etimedout_on_a_stepper_command_mark_every_axis(error):
    refusal = _refusal(RefusedCommandClass.STEPPER_MOVE, error, motors=(Motor.PELLET_X_MOTOR,))

    assert refusal.needs_home == _XYZ
    assert "Home the pellet axes" in refusal.message


def test_a_home_refused_with_eio_says_to_home_once_more_then_reboot():
    """-5 on a home has two causes the number cannot tell apart.

    A fault latched while the home ran: another home recovers. The drivers
    failed to re-prove: a further home repeats the check and is refused again.
    So home once more, then reboot (release note v2.3.1, "Faults and recovery").
    """
    message = _refusal(RefusedCommandClass.STEPPER_HOME, -5, motors=(Motor.PELLET_Y_MOTOR,)).message

    assert "once more" in message
    assert "reboot the pellet board" in message.lower()
    assert "12 V" in message
    assert message.index("once more") < message.lower().index("reboot")


def test_a_home_refused_with_ebusy_says_the_axis_is_stuck():
    refusal = _refusal(RefusedCommandClass.STEPPER_HOME, -16, motors=(Motor.PELLET_Z_MOTOR,))

    assert refusal.needs_home == (Motor.PELLET_Z_MOTOR,)
    assert "reboot the pellet board" in refusal.message.lower()


def test_an_unnamed_error_on_any_command_is_a_refusal_that_marks_no_axis():
    refusal = _refusal(RefusedCommandClass.OTHER, -22)

    assert refusal.needs_home == ()
    assert "board error -22" in refusal.message


def test_a_command_refused_message_reaches_command_refused_and_not_ack_received():
    messages = queue.Queue()
    handler = MessageHandler(messages)
    acked, refused = [], []
    handler.ack_received += lambda tok, *, perf_c=None: acked.append(tok)
    handler.command_refused += (
        lambda tok, *, perf_c=None, refusal=None: refused.append((tok, perf_c, refusal)))
    refusal = _refusal(RefusedCommandClass.STEPPER_MOVE, -5)

    messages.put((SystemStatusMessageKind.COMMAND_REFUSED, ("tok", 1.5, refusal)))
    messages.put((TERMINATE, None))
    handler.run()

    assert acked == []
    assert refused == [("tok", 1.5, refusal)]


def _connection_whose_device_answers(kind, make_data):
    """A DeviceConnection whose device answers each command with one message."""
    connection = None

    class _Device:
        device_interface = None

        def notify_message(self, _kind, _data, context):
            connection._api.send_message(kind, make_data(context))

    connection = DeviceConnection(_Device(), message_queue=queue.Queue())
    return connection


def test_await_acknowledge_raises_naming_the_code_when_the_command_is_refused():
    refusal = _refusal(RefusedCommandClass.STEPPER_HOME, -5)
    connection = _connection_whose_device_answers(
        SystemStatusMessageKind.COMMAND_REFUSED, lambda tok: (tok, 0.0, refusal))

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="EIO") as raised:
        with connection.await_acknowledge({"home"}, timeout=3):
            connection.send_message(SystemCommandKind.SEND_HOME, None, "home")

    assert time.perf_counter() - started < 1, "a refusal must not wait for the timeout"
    assert raised.value.refusal is refusal


def test_await_acknowledge_still_returns_on_an_ack():
    connection = _connection_whose_device_answers(
        SystemStatusMessageKind.ACKNOWLEDGE, lambda tok: (tok, 0.0))

    with connection.await_acknowledge({"home"}, timeout=1):
        connection.send_message(SystemCommandKind.SEND_HOME, None, "home")
