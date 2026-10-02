import contextlib
import logging
import math
import errno
import queue
import threading
import time
import uuid
from functools import partial
from unittest import mock

import pytest
from autotrainer.core import RawValueHolder

from autotrainer.core.message import SystemStatusMessageKind, SystemCommandKind
from autotrainer.device import (
    CanFailure,
    CanFailureKind,
    CanDevice,
    DeviceApi,
    Target,
    Motor,
    StepperStatus,
    ServoStatus,
    ServoConfig,
    StepperConfig,
    MotorSteps,
    DeviceConnection,
    DigitalOutputs,
    MotorConfigurationFile,
    Tone,
)
from autotrainer.device import CanInterface, CanTransportConfiguration
from autotrainer.device import can_device
from autotrainer.device.can_device import (
    default_move_retract,
    default_load_pellet,
    default_send_pellet,
    _no_op,
    _retry_compound,
    _shutdown_requested,
)
from autotrainer.device.device_connection import _REQUEST_DISCONNECT
from autotrainer.device.device_interface import Acknowledge
from autotrainer.device.emulation_interface import EmulationInterface

_expected = None

def data_callback(kind: int, response):
    assert kind == _expected
    del response  # uncheck atm


def test_device_connection_retains_live_reader_after_join_timeout():
    connection = object.__new__(DeviceConnection)
    reader = mock.Mock()
    reader.is_alive.return_value = True
    connection._current_thread = reader

    assert connection.join() is False

    reader.join.assert_called_once_with(3)
    assert connection._current_thread is reader


def test_device_connection_throttles_empty_reads():
    """An idle CAN backend must not spin and starve the application's GUI thread."""
    interface = mock.Mock()
    interface.can_read.return_value = True
    interface.read.return_value = []
    interface.is_open = True

    device = mock.Mock()
    device.device_interface = interface
    connection = DeviceConnection(device, message_queue=queue.Queue())
    connection._collect_ms = 5

    thread = threading.Thread(target=connection._run_connected)
    thread.start()
    time.sleep(0.055)
    connection._cmd_queue.put((_REQUEST_DISCONNECT, None, None))
    thread.join(1)

    assert not thread.is_alive()
    # Disconnect commands are intentionally checked every 250 ms, so include
    # that interval while still proving this was a throttled poll, not a spin.
    assert 2 <= interface.read.call_count <= 80


def test_pellet_only_connection_skips_unused_motor_configurations():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    connection = DeviceConnection(device, message_queue=queue.Queue())
    connection.send_message = mock.Mock()

    @contextlib.contextmanager
    def no_wait(*args, **kwargs):
        yield

    connection.await_acknowledge = no_wait
    connection.use_motor_configurations(MotorConfigurationFile())

    configured_motors = {
        call.args[1][0]
        for call in connection.send_message.call_args_list
    }
    assert {
        Motor.PELLET_X_MOTOR,
        Motor.PELLET_Y_MOTOR,
        Motor.PELLET_Z_MOTOR,
        Motor.PELLET_LOAD_SERVO,
        Motor.PELLET_COVER_SERVO,
    } <= configured_motors


def test_disconnect_discards_pending_and_retry_state():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    device._commands_queue.put((SystemCommandKind.SET_X, 1, "queued"))
    device._compound_movement = [{"x": 1}]
    device._prev_command = ("retry", 1, "cached")
    board = device._boards_pending_ctx[Target.PELLET_DEVICE]
    board.ctx = "pending"
    board.kind = SystemCommandKind.SET_X
    board.uuid = 42
    board.prev_command = ("retry", 1, "pending")
    board.compound_steps = [{"x": 2}]
    board.repeated_command_count = 2

    device.disconnect()
    device.disconnect()

    assert device._commands_queue.empty()
    assert device._compound_movement is None
    assert device._prev_command is None
    assert board.ctx is None
    assert board.kind is None
    assert board.uuid is None
    assert board.prev_command is None
    assert board.compound_steps is None
    assert board.repeated_command_count == 0

    device.notify_message(SystemCommandKind.SET_X, 2, "after-shutdown")
    assert device._commands_queue.empty()


def test_compound_move_keeps_configured_motor_coordinate_unchanged():
    """UI-only coordinate changes must not reinterpret move_config values."""
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    device.device_interface.move_motor_x = mock.Mock(return_value=True)
    steps = [{"x": 25}]
    board = device._boards_pending_ctx[Target.PELLET_DEVICE]

    assert device._perform_next_compound_step(board, steps)

    device.device_interface.move_motor_x.assert_called_once_with(
        25,
        save_as_fixed=False,
    )
    assert steps == []


def test_send_sequence_can_append_one_embedded_tone_without_mutating_template():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    original = device._send_pellet.steps
    device._start_sequence = mock.Mock(return_value=True)

    assert device._start_send_pellet_sequence({"embedded_tone": (6000, 125)})

    sequence = device._start_sequence.call_args.args[0]
    assert sequence.steps[-1] == {"tone": "6000,0.125"}
    assert device._send_pellet.steps == original


def test_send_sequence_can_prepend_board_timed_pre_reveal_stimulus():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    original = device._send_pellet.steps
    device._start_sequence = mock.Mock(return_value=True)

    assert device._start_send_pellet_sequence({
        "pre_reveal_stimulus": (200, 1000),
    })

    sequence = device._start_sequence.call_args.args[0]
    assert sequence.steps[:3] == [
        {"stim3": 1000},
        {"delay": 0.199},
        {"predefined": "release"},
    ]
    assert device._send_pellet.steps == original


def test_send_sequence_rejects_pulse_longer_than_pre_reveal_interval():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )

    with pytest.raises(ValueError, match="shorter than the pre-reveal delay"):
        device._start_send_pellet_sequence({
            "pre_reveal_stimulus": (1, 1000),
        })


def _pre_reveal_device(operations):
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
        operation_callback=lambda *args: operations.append(args),
    )
    interface = device.device_interface
    interface.open()
    # Wrapped rather than replaced, so the emulator still refuses STIM0 and
    # STIM1 the way the board does.
    interface.pulse_digital_output = mock.Mock(wraps=interface.pulse_digital_output)
    device._boards_pending_ctx[Target.PELLET_DEVICE].ctx = "pellet-cycle"
    return device


@pytest.mark.parametrize("stimulus, output", [
    # christielab10 laser 2 is boardStimLine 2; board STIM2 is STIMULUS_3.
    ((200, 1000, 2), DigitalOutputs.STIMULUS_3),
    # Laser 1 is boardStimLine 3.
    ((200, 1000, 3), DigitalOutputs.STIMULUS_4),
    # Data from before the line was carried meant STIM3, and still does.
    ((200, 1000), DigitalOutputs.STIMULUS_4),
])
def test_pre_reveal_pulse_goes_out_on_the_trial_lasers_board_line(stimulus, output):
    operations = []
    device = _pre_reveal_device(operations)

    assert device._start_send_pellet_sequence({"pre_reveal_stimulus": stimulus})

    device.device_interface.pulse_digital_output.assert_called_once_with(output, 1000)
    kind, data, context, target, _perf_time, _wall_time = operations[0]
    assert kind is SystemCommandKind.PULSE_DIGITAL_OUTPUT
    # The (output, duration) a direct HardwareModel.pulse_stim records, so
    # the session's device events name the line that was actually pulsed.
    assert data == (int(output.value), 1000)
    assert context == "pellet-cycle"
    assert target is Target.PELLET_DEVICE


@pytest.mark.parametrize("stim_line", [0, 1, 4])
def test_pre_reveal_refuses_a_line_that_cannot_carry_a_stimulus(stim_line):
    operations = []
    device = _pre_reveal_device(operations)

    with pytest.raises(
        ValueError,
        match=f"Board STIM{stim_line} cannot carry a stimulus pulse; "
              "use board STIM2 or STIM3",
    ):
        device._start_send_pellet_sequence({
            "pre_reveal_stimulus": (200, 1000, stim_line),
        })

    device.device_interface.pulse_digital_output.assert_not_called()
    assert operations == []


def test_pre_reveal_refusals_name_the_line_they_would_pulse():
    device = _pre_reveal_device([])

    with pytest.raises(ValueError, match="STIM2 pulse duration must be within"):
        device._start_send_pellet_sequence({
            "pre_reveal_stimulus": (200, 99, 2),
        })
    with pytest.raises(
        ValueError, match="STIM2 pulse duration must be shorter than the pre-reveal delay",
    ):
        device._start_send_pellet_sequence({
            "pre_reveal_stimulus": (1, 1000, 2),
        })


def test_immediate_tone_status_is_forwarded_with_source_timestamp():
    received = []
    device = CanDevice(
        api=DeviceApi(
            message_callback=lambda kind, data: received.append((kind, data)),
        ),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    tone = Tone(Target.PELLET_DEVICE, time_remaining_ms=300, frequency_hz=6000)
    tone.index = 123_456_789
    tone.timestamp_ns = 987_654_321

    device.notify_data([tone])

    assert received == [(SystemStatusMessageKind.TONE_STATUS, tone)]
    assert received[0][1].index == 123_456_789
    assert received[0][1].timestamp_ns == 987_654_321


def test_repeated_tone_start_is_forwarded_before_an_off_report():
    received = []
    device = CanDevice(
        api=DeviceApi(
            message_callback=lambda kind, data: received.append((kind, data)),
        ),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )

    device.notify_data([
        Tone(Target.PELLET_DEVICE, time_remaining_ms=300, frequency_hz=6000),
    ])
    device.notify_data([
        Tone(Target.PELLET_DEVICE, time_remaining_ms=100, frequency_hz=6000),
    ])
    device.notify_data([
        Tone(Target.PELLET_DEVICE, time_remaining_ms=300, frequency_hz=6000),
    ])

    assert [data.time_remaining_ms for _, data in received] == [300, 300]


def test_compound_tone_reports_the_executed_play_tone_command():
    operations = []
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
        operation_callback=lambda *args: operations.append(args),
    )
    device.device_interface.emit_tone = mock.Mock(return_value=True)
    board = device._boards_pending_ctx[Target.PELLET_DEVICE]
    board.ctx = "pellet-cycle"
    steps = [{"tone": "5000,0.3"}]

    assert device._perform_next_compound_step(board, steps)

    device.device_interface.emit_tone.assert_called_once_with(5000, 300)
    assert steps == []
    kind, data, context, target, perf_time, wall_time = operations[0]
    assert kind is SystemCommandKind.PLAY_TONE
    assert data == (5000, 300)
    assert context == "pellet-cycle"
    assert target is Target.PELLET_DEVICE
    assert perf_time > 0
    assert wall_time > 0


# The recorder callback costs milliseconds (the hardware model, the session
# recorder's lock, JSON), and christielab10 measured the board answering a STIM
# frame in about 0.3 ms. A compound step must put the frame on the bus first
# and record it after, with the stamp the row carries still taken before the
# send. Each step is checked on one list shared by the send and the recorder.

_SEND_THEN_RECORD_STEPS = {
    "stim": ({"stim3": 1000}, "pulse_digital_output"),
    "tone": ({"tone": "5000,0.3"}, "emit_tone"),
}


def _device_that_logs_sends_and_rows(events, method_name, send_result, record_error=None):
    """An emulated device whose send and recorder append to ``events``.

    A send is ("send", time.perf_counter() at its entry). A recorded row is
    ("record", the perf_time stamp it carries). ``send_result`` is the value
    the send returns, or an exception to raise. ``record_error``, if given, is
    raised by the recorder once it has logged its row.
    """
    def record(kind, data, context, target, perf_time, wall_time):
        events.append(("record", perf_time))
        if record_error is not None:
            raise record_error

    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
        operation_callback=record,
    )

    def send(*args, **kwargs):
        events.append(("send", time.perf_counter()))
        if isinstance(send_result, BaseException):
            raise send_result
        return send_result

    setattr(device.device_interface, method_name, mock.Mock(side_effect=send))
    board = device._boards_pending_ctx[Target.PELLET_DEVICE]
    board.ctx = "pellet-cycle"
    return device, board


@pytest.mark.parametrize("step_name", sorted(_SEND_THEN_RECORD_STEPS))
def test_a_compound_step_is_sent_before_it_is_recorded(step_name):
    step, method_name = _SEND_THEN_RECORD_STEPS[step_name]
    events = []
    device, board = _device_that_logs_sends_and_rows(events, method_name, True)

    assert device._perform_next_compound_step(board, [dict(step)])

    assert [name for name, _ in events] == ["send", "record"]
    (_, sent_at), (_, row_stamp) = events
    # The row still carries the stamp taken before the frame, not one taken
    # once the recorder ran.
    assert row_stamp <= sent_at


@pytest.mark.parametrize("step_name", sorted(_SEND_THEN_RECORD_STEPS))
def test_a_refused_compound_send_is_still_recorded_once(step_name):
    step, method_name = _SEND_THEN_RECORD_STEPS[step_name]
    events = []
    device, board = _device_that_logs_sends_and_rows(events, method_name, False)
    steps = [dict(step)]

    assert not device._perform_next_compound_step(board, steps)

    assert sorted(name for name, _ in events) == ["record", "send"]
    assert steps == [step]
    assert board.skip_uuid_ack_perf_c is False


@pytest.mark.parametrize("step_name", sorted(_SEND_THEN_RECORD_STEPS))
def test_a_compound_send_that_raises_is_still_recorded_once(step_name):
    step, method_name = _SEND_THEN_RECORD_STEPS[step_name]
    events = []
    device, board = _device_that_logs_sends_and_rows(
        events, method_name, OSError("CAN adapter lost"))

    with pytest.raises(OSError, match="CAN adapter lost"):
        device._perform_next_compound_step(board, [dict(step)])

    assert sorted(name for name, _ in events) == ["record", "send"]


@pytest.mark.parametrize("step_name", sorted(_SEND_THEN_RECORD_STEPS))
def test_a_failing_recorder_does_not_mask_the_send_that_raised(step_name, caplog):
    step, method_name = _SEND_THEN_RECORD_STEPS[step_name]
    events = []
    send_error = OSError("CAN adapter lost")
    recorder_error = RuntimeError("recorder failed")
    device, board = _device_that_logs_sends_and_rows(
        events, method_name, send_error, record_error=recorder_error)

    with caplog.at_level(logging.ERROR, logger="autotrainer.device.can_device"):
        with pytest.raises(OSError, match="CAN adapter lost") as raised:
            device._perform_next_compound_step(board, [dict(step)])

    # The caller sees the send's own failure, whole.
    assert raised.value is send_error
    assert sorted(name for name, _ in events) == ["record", "send"]
    # The recorder's failure is not lost: one ERROR record with its traceback,
    # and what is needed to trace it to the command.
    (logged,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert logged.exc_info[1] is recorder_error
    message = logged.getMessage()
    assert "CAN adapter lost" in message
    assert "pellet-cycle" in message
    assert "PELLET_DEVICE" in message
    expected_kind = {"stim": "PULSE_DIGITAL_OUTPUT", "tone": "PLAY_TONE"}[step_name]
    assert expected_kind in message


@pytest.mark.parametrize("step_name", sorted(_SEND_THEN_RECORD_STEPS))
def test_a_failing_recorder_after_a_good_send_still_raises(step_name):
    # What happened before, and still does: the failure ends the handler
    # thread. The frame is already out when it does.
    step, method_name = _SEND_THEN_RECORD_STEPS[step_name]
    events = []
    recorder_error = RuntimeError("recorder failed")
    device, board = _device_that_logs_sends_and_rows(
        events, method_name, True, record_error=recorder_error)

    with pytest.raises(RuntimeError, match="recorder failed") as raised:
        device._perform_next_compound_step(board, [dict(step)])

    assert raised.value is recorder_error
    assert [name for name, _ in events] == ["send", "record"]


def test_a_sent_compound_tone_skips_the_next_ack_stamp():
    events = []
    device, board = _device_that_logs_sends_and_rows(events, "emit_tone", True)

    assert device._perform_next_compound_step(board, [{"tone": "5000,0.3"}])

    assert board.skip_uuid_ack_perf_c is True


# A host-only compound step (the predefined cover, release, retrieve and scoop,
# load_arm and barrier_arm, and _no_op) sends nothing: it only injects the servo
# steps. The command loop used to go back to its 50 ms idle wait after one, so
# the frame came out 50 ms late (christielab10 session004: the pre-reveal
# release, +69..+111 ms against its 1500 ms target). The tests below say which
# steps may now continue in the same pass and which must not.

_SENDERS = (
    "release_pellet", "cover_pellet", "retrieve_pellet", "scoop_pellet",
    "move_servo_motor", "fixed_position", "servo_attach", "servo_detach",
)


def _open_compound_device():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    interface = device.device_interface
    interface.open()
    # Wrapped, so the emulator still answers and a test can count the frames.
    for name in _SENDERS:
        setattr(interface, name, mock.Mock(wraps=getattr(interface, name)))
    board = device._boards_pending_ctx[Target.PELLET_DEVICE]
    board.ctx = "pellet-cycle"
    return device, interface, board


@pytest.mark.parametrize("step, sender, arguments", [
    ({"predefined": "release"}, "release_pellet", ()),
    ({"predefined": "cover"}, "cover_pellet", ()),
    ({"predefined": "retrieve"}, "retrieve_pellet", ()),
    ({"predefined": "scoop"}, "scoop_pellet", ()),
    ({"load_arm": 45.0}, "move_servo_motor", (Motor.PELLET_LOAD_SERVO, 45.0)),
    ({"barrier_arm": 45.0}, "move_servo_motor", (Motor.PELLET_COVER_SERVO, 45.0)),
], ids=["release", "cover", "retrieve", "scoop", "load_arm", "barrier_arm"])
def test_a_host_only_step_sends_the_servo_command_in_the_same_pass(step, sender, arguments):
    device, interface, board = _open_compound_device()
    steps = [dict(step), {"predefined": "send"}]

    device._perform_compound_until_sent(board, steps)

    getattr(interface, sender).assert_called_once_with(*arguments)
    # It stops at the step that sent: the step after it is left for the ack.
    interface.fixed_position.assert_not_called()
    assert steps == [{"predefined": "send"}]


def test_a_no_op_step_runs_into_the_next_step_in_the_same_pass():
    device, interface, board = _open_compound_device()
    no_op = {"_internal_func": _no_op, "_internal_func_motor": Motor.PELLET_COVER_SERVO}
    steps = [dict(no_op), dict(no_op), dict(no_op), {"predefined": "send"}]

    device._perform_compound_until_sent(board, steps)

    interface.fixed_position.assert_called_once_with()
    assert steps == []


def test_host_only_steps_with_nothing_to_send_run_out_and_return():
    device, interface, board = _open_compound_device()
    no_op = {"_internal_func": _no_op, "_internal_func_motor": Motor.PELLET_COVER_SERVO}
    steps = [dict(no_op), dict(no_op), dict(no_op)]

    device._perform_compound_until_sent(board, steps)

    assert steps == []
    for name in _SENDERS:
        getattr(interface, name).assert_not_called()


@pytest.mark.parametrize("step, sender", [
    ({"servo_attach": Motor.PELLET_COVER_SERVO}, "servo_attach"),
    ({"servo_detach": Motor.PELLET_COVER_SERVO}, "servo_detach"),
], ids=["attach", "detach"])
def test_a_step_that_sends_without_a_uuid_still_returns_to_the_loop(step, sender):
    # Attach and detach send a frame but take no uuid. The loop's wait after
    # them is kept, so no frame-to-frame interval on the bus changes.
    device, interface, board = _open_compound_device()
    following = {"_servo_move": (Motor.PELLET_COVER_SERVO, 10.0)}
    steps = [dict(step), dict(following)]

    device._perform_compound_until_sent(board, steps)

    getattr(interface, sender).assert_called_once_with(Motor.PELLET_COVER_SERVO)
    interface.move_servo_motor.assert_not_called()
    assert steps == [following]


def test_an_injected_attach_still_returns_to_the_loop_before_the_move():
    device, interface, board = _open_compound_device()
    device._motor_configs[Motor.PELLET_COVER_SERVO].detach = True
    steps = [{"predefined": "release"}, {"predefined": "send"}]

    device._perform_compound_until_sent(board, steps)

    # The release injected [attach, release, detach]. The attach went out and
    # the pass ended there, as the loop's idle wait followed it before.
    interface.servo_attach.assert_called_once_with(Motor.PELLET_COVER_SERVO)
    interface.release_pellet.assert_not_called()
    assert len(steps) == 3
    assert steps[1:] == [
        {"servo_detach": Motor.PELLET_COVER_SERVO},
        {"predefined": "send"},
    ]


def test_the_retry_after_an_injection_targets_the_servo_command():
    device, interface, board = _open_compound_device()
    steps = [{"predefined": "release"}, {"predefined": "send"}]

    device._perform_compound_until_sent(board, steps)

    # An ack timeout retries _prev_command. It has to be the command that was
    # sent, not the host-only step that came before it.
    retry_kind, (_kind, retry_step, retry_steps), _ctx, _perf = device._prev_command
    assert retry_kind is _retry_compound
    assert retry_step["_internal_func"] is interface.release_pellet
    assert retry_step["_internal_func_motor"] is Motor.PELLET_COVER_SERVO
    assert retry_steps is steps


def test_a_step_that_keeps_failing_after_a_host_only_step_still_raises():
    device, interface, board = _open_compound_device()
    interface.close()  # the emulated board then refuses every servo command
    steps = [{"predefined": "release"}]

    with pytest.raises(RuntimeError, match="too many failure"):
        device._perform_compound_until_sent(board, steps)

    # One try and the configured repeats, as before.
    assert interface.release_pellet.call_count == (
        device.default_command_write_failed_repeat_count + 1
    )


def _device_with_a_host_only_step_stand_in(steps, *, takes_uuid, pops):
    """A device whose every step is reported host-only, to test the pass's guards."""
    device, interface, board = _open_compound_device()
    performed = []

    def perform(_board, compound_movements):
        performed.append(len(compound_movements))
        if pops:
            compound_movements.pop(0)
        if takes_uuid:
            EmulationInterface.next_uuid()
        device._compound_step_host_only = True
        return True

    device._perform_next_compound_step = perform
    return device, board, performed


def test_a_pass_ends_at_a_step_that_took_a_uuid_even_if_it_was_marked_host_only():
    steps = [{"x": 1}, {"x": 2}, {"x": 3}]
    device, board, performed = _device_with_a_host_only_step_stand_in(
        steps, takes_uuid=True, pops=True)

    device._perform_compound_until_sent(board, steps)

    assert performed == [3]
    assert steps == [{"x": 2}, {"x": 3}]


def test_a_chain_of_host_only_steps_cannot_loop_for_ever():
    # Real host-only steps always pop themselves, so this one cannot happen;
    # the pass is still bounded by the steps it was given.
    steps = [{"x": 1}]
    device, board, performed = _device_with_a_host_only_step_stand_in(
        steps, takes_uuid=False, pops=False)

    device._perform_compound_until_sent(board, steps)

    assert len(performed) == 1 + len(steps)


def test_a_shutdown_request_stops_the_pass_after_a_host_only_step():
    device, interface, board = _open_compound_device()
    device._want_exit.set()
    steps = [{"predefined": "release"}, {"predefined": "send"}]

    device._perform_compound_until_sent(board, steps)

    for name in _SENDERS:
        getattr(interface, name).assert_not_called()


def _times_through_the_handler(kind, data, *, detach=False):
    """Run one command on a connected emulated device and time what it sends.

    The command handler runs on its own thread, and a pump thread hands the
    emulator's acks back as DeviceConnection's reader would. Returns the
    perf_counter at "queued" (just before the command is queued), at each
    servo command ("release_sent", "cover_sent", "attach_sent") and at
    "delay_acked" (when the delay returns, which is when its ack is queued).
    """
    token = "handler-run"
    finished = threading.Event()
    times = {}

    def callback(message_kind, message):
        if message_kind == SystemStatusMessageKind.ACKNOWLEDGE and message[0] == token:
            finished.set()

    device = CanDevice(
        api=DeviceApi(message_callback=callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    interface = device.device_interface
    interface.open()
    device._motor_configs[Motor.PELLET_COVER_SERVO].detach = detach

    def stamped(name, method, *, after=False):
        def call(*args):
            if not after:
                times[name] = time.perf_counter()
            result = method(*args)
            if after:
                times[name] = time.perf_counter()
            return result
        return call

    # The delay waits its time, then queues the board's ack.
    interface.delay = stamped("delay_acked", interface.delay, after=True)
    interface.release_pellet = stamped("release_sent", interface.release_pellet)
    interface.cover_pellet = stamped("cover_sent", interface.cover_pellet)
    interface.servo_attach = stamped("attach_sent", interface.servo_attach)

    stop = threading.Event()

    def deliver_acks():
        while not stop.is_set():
            for message in interface.read(64):
                if isinstance(message, Acknowledge):
                    device._handle_ack(message)
            time.sleep(0.001)

    reader = threading.Thread(target=deliver_acks, daemon=True)
    reader.start()
    try:
        device.connect()
        time.sleep(0.05)  # the handler thread is then waiting on its queue
        times["queued"] = time.perf_counter()
        device.notify_message(kind, data, context=token)
        assert finished.wait(5), f"{kind} never finished"
    finally:
        stop.set()
        reader.join(1)
        device.disconnect()
        interface.close()
    return times


def test_a_pre_reveal_release_goes_out_without_the_loops_idle_wait():
    """The release follows the board's ack of the delay by milliseconds, not 50 ms."""
    times = _times_through_the_handler(
        SystemCommandKind.SEND_PELLET, {"pre_reveal_stimulus": (200, 1000, 3)})

    # The loop's idle wait is 50 ms, so before this change the gap was at
    # least that. The bound is loose on purpose: it has to hold on a busy rig.
    assert times["release_sent"] - times["delay_acked"] < 0.03


# _start_sequence runs a sequence's first step itself, before the command loop
# sees it, so a sequence that opens with a host-only step (a standalone COVER or
# RELEASE is one) waited the loop's 50 ms before its servo command. It follows
# the same rule as _perform_compound_until_sent.

@pytest.mark.parametrize("steps, sender, arguments", [
    ([{"predefined": "cover"}], "cover_pellet", ()),
    ([{"predefined": "release"}], "release_pellet", ()),
    ([{"predefined": "retrieve"}, {"predefined": "send"}], "retrieve_pellet", ()),
    ([{"predefined": "scoop"}, {"predefined": "send"}], "scoop_pellet", ()),
    ([{"load_arm": 45.0}, {"predefined": "send"}],
     "move_servo_motor", (Motor.PELLET_LOAD_SERVO, 45.0)),
    ([{"barrier_arm": 45.0}, {"predefined": "send"}],
     "move_servo_motor", (Motor.PELLET_COVER_SERVO, 45.0)),
], ids=["cover", "release", "retrieve", "scoop", "load_arm", "barrier_arm"])
def test_a_sequence_that_opens_with_a_host_only_step_sends_the_servo_command_at_once(
    steps, sender, arguments,
):
    device, interface, board = _open_compound_device()

    assert device._start_sequence(MotorSteps("sequence", [dict(step) for step in steps]))

    getattr(interface, sender).assert_called_once_with(*arguments)
    interface.fixed_position.assert_not_called()
    # What the loop attaches to the board once the handler returns.
    assert device._compound_movement == steps[1:]
    # An ack timeout retries the servo command, not the host-only step.
    retry_kind, (_kind, retry_step, retry_steps), _ctx, _perf = device._prev_command
    assert retry_kind is _retry_compound
    assert "_internal_func" in retry_step or "_servo_move" in retry_step
    assert "predefined" not in retry_step and "load_arm" not in retry_step
    assert "barrier_arm" not in retry_step
    assert retry_steps is device._compound_movement


@pytest.mark.parametrize("step, sender", [
    ({"servo_attach": Motor.PELLET_COVER_SERVO}, "servo_attach"),
    ({"servo_detach": Motor.PELLET_COVER_SERVO}, "servo_detach"),
], ids=["attach", "detach"])
def test_a_sequence_that_opens_with_an_attach_or_detach_still_returns_to_the_loop(step, sender):
    device, interface, board = _open_compound_device()
    following = {"_servo_move": (Motor.PELLET_COVER_SERVO, 10.0)}

    assert device._start_sequence(MotorSteps("sequence", [dict(step), dict(following)]))

    getattr(interface, sender).assert_called_once_with(Motor.PELLET_COVER_SERVO)
    interface.move_servo_motor.assert_not_called()
    assert device._compound_movement == [following]


def test_a_sequence_with_an_injected_attach_still_returns_to_the_loop_before_the_move():
    device, interface, board = _open_compound_device()
    device._motor_configs[Motor.PELLET_COVER_SERVO].detach = True

    assert device._start_sequence(device._release_pellet)

    interface.servo_attach.assert_called_once_with(Motor.PELLET_COVER_SERVO)
    interface.release_pellet.assert_not_called()
    assert len(device._compound_movement) == 2


def test_a_shutdown_request_stops_a_sequence_after_its_host_only_first_step():
    device, interface, board = _open_compound_device()
    device._want_exit.set()

    assert device._start_sequence(device._cover_pellet)

    for name in _SENDERS:
        getattr(interface, name).assert_not_called()


@pytest.mark.parametrize("kind, sent", [
    (SystemCommandKind.COVER_PELLET, "cover_sent"),
    (SystemCommandKind.RELEASE_PELLET, "release_sent"),
], ids=["cover", "release"])
def test_a_standalone_cover_or_release_goes_out_without_the_loops_idle_wait(kind, sent):
    times = _times_through_the_handler(kind, None)

    # Before this change the command waited the loop's 50 ms. The bound is
    # loose on purpose: it has to hold on a busy rig.
    assert times[sent] - times["queued"] < 0.03


def test_a_release_with_detach_keeps_the_loops_wait_between_attach_and_move():
    times = _times_through_the_handler(SystemCommandKind.RELEASE_PELLET, None, detach=True)

    assert times["attach_sent"] - times["queued"] < 0.03
    # The attach takes no uuid, so the loop's 50 ms stays between it and the
    # move: no frame-to-frame interval on the bus changes.
    assert times["release_sent"] - times["attach_sent"] >= 0.04


def test_command_queued_immediately_before_connect_survives_startup():
    token = "queued-before-connect"
    acknowledged = threading.Event()

    def callback(kind, data):
        if kind == SystemStatusMessageKind.ACKNOWLEDGE and data[0] == token:
            acknowledged.set()

    device = CanDevice(
        api=DeviceApi(message_callback=callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    device.device_interface.open()
    device.notify_message(SystemCommandKind.REQUEST_VERSION, None, context=token)

    try:
        device.connect()
        assert acknowledged.wait(1), "pre-connect command was discarded during startup"
    finally:
        device.disconnect()
        device.device_interface.close()


def test_wait_connected_reports_connection_timeout_separately():
    device = mock.Mock()
    device.device_interface = mock.Mock()
    device.connected = False
    connection = DeviceConnection(device, message_queue=queue.Queue())

    with pytest.raises(TimeoutError, match="device connection timeout"):
        connection.wait_connected(timeout=0)


def test_await_acknowledge_reports_command_timeout_separately():
    device = mock.Mock()
    device.device_interface = mock.Mock()
    connection = DeviceConnection(device, message_queue=queue.Queue())

    with pytest.raises(RuntimeError, match="command timeout"):
        with connection.await_acknowledge({"not-acknowledged"}, timeout=0):
            pass


def test_command_handler_failure_requests_safety_shutdown():
    shutdown = mock.Mock()
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
        shutdown_callback=shutdown,
    )
    device._CanDevice__command_handler = mock.Mock(side_effect=RuntimeError("ack exhausted"))

    with pytest.raises(RuntimeError, match="ack exhausted"):
        device._command_handler()

    shutdown.assert_called_once()
    assert "ack exhausted" in shutdown.call_args.args[0]


def test_command_handler_reports_structured_terminal_failure():
    reported = []
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
        failure_callback=reported.append,
    )
    failure = CanFailure(
        CanFailureKind.ACKNOWLEDGEMENT_TIMEOUT,
        "send acknowledgement timed out",
        command=SystemCommandKind.SEND_PELLET,
        context="send-1",
    )
    error = RuntimeError(failure.error)
    error.can_failure = failure
    device._CanDevice__command_handler = mock.Mock(side_effect=error)

    with pytest.raises(RuntimeError, match="acknowledgement timed out"):
        device._command_handler()

    assert reported == [failure]


def test_device_connection_reports_reader_transport_failure():
    reported = []
    interface = mock.Mock()
    interface.is_open = True
    device = mock.Mock()
    device.device_interface = interface
    connection = DeviceConnection(
        device,
        message_queue=queue.Queue(),
        failure_callback=reported.append,
    )
    connection._run_unconnected = mock.Mock(return_value=True)
    connection._run_connected = mock.Mock(side_effect=OSError("CAN adapter lost"))

    connection._run()

    assert len(reported) == 1
    assert reported[0].kind is CanFailureKind.TRANSPORT
    assert reported[0].error == "CAN adapter lost"
    assert connection.first_failure is reported[0]
    device.disconnect.assert_called_once_with()
    interface.close.assert_called_once_with()


def test_device_connection_suppresses_enetdown_only_for_intentional_shutdown():
    reported = []
    interface = mock.Mock()
    interface.is_open = True
    device = mock.Mock()
    device.device_interface = interface
    connection = DeviceConnection(
        device,
        message_queue=queue.Queue(),
        failure_callback=reported.append,
    )
    connection._intentional_shutdown.set()
    error = RuntimeError("network down")
    error.error_code = errno.ENETDOWN
    connection._run_unconnected = mock.Mock(return_value=True)
    connection._run_connected = mock.Mock(side_effect=error)

    connection._run()

    assert reported == []
    assert connection.first_failure is None


def test_disconnect_waits_for_inflight_send_and_rejects_following_send():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    entered = threading.Event()
    release = threading.Event()

    def inflight():
        entered.set()
        release.wait(1)
        return True

    command_thread = threading.Thread(target=device._execute_if_active, args=(inflight,))
    command_thread.start()
    assert entered.wait(1)

    shutdown_thread = threading.Thread(target=device.disconnect)
    shutdown_thread.start()
    assert device._want_exit.wait(1)
    assert shutdown_thread.is_alive()

    release.set()
    command_thread.join(1)
    shutdown_thread.join(1)

    assert not command_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert device._execute_if_active(lambda: True) is _shutdown_requested


@pytest.fixture
def expected_tok() -> RawValueHolder:
    value = RawValueHolder(value=None)
    return value


def api_msg_cb(msg_kind, data, *, event, tokens_acked, expected_tok: RawValueHolder):
    # print(msg_kind, data)
    if msg_kind == SystemStatusMessageKind.ACKNOWLEDGE:
        tok, perf_c = data
        tokens_acked.append(tok)
        if tok is not None and expected_tok is not None and tok == expected_tok.value:
            expected_tok.value = None
            if event is not None:
                event.set()


@pytest.fixture
def tokens_acked():
    return []


@pytest.fixture
def expected_tok_event():
    return threading.Event()


@pytest.fixture
def device_conn(device):
    msg_q = queue.Queue()
    msg_cb = device.api.message_callback
    dc = DeviceConnection(device, message_queue=msg_q)
    dc.request_connect()
    device.api.message_callback = msg_cb
    try:
        yield dc
    finally:
        dc.request_disconnect()


@pytest.fixture  # (scope="module")
def device(expected_tok_event, expected_tok, tokens_acked) -> CanDevice:  # noqa
    device = CanDevice(api=DeviceApi(message_callback=data_callback), force_emulation=True)
    # unneeded, at least with emulation iface:
    # device._interface.pellet_address = 0x01
    # device.notify_message(_REQUEST_CONNECT)
    device.api.message_callback = partial(
        api_msg_cb,
        tokens_acked=tokens_acked,
        expected_tok=expected_tok,
        event=expected_tok_event,
    )
    device.connect()
    device.device_interface.open()
    try:
        yield device  # noqa
    finally:
        device.disconnect()


@pytest.mark.parametrize("kind, tag, data", [
    (SystemCommandKind.REQUEST_VERSION, 101, None),
    (SystemCommandKind.SET_X, 103, 10),
    (SystemCommandKind.SET_Y, 104, 15),
    (SystemCommandKind.SET_Z, 105, 20),
    (SystemCommandKind.MOVE_X, 106, 10),
    (SystemCommandKind.MOVE_Y, 106, 15),
    (SystemCommandKind.MOVE_Z, 108, 20),
    (SystemCommandKind.MOVE_LOAD_SERVO, 111, 35),
    (SystemCommandKind.MOVE_COVER_SERVO, 112, 40),
    (SystemCommandKind.SEND_HOME, 113, None),
    (SystemCommandKind.LOAD_PELLET, 114, None),
    (SystemCommandKind.SEND_PELLET, 115, None),
    (SystemCommandKind.RELEASE_PELLET, 116, None),
    (SystemCommandKind.COVER_PELLET, 117, None),
    (SystemCommandKind.PLAY_TONE, 118, None),
    (SystemCommandKind.DELAY, 119, 0.5),
    (SystemCommandKind.READ_MOTOR_CONFIGURATION, 120, Motor.PELLET_X_MOTOR),
    (SystemCommandKind.WRITE_MOTOR_CONFIGURATION, 121, (Motor.PELLET_X_MOTOR, StepperConfig())),
    (SystemCommandKind.SEND_FIXED_XYZ, 122, None),
    (SystemCommandKind.SEND_FIXED_XYZ, 123, None),
    (SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE, 124, default_move_retract()),
])
def test_notify_command(device, kind, tag, data):
    device.notify_message(kind, data, tag)


@pytest.mark.parametrize("data, kind", [
    (StepperStatus(Target.PELLET_DEVICE, Motor.PELLET_X_MOTOR, 10, 2.0, False),
     SystemStatusMessageKind.PELLET_MOTOR_X),
    (ServoStatus(Target.PELLET_DEVICE, Motor.PELLET_LOAD_SERVO, 40),
     SystemStatusMessageKind.PELLET_LOAD),
    (ServoConfig(Target.PELLET_DEVICE, Motor.PELLET_LOAD_SERVO, 0, 0, 0, 0, 0, 0),
     SystemStatusMessageKind.MOTOR_CONFIGURATION),
    (StepperConfig(Target.PELLET_DEVICE, Motor.PELLET_X_MOTOR, 0, 0, 0, 0, False),
     SystemStatusMessageKind.MOTOR_CONFIGURATION),
])
def test_notify_data(device, data, kind):
    global _expected

    if kind:
        _expected = kind

    device.notify_data([data])


@pytest.mark.parametrize("kind,data", (
    (SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE, default_move_retract()),
    (SystemCommandKind.SET_LOAD_PELLET_PROCEDURE, default_load_pellet()),
    (SystemCommandKind.SET_SEND_PELLET_PROCEDURE, default_send_pellet()),
))
def test_set_procedures(
    expected_tok,
    expected_tok_event,
    tokens_acked,
    device,
    kind,
    data,
):
    ctx = uuid.uuid4()
    expected_tok.value = ctx
    device.notify_message(kind, data, context=ctx)
    expected_tok_event.wait(3)  # should be quite faster
    assert ctx in tokens_acked


def test_move_relative(
    expected_tok,
    expected_tok_event,
    tokens_acked,
    device,
    device_conn,
):
    # we rely on that on start:
    dev_positions = device.device_interface._positions  # noqa
    assert dev_positions[Motor.PELLET_X_MOTOR] == 0
    assert dev_positions[Motor.PELLET_Y_MOTOR] == 0
    ctx = uuid.uuid4()
    expected_tok.value = ctx
    device.notify_message(SystemCommandKind.SEND_RETRACT, None, context=ctx)
    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    tokens_acked.clear()
    # emulation iface doesn't check motor limits, so the result position is 0 + retract_offset,
    # default one being -15, so we get -15 :
    assert math.isclose(dev_positions[Motor.PELLET_Y_MOTOR], -15, abs_tol=0.1)
    #
    expected_tok_event.clear()
    expected_tok.value = ctx
    device.notify_message(SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE,
                          MotorSteps("custom", [{'y_rel': 20}, {'x_rel': -5}]),
                          context=ctx)
    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    tokens_acked.clear()
    expected_tok_event.clear()
    expected_tok.value = ctx
    device.notify_message(SystemCommandKind.SEND_RETRACT, None, context=ctx)
    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    # tokens_acked.clear()
    assert math.isclose(dev_positions[Motor.PELLET_X_MOTOR], -5, abs_tol=0.1)  # -5
    assert math.isclose(dev_positions[Motor.PELLET_Y_MOTOR], 5, abs_tol=0.1)  # -15 + 20 == 5


def test_emulated_laser_2_pre_reveal_send_pulses_board_stim2_and_completes(
    expected_tok,
    expected_tok_event,
    tokens_acked,
    device,
    device_conn,
):
    # The whole SEND through the command queue: STIM2 pulse, board delay,
    # reveal, then the configured send, acknowledged as one command.
    interface = device.device_interface
    interface.pulse_digital_output = mock.Mock(wraps=interface.pulse_digital_output)
    ctx = uuid.uuid4()
    expected_tok.value = ctx

    device.notify_message(
        SystemCommandKind.SEND_PELLET,
        {"pre_reveal_stimulus": (200, 1000, 2)},
        context=ctx,
    )
    expected_tok_event.wait(3)

    assert ctx in tokens_acked
    interface.pulse_digital_output.assert_called_once_with(
        DigitalOutputs.STIMULUS_3, 1000,
    )


def test_can_connect_twice(device, caplog):
    with caplog.at_level(logging.DEBUG):
        device.connect()
    assert "CAN command Handler thread already alive" in caplog.text
    assert device.connected
    assert device.device_interface.is_open is True


def test_rel_move_succeed_after_uuid_ack_timeout(
    expected_tok,
    expected_tok_event,
    tokens_acked,
    device,
    device_conn,
    monkeypatch,
    caplog,
):
    orig_move_y = device.device_interface.move_motor_y
    def ret_move(*args, **kwargs):
        # consume one uuid, but don't insert ack into return messages as with emulation iface
        device.device_interface.next_uuid()
        # restore orig move:
        device.device_interface.move_motor_y = orig_move_y
        # return True to fake command written to CAN bus ok:
        return True
    m = mock.MagicMock()
    m.side_effect = ret_move
    device.device_interface.move_motor_y = m
    ctx = uuid.uuid4()
    expected_tok.value = ctx
    device.default_command_ack_timeout_duration = 0.5

    ack_timeout_engaged = False
    ack_timeout_engaged_count = 0
    def dev_prop_changed(name, value, old):
        if name == device.UUID_ACK_TIMEOUT_ENGAGED:
            nonlocal ack_timeout_engaged, ack_timeout_engaged_count
            ack_timeout_engaged = value
            if value:
                ack_timeout_engaged_count += 1

    device.property_changed += dev_prop_changed

    device.notify_message(SystemCommandKind.SEND_RETRACT, None, context=ctx)

    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    assert not ack_timeout_engaged
    assert ack_timeout_engaged_count == 1
    assert device._commands_handler_thread.is_alive()


def test_pyjerrycan_interface_keeps_the_configured_transport():
    """A non-SocketCAN transport must not be replaced by bare constructor defaults.

    The pyjerrycan branch used to drop the caller's configuration entirely, so a
    custom channel or receive timeout was silently ignored.
    """
    transport = CanTransportConfiguration(
        kind="pyjerrycan",
        channel="can7",
        receive_timeout_seconds=0.25,
    )
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        can_transport=transport,
    )
    with mock.patch.object(can_device, "HAVE_CAN_DEVICE", True):
        interface = device._make_device_interface(force_emulation=False)

    assert isinstance(interface, CanInterface)
    assert interface._can_transport.channel == "can7"
    assert interface._can_transport.receive_timeout_seconds == 0.25
