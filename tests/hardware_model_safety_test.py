import contextlib
import threading
import os
from queue import Queue
from unittest import mock
from uuid import uuid4

import pytest

from autotrainer.core import Motor, SystemCommandKind
from autotrainer.device import (
    CanFailure,
    CanFailureKind,
    CanTransportConfiguration,
    CanTransportKind,
    CommandRefusal,
    CommandRefusedError,
    RefusedCommandClass,
)
from tools.acquisition.model.hardware_model import HardwareModel


def test_test_harness_forces_can_emulation():
    assert os.environ["AUTOTRAINER_CAN_TRANSPORT"] == "emulation"
    assert (
        CanTransportConfiguration.from_environment().kind
        is CanTransportKind.EMULATION
    )


def test_safety_shutdown_is_idempotent_and_disconnects_before_reset():
    events = []
    hardware = object.__new__(HardwareModel)
    hardware._safety_shutdown_lock = threading.Lock()
    hardware._safety_shutdown_started = False
    hardware._safety_shutdown_thread = None
    hardware._can_recovery_cancel = threading.Event()
    hardware._can_recovery_lock = threading.Lock()
    hardware._can_recovery_generation = 0
    hardware._can_device = mock.Mock()
    hardware._can_device.can_transport_configuration = mock.sentinel.transport
    hardware._disconnect_transport = mock.Mock(side_effect=lambda: events.append("disconnect"))
    hardware._reset_socketcan = mock.Mock(side_effect=lambda _: events.append("reset"))

    hardware.safety_shutdown("first")
    hardware.safety_shutdown("second")

    assert events == ["disconnect", "reset"]
    assert hardware._safety_shutdown_thread.daemon is False
    hardware._disconnect_transport.assert_called_once_with()
    hardware._reset_socketcan.assert_called_once_with(mock.sentinel.transport)


def test_confirmed_can_safety_shutdown_resets_configured_transport_without_active_device(monkeypatch):
    hardware = object.__new__(HardwareModel)
    hardware._safety_shutdown_lock = threading.Lock()
    hardware._safety_shutdown_started = False
    hardware._safety_shutdown_thread = None
    hardware._can_recovery_cancel = threading.Event()
    hardware._can_recovery_lock = threading.Lock()
    hardware._can_recovery_generation = 0
    hardware._can_device = None
    hardware._disconnect_transport = mock.Mock()
    hardware._reset_socketcan = mock.Mock()
    monkeypatch.setattr(
        "tools.acquisition.model.hardware_model.CanTransportConfiguration.from_environment",
        lambda: mock.sentinel.transport,
    )

    hardware.safety_shutdown("confirmed CAN failure")

    hardware._disconnect_transport.assert_called_once_with()
    hardware._reset_socketcan.assert_called_once_with(mock.sentinel.transport)


def test_ordinary_disconnect_closes_only_owned_transport():
    hardware = object.__new__(HardwareModel)
    hardware._can_recovery_cancel = threading.Event()
    hardware._can_recovery_lock = threading.Lock()
    hardware._can_recovery_generation = 0
    hardware._disconnect_transport = mock.Mock()
    hardware._reset_socketcan = mock.Mock()
    hardware._can_connection_state = {"state": "ready", "error": ""}
    hardware._on_property_changed = mock.Mock()

    hardware.disconnect()

    hardware._disconnect_transport.assert_called_once_with()
    hardware._reset_socketcan.assert_not_called()


def test_privileged_reset_is_prohibited_under_pytest():
    hardware = object.__new__(HardwareModel)
    transport = CanTransportConfiguration(
        kind=CanTransportKind.SOCKETCAN,
        channel="can0",
        fd=True,
    )

    with pytest.raises(RuntimeError, match="prohibited under automated tests"):
        hardware._reset_socketcan(transport)


def test_safety_shutdown_latch_rejects_new_commands():
    hardware = object.__new__(HardwareModel)
    hardware._safety_shutdown_lock = threading.Lock()
    hardware._safety_shutdown_started = True
    device = mock.Mock()

    sent = hardware._send_command(
        device,
        SystemCommandKind.SET_RGB_LED,
        (0, 0, 0),
        context="after-shutdown",
    )

    assert sent is False
    device.send_message.assert_not_called()


def test_transport_loss_marks_inflight_command_unknown_without_replay(hardware_model):
    token = uuid4()
    hardware_model._pending_tokens[token] = (SystemCommandKind.SEND_PELLET, 1.0)
    hardware_model._run_can_recovery = mock.Mock()
    reported = []
    hardware_model.command_failed += reported.append

    hardware_model._on_can_failure(CanFailure(
        CanFailureKind.TRANSPORT,
        "adapter removed",
        category="device_removed",
    ))
    hardware_model._can_recovery_thread.join(1)

    assert len(reported) == 1
    assert reported[0].kind is CanFailureKind.OPERATION_UNKNOWN
    assert reported[0].command is SystemCommandKind.SEND_PELLET
    assert reported[0].context == str(token)
    assert "not replayed" in reported[0].error
    hardware_model._run_can_recovery.assert_called_once()


def test_bounded_recovery_reuses_full_connection_initialization(hardware_model):
    command_queue = Queue()
    hardware_model._command_queue = command_queue
    hardware_model._can_device = mock.Mock()
    hardware_model._can_device.can_transport_configuration = CanTransportConfiguration(
        kind=CanTransportKind.EMULATION,
    )
    hardware_model._disconnect_transport = mock.Mock()
    hardware_model.connect = mock.Mock(
        side_effect=(RuntimeError("first reopen failed"), True),
    )
    hardware_model._first_can_failure = CanFailure(
        CanFailureKind.TRANSPORT,
        "reader failed",
    )

    hardware_model._run_can_recovery(hardware_model._first_can_failure)

    assert hardware_model.connect.call_count == 2
    hardware_model.connect.assert_called_with(
        command_queue,
        _recovery=True,
        _generation=1,
    )
    assert hardware_model._can_connection_state == {"state": "ready", "error": ""}
    assert hardware_model._first_can_failure.error == "reader failed"
    assert hardware_model._can_recovery_failure is None


def test_ack_failure_captures_interface_counters_before_recovery(
    hardware_model,
    monkeypatch,
):
    hardware_model._can_device = mock.Mock()
    hardware_model._can_device.can_transport_configuration = CanTransportConfiguration(
        kind=CanTransportKind.SOCKETCAN,
        channel="can7",
        fd=True,
    )
    hardware_model._run_can_recovery = mock.Mock()
    captured = {"channel": "can7", "interface_state": "ERROR-ACTIVE"}
    monkeypatch.setattr(
        "tools.acquisition.model.hardware_model.capture_can_diagnostics",
        lambda channel: captured,
    )
    reported = []
    hardware_model.command_failed += reported.append

    hardware_model._on_can_failure(CanFailure(
        CanFailureKind.ACKNOWLEDGEMENT_TIMEOUT,
        "pellet board acknowledgement timed out",
        command=SystemCommandKind.SEND_PELLET,
        context="send-1",
    ))
    hardware_model._can_recovery_thread.join(1)

    assert reported[0].diagnostics == captured
    assert hardware_model._first_can_failure.diagnostics == captured
    hardware_model._run_can_recovery.assert_called_once()


def test_command_failure_does_not_latch_or_suppress_later_transport_recovery(
    hardware_model,
):
    hardware_model._run_can_recovery = mock.Mock()

    hardware_model._on_can_failure(CanFailure(
        CanFailureKind.COMMAND,
        "motor command rejected",
        command=SystemCommandKind.SEND_PELLET,
    ))

    assert hardware_model._first_can_failure.error == "motor command rejected"
    assert hardware_model._can_recovery_failure is None
    assert hardware_model._can_connection_state["state"] != "failed"

    hardware_model._on_can_failure(CanFailure(
        CanFailureKind.TRANSPORT,
        "adapter removed",
    ))
    hardware_model._can_recovery_thread.join(1)

    assert hardware_model._can_recovery_failure.error == "adapter removed"
    hardware_model._run_can_recovery.assert_called_once()


def test_stale_recovery_generation_cannot_publish_ready(hardware_model):
    hardware_model._can_recovery_generation = 4
    before = dict(hardware_model._can_connection_state)

    assert not hardware_model._set_can_connection_state("ready", generation=3)

    assert hardware_model._can_connection_state == before


def test_stale_recovery_cannot_enter_connect_lifecycle(hardware_model):
    hardware_model._can_recovery_generation = 4
    before = dict(hardware_model._can_connection_state)

    connected = hardware_model.connect(
        Queue(),
        _recovery=True,
        _generation=3,
    )

    assert connected is False
    assert hardware_model._can_connection_state == before


def test_recovery_handle_is_cleared_when_command_queue_is_missing(hardware_model):
    failure = CanFailure(CanFailureKind.TRANSPORT, "reader failed")
    hardware_model._command_queue = None
    hardware_model._can_recovery_generation = 7
    hardware_model._can_recovery_failure = failure
    hardware_model._can_recovery_thread = mock.sentinel.thread

    hardware_model._run_can_recovery(failure, 7)

    assert hardware_model._can_recovery_thread is None


def test_transport_close_cannot_be_mistaken_for_command_ack(hardware_model):
    token = uuid4()
    hardware_model._pending_tokens[token] = (SystemCommandKind.SEND_HOME, 1.0)
    hardware_model._command_outcomes[token] = ("pending", "")
    hardware_model._device_conn = None
    hardware_model._can_device = None

    hardware_model._disconnect_transport(command_outcome="unknown")

    with pytest.raises(RuntimeError, match="ended as unknown"):
        hardware_model.wait_pending_command_acked(token, timeout=0.01)


def test_acknowledged_command_has_explicit_terminal_outcome(hardware_model):
    token = uuid4()
    hardware_model._pending_tokens[token] = (SystemCommandKind.SEND_HOME, 1.0)
    hardware_model._command_outcomes[token] = ("pending", "")

    hardware_model._ack_received(token)

    hardware_model.wait_pending_command_acked(token, timeout=0.01)
    assert hardware_model._command_outcomes[token] == ("acknowledged", "")


_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)


def _refusal(
    token,
    error=-5,
    *,
    command=SystemCommandKind.SEND_PELLET,
    command_class=RefusedCommandClass.FIXED_XYZ,
    motors=_XYZ,
    resent=False,
):
    return CommandRefusal(
        token=token, error=error, command=command, step="{'predefined': 'send'}",
        command_class=command_class, motors=motors, resent=resent,
    )


def _pending(hardware_model, command=SystemCommandKind.SEND_PELLET, started=1.0):
    token = uuid4()
    hardware_model._pending_tokens[token] = (command, started)
    hardware_model._command_outcomes[token] = ("pending", "")
    return token


def test_a_refused_command_has_explicit_terminal_outcome(hardware_model):
    token = _pending(hardware_model)

    hardware_model._command_refused(token, refusal=_refusal(token))

    with pytest.raises(RuntimeError, match="ended as refused"):
        hardware_model.wait_pending_command_acked(token, timeout=0.01)
    outcome, error = hardware_model._command_outcomes[token]
    assert outcome == "refused"
    assert "EIO" in error


def test_a_refusal_removes_the_pending_token_and_refreshes_the_in_progress_property(hardware_model):
    token = _pending(hardware_model)
    _pending(hardware_model, SystemCommandKind.SET_X, started=2.0)
    shown = []
    hardware_model.property_changed += (
        lambda name, value, old:
        shown.append(value) if name == HardwareModel.PENDING_COMMAND_PROPERTY else None)

    hardware_model._command_refused(token, refusal=_refusal(token))

    assert token not in hardware_model._pending_tokens
    assert shown == ["SET_X"]


def test_a_refusal_is_reported_as_a_command_failure_naming_the_code(hardware_model):
    token = _pending(hardware_model)
    reported = []
    hardware_model.command_failed += reported.append

    hardware_model._command_refused(token, refusal=_refusal(token, -1))

    (failure,) = reported
    assert failure.kind is CanFailureKind.COMMAND
    assert failure.command is SystemCommandKind.SEND_PELLET
    assert failure.context == str(token)
    assert failure.category == "board_refusal"
    assert failure.error_code == -1
    assert "E-stop" in failure.error and "EPERM" in failure.error


def test_a_resent_ebusy_refusal_is_reported_as_an_unknown_outcome(hardware_model):
    token = _pending(hardware_model, SystemCommandKind.SEND_HOME)
    reported = []
    hardware_model.command_failed += reported.append

    hardware_model._command_refused(token, refusal=_refusal(
        token, -16, command=SystemCommandKind.SEND_HOME,
        command_class=RefusedCommandClass.STEPPER_HOME,
        motors=(Motor.PELLET_Y_MOTOR,), resent=True,
    ))

    assert reported[0].kind is CanFailureKind.OPERATION_UNKNOWN
    assert hardware_model.pellet_axes_needing_home == ()


def test_a_refusal_does_not_start_can_recovery(hardware_model):
    hardware_model._run_can_recovery = mock.Mock()
    token = _pending(hardware_model)

    hardware_model._command_refused(token, refusal=_refusal(token, -116))

    hardware_model._run_can_recovery.assert_not_called()
    assert hardware_model._can_recovery_thread is None
    assert hardware_model._can_connection_state["state"] != "failed"


def test_a_home_ack_clears_needs_home_and_a_refusal_sets_it(hardware_model):
    published = []
    hardware_model.property_changed += (
        lambda name, value, old:
        published.append(value) if name == HardwareModel.PELLET_NEEDS_HOME_PROPERTY else None)
    assert hardware_model.pellet_axes_needing_home == ()
    move = _pending(hardware_model, SystemCommandKind.MOVE_X)

    hardware_model._command_refused(move, refusal=_refusal(
        move, -16, command=SystemCommandKind.MOVE_X,
        command_class=RefusedCommandClass.STEPPER_MOVE, motors=(Motor.PELLET_X_MOTOR,),
    ))
    assert hardware_model.pellet_axes_needing_home == ("X",)
    assert "needs homing" in hardware_model.last_command_refusal

    hardware_model._ack_received(_pending(hardware_model, SystemCommandKind.SET_Y))
    assert hardware_model.pellet_axes_needing_home == ("X",), "only a home clears it"

    hardware_model._ack_received(_pending(hardware_model, SystemCommandKind.SEND_HOME))
    assert hardware_model.pellet_axes_needing_home == ()
    assert published == [("X",), ()]


def test_a_refusal_of_a_token_this_model_did_not_send_still_marks_needs_home(hardware_model):
    """The connect-time home is sent outside the pending-token table."""
    reported = []
    hardware_model.command_failed += reported.append

    hardware_model._command_refused("connect-home", refusal=_refusal(
        "connect-home", -5, command=SystemCommandKind.SEND_HOME,
        command_class=RefusedCommandClass.STEPPER_HOME, motors=(Motor.PELLET_Y_MOTOR,),
    ))

    assert hardware_model.pellet_axes_needing_home == ("X", "Y", "Z")
    assert reported[0].command is SystemCommandKind.SEND_HOME


class _ConnectionWhoseHomeIsRefused:
    """Stands in for DeviceConnection: version and configuration succeed, the home is refused."""

    refusal = None
    on_version = None

    def __init__(self, device, cmd_queue, *, name, failure_callback):
        self.device = device
        self.sent = []

    def request_connect(self):
        pass

    def wait_connected(self, *, timeout):
        pass

    @contextlib.contextmanager
    def await_acknowledge(self, tokens, *, timeout):
        yield
        if self.sent[-1] is SystemCommandKind.SEND_HOME:
            raise CommandRefusedError(self.refusal)

    def send_message(self, kind, data=None, context=None):
        self.sent.append(kind)
        if kind is SystemCommandKind.REQUEST_VERSION:
            type(self).on_version()

    def load_default_motor_config(self):
        pass

    def load_default_move_config(self):
        pass

    def request_disconnect(self):
        pass

    def join(self):
        return True


def test_connect_fails_with_the_reason_when_the_connect_time_home_is_refused(
    hardware_model, monkeypatch,
):
    import tools.acquisition.model.hardware_model as hardware_model_module
    from autotrainer.core import MessageHandler

    connection = _ConnectionWhoseHomeIsRefused
    monkeypatch.setattr(connection, "refusal", _refusal(
        "home", -5, command=SystemCommandKind.SEND_HOME,
        command_class=RefusedCommandClass.STEPPER_HOME, motors=(Motor.PELLET_Y_MOTOR,),
    ))
    monkeypatch.setattr(connection, "on_version", staticmethod(
        lambda: hardware_model._message_handler_property_changed(
            MessageHandler.FIRMWARE_VERSION_PROPERTY, "Pellet Emulator v0.1.0", None)))
    monkeypatch.setattr(hardware_model_module, "DeviceConnection", connection)
    monkeypatch.setattr(HardwareModel, "_request_board_capabilities", lambda self, device: None)

    try:
        with pytest.raises(CommandRefusedError, match="once more"):
            hardware_model.connect(Queue())
        assert hardware_model._device_conn.sent[-1] is SystemCommandKind.SEND_HOME
        assert hardware_model._can_connection_state["state"] != "ready"
    finally:
        hardware_model.disconnect()


def test_recovery_does_not_retry_a_connect_whose_home_was_refused(hardware_model):
    hardware_model._command_queue = Queue()
    hardware_model._can_device = mock.Mock()
    hardware_model._can_device.can_transport_configuration = CanTransportConfiguration(
        kind=CanTransportKind.EMULATION,
    )
    hardware_model._disconnect_transport = mock.Mock()
    refusal = _refusal(
        "home", -1, command=SystemCommandKind.SEND_HOME,
        command_class=RefusedCommandClass.STEPPER_HOME, motors=(Motor.PELLET_Y_MOTOR,),
    )
    hardware_model.connect = mock.Mock(side_effect=CommandRefusedError(refusal))
    failure = CanFailure(CanFailureKind.TRANSPORT, "reader failed")
    hardware_model._first_can_failure = failure

    hardware_model._run_can_recovery(failure)

    assert hardware_model.connect.call_count == 1
    assert hardware_model._can_connection_state["state"] == "failed"
    assert "E-stop" in hardware_model._can_connection_state["error"]


def test_a_refusal_is_logged_below_error_so_the_footer_keeps_the_operator_notice(
    hardware_model, caplog,
):
    """Every ERROR record reaches the status bar, and the last one wins there.

    The device logs the refusal at ERROR first and AppModel shows the operator
    notice last; a second ERROR copy from here would replace the notice with a
    line carrying the token.
    """
    import logging

    token = _pending(hardware_model)
    with caplog.at_level(logging.DEBUG, logger="tools.acquisition.model.hardware_model"):
        hardware_model._command_refused(token, refusal=_refusal(token))

    refused = [r for r in caplog.records if "refused" in r.getMessage()]
    assert refused, "the refusal is still logged"
    assert all(r.levelno < logging.ERROR for r in refused)
