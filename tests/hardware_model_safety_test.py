import threading
import os
from queue import Queue
from unittest import mock
from uuid import uuid4

import pytest

from autotrainer.core import SystemCommandKind
from autotrainer.device import (
    CanFailure,
    CanFailureKind,
    CanTransportConfiguration,
    CanTransportKind,
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
        side_effect=(RuntimeError("first reopen failed"), None),
    )
    hardware_model._first_can_failure = CanFailure(
        CanFailureKind.TRANSPORT,
        "reader failed",
    )

    hardware_model._run_can_recovery(hardware_model._first_can_failure)

    assert hardware_model.connect.call_count == 2
    hardware_model.connect.assert_called_with(command_queue, _recovery=True)
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


def test_recovery_handle_is_cleared_when_command_queue_is_missing(hardware_model):
    failure = CanFailure(CanFailureKind.TRANSPORT, "reader failed")
    hardware_model._command_queue = None
    hardware_model._can_recovery_generation = 7
    hardware_model._can_recovery_failure = failure
    hardware_model._can_recovery_thread = mock.sentinel.thread

    hardware_model._run_can_recovery(failure, 7)

    assert hardware_model._can_recovery_thread is None
