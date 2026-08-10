import threading
import os
from unittest import mock

from autotrainer.core import SystemCommandKind
from autotrainer.device import CanTransportConfiguration, CanTransportKind
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


def test_safety_shutdown_resets_configured_transport_without_active_device(monkeypatch):
    hardware = object.__new__(HardwareModel)
    hardware._safety_shutdown_lock = threading.Lock()
    hardware._safety_shutdown_started = False
    hardware._safety_shutdown_thread = None
    hardware._can_device = None
    hardware._disconnect_transport = mock.Mock()
    hardware._reset_socketcan = mock.Mock()
    monkeypatch.setattr(
        "tools.acquisition.model.hardware_model.CanTransportConfiguration.from_environment",
        lambda: mock.sentinel.transport,
    )

    hardware.safety_shutdown("application close")

    hardware._disconnect_transport.assert_called_once_with()
    hardware._reset_socketcan.assert_called_once_with(mock.sentinel.transport)


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
