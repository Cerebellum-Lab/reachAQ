import logging

import pytest

from autotrainer.device import can_transport
from autotrainer.device import (
    CanDevice,
    CanTransportConfiguration,
    CanTransportKind,
    EmulationInterface,
    Target,
    normalize_can_transport_kind,
)


def test_can_transport_configuration_defaults_to_current_backend():
    config = CanTransportConfiguration()

    assert config.kind == CanTransportKind.PYJERRYCAN
    assert config.channel == "can0"
    assert config.uses_linux_can_stack is False


def test_can_transport_configuration_accepts_socketcan_mapping():
    config = CanTransportConfiguration.from_mapping({
        "type": "socketcan",
        "channel": "can1",
        "bitrate": 500000,
        "fd": True,
        "data_bitrate": 2000000,
    })

    assert config.kind == CanTransportKind.SOCKETCAN
    assert config.channel == "can1"
    assert config.uses_linux_can_stack is True


def test_normalize_can_transport_kind_rejects_unknown_kind():
    with pytest.raises(ValueError):
        normalize_can_transport_kind("not-a-backend")


def test_can_transport_configuration_rejects_invalid_bitrate():
    with pytest.raises(ValueError):
        CanTransportConfiguration(kind="socketcan", bitrate=0)


def test_can_transport_configuration_rejects_data_bitrate_without_fd():
    with pytest.raises(ValueError, match="data_bitrate requires CAN FD"):
        CanTransportConfiguration(
            kind="socketcan",
            data_bitrate=2000000,
            fd=False,
        )


def test_can_transport_configuration_reads_environment(monkeypatch):
    monkeypatch.setenv("AUTOTRAINER_CAN_TRANSPORT", "socketcan")
    monkeypatch.setenv("AUTOTRAINER_CAN_CHANNEL", "can1")
    monkeypatch.setenv("AUTOTRAINER_CAN_BITRATE", "500000")
    monkeypatch.setenv("AUTOTRAINER_CAN_DATA_BITRATE", "2000000")
    monkeypatch.setenv("AUTOTRAINER_CAN_FD", "true")

    config = CanTransportConfiguration.from_environment()

    assert config.kind == CanTransportKind.SOCKETCAN
    assert config.channel == "can1"
    assert config.bitrate == 500000
    assert config.data_bitrate == 2000000
    assert config.fd is True


def test_can_transport_configuration_defaults_linux_x86_to_socketcan_fd(monkeypatch):
    for name in (
        "TRANSPORT",
        "TYPE",
        "CHANNEL",
        "BITRATE",
        "DATA_BITRATE",
        "FD",
        "RECEIVE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(f"AUTOTRAINER_CAN_{name}", raising=False)
    monkeypatch.setattr(can_transport.platform, "system", lambda: "Linux")
    monkeypatch.setattr(can_transport.platform, "machine", lambda: "x86_64")

    config = CanTransportConfiguration.from_environment()

    assert config.kind == CanTransportKind.SOCKETCAN
    assert config.channel == "can0"
    assert config.bitrate == 1000000
    assert config.data_bitrate == 5000000
    assert config.fd is True


def test_can_transport_configuration_keeps_legacy_default_on_linux_arm(monkeypatch):
    monkeypatch.delenv("AUTOTRAINER_CAN_TRANSPORT", raising=False)
    monkeypatch.delenv("AUTOTRAINER_CAN_TYPE", raising=False)
    monkeypatch.setattr(can_transport.platform, "system", lambda: "Linux")
    monkeypatch.setattr(can_transport.platform, "machine", lambda: "aarch64")

    config = CanTransportConfiguration.from_environment()

    assert config.kind == CanTransportKind.PYJERRYCAN
    assert config.fd is False


def test_can_device_accepts_explicit_emulation_transport(caplog):
    with caplog.at_level(logging.WARNING):
        device = CanDevice(can_transport=CanTransportConfiguration(kind="emulation"))

    assert device.can_transport_configuration.kind == CanTransportKind.EMULATION
    assert isinstance(device.device_interface, EmulationInterface)
    assert "Using emulation interface" in caplog.text


def test_can_device_accepts_linux_transport_backend(caplog):
    with caplog.at_level(logging.WARNING):
        device = CanDevice(can_transport=CanTransportConfiguration(kind="socketcan"))

    assert device.can_transport_configuration.kind == CanTransportKind.SOCKETCAN
    assert not isinstance(device.device_interface, EmulationInterface)
    assert "Using emulation interface" not in caplog.text


def test_can_device_accepts_pellet_only_required_targets():
    device = CanDevice(
        can_transport=CanTransportConfiguration(kind="emulation"),
        required_targets=(Target.PELLET_DEVICE,),
    )

    assert device.required_targets == (Target.PELLET_DEVICE,)
    assert device.is_target_required(Target.PELLET_DEVICE)
    assert not device.is_target_required(Target.MAGNET_DEVICE)
