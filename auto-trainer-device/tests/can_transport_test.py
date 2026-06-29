import pytest

from autotrainer.device import (
    CanTransportConfiguration,
    CanTransportKind,
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
