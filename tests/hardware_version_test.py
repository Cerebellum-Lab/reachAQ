import pytest

from autotrainer.device import (
    CanDevice,
    CanTransportConfiguration,
    CanTransportKind,
    EmulationInterface,
)
from autotrainer.device import can_device
from autotrainer.model import EnvironmentProvider, HardwareVersion
from autotrainer.model import hardware_version as hardware_version_module


@pytest.fixture(autouse=True)
def reset_hardware_version(monkeypatch):
    monkeypatch.delenv("AUTOTRAINER_HARDWARE_VERSION", raising=False)
    yield
    if hasattr(EnvironmentProvider, "set_hardware_version"):
        EnvironmentProvider.set_hardware_version(None)


@pytest.mark.parametrize("have_pyjerrycan", (False, True))
def test_default_hardware_version_is_alogus_regardless_of_pyjerrycan(
    monkeypatch,
    have_pyjerrycan,
):
    monkeypatch.setattr(
        hardware_version_module,
        "HAVE_CAN_DEVICE",
        have_pyjerrycan,
        raising=False,
    )

    assert hardware_version_module.default_determine_hardware_version() == HardwareVersion.ALOGUS_V1


def test_environment_provider_does_not_cache_default_before_explicit_profile_selection(
    monkeypatch,
):
    assert EnvironmentProvider.hardware_version() == HardwareVersion.ALOGUS_V1

    monkeypatch.setenv("AUTOTRAINER_HARDWARE_VERSION", "anschutz")
    assert EnvironmentProvider.hardware_version() == HardwareVersion.ANSHUTZ

    monkeypatch.setenv("AUTOTRAINER_HARDWARE_VERSION", "alogus")
    assert EnvironmentProvider.hardware_version() == HardwareVersion.ALOGUS_V1


def test_environment_provider_accepts_an_explicit_runtime_profile():
    EnvironmentProvider.set_hardware_version(HardwareVersion.ANSHUTZ)
    assert EnvironmentProvider.hardware_version() == HardwareVersion.ANSHUTZ

    EnvironmentProvider.set_hardware_version(HardwareVersion.ALOGUS_V1)
    assert EnvironmentProvider.hardware_version() == HardwareVersion.ALOGUS_V1


def test_socketcan_without_pyjerrycan_uses_real_interface_and_alogus_profile(
    monkeypatch,
):
    monkeypatch.setattr(can_device, "HAVE_CAN_DEVICE", False)

    device = CanDevice(
        can_transport=CanTransportConfiguration(kind=CanTransportKind.SOCKETCAN)
    )

    assert not isinstance(device.device_interface, EmulationInterface)
    assert EnvironmentProvider.hardware_version() == HardwareVersion.ALOGUS_V1


def test_invalid_explicit_hardware_profile_does_not_silently_fallback(monkeypatch):
    monkeypatch.setenv("AUTOTRAINER_HARDWARE_VERSION", "not-a-profile")

    with pytest.raises(ValueError, match="Unsupported hardware version"):
        EnvironmentProvider.hardware_version()
