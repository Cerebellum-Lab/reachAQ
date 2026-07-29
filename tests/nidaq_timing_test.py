from autotrainer.core import (
    NidaqDeviceIdentity,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    NidaqTimingConfiguration,
)
from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts
from tools.acquisition.model.nidaq_timing import build_nidaq_timing_plan


def _stream(*channels):
    return NidaqSignalStreamConfiguration(
        channels=tuple(
            NidaqSignalChannelConfiguration(name, physical, kind)
            for name, physical, kind in channels
        ),
        is_enabled=True,
    )


def _pxi(name, serial):
    return NidaqDevicePorts(
        name=name,
        product_type=f"Model-{serial}",
        serial_number=serial,
        bus_type="PXI",
        pxi_chassis_number=1,
        analog_output_sample_clock_supported=True,
        digital_trigger_supported=True,
    )


def test_single_device_uses_same_device_hardware_timeline():
    configuration = _stream(
        ("cam_frames", "InputCard/port0/line0", "digital"),
        ("laser_feedback", "InputCard/ai0", "analog"),
    )

    plan = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(),
        (NidaqDevicePorts(name="InputCard"),),
    )

    assert plan.is_valid
    assert plan.resolved_mode == "same_device"
    assert plan.master_device == "InputCard"
    assert plan.task_start_order == ("InputCard",)


def test_pxi_input_master_and_hardware_timed_output_slave_use_backplane():
    configuration = _stream(
        ("cam_frames", "Acquire/port0/line0", "digital"),
        ("laser_feedback", "Acquire/ai0", "analog"),
    )

    plan = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(),
        (_pxi("LaserOut", 40), _pxi("Acquire", 50)),
        hardware_timed_output_devices=("LaserOut",),
    )

    assert plan.is_valid
    assert plan.resolved_mode == "backplane"
    assert plan.master_device == "Acquire"
    assert plan.slave_devices == ("LaserOut",)
    assert plan.reference_clock_source == "PXI_CLK10"
    assert plan.sample_clock_source == "/Acquire/ai/SampleClock"
    assert plan.start_trigger_source == "/Acquire/ai/StartTrigger"
    assert plan.task_start_order == ("LaserOut", "Acquire")


def test_manual_master_resolves_by_serial_after_runtime_alias_changes():
    configuration = _stream(
        ("first", "RenamedA/ai0", "analog"),
        ("second", "RenamedB/ai0", "analog"),
    )
    timing = NidaqTimingConfiguration(
        timing_master=NidaqDeviceIdentity(
            logical_name="canonical-input",
            runtime_name="OldAlias",
            product_type="Model-22",
            serial_number=22,
        ),
    )

    plan = build_nidaq_timing_plan(
        configuration,
        timing,
        (_pxi("RenamedA", 11), _pxi("RenamedB", 22)),
    )

    assert plan.is_valid
    assert plan.master_device == "RenamedB"


def test_missing_or_substituted_manual_master_is_not_silently_rebound():
    configuration = _stream(
        ("first", "Acquire/ai0", "analog"),
        ("second", "Other/ai0", "analog"),
    )
    timing = NidaqTimingConfiguration(
        timing_master=NidaqDeviceIdentity(
            logical_name="canonical-input",
            product_type="Expected",
            serial_number=999,
        ),
    )

    plan = build_nidaq_timing_plan(
        configuration,
        timing,
        (_pxi("Acquire", 11), _pxi("Other", 22)),
    )

    assert not plan.is_valid
    assert "did not resolve" in plan.reason


def test_non_pxi_multi_device_requires_explicit_external_routes():
    configuration = _stream(
        ("first", "DevA/ai0", "analog"),
        ("second", "DevB/ai0", "analog"),
    )
    devices = (
        NidaqDevicePorts(name="DevA", bus_type="USB"),
        NidaqDevicePorts(name="DevB", bus_type="USB"),
    )

    unavailable = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(),
        devices,
    )
    external = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(
            sync_mode="external",
            reference_clock_source="/DevA/PFI0",
            start_trigger_source="/DevA/PFI1",
            sample_clock_source="/DevA/PFI2",
        ),
        devices,
    )

    assert not unavailable.is_valid
    assert external.is_valid
    assert external.resolved_mode == "external"


def test_independent_mode_is_diagnostic_only_when_alignment_is_required():
    configuration = _stream(
        ("first", "DevA/ai0", "analog"),
        ("second", "DevB/ai0", "analog"),
    )
    devices = (
        NidaqDevicePorts(name="DevA"),
        NidaqDevicePorts(name="DevB"),
    )

    plan = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(sync_mode="independent"),
        devices,
    )

    assert not plan.is_valid
    assert plan.resolved_mode == "independent"
    assert plan.synchronization_quality == "independent_host_estimated"
