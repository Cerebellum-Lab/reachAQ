from autotrainer.core import (
    NidaqDeviceIdentity,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    NidaqTimingConfiguration,
    NidaqTimingRoute,
)
from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts
from tools.acquisition.model.nidaq_timing import build_nidaq_timing_plan
from tools.acquisition.model.nidaq_timing import (
    resolve_nidaq_device_aliases,
    resolve_nidaq_stream_configuration,
)


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
        analog_inputs=(f"{name}/ai0",),
        digital_inputs=(f"{name}/port0/line0",),
        counter_outputs=(f"{name}/ctr0",),
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
    assert plan.clock_producer == "ai"
    assert plan.consumer_devices == ("Acquire",)
    assert plan.hardware_output_devices == ("LaserOut",)
    assert plan.hardware_output_timing_status == "declared_not_armed"


def test_digital_only_pxi_master_uses_counter_without_fake_ai_trigger():
    configuration = _stream(
        ("cam_frames", "Acquire/port0/line0", "digital"),
        ("tone1", "Confirm/port0/line0", "digital"),
    )

    plan = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(),
        (_pxi("Acquire", 50), _pxi("Confirm", 51)),
    )

    assert plan.is_valid
    assert plan.clock_producer == "counter"
    assert plan.clock_producer_device == "Acquire"
    assert plan.sample_clock_source == "/Acquire/Ctr0InternalOutput"
    assert plan.start_trigger_source is None
    assert all("ai/StartTrigger" not in route.source for route in plan.routes)


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
    assert {task.task_id for task in external.task_graph.tasks} == {
        "DevA.ai", "DevB.ai",
    }


def test_first_release_rejects_three_active_nidaq_devices():
    configuration = _stream(
        ("one", "DevA/ai0", "analog"),
        ("two", "DevB/ai0", "analog"),
        ("three", "DevC/ai0", "analog"),
    )
    plan = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(sync_mode="independent", require_hardware_synchronization=False),
        (NidaqDevicePorts(name="DevA"), NidaqDevicePorts(name="DevB"), NidaqDevicePorts(name="DevC")),
    )
    assert not plan.is_valid
    assert "at most two" in plan.reason


def test_timing_graph_retains_explicit_routes_and_strategy():
    configuration = _stream(
        ("first", "DevA/ai0", "analog"),
        ("second", "DevB/ai0", "analog"),
    )
    route = NidaqTimingRoute("sample_clock", "/DevA/PFI0", ("/DevB/PFI0",))
    timing = NidaqTimingConfiguration(
        sync_mode="external",
        sample_clock_source="/DevA/PFI0",
        start_trigger_source="/DevA/PFI1",
        external_routes=(route,),
        task_strategy="auto_multidevice",
    )
    plan = build_nidaq_timing_plan(
        configuration,
        timing,
        (NidaqDevicePorts(name="DevA"), NidaqDevicePorts(name="DevB")),
    )
    assert plan.task_graph.strategy == "auto_multidevice"
    assert route in plan.task_graph.routes
    assert plan.multidevice_probe_status == "pending_exact_probe"


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


def test_channel_alias_resolves_by_stable_device_identity():
    configuration = _stream(
        ("cam_frames", "OldAlias/port0/line0", "digital"),
    )
    devices = (
        NidaqDevicePorts(
            name="RenamedDevice",
            product_type="InputModel",
            serial_number=123,
            digital_inputs=("RenamedDevice/port0/line0",),
        ),
    )
    aliases = resolve_nidaq_device_aliases(
        (
            NidaqDeviceIdentity(
                logical_name="acquisition",
                runtime_name="OldAlias",
                product_type="InputModel",
                serial_number=123,
            ),
        ),
        devices,
    )

    resolved = resolve_nidaq_stream_configuration(
        configuration,
        aliases,
    )

    assert aliases["OldAlias"] == "RenamedDevice"
    assert (
        resolved.channels[0].physical_channel
        == "RenamedDevice/port0/line0"
    )


def test_discovered_channel_and_rate_capabilities_are_validated():
    missing_channel = build_nidaq_timing_plan(
        _stream(("feedback", "Acquire/ai1", "analog")),
        NidaqTimingConfiguration(),
        (
            NidaqDevicePorts(
                name="Acquire",
                analog_inputs=("Acquire/ai0",),
            ),
        ),
    )
    excessive_rate = build_nidaq_timing_plan(
        NidaqSignalStreamConfiguration(
            channels=(
                NidaqSignalChannelConfiguration(
                    "feedback",
                    "Acquire/ai0",
                    "analog",
                ),
            ),
            is_enabled=True,
            sample_rate_hz=20_000,
        ),
        NidaqTimingConfiguration(),
        (
            NidaqDevicePorts(
                name="Acquire",
                analog_inputs=("Acquire/ai0",),
                analog_input_max_single_channel_rate=10_000,
            ),
        ),
    )

    assert not missing_channel.is_valid
    assert "is not available" in missing_channel.reason
    assert not excessive_rate.is_valid
    assert "exceeds" in excessive_rate.reason


def test_explicit_route_must_be_present_when_terminals_are_discovered():
    plan = build_nidaq_timing_plan(
        _stream(
            ("first", "DevA/ai0", "analog"),
            ("second", "DevB/ai0", "analog"),
        ),
        NidaqTimingConfiguration(
            sync_mode="external",
            start_trigger_source="/DevA/PFI7",
            sample_clock_source="/DevA/PFI1",
        ),
        (
            NidaqDevicePorts(
                name="DevA",
                bus_type="USB",
                analog_inputs=("DevA/ai0",),
                terminals=("/DevA/PFI0", "/DevA/PFI1"),
            ),
            NidaqDevicePorts(
                name="DevB",
                bus_type="USB",
                analog_inputs=("DevB/ai0",),
            ),
        ),
    )

    assert not plan.is_valid
    assert "/DevA/PFI7" in plan.reason
