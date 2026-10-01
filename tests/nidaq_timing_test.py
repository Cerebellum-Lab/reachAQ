import dataclasses
import hashlib
import json

import pytest

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
    resolve_nidaq_multidevice_probe,
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
        # A board that clocks the digital line it lists, as a 6221 does at
        # 1 MHz. Discovery's None means it cannot, and the plan refuses it.
        digital_input_max_rate=1_000_000.0,
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
        # Inputs on two boards take the clock over a backplane line.
        NidaqTimingConfiguration(sample_clock_export_terminal="/Acquire/PXI_Trig4"),
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
        # Bare, as the master is not known by name until it resolves.
        sample_clock_export_terminal="PXI_Trig4",
        start_trigger_export_terminal="PXI_Trig5",
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

    selected = resolve_nidaq_multidevice_probe(plan, "verified")
    analog = [task for task in selected.task_graph.tasks if task.subsystem == "ai"]
    assert len(analog) == 1
    assert analog[0].device == "DevA"
    assert analog[0].channels == ("DevA/ai0", "DevB/ai0")
    assert selected.multidevice_probe_status == "verified"

    fallback = resolve_nidaq_multidevice_probe(plan, "fallback_per_device")
    assert len([task for task in fallback.task_graph.tasks if task.subsystem == "ai"]) == 2


def test_timing_graph_retains_exact_laser_output_channels():
    plan = build_nidaq_timing_plan(
        _stream(("cam", "Dev1/ai0", "analog")),
        NidaqTimingConfiguration(),
        (NidaqDevicePorts(name="Dev1"),),
        hardware_timed_output_channels=("Dev1/ao0",),
    )
    output = next(task for task in plan.task_graph.tasks if task.subsystem == "ao")
    assert output.channels == ("Dev1/ao0",)


def test_independent_mode_is_valid_and_says_its_alignment_is_estimated():
    """Asking for independent is accepting host-estimated alignment.

    This asserted an invalid plan, which it got because
    require_hardware_synchronization defaulted to true and nobody writing
    "independent" also wrote "and I do not need synchronization". The
    boolean now derives from the mode, so the plan is valid and says plainly
    what its alignment is worth; the contradiction is refused where it would
    be written instead of surfacing as an invalid plan three layers away.
    """
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

    assert plan.is_valid
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


def test_a_digital_line_on_a_board_that_cannot_clock_it_is_refused_by_name():
    # Run and the DAQ Monitor refused it. Every stream start builds this plan
    # from fresh discovery, Idle's included, and it let the line through to
    # the preflight, which failed at -200452 naming neither line nor board.
    # One check for all three, so one wording.
    output_board = dataclasses.replace(
        _pxi("Output", 40), digital_input_max_rate=None)

    plan = build_nidaq_timing_plan(
        _stream(("tone1", "Output/port0/line0", "digital")),
        NidaqTimingConfiguration(),
        (output_board,),
    )

    assert not plan.is_valid
    assert "stream channel 'tone1' is 'Output/port0/line0'" in plan.reason
    assert "Output, which cannot clock digital input" in plan.reason


def test_a_plan_refused_for_a_line_keeps_the_master_it_would_have_had():
    # What reads the clock's board off a refused plan, Run's laser route
    # check, gets the plan's own choice: cam_frames' board over the first
    # analog input's, as a valid plan would have it.
    clocks = _pxi("Clocks", 41)
    lines = dataclasses.replace(_pxi("Lines", 42), digital_input_max_rate=None)

    plan = build_nidaq_timing_plan(
        _stream(("laser_feedback", "Clocks/ai0", "analog"),
                ("cam_frames", "Lines/port0/line0", "digital")),
        NidaqTimingConfiguration(),
        (clocks, lines),
    )

    assert not plan.is_valid
    assert "Lines, which cannot clock digital input" in plan.reason
    assert plan.master_device == "Lines"
    # Nothing starts on it: a refused plan runs no task.
    assert plan.sample_clock_source is None


def test_christielab10s_plan_is_unchanged_by_the_digital_clock_rule():
    # Every stream line is on the 6221, which clocks digital input at 1 MHz.
    # The 6713, which cannot, carries only the lasers' outputs, and is not
    # asked about digital input at all.
    from nidaq_channel_plan_test import _christielab10_stream

    inputs = dataclasses.replace(
        _pxi("PXI1Slot5", 21803707),
        analog_inputs=tuple(f"PXI1Slot5/ai{pin}" for pin in range(16)),
        digital_inputs=tuple(f"PXI1Slot5/port0/line{line}" for line in range(8)),
    )
    outputs = dataclasses.replace(
        _pxi("PXI1Slot4", 27056752),
        analog_inputs=(),
        digital_inputs=tuple(f"PXI1Slot4/port0/line{line}" for line in range(8)),
        digital_input_max_rate=None,
    )

    plan = build_nidaq_timing_plan(
        _christielab10_stream(),
        NidaqTimingConfiguration(),
        (outputs, inputs),
        hardware_timed_output_devices=("PXI1Slot4",),
        hardware_timed_output_channels=("PXI1Slot4/ao0", "PXI1Slot4/ao1"),
    )

    assert plan.is_valid, plan.reason
    assert plan.resolved_mode == "backplane"
    assert plan.master_device == "PXI1Slot5"
    assert plan.sample_clock_source == "/PXI1Slot5/ai/SampleClock"


# ------------------------------------- inputs on two boards, over the backplane


def _two_input_boards(**timing):
    """Inputs on Acquire, the master (it has cam_frames), and on Feedback."""
    return build_nidaq_timing_plan(
        _stream(
            ("cam_frames", "Acquire/port0/line0", "digital"),
            ("laser_feedback", "Acquire/ai0", "analog"),
            ("slave_feedback", "Feedback/ai0", "analog"),
            ("slave_tone", "Feedback/port0/line0", "digital"),
        ),
        NidaqTimingConfiguration(**timing),
        (_pxi("Acquire", 50), _pxi("Feedback", 51)),
    )


def _task(plan, task_id):
    return next(task for task in plan.task_graph.tasks if task.task_id == task_id)


@pytest.mark.parametrize(("sample_export", "start_export"), [
    ("/Acquire/PXI_Trig4", "/Acquire/PXI_Trig5"),
    # Bare, and spelt otherwise: the line is the same, named as DAQmx does.
    ("pxi_trig4", "PXI_Trig5"),
])
def test_a_slave_input_board_names_the_masters_clock_and_trigger_on_its_own_lines(
        sample_export, start_export):
    # Named on the master, they are a cross-board route, which DAQmx refuses
    # on christielab10's unidentified chassis (-89125), as the laser found.
    # The master exports them onto the two lines; the slave names each line
    # as its own.
    plan = _two_input_boards(
        sample_clock_export_terminal=sample_export,
        start_trigger_export_terminal=start_export)

    assert plan.is_valid, plan.reason
    assert plan.resolved_mode == "backplane"
    assert (plan.master_device, plan.slave_devices) == ("Acquire", ("Feedback",))
    for task_id in ("Feedback.ai", "Feedback.di"):
        task = _task(plan, task_id)
        assert (task.sample_clock_source, task.start_trigger_source) == (
            "/Feedback/PXI_Trig4", "/Feedback/PXI_Trig5")
    # The master is unchanged: its own clock, armed by nothing.
    master_ai, master_di = _task(plan, "Acquire.ai"), _task(plan, "Acquire.di")
    assert (master_ai.sample_clock_source, master_ai.start_trigger_source) == (None, None)
    assert (master_di.sample_clock_source, master_di.start_trigger_source) == (
        "/Acquire/ai/SampleClock", None)
    # What the laser reads stays the master's own terminals, which it routes
    # itself (NidaqLaserController._shared_clock_for).
    assert plan.sample_clock_source == "/Acquire/ai/SampleClock"
    assert plan.start_trigger_source == "/Acquire/ai/StartTrigger"
    assert (plan.sample_clock_export_terminal, plan.start_trigger_export_terminal) == (
        sample_export, start_export)


def test_a_counter_clocked_master_needs_no_start_trigger_line():
    # A master with no analog input clocks the inputs from its counter and
    # has no start trigger to export, so only the clock takes a line.
    plan = build_nidaq_timing_plan(
        _stream(
            ("cam_frames", "Acquire/port0/line0", "digital"),
            ("tone1", "Confirm/port0/line0", "digital"),
        ),
        NidaqTimingConfiguration(sample_clock_export_terminal="/Acquire/PXI_Trig4"),
        (_pxi("Acquire", 50), _pxi("Confirm", 51)),
    )

    assert plan.is_valid, plan.reason
    assert plan.clock_producer == "counter"
    slave = _task(plan, "Confirm.di")
    assert (slave.sample_clock_source, slave.start_trigger_source) == (
        "/Confirm/PXI_Trig4", None)
    assert plan.sample_clock_source == "/Acquire/Ctr0InternalOutput"


@pytest.mark.parametrize(("timing", "field", "words"), [
    (dict(), "timing sampleClockExportTerminal", "is unset"),
    (dict(sample_clock_export_terminal="/Acquire/PFI3"),
     "timing sampleClockExportTerminal", "'/Acquire/PFI3' is not a PXI_Trig line"),
    # PXI has PXI_Trig0 to PXI_Trig7.
    (dict(sample_clock_export_terminal="/Acquire/PXI_Trig9"),
     "timing sampleClockExportTerminal", "'/Acquire/PXI_Trig9' is not a PXI_Trig line"),
    # The master drives the line, so it is named on the master.
    (dict(sample_clock_export_terminal="/Feedback/PXI_Trig4"),
     "timing sampleClockExportTerminal", "'/Feedback/PXI_Trig4' names Feedback"),
    (dict(sample_clock_export_terminal="/Acquire/PXI_Trig4"),
     "timing startTriggerExportTerminal", "is unset"),
])
def test_a_slave_input_board_without_a_line_is_an_invalid_plan_naming_the_field(
        timing, field, words):
    # No default line: one could collide with backplane wiring the plan
    # cannot see (Ben, 2026-09-30).
    plan = _two_input_boards(**timing)

    assert not plan.is_valid
    assert field in plan.reason and words in plan.reason
    # An example of a line on the master, and not the one the other field
    # already takes.
    example = "/Acquire/PXI_Trig5" if "startTrigger" in field else "/Acquire/PXI_Trig4"
    assert f"such as {example}" in plan.reason
    # A refused plan keeps the master it would have had, for Run's laser
    # route check (refused_plan_clock_source).
    assert (plan.master_device, plan.slave_devices) == ("Acquire", ("Feedback",))


def test_a_forced_multidevice_slave_needs_a_line_only_for_a_task_it_keeps():
    # Forced, every subsystem on both boards is merged into one task on the
    # master, or the start fails: no slave task is left to take a line. A
    # subsystem on the slave alone stays a slave task.
    devices = (_pxi("Acquire", 50), _pxi("Feedback", 51))
    forced = NidaqTimingConfiguration(task_strategy="forced_multidevice")
    merged = build_nidaq_timing_plan(
        _stream(("first", "Acquire/ai0", "analog"), ("second", "Feedback/ai0", "analog")),
        forced, devices)
    kept = build_nidaq_timing_plan(
        _stream(("first", "Acquire/ai0", "analog"),
                ("tone1", "Feedback/port0/line0", "digital")),
        forced, devices)

    assert merged.is_valid, merged.reason
    assert not kept.is_valid
    assert "timing sampleClockExportTerminal" in kept.reason


# ------------------------------- every plan without a slave input, as it was


def _pinned_christielab10():
    """christielab10's own blocks, on its own two boards, as a start builds it."""
    from nidaq_channel_plan_test import _christielab10_from_yaml, _christielab10_lasers
    from tools.acquisition.model.nidaq_channel_plan import (
        build_nidaq_acquisition_configuration,
    )
    from tools.acquisition.model.nidaq_routing import UNIDENTIFIED

    loaded = _christielab10_from_yaml()
    lasers = _christielab10_lasers()
    stream = build_nidaq_acquisition_configuration(
        loaded.nidaq_stream, loaded.nidaq_ports, lasers)

    def board(name, product, serial, **values):
        # Its chassis is unidentified, both boards alike.
        return NidaqDevicePorts(
            name=name, product_type=product, serial_number=serial,
            bus_type="PXI", pxi_chassis_number=UNIDENTIFIED,
            pxi_slot_number=UNIDENTIFIED, digital_trigger_supported=True,
            digital_inputs=tuple(f"{name}/port0/line{line}" for line in range(8)),
            **values)

    devices = (
        board("PXI1Slot4", "PXI-6713", 27056752,
              analog_outputs=tuple(f"PXI1Slot4/ao{pin}" for pin in range(8)),
              counter_outputs=("PXI1Slot4/ctr0", "PXI1Slot4/ctr1"),
              analog_output_sample_clock_supported=True),
        board("PXI1Slot5", "PXI-6221", 21803707,
              analog_inputs=tuple(f"PXI1Slot5/ai{pin}" for pin in range(16)),
              analog_outputs=("PXI1Slot5/ao0", "PXI1Slot5/ao1"),
              counter_outputs=("PXI1Slot5/ctr0", "PXI1Slot5/ctr1"),
              analog_output_sample_clock_supported=True,
              digital_input_max_rate=1_000_000.0),
    )
    return build_nidaq_timing_plan(
        stream, loaded.nidaq_ports.timing, devices,
        hardware_timed_output_devices=("PXI1Slot4",),
        hardware_timed_output_channels=tuple(
            channel.analog_output for channel in lasers.channels),
    )


def _pinned_merged(status):
    plan = build_nidaq_timing_plan(
        _stream(("first", "Acquire/ai0", "analog"), ("second", "Feedback/ai0", "analog")),
        NidaqTimingConfiguration(task_strategy="forced_multidevice"),
        (_pxi("Acquire", 50), _pxi("Feedback", 51)))
    return plan if status is None else resolve_nidaq_multidevice_probe(plan, status)


_PINNED_PLANS = {
    "christielab10": _pinned_christielab10,
    "one board": lambda: build_nidaq_timing_plan(
        _stream(("cam_frames", "InputCard/port0/line0", "digital"),
                ("laser_feedback", "InputCard/ai0", "analog")),
        NidaqTimingConfiguration(),
        (NidaqDevicePorts(name="InputCard"),)),
    "inputs on one board, laser output on another": lambda: build_nidaq_timing_plan(
        _stream(("cam_frames", "Acquire/port0/line0", "digital"),
                ("laser_feedback", "Acquire/ai0", "analog")),
        NidaqTimingConfiguration(),
        (_pxi("LaserOut", 40), _pxi("Acquire", 50)),
        hardware_timed_output_devices=("LaserOut",),
        hardware_timed_output_channels=("LaserOut/ao0",)),
    "external, inputs on two boards": lambda: build_nidaq_timing_plan(
        _stream(("first", "DevA/ai0", "analog"), ("second", "DevB/ai0", "analog")),
        NidaqTimingConfiguration(
            sync_mode="external", reference_clock_source="/DevA/PFI0",
            start_trigger_source="/DevA/PFI1", sample_clock_source="/DevA/PFI2"),
        (NidaqDevicePorts(name="DevA", bus_type="USB"),
         NidaqDevicePorts(name="DevB", bus_type="USB"))),
    "independent, inputs on two boards": lambda: build_nidaq_timing_plan(
        _stream(("first", "DevA/ai0", "analog"), ("second", "DevB/ai0", "analog")),
        NidaqTimingConfiguration(sync_mode="independent"),
        (NidaqDevicePorts(name="DevA"), NidaqDevicePorts(name="DevB"))),
    "forced multidevice, before its probe": lambda: _pinned_merged(None),
    "forced multidevice, merged": lambda: _pinned_merged("verified"),
}

#: Each plan's graph_id, and the first 16 hex digits of the SHA-256 of the
#: plan as a session file stores it (session_data_recorder: timing_plan_json),
#: as the code before inputs on two boards took backplane lines built them.
_PINS = {
    "christielab10": ("5a5fd43ee352a4d9", "d40375201d4d7fc7"),
    "one board": ("1bbb5feb84f951aa", "9f2dcf80a722e3fa"),
    "inputs on one board, laser output on another": (
        "c769e0f6d590de48", "1dad49d00685bb3d"),
    "external, inputs on two boards": ("f0a0e370d4f445a9", "15f79f6f57b17f23"),
    "independent, inputs on two boards": ("9dc7f79c01451e85", "1174780911cfa6d9"),
    # The probe's choice keeps the graph_id; the merged task is in the digest.
    "forced multidevice, before its probe": ("9e071173adfdf1f6", "172ecab7f68fbadd"),
    "forced multidevice, merged": ("9e071173adfdf1f6", "0e5429d4e9592be6"),
}


@pytest.mark.parametrize("name", sorted(_PINNED_PLANS))
def test_a_plan_with_no_slave_input_task_is_byte_identical(name):
    plan = _PINNED_PLANS[name]()

    assert plan.is_valid, plan.reason
    stored = json.dumps(dataclasses.asdict(plan), sort_keys=True)
    digest = hashlib.sha256(stored.encode("utf-8")).hexdigest()[:16]
    built = (plan.task_graph.graph_id, digest)
    assert built == _PINS[name], f"{name}: {built}"


def test_christielab10s_pinned_plan_is_its_own():
    # What the pin above holds: inputs on the 6221, the 6713 an output-only
    # slave whose reservation names the 6221's own clock and trigger.
    plan = _pinned_christielab10()

    assert (plan.master_device, plan.slave_devices) == ("PXI1Slot5", ("PXI1Slot4",))
    assert plan.resolved_mode == "backplane"
    assert plan.sample_clock_export_terminal is None
    assert plan.start_trigger_export_terminal is None
    assert {task.task_id for task in plan.task_graph.tasks} == {
        "PXI1Slot5.ai", "PXI1Slot5.di", "PXI1Slot4.ao-reservation"}
    reservation = _task(plan, "PXI1Slot4.ao-reservation")
    assert reservation.channels == ("PXI1Slot4/ao0", "PXI1Slot4/ao1")
    assert (reservation.sample_clock_source, reservation.start_trigger_source) == (
        "/PXI1Slot5/ai/SampleClock", "/PXI1Slot5/ai/StartTrigger")
