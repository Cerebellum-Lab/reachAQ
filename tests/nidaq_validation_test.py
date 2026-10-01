"""The configuration is checked against the boards before a task uses it.

Each case here is a failure that happened, and the cost of each was the same:
the problem surfaced as a DAQmx error code from inside a running task, naming
neither the configuration line responsible nor what to do about it.
"""

from types import SimpleNamespace

import pytest

from tools.acquisition.model.nidaq_validation import (
    NidaqConfigurationInvalid,
    chassis_identification_note,
    require_valid_nidaq_configuration,
    validate_nidaq_configuration,
)
from tools.acquisition.model.nidaq_routing import UNIDENTIFIED


def _device(name="PXI1Slot5", terminals=("PFI0", "PFI1", "PXI_Clk10"),
            analog_inputs=("PXI1Slot5/ai0", "PXI1Slot5/ai3"),
            analog_outputs=(), term_cfgs=("RSE", "NRSE", "DIFF"),
            analog_trigger=False, chassis=1, bus="BusType.PXI",
            per_channel_cfgs=None):
    return SimpleNamespace(
        name=name,
        terminals=tuple(f"/{name}/{t}" for t in terminals),
        analog_inputs=tuple(analog_inputs),
        analog_outputs=tuple(analog_outputs),
        digital_inputs=(), digital_outputs=(),
        counter_inputs=(), counter_outputs=(),
        # Per channel, because a board's answer is not uniform across its
        # range: ai0-ai7 do DIFF on a 6221 and ai8 upwards do not.
        analog_input_terminal_configs=tuple(
            per_channel_cfgs if per_channel_cfgs is not None
            else ((channel, tuple(term_cfgs)) for channel in analog_inputs)),
        analog_trigger_supported=analog_trigger,
        bus_type=bus,
        pxi_chassis_number=chassis,
        pxi_slot_number=5,
    )


def _stream(channels=(), terminal_config="rse"):
    return SimpleNamespace(channels=tuple(channels),
                           analog_terminal_config=terminal_config)


def _channel(name, physical_channel, kind="analog"):
    return SimpleNamespace(name=name, physical_channel=physical_channel,
                           kind=kind)


def _ports(**timing):
    defaults = dict(reference_clock_source=None, start_trigger_source=None,
                    sample_clock_source=None,
                    sample_clock_export_terminal=None,
                    start_trigger_export_terminal=None)
    defaults.update(timing)
    return SimpleNamespace(timing=SimpleNamespace(**defaults))


def _laser_channel(number=1, output="PXI1Slot5/ao0", trigger=None, route=None):
    return SimpleNamespace(
        channel_id=SimpleNamespace(value=number),
        analog_output=output, trigger_source=trigger,
        trigger_route_source=route)


def _laser(channels=(), hardware_timed=True):
    return SimpleNamespace(channels=tuple(channels),
                           hardware_timed=hardware_timed)


def test_a_terminal_the_board_does_not_expose_is_refused_with_the_alternatives():
    """PXI_CLK10 on a board with no Clk10 cost a whole signal stream."""
    devices = [_device("PXI1Slot4", terminals=("PFI0", "ao/StartTrigger"))]

    issues = validate_nidaq_configuration(
        devices, ports=_ports(
            reference_clock_source="/PXI1Slot4/PXI_Clk10"))

    assert len(issues) == 1
    assert "does not expose" in issues[0].problem
    assert "PFI0" in issues[0].possible


def test_a_terminal_on_a_board_that_is_not_installed_is_refused():
    devices = [_device("PXI1Slot5")]

    issues = validate_nidaq_configuration(
        devices, ports=_ports(sample_clock_source="/PXI1Slot9/ai/SampleClock"))

    assert "not installed" in issues[0].problem


def test_a_bare_terminal_name_is_left_to_the_stream_to_resolve():
    """PXI_CLK10 carries no device, so there is nothing to check it against."""
    devices = [_device("PXI1Slot5")]

    assert validate_nidaq_configuration(
        devices, ports=_ports(reference_clock_source="PXI_CLK10")) == ()


def test_a_channel_the_board_does_not_have_is_refused():
    devices = [_device("PXI1Slot5", analog_inputs=("PXI1Slot5/ai0",))]

    issues = validate_nidaq_configuration(
        devices, stream=_stream([_channel("diode", "PXI1Slot5/ai31")]))

    assert "does not have" in issues[0].problem


def test_referencing_the_channel_cannot_do_is_refused():
    devices = [_device("PXI1Slot5", term_cfgs=("RSE",))]

    issues = validate_nidaq_configuration(
        devices,
        stream=_stream([_channel("diode", "PXI1Slot5/ai0")],
                       terminal_config="diff"))

    assert "does not accept that referencing" in issues[0].problem
    assert issues[0].possible == ("rse",)


def test_referencing_is_judged_per_channel_not_per_board():
    """A 6221 offers DIFF on ai0-ai7 and not above, so the board-wide answer lies."""
    devices = [_device(
        "PXI1Slot5",
        analog_inputs=("PXI1Slot5/ai3", "PXI1Slot5/ai9"),
        per_channel_cfgs=(
            ("PXI1Slot5/ai3", ("RSE", "NRSE", "DIFF")),
            ("PXI1Slot5/ai9", ("RSE", "NRSE")),
        ))]

    issues = validate_nidaq_configuration(
        devices,
        stream=_stream([_channel("copy", "PXI1Slot5/ai3"),
                        _channel("readback", "PXI1Slot5/ai9")],
                       terminal_config="diff"))

    assert len(issues) == 1
    assert "ai9" in issues[0].problem
    assert issues[0].possible == ("rse", "nrse")


def test_an_analog_trigger_on_a_board_without_one_is_refused():
    """The APFI connector exists on the breakout whatever board is behind it."""
    devices = [_device("PXI1Slot5", terminals=("APFI0",), analog_trigger=False)]

    issues = validate_nidaq_configuration(
        devices,
        laser=_laser([_laser_channel(trigger="/PXI1Slot5/APFI0")]))

    assert any("no analog trigger circuit" in issue.problem for issue in issues)


def test_cross_board_timing_without_an_explicit_route_is_refused():
    """The -89125 case: possible here, but not by itself."""
    devices = [
        _device("PXI1Slot5", chassis=UNIDENTIFIED),
        _device("PXI1Slot4", chassis=UNIDENTIFIED, analog_inputs=(),
                analog_outputs=("PXI1Slot4/ao0",)),
    ]

    issues = validate_nidaq_configuration(
        devices,
        laser=_laser([_laser_channel(output="PXI1Slot4/ao0")]),
        timing_plan=SimpleNamespace(
            sample_clock_source="/PXI1Slot5/ai/SampleClock"))

    assert len(issues) == 1
    assert "no explicit route" in issues[0].problem
    assert "triggerRouteSource" in issues[0].remedy


def test_cross_board_timing_with_an_explicit_route_is_allowed():
    """This rig runs this way; refusing it would be wrong."""
    devices = [
        _device("PXI1Slot5", chassis=UNIDENTIFIED),
        _device("PXI1Slot4", chassis=UNIDENTIFIED, analog_inputs=(),
                analog_outputs=("PXI1Slot4/ao0",),
                terminals=("PFI0", "PXI_Trig0")),
    ]

    issues = validate_nidaq_configuration(
        devices,
        laser=_laser([_laser_channel(
            output="PXI1Slot4/ao0", trigger="/PXI1Slot4/PXI_Trig0",
            route="/PXI1Slot5/PFI0")]),
        timing_plan=SimpleNamespace(
            sample_clock_source="/PXI1Slot5/ai/SampleClock"))

    assert issues == ()


def test_a_software_timed_laser_needs_no_route():
    devices = [_device("PXI1Slot5"),
               _device("PXI1Slot4", analog_outputs=("PXI1Slot4/ao0",))]

    assert validate_nidaq_configuration(
        devices,
        laser=_laser([_laser_channel(output="PXI1Slot4/ao0")],
                     hardware_timed=False),
        timing_plan=SimpleNamespace(
            sample_clock_source="/PXI1Slot5/ai/SampleClock")) == ()


def test_refusing_names_every_problem_at_once():
    """One run should not mean one fix and another failure.

    Two problems here, not three: a channel the board does not have is
    reported once, and its referencing is not piled on top, because checking
    a mode against a channel that does not exist adds noise rather than
    information.
    """
    devices = [_device("PXI1Slot5", analog_inputs=("PXI1Slot5/ai0",),
                       term_cfgs=("RSE",))]

    with pytest.raises(NidaqConfigurationInvalid) as raised:
        require_valid_nidaq_configuration(
            devices,
            stream=_stream([_channel("a", "PXI1Slot5/ai31")],
                           terminal_config="diff"),
            ports=_ports(reference_clock_source="/PXI1Slot5/nonexistent"))

    subjects = {issue.subject for issue in raised.value.issues}
    assert subjects == {"stream channel 'a'", "timing referenceClockSource"}
    assert "does not match the installed hardware" in str(raised.value)


def test_nothing_is_asserted_when_no_devices_were_discovered():
    """An empty probe means unknown, and unknown must not read as invalid."""
    assert validate_nidaq_configuration(
        (), stream=_stream([_channel("a", "PXI1Slot5/ai0")])) == ()


def test_an_unidentified_chassis_is_noted_without_being_an_error():
    devices = [_device("PXI1Slot5", chassis=UNIDENTIFIED)]

    note = chassis_identification_note(devices)

    assert "PXI1Slot5" in note and "explicit route" in note
    assert chassis_identification_note([_device("PXI1Slot5", chassis=1)]) == ""


def _digital_device(name, *, digital_input_max_rate):
    device = _device(name, analog_inputs=())
    device.digital_inputs = (f"{name}/port0/line0", f"{name}/port0/line1")
    device.digital_input_max_rate = digital_input_max_rate
    return device


def test_a_digital_stream_line_on_a_board_that_cannot_clock_it_is_refused():
    # A PXI-6713 port0 line passed every check here and was caught only when
    # the stream started, as -200452, naming neither the line nor the board.
    # Discovery says it: the 6713 reports no digital-input rate, the 6221
    # 1 MHz (christielab10, 2026-09-25).
    devices = [
        _digital_device("PXI1Slot4", digital_input_max_rate=None),
        _digital_device("PXI1Slot5", digital_input_max_rate=1_000_000.0),
    ]
    stream = _stream([
        _channel("tone1", "PXI1Slot4/port0/line0", kind="digital"),
        _channel("tone2", "PXI1Slot5/port0/line1", kind="digital"),
    ])

    issues = validate_nidaq_configuration(devices, stream=stream)

    issue, = issues
    assert issue.subject == "stream channel 'tone1'"
    assert issue.value == "PXI1Slot4/port0/line0"
    assert "PXI1Slot4, which cannot clock digital input" in issue.problem
    with pytest.raises(NidaqConfigurationInvalid,
                       match="PXI1Slot4, which cannot clock digital input"):
        require_valid_nidaq_configuration(devices, stream=stream)


# ------------------------------------------- the clock a refused plan gives Run


def _refused_plan(**values):
    from autotrainer.core import NidaqTimingPlan

    fields = dict(requested_mode="auto", resolved_mode="unavailable",
                  is_valid=False, master_device="Clocks", reason="refused")
    fields.update(values)
    return NidaqTimingPlan(**fields)


def test_a_refused_plan_gives_its_masters_board():
    from autotrainer.core import NidaqTimingConfiguration
    from tools.acquisition.model.nidaq_validation import refused_plan_clock_source

    assert refused_plan_clock_source(
        _refused_plan(), NidaqTimingConfiguration()) == "/Clocks/ai/SampleClock"


def test_an_explicit_sample_clock_source_is_the_clock():
    from autotrainer.core import NidaqTimingConfiguration
    from tools.acquisition.model.nidaq_validation import refused_plan_clock_source

    timing = NidaqTimingConfiguration(sync_mode="external",
                                      sample_clock_source="/Other/PFI3",
                                      start_trigger_source="/Other/PFI4")

    assert refused_plan_clock_source(_refused_plan(), timing) == "/Other/PFI3"


def test_a_timing_master_override_is_the_refused_plans_master():
    # The plan's own choice, overridden by timingMaster, as a valid plan's.
    import dataclasses

    from autotrainer.core import (
        NidaqDeviceIdentity,
        NidaqSignalChannelConfiguration,
        NidaqSignalStreamConfiguration,
        NidaqTimingConfiguration,
    )
    from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts
    from tools.acquisition.model.nidaq_timing import build_nidaq_timing_plan
    from tools.acquisition.model.nidaq_validation import refused_plan_clock_source

    def board(name, serial, **values):
        return NidaqDevicePorts(
            name=name, product_type=f"Model-{serial}", serial_number=serial,
            bus_type="PXI", pxi_chassis_number=1, analog_inputs=(f"{name}/ai0",),
            digital_inputs=(f"{name}/port0/line0",), counter_outputs=(f"{name}/ctr0",),
            analog_output_sample_clock_supported=True, digital_trigger_supported=True,
            digital_input_max_rate=1_000_000.0, **values)

    stream = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration("cam_frames", "Lines/port0/line0", "digital"),
            NidaqSignalChannelConfiguration("tone1", "Lines/port0/line1", "digital"),
            NidaqSignalChannelConfiguration("laser_feedback", "Clocks/ai0", "analog"),
        ),
        is_enabled=True)
    timing = NidaqTimingConfiguration(timing_master=NidaqDeviceIdentity(
        logical_name="Clocks", runtime_name="Clocks", serial_number=41))
    lines = dataclasses.replace(board("Lines", 42), digital_input_max_rate=None)

    plan = build_nidaq_timing_plan(stream, timing, (board("Clocks", 41), lines))

    assert not plan.is_valid and plan.master_device == "Clocks"
    assert refused_plan_clock_source(plan, timing) == "/Clocks/ai/SampleClock"


def test_independent_boards_a_masterless_plan_a_valid_one_and_none_give_no_clock():
    from autotrainer.core import NidaqTimingConfiguration
    from tools.acquisition.model.nidaq_validation import refused_plan_clock_source

    timing = NidaqTimingConfiguration()
    assert refused_plan_clock_source(
        _refused_plan(requested_mode="independent"), timing) is None
    assert refused_plan_clock_source(_refused_plan(master_device=None), timing) is None
    assert refused_plan_clock_source(
        _refused_plan(is_valid=True, resolved_mode="same_device"), timing) is None
    assert refused_plan_clock_source(None, timing) is None


# ------------------------------------------- the stream's two export lines


def _christielab10_boards():
    lines = tuple(f"PXI_Trig{number}" for number in range(8))
    return [
        _device("PXI1Slot5", terminals=("PFI0", "PFI1", "PFI3", *lines),
                analog_inputs=tuple(f"PXI1Slot5/ai{pin}" for pin in range(16)),
                chassis=UNIDENTIFIED),
        _device("PXI1Slot4", terminals=("PFI0", *lines), analog_inputs=(),
                analog_outputs=("PXI1Slot4/ao0", "PXI1Slot4/ao1"),
                chassis=UNIDENTIFIED),
    ]


def _christielab10_lasers(**changes):
    """christielab10's lasers: clock lines PXI_Trig1 and PXI_Trig3, triggers
    and trigger inputs on PXI_Trig0 and PXI_Trig2, routed from PFI0/PFI1."""
    import dataclasses

    from nidaq_channel_plan_test import _christielab10_lasers as lasers

    return dataclasses.replace(lasers(), **changes)


def _export_issues(sample=None, start=None, laser=None):
    return validate_nidaq_configuration(
        _christielab10_boards(),
        ports=_ports(sample_clock_export_terminal=sample,
                     start_trigger_export_terminal=start),
        laser=_christielab10_lasers() if laser is None else laser)


def test_christielab10_with_no_export_lines_or_two_free_ones_is_accepted():
    assert _export_issues() == ()
    assert _export_issues("/PXI1Slot5/PXI_Trig4", "/PXI1Slot5/PXI_Trig5") == ()
    # A PFI is not a backplane line, and never clashes with one.
    assert _export_issues("/PXI1Slot5/PFI3") == ()


def _route_on_trig7():
    import dataclasses

    lasers = _christielab10_lasers()
    first, second = lasers.channels
    return dataclasses.replace(lasers, channels=(
        dataclasses.replace(first, trigger_route_source="/PXI1Slot5/PXI_Trig7"),
        second))


@pytest.mark.parametrize(("sample", "start", "laser", "clashes"), [
    ("/PXI1Slot5/PXI_Trig4", "PXI_Trig4", None, [
        ("timing startTriggerExportTerminal", "timing sampleClockExportTerminal")]),
    ("/PXI1Slot5/PXI_Trig1", None, None, [
        ("timing sampleClockExportTerminal", "the laser backplaneClockLine")]),
    ("pxi_trig3", None, None, [
        ("timing sampleClockExportTerminal", "the laser pulseClockLine")]),
    (None, "/PXI1Slot5/PXI_Trig0", _christielab10_lasers(trigger_listener_inputs=()), [
        ("timing startTriggerExportTerminal",
         "laser 1 triggerSource /PXI1Slot4/PXI_Trig0")]),
    # One refusal each: laser 2's trigger and the trigger input on its line.
    (None, "/PXI1Slot5/PXI_Trig2", None, [
        ("timing startTriggerExportTerminal",
         "laser 2 triggerSource /PXI1Slot4/PXI_Trig2"),
        ("timing startTriggerExportTerminal",
         "triggerListenerInputs /PXI1Slot4/PXI_Trig2")]),
    ("/PXI1Slot5/PXI_Trig7", None, _route_on_trig7(), [
        ("timing sampleClockExportTerminal",
         "laser 1 triggerRouteSource /PXI1Slot5/PXI_Trig7")]),
    (None, "/PXI1Slot5/PXI_Trig6",
     _christielab10_lasers(trigger_listener_inputs=("/PXI1Slot4/PXI_Trig6",)), [
        ("timing startTriggerExportTerminal",
         "triggerListenerInputs /PXI1Slot4/PXI_Trig6")]),
])
def test_an_export_line_another_signal_takes_is_refused_once_per_clash(
        sample, start, laser, clashes):
    # The stream's master drives both lines for as long as it runs. A second
    # driver on one corrupts both signals, and DAQmx does not see it across
    # these boards; a trigger input on one reads the stream's signal instead
    # of a trigger. Refused before any task, as clock_line_clashes refuses
    # the laser's own lines.
    issues = _export_issues(sample, start, laser)

    assert [(issue.subject, other) for issue in issues
            for _subject, other in clashes if other in issue.problem] == clashes
    assert len(issues) == len(clashes)
    for issue in issues:
        assert issue.value in (sample, start)
        # A line no other field takes here, as the remedy's example.
        assert "such as PXI_Trig" in issue.remedy
        for taken in ("PXI_Trig0", "PXI_Trig1", "PXI_Trig2", "PXI_Trig3"):
            assert f"such as {taken}" not in issue.remedy
