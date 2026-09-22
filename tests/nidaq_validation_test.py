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
