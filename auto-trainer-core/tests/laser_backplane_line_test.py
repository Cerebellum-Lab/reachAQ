"""The backplane clock line may not be a line a laser trigger takes.

backplaneClockLine is the PXI_Trig line the shared sample clock is driven
onto. Its comment said it was distinct from the stimulus trigger's line, and
nothing checked it. A trigger on the same line puts two drivers on it, which
DAQmx cannot see across christielab10's boards, so the clock or the trigger
is corrupted rather than refused.
"""

import io

import pytest

from autotrainer.core import (
    LaserChannelConfiguration,
    LaserSystemConfiguration,
    SystemConfiguration,
)


def _channel(channel_id=1, **values):
    fields = dict(
        channel_id=channel_id,
        analog_output=f"PXI1Slot4/ao{channel_id - 1}",
        diode_input="PXI1Slot5/ai8",
        shutter_output=f"PXI1Slot5/port0/line{3 + channel_id}",
    )
    fields.update(values)
    return LaserChannelConfiguration(**fields)


def _lasers(*channels, **values):
    fields = dict(backend="nidaq", hardware_timed=True, sample_rate_hz=100_000.0)
    fields.update(values)
    return LaserSystemConfiguration(channels=channels, **fields)


def test_a_trigger_on_the_backplane_clock_line_is_refused_naming_both():
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(trigger_source="/PXI1Slot4/PXI_Trig1"))

    assert str(refused.value) == (
        "laser 1 triggerSource /PXI1Slot4/PXI_Trig1 uses PXI_Trig1, the "
        "backplaneClockLine; choose a different trigger line or "
        "backplaneClockLine")


def test_a_trigger_listener_on_the_backplane_clock_line_is_refused():
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(), trigger_listener_inputs=("/PXI1Slot4/PXI_Trig3",),
                backplane_clock_line="PXI_Trig3")

    message = str(refused.value)
    assert "triggerListenerInputs /PXI1Slot4/PXI_Trig3 uses PXI_Trig3" in message
    assert "backplaneClockLine" in message


@pytest.mark.parametrize(
    ("terminal", "clock_line"),
    [
        ("/PXI1Slot5/pxi_trig1", "PXI_Trig1"),
        ("PXI_TRIG1", "PXI_Trig1"),
        ("/PXI1Slot4/PXI_Trig1", "/PXI1Slot5/pxi_trig1"),
    ],
    ids=["lower_case_other_board", "bare_upper_case", "clock_named_on_a_board"],
)
def test_a_clash_ignores_case_and_the_device(terminal, clock_line):
    with pytest.raises(ValueError, match="backplaneClockLine"):
        _lasers(_channel(trigger_source=terminal), backplane_clock_line=clock_line)
    with pytest.raises(ValueError, match="backplaneClockLine"):
        _lasers(_channel(), trigger_listener_inputs=(terminal,),
                backplane_clock_line=clock_line)


@pytest.mark.parametrize(
    "terminal", ["/PXI1Slot4/PFI1", "/PXI1Slot5/PFI1", "/PXI1Slot4/PXI_Trig11",
                 "/PXI1Slot4/RTSI1"])
def test_a_trigger_on_another_kind_of_terminal_never_clashes(terminal):
    lasers = _lasers(_channel(trigger_source=terminal),
                     trigger_listener_inputs=(terminal,))

    assert lasers.backplane_clock_line == "PXI_Trig1"


def test_the_route_source_is_not_compared_because_nothing_drives_it():
    # _connect_trigger_route drives the trigger's own line on the route
    # source's board, and only reads the route source.
    lasers = _lasers(_channel(trigger_source="/PXI1Slot4/PXI_Trig0",
                              trigger_route_source="/PXI1Slot5/PXI_Trig1"))

    assert lasers.channels[0].trigger_route_source == "/PXI1Slot5/PXI_Trig1"


def _christielab10_lasers():
    """christielab10's laser trigger wiring, as of 2026-09-25."""
    return LaserSystemConfiguration(
        channels=(
            _channel(1, trigger_source="/PXI1Slot4/PXI_Trig0",
                     trigger_route_source="/PXI1Slot5/PFI0", board_stim_line=3),
            _channel(2, trigger_source="/PXI1Slot4/PXI_Trig2",
                     trigger_route_source="/PXI1Slot5/PFI1", board_stim_line=2),
        ),
        backend="nidaq",
        hardware_timed=True,
        sample_rate_hz=100_000.0,
        trigger_listener_inputs=("/PXI1Slot4/PXI_Trig0", "/PXI1Slot4/PXI_Trig2"),
        backplane_clock_line="PXI_Trig1",
    )


def test_christielab10s_laser_configuration_still_loads():
    configuration = SystemConfiguration(laser=_christielab10_lasers())

    loaded = SystemConfiguration.load_yaml(io.StringIO(configuration.dump_yaml()))

    assert loaded.laser == _christielab10_lasers()


def test_a_file_with_a_clash_is_refused_as_it_loads():
    text = SystemConfiguration(laser=_christielab10_lasers()).dump_yaml()
    assert "backplaneClockLine: PXI_Trig1" in text

    with pytest.raises(ValueError, match="PXI_Trig0, the backplaneClockLine"):
        SystemConfiguration.load_yaml(io.StringIO(
            text.replace("backplaneClockLine: PXI_Trig1", "backplaneClockLine: PXI_Trig0")))


# ------------------------------------------------------------ pulseClockLine


def test_the_pulse_clock_line_defaults_to_pxi_trig3_and_christielab10_keeps_it():
    # christielab10's triggers and trigger inputs are on PXI_Trig0 and
    # PXI_Trig2, and the shared clock on PXI_Trig1.
    assert _christielab10_lasers().pulse_clock_line == "PXI_Trig3"


def test_a_file_without_a_pulse_clock_line_loads_with_the_default():
    text = SystemConfiguration(laser=_christielab10_lasers()).dump_yaml()
    assert "pulseClockLine: PXI_Trig3" in text

    loaded = SystemConfiguration.load_yaml(io.StringIO(
        text.replace("  pulseClockLine: PXI_Trig3\n", "")))

    assert loaded.laser.pulse_clock_line == "PXI_Trig3"
    assert loaded.laser == _christielab10_lasers()


def test_the_pulse_clock_line_cannot_be_the_backplane_clock_line():
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(), backplane_clock_line="PXI_Trig1",
                pulse_clock_line="/PXI1Slot4/pxi_trig1")

    message = str(refused.value)
    assert "pulseClockLine /PXI1Slot4/pxi_trig1" in message
    assert "backplaneClockLine" in message


@pytest.mark.parametrize("terminal", ["/PXI1Slot4/PXI_Trig3", "/PXI1Slot5/pxi_trig3"])
def test_a_trigger_on_the_pulse_clock_line_is_refused_naming_both(terminal):
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(trigger_source=terminal))
    assert str(refused.value) == (
        f"laser 1 triggerSource {terminal} uses PXI_Trig3, the pulseClockLine; "
        "choose a different trigger line or pulseClockLine")

    with pytest.raises(ValueError, match="triggerListenerInputs .* the pulseClockLine"):
        _lasers(_channel(), trigger_listener_inputs=(terminal,))
