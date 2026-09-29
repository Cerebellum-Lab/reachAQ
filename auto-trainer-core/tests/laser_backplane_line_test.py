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
        ("/PXI1Slot4/PXI_Trig1", "pxi_trig1"),
    ],
    ids=["lower_case_other_board", "bare_upper_case", "clock_in_lower_case"],
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


@pytest.mark.parametrize(
    ("route_source", "field", "line"),
    [("/PXI1Slot5/PXI_Trig1", "backplaneClockLine", "PXI_Trig1"),
     ("/PXI1Slot5/pxi_trig3", "pulseClockLine", "PXI_Trig3")],
)
def test_a_route_source_on_a_clock_line_is_refused(route_source, field, line):
    # _connect_trigger_route reads the route source and drives the trigger's
    # own line from it: from a clock's line it would carry that clock into
    # the trigger, and the laser would arm on the clock's first edge.
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(trigger_source="/PXI1Slot4/PXI_Trig0",
                         trigger_route_source=route_source))

    message = str(refused.value)
    assert f"laser 1 triggerRouteSource {route_source} uses {line}, the {field}" in message
    assert "arm on" in message


@pytest.mark.parametrize("route_source", ["/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig5"])
def test_a_route_source_off_the_clock_lines_is_kept(route_source):
    lasers = _lasers(_channel(trigger_source="/PXI1Slot4/PXI_Trig0",
                              trigger_route_source=route_source))

    assert lasers.channels[0].trigger_route_source == route_source


@pytest.mark.parametrize("field", ["backplane_clock_line", "pulse_clock_line"])
@pytest.mark.parametrize(
    ("value", "why"),
    [("/PXI1Slot4/PXI_Trig1", "names a board"), ("PXI1Slot4/PXI_Trig5", "names a board"),
     ("/PXI1Slot4/PFI3", "names a board"),
     ("PFI3", "not a PXI_Trig line"), ("RTSI1", "not a PXI_Trig line"),
     ("PXI_Trig8", "PXI_Trig0 to PXI_Trig7"), ("PXI_Trig12", "PXI_Trig0 to PXI_Trig7"),
     ("", "empty"), (None, "empty"), ("   ", "empty")],
)
def test_a_clock_line_is_a_bare_pxi_trig_line(field, value, why):
    # The driver names the line on a board of its choosing; a board's name
    # given here was taken as the line's, or ignored.
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(), **{field: value})

    message = str(refused.value)
    camel = {"backplane_clock_line": "backplaneClockLine",
             "pulse_clock_line": "pulseClockLine"}[field]
    assert camel in message and why in message


def test_a_board_named_on_a_clock_line_is_refused_with_a_line_that_is_allowed():
    # The suggestion was the value's own tail, so "/PXI1Slot4/PFI3" was
    # told to give PFI3, which is refused too.
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(), pulse_clock_line="/PXI1Slot4/PFI3")

    message = str(refused.value)
    assert "PFI3," not in message and "such as PFI3" not in message
    assert "a bare PXI_Trig line, such as PXI_Trig3" in message


def test_a_board_named_on_a_pxi_trig_line_suggests_that_line():
    # The field's default was the example, and it can itself be taken:
    # here backplaneClockLine is PXI_Trig1 already.
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(), pulse_clock_line="/PXI1Slot4/PXI_Trig5")

    assert "a bare PXI_Trig line, such as PXI_Trig5" in str(refused.value)


def test_a_clock_line_is_named_as_the_driver_spells_it():
    lasers = _lasers(_channel(), backplane_clock_line=" pxi_trig1 ",
                     pulse_clock_line="PXI_TRIG4")

    assert (lasers.backplane_clock_line, lasers.pulse_clock_line) == (
        "PXI_Trig1", "PXI_Trig4")


def test_a_file_with_a_null_clock_line_is_refused():
    text = SystemConfiguration(laser=_christielab10_lasers()).dump_yaml()

    with pytest.raises(ValueError, match="backplaneClockLine"):
        SystemConfiguration.load_yaml(io.StringIO(
            text.replace("backplaneClockLine: PXI_Trig1", "backplaneClockLine: null")))


def test_a_file_without_a_backplane_clock_line_loads_with_the_default():
    text = SystemConfiguration(laser=_christielab10_lasers()).dump_yaml()
    stripped = text.replace("  backplaneClockLine: PXI_Trig1\n", "")
    assert "backplaneClockLine" not in stripped

    loaded = SystemConfiguration.load_yaml(io.StringIO(stripped))

    assert loaded.laser.backplane_clock_line == "PXI_Trig1"
    assert loaded.laser == _christielab10_lasers()


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

    stripped = text.replace("  pulseClockLine: PXI_Trig3\n", "")
    assert "pulseClockLine" not in stripped

    loaded = SystemConfiguration.load_yaml(io.StringIO(stripped))

    assert loaded.laser.pulse_clock_line == "PXI_Trig3"
    assert loaded.laser == _christielab10_lasers()


def test_the_pulse_clock_line_cannot_be_the_backplane_clock_line():
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(), backplane_clock_line="PXI_Trig1",
                pulse_clock_line="pxi_trig1")

    message = str(refused.value)
    assert "pulseClockLine PXI_Trig1 is the backplaneClockLine" in message


@pytest.mark.parametrize("terminal", ["/PXI1Slot4/PXI_Trig3", "/PXI1Slot5/pxi_trig3"])
def test_a_trigger_on_the_pulse_clock_line_is_refused_naming_both(terminal):
    with pytest.raises(ValueError) as refused:
        _lasers(_channel(trigger_source=terminal))
    assert str(refused.value) == (
        f"laser 1 triggerSource {terminal} uses PXI_Trig3, the pulseClockLine; "
        "choose a different trigger line or pulseClockLine")

    with pytest.raises(ValueError, match="triggerListenerInputs .* the pulseClockLine"):
        _lasers(_channel(), trigger_listener_inputs=(terminal,))


def test_a_configuration_from_channels_keeps_its_clock_lines():
    # from_channels built the configuration with the default lines whatever
    # it was given, and had nowhere to take them.
    lasers = LaserSystemConfiguration.from_channels(
        (_channel(),), backend="nidaq", hardware_timed=True,
        sample_rate_hz=100_000.0, backplane_clock_line="PXI_Trig5",
        pulse_clock_line="PXI_Trig4")

    assert (lasers.backplane_clock_line, lasers.pulse_clock_line) == (
        "PXI_Trig5", "PXI_Trig4")
    defaults = LaserSystemConfiguration.from_channels((_channel(),))
    assert (defaults.backplane_clock_line, defaults.pulse_clock_line) == (
        "PXI_Trig1", "PXI_Trig3")


@pytest.mark.parametrize("field", ["backplane_clock_line", "pulse_clock_line"])
def test_a_configuration_from_channels_refuses_a_null_clock_line_as_the_constructor_does(field):
    with pytest.raises(ValueError, match="empty"):
        LaserSystemConfiguration.from_channels((_channel(),), **{field: None})
