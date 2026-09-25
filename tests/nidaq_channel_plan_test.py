import dataclasses

import pytest

from autotrainer.core import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NidaqPortConfiguration,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from tools.acquisition.model.nidaq_channel_plan import (
    build_nidaq_acquisition_configuration,
    with_display_channels,
)


def test_all_mapped_inputs_are_acquired_independently_from_display_selection():
    configured = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration(
                name="barcode",
                physical_channel="InputCard/port0/line1",
                kind="digital",
            ),
        ),
        is_enabled=True,
        display_channels=("barcode",),
    )
    ports = NidaqPortConfiguration(
        cam_frames="InputCard/port0/line0",
        barcode="InputCard/port0/line1",
        tone1="InputCard/port0/line2",
    )
    laser = LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="OutputCard/ao0",
                diode_input="InputCard/ai0",
                shutter_output="OutputCard/port0/line0",
                command_copy_input="InputCard/ai1",
                feedback_scale=2.0,
                command_copy_scale=3.0,
            ),
        ),
        backend="nidaq",
    )

    result = build_nidaq_acquisition_configuration(configured, ports, laser)

    assert tuple(channel.name for channel in result.channels) == (
        "cam_frames",
        "barcode",
        "tone1",
        "laser1_diode",
        "laser1_command_copy",
    )
    assert result.display_channels == ("barcode",)
    assert result.is_enabled
    assert result.channels[-2].scale == 2.0
    assert result.channels[-1].scale == 3.0


def test_display_selection_does_not_change_acquisition_channels():
    configuration = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration("first", "Dev1/ai0"),
            NidaqSignalChannelConfiguration("second", "Dev1/ai1"),
        ),
        is_enabled=True,
    )

    result = with_display_channels(configuration, ("second",))

    assert result.channels == configuration.channels
    assert result.display_channels == ("second",)


def test_existing_unmapped_channel_is_retained_as_custom_input():
    custom = NidaqSignalChannelConfiguration("force", "Dev2/ai3")
    configured = NidaqSignalStreamConfiguration(
        channels=(custom,),
        is_enabled=True,
    )

    result = build_nidaq_acquisition_configuration(
        configured,
        NidaqPortConfiguration(),
        LaserSystemConfiguration(),
    )

    assert result.channels == (custom,)


def _laser_with(**overrides):
    values = dict(
        channel_id=LaserChannelId.LASER_1,
        analog_output="OutputCard/ao0",
        diode_input="InputCard/ai0",
        shutter_output="OutputCard/port0/line0",
        command_copy_input="InputCard/ai1",
    )
    values.update(overrides)
    return LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(**values),), backend="nidaq")


def _empty_stream():
    return NidaqSignalStreamConfiguration()


def test_a_configured_trigger_readback_is_acquired_beside_the_laser_inputs():
    # Streamed and recorded like the diode and the command copy: it was
    # never in the plan, so the Board trigger graph could never show it.
    result = build_nidaq_acquisition_configuration(
        _empty_stream(),
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input="InputCard/ai9"),
    )

    assert tuple(channel.name for channel in result.channels) == (
        "laser1_diode",
        "laser1_command_copy",
        "laser1_trigger",
    )
    trigger = result.channels[-1]
    assert trigger.physical_channel == "InputCard/ai9"
    assert trigger.kind == "analog"
    assert trigger.unit == "V"
    assert trigger.scale == 1.0


def test_each_laser_gets_its_own_trigger_readback():
    laser = LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="OutputCard/ao0",
                diode_input="InputCard/ai8",
                shutter_output="InputCard/port0/line4",
                trigger_monitor_input="InputCard/ai9",
            ),
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_2,
                analog_output="OutputCard/ao1",
                diode_input="InputCard/ai4",
                shutter_output="InputCard/port0/line5",
                trigger_monitor_input="InputCard/ai10",
            ),
        ),
        backend="nidaq",
    )

    result = build_nidaq_acquisition_configuration(
        _empty_stream(), NidaqPortConfiguration(), laser)

    by_name = {channel.name: channel.physical_channel for channel in result.channels}
    assert by_name["laser1_trigger"] == "InputCard/ai9"
    assert by_name["laser2_trigger"] == "InputCard/ai10"


def test_a_trigger_readback_on_a_digital_line_is_acquired_as_digital():
    result = build_nidaq_acquisition_configuration(
        _empty_stream(),
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input="InputCard/port0/line6"),
    )

    trigger = result.channels[-1]
    assert trigger.name == "laser1_trigger"
    assert trigger.kind == "digital"
    assert trigger.unit == "logic"


def test_without_a_trigger_readback_the_plan_is_what_it_was():
    # Whole channels, so a unit, scale or offset that moved would show.
    stored_tone = NidaqSignalChannelConfiguration(
        "tone1", "InputCard/port0/line2", kind="digital", scale=3.0, offset=0.25)
    stored_diode = NidaqSignalChannelConfiguration(
        "laser1_diode", "InputCard/ai0", unit="mV", offset=0.5,
        minimum=-1.0, maximum=6.0)
    custom = NidaqSignalChannelConfiguration("force", "InputCard/ai12", unit="N")
    configured = NidaqSignalStreamConfiguration(
        channels=(stored_tone, stored_diode, custom), is_enabled=True)

    result = build_nidaq_acquisition_configuration(
        configured,
        NidaqPortConfiguration(tone1="InputCard/port0/line2"),
        _laser_with(feedback_scale=2.0, command_copy_scale=4.0),
    )

    assert result.channels == (
        NidaqSignalChannelConfiguration(
            "tone1", "InputCard/port0/line2", kind="digital", unit="logic",
            scale=3.0, offset=0.25),
        NidaqSignalChannelConfiguration(
            "laser1_diode", "InputCard/ai0", kind="analog", unit="mV",
            scale=2.0, offset=0.5, minimum=-1.0, maximum=6.0),
        NidaqSignalChannelConfiguration(
            "laser1_command_copy", "InputCard/ai1", kind="analog", unit="V",
            scale=4.0, offset=0.0),
        custom,
    )


def test_trigger_readbacks_are_scanned_after_every_other_input():
    # Appended last, so every channel already acquired keeps its place in
    # the multiplexed scan, and the fast STIM edge is not converted just
    # before a diode.
    custom = NidaqSignalChannelConfiguration("force", "InputCard/ai12")
    laser = LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="OutputCard/ao0",
                diode_input="InputCard/ai8",
                shutter_output="InputCard/port0/line4",
                command_copy_input="InputCard/ai3",
                trigger_monitor_input="InputCard/ai9",
            ),
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_2,
                analog_output="OutputCard/ao1",
                diode_input="InputCard/ai4",
                shutter_output="InputCard/port0/line5",
                command_copy_input="InputCard/ai5",
                trigger_monitor_input="InputCard/ai10",
            ),
        ),
        backend="nidaq",
    )

    result = build_nidaq_acquisition_configuration(
        NidaqSignalStreamConfiguration(channels=(custom,), is_enabled=True),
        NidaqPortConfiguration(tone1="InputCard/port0/line0"),
        laser,
    )

    assert tuple(channel.name for channel in result.channels) == (
        "tone1",
        "laser1_diode",
        "laser1_command_copy",
        "laser2_diode",
        "laser2_command_copy",
        "force",
        "laser1_trigger",
        "laser2_trigger",
    )


def _stream_of(*channels):
    return NidaqSignalStreamConfiguration(channels=channels, is_enabled=True)


def test_moving_the_trigger_readback_to_another_input_keeps_one_channel():
    # A load or save stores the plan, laser1_trigger on ai9 included. With
    # the readback moved to ai11, the stored ai9 entry came back as a custom
    # input under the role's own name, and the load failed: "maps to both".
    stored = build_nidaq_acquisition_configuration(
        _empty_stream(),
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input="InputCard/ai9"),
    )

    result = build_nidaq_acquisition_configuration(
        stored,
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input="InputCard/ai11"),
    )

    assert tuple(
        (channel.name, channel.physical_channel) for channel in result.channels
    ) == (
        ("laser1_diode", "InputCard/ai0"),
        ("laser1_command_copy", "InputCard/ai1"),
        ("laser1_trigger", "InputCard/ai11"),
    )


def test_moving_a_laser_input_to_another_input_keeps_one_channel():
    stored = _stream_of(
        NidaqSignalChannelConfiguration("laser1_diode", "InputCard/ai0"),
        NidaqSignalChannelConfiguration("laser1_command_copy", "InputCard/ai1"),
    )

    result = build_nidaq_acquisition_configuration(
        stored, NidaqPortConfiguration(), _laser_with(diode_input="InputCard/ai2"))

    assert tuple(
        (channel.name, channel.physical_channel) for channel in result.channels
    ) == (
        ("laser1_diode", "InputCard/ai2"),
        ("laser1_command_copy", "InputCard/ai1"),
    )


def test_moving_a_port_role_to_another_line_keeps_one_channel():
    # The same failure for every role: Edit DAQ Ports moving tone1 to a new
    # line was refused with "maps to both".
    stored = _stream_of(
        NidaqSignalChannelConfiguration("tone1", "InputCard/port0/line0", kind="digital"),
    )

    result = build_nidaq_acquisition_configuration(
        stored, NidaqPortConfiguration(tone1="InputCard/port0/line3"),
        LaserSystemConfiguration(),
    )

    assert tuple(
        (channel.name, channel.physical_channel) for channel in result.channels
    ) == (("tone1", "InputCard/port0/line3"),)


def test_a_stored_channel_named_like_an_unconfigured_role_stays_a_custom_input():
    stored = _stream_of(
        NidaqSignalChannelConfiguration("tone2", "InputCard/port0/line5", kind="digital"),
    )

    result = build_nidaq_acquisition_configuration(
        stored, NidaqPortConfiguration(), LaserSystemConfiguration())

    assert result.channels == stored.channels


@pytest.mark.parametrize(
    "terminal",
    ["/InputCard/PFI0", "InputCard/ao0", "InputCard/ctr0", "InputCard/ai0:3",
     "InputCard/port0"],
    ids=["pfi", "analog_output", "counter", "range", "whole_port"],
)
def test_a_trigger_readback_that_is_not_one_input_is_refused(terminal):
    # A PFI terminal read as an analog input: the stream task failed and
    # took every NI input down with it.
    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(),
            NidaqPortConfiguration(),
            _laser_with(trigger_monitor_input=terminal),
        )

    message = str(refused.value)
    assert "Laser 1" in message and terminal in message
    assert "analog input" in message and "port0 line" in message


@pytest.mark.parametrize(
    "terminal", ["PXI1Slot5/port1/line0", "/PXI1Slot5/port2/line7"], ids=["port1", "port2"])
def test_a_trigger_readback_on_a_pfi_pin_line_is_refused(terminal):
    # The stream clocks every digital input in one task, and an M Series
    # board clocks port0 only: port1 and port2 are the static PFI pins.
    # STIM3's /PXI1Slot5/PFI0 is PXI1Slot5/port1/line0 by another name.
    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(),
            NidaqPortConfiguration(),
            _laser_with(trigger_monitor_input=terminal),
        )

    message = str(refused.value)
    assert terminal in message
    assert "PFI pins" in message and "port0 line" in message


def test_a_laser_without_a_readback_drops_its_stored_trigger_channel(caplog):
    # Choosing "(none)" in Edit DAQ Ports saved no readback, but the stored
    # plan still held laser1_trigger on ai9, and a laser with no readback
    # claimed neither the pin nor the name: it came back as a hidden custom
    # input, recorded indefinitely.
    stored = build_nidaq_acquisition_configuration(
        _empty_stream(),
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input="InputCard/ai9"),
    )

    with caplog.at_level("WARNING"):
        result = build_nidaq_acquisition_configuration(
            stored, NidaqPortConfiguration(), _laser_with())

    assert "InputCard/ai9" not in {channel.physical_channel for channel in result.channels}
    assert tuple(channel.name for channel in result.channels) == (
        "laser1_diode", "laser1_command_copy")
    warning, = [record.getMessage() for record in caplog.records
                if "drops the stored channel" in record.getMessage()]
    assert "laser1_trigger" in warning and "InputCard/ai9" in warning
    assert "not set" in warning


def test_a_dropped_stored_channel_names_where_its_role_went(caplog):
    stored = build_nidaq_acquisition_configuration(
        _empty_stream(),
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input="InputCard/ai9"),
    )

    with caplog.at_level("WARNING"):
        build_nidaq_acquisition_configuration(
            stored,
            NidaqPortConfiguration(),
            _laser_with(trigger_monitor_input="InputCard/ai11"),
        )

    warning, = [record.getMessage() for record in caplog.records
                if "drops the stored channel" in record.getMessage()]
    assert "InputCard/ai9" in warning and "now on InputCard/ai11" in warning


def test_a_readback_does_not_inherit_another_roles_settings():
    # The command copy moved off ai9 and the readback moved onto it: the old
    # command copy's scale and offset are not the readback's.
    stored = _stream_of(
        NidaqSignalChannelConfiguration(
            "laser1_command_copy", "InputCard/ai9", scale=3.0, offset=0.5, unit="mV"),
    )

    result = build_nidaq_acquisition_configuration(
        stored,
        NidaqPortConfiguration(),
        _laser_with(command_copy_input="InputCard/ai3", trigger_monitor_input="InputCard/ai9"),
    )

    trigger = next(channel for channel in result.channels if channel.name == "laser1_trigger")
    assert (trigger.unit, trigger.scale, trigger.offset) == ("V", 1.0, 0.0)


def test_a_readback_keeps_its_own_settings_from_one_load_to_the_next():
    # Taken over from christielab10's custom channel on the first load, and
    # stored as laser1_trigger: the next load must not reset what it kept.
    custom = NidaqSignalChannelConfiguration(
        "laser1_trigger_readback", "InputCard/ai9", scale=2.0, offset=0.5)
    laser = _laser_with(trigger_monitor_input="InputCard/ai9")
    first = build_nidaq_acquisition_configuration(
        _stream_of(custom), NidaqPortConfiguration(), laser)

    second = build_nidaq_acquisition_configuration(first, NidaqPortConfiguration(), laser)

    for plan in (first, second):
        trigger = plan.channels[-1]
        assert (trigger.name, trigger.scale, trigger.offset) == ("laser1_trigger", 2.0, 0.5)


@pytest.mark.parametrize("terminal", ["/Dev1/ai7", "Dev1/ai15", "Dev1/port0/line3"])
def test_a_trigger_readback_on_one_input_is_accepted(terminal):
    result = build_nidaq_acquisition_configuration(
        _empty_stream(),
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input=terminal),
    )

    assert result.channels[-1].physical_channel == terminal


def test_a_trigger_readback_on_a_shutter_line_is_refused():
    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(),
            NidaqPortConfiguration(),
            _laser_with(
                shutter_output="InputCard/port0/line4",
                trigger_monitor_input="InputCard/port0/line4",
            ),
        )

    message = str(refused.value)
    assert "InputCard/port0/line4" in message
    assert "laser 1 shutter output" in message


def test_a_trigger_readback_on_the_pmt_shutter_line_is_refused():
    laser = dataclasses.replace(
        _laser_with(trigger_monitor_input="InputCard/port0/line7"),
        pmt_shutter_output="InputCard/port0/line7",
    )

    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(), NidaqPortConfiguration(), laser)

    assert "PMT shutter output" in str(refused.value)


def test_a_custom_input_on_the_trigger_terminal_becomes_the_laser_trigger():
    # christielab10 acquires ai9 as a custom channel, laser1_trigger_readback.
    # Named as this laser's readback, it is claimed like any laser input: one
    # channel under the role's name, keeping its unit and scaling.
    custom = NidaqSignalChannelConfiguration(
        "laser1_trigger_readback", "InputCard/ai9", unit="V", scale=2.0, offset=0.5)
    configured = NidaqSignalStreamConfiguration(
        channels=(custom,),
        is_enabled=True,
        display_channels=("laser1_trigger_readback",),
    )

    result = build_nidaq_acquisition_configuration(
        configured,
        NidaqPortConfiguration(),
        _laser_with(trigger_monitor_input="InputCard/ai9"),
    )

    names = tuple(channel.name for channel in result.channels)
    assert names == ("laser1_diode", "laser1_command_copy", "laser1_trigger")
    trigger = result.channels[-1]
    assert (trigger.scale, trigger.offset) == (2.0, 0.5)


@pytest.mark.parametrize(
    ("clashing_input", "owner"),
    [("InputCard/ai0", "laser1_diode"), ("InputCard/ai1", "laser1_command_copy")],
    ids=["diode", "command_copy"],
)
def test_a_trigger_readback_on_another_laser_input_is_refused(clashing_input, owner):
    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(),
            NidaqPortConfiguration(),
            _laser_with(trigger_monitor_input=clashing_input),
        )

    message = str(refused.value)
    assert clashing_input in message
    assert owner in message and "laser1_trigger" in message


def test_a_trigger_readback_on_a_port_role_line_is_refused():
    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(),
            NidaqPortConfiguration(tone1="InputCard/port0/line2"),
            _laser_with(trigger_monitor_input="InputCard/port0/line2"),
        )

    assert "tone1" in str(refused.value)
    assert "laser1_trigger" in str(refused.value)
