import dataclasses
from pathlib import Path

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
    nidaq_channel_kind,
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


def test_a_configured_lasers_stored_channel_stays_a_custom_input_with_the_backend_disabled():
    # This asserted the same for a stored tone2 with its role unset; that is
    # a cleared port role now, and dropped (see below). A disabled laser
    # backend plans no laser role, so a configured laser's stored channel
    # stays a custom input, which its tab still finds by pin. One of a laser
    # not configured is dropped (below).
    stored = _stream_of(
        NidaqSignalChannelConfiguration("laser1_diode", "InputCard/ai0"),
    )
    lasers = dataclasses.replace(_laser_with(), backend="disabled")

    result = build_nidaq_acquisition_configuration(
        stored, NidaqPortConfiguration(), lasers)

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


# ------------------------------------------------------------ port-role lines


@pytest.mark.parametrize(
    ("attribute", "field"),
    [("tone1", "tone1"), ("tone2", "tone2"), ("tone3_r", "tone3R"),
     ("tone3_l", "tone3L"), ("cam_frames", "camFrames"), ("barcode", "barcode")],
)
def test_a_port_role_on_a_pfi_pin_line_is_refused(attribute, field):
    # The stream samples every digital role in one clocked task, and an M
    # Series board clocks port0 only; a port1 or port2 line took every NI
    # input down (-200452).
    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(),
            NidaqPortConfiguration(**{attribute: "PXI1Slot5/port1/line0"}),
            LaserSystemConfiguration(),
        )

    message = str(refused.value)
    assert field in message and "PXI1Slot5/port1/line0" in message
    assert "PFI pins" in message and "port0 line" in message


def test_a_port_role_on_an_analog_input_is_refused():
    with pytest.raises(ValueError) as refused:
        build_nidaq_acquisition_configuration(
            _empty_stream(),
            NidaqPortConfiguration(cam_frames="PXI1Slot5/ai0"),
            LaserSystemConfiguration(),
        )

    assert "camFrames" in str(refused.value) and "port0 line" in str(refused.value)


def _christielab10_stream():
    """christielab10's saved nidaqStream channels, as of 2026-09-24."""
    def digital(name, line):
        return NidaqSignalChannelConfiguration(
            name, f"PXI1Slot5/port0/line{line}", kind="digital", unit="logic")

    def analog(name, pin):
        return NidaqSignalChannelConfiguration(name, f"PXI1Slot5/{pin}", kind="analog", unit="V")

    return NidaqSignalStreamConfiguration(
        channels=(
            digital("cam_frames", 2),
            digital("barcode", 3),
            digital("tone1", 0),
            digital("tone2", 1),
            analog("laser1_diode", "ai8"),
            analog("laser1_command_copy", "ai3"),
            analog("laser2_diode", "ai4"),
            analog("laser2_command_copy", "ai5"),
            analog("laser1_trigger_readback", "ai9"),
            analog("laser2_trigger_readback", "ai10"),
        ),
        is_enabled=True,
        sample_rate_hz=10000.0,
        read_chunk_size=500,
        display_channels=("laser1_command_copy", "cam_frames", "barcode", "tone2", "tone1"),
    )


def _christielab10_lasers():
    return LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="PXI1Slot4/ao0",
                diode_input="PXI1Slot5/ai8",
                shutter_output="PXI1Slot5/port0/line4",
                command_copy_input="PXI1Slot5/ai3",
                trigger_source="/PXI1Slot4/PXI_Trig0",
                trigger_route_source="/PXI1Slot5/PFI0",
                board_stim_line=3,
            ),
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_2,
                analog_output="PXI1Slot4/ao1",
                diode_input="PXI1Slot5/ai4",
                shutter_output="PXI1Slot5/port0/line5",
                command_copy_input="PXI1Slot5/ai5",
                trigger_source="/PXI1Slot4/PXI_Trig2",
                trigger_route_source="/PXI1Slot5/PFI1",
                board_stim_line=2,
            ),
        ),
        backend="nidaq",
        hardware_timed=True,
        sample_rate_hz=100000.0,
        trigger_listener_inputs=("/PXI1Slot4/PXI_Trig0", "/PXI1Slot4/PXI_Trig2"),
    )


CHRISTIELAB10_PORTS = NidaqPortConfiguration(
    device_name="PXI1Slot5",
    tone1="PXI1Slot5/port0/line0",
    tone2="PXI1Slot5/port0/line1",
    cam_frames="PXI1Slot5/port0/line2",
    barcode="PXI1Slot5/port0/line3",
    tone3_r=None,
    tone3_l=None,
)


def test_christielab10s_port_block_loads_unchanged():
    # The rules for digital lines and cleared roles must leave the rig's own
    # configuration exactly as it is: every channel, in order, with its
    # settings and display selection.
    stored = _christielab10_stream()

    result = build_nidaq_acquisition_configuration(
        stored, CHRISTIELAB10_PORTS, _christielab10_lasers())

    assert result.channels == stored.channels
    assert result.display_channels == stored.display_channels
    assert result == stored


# -------------------------------------------------------- cleared port roles


def test_a_cleared_port_role_stops_being_acquired(caplog):
    # Cleared in Edit DAQ Ports, tone1 claimed neither its pin nor its name,
    # so its stored channel came back as a custom input and was recorded
    # from then on - the readback's "(none)" problem, for the port roles.
    stored = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration(
                "tone1", "InputCard/port0/line0", kind="digital"),
            NidaqSignalChannelConfiguration("stim_readback", "InputCard/ai6"),
        ),
        is_enabled=True,
        display_channels=("tone1", "stim_readback"),
    )

    with caplog.at_level("WARNING"):
        result = build_nidaq_acquisition_configuration(
            stored, NidaqPortConfiguration(tone1=None), LaserSystemConfiguration())

    assert "tone1" not in {channel.name for channel in result.channels}
    assert "InputCard/port0/line0" not in {
        channel.physical_channel for channel in result.channels}
    # A channel no role owns is left alone, and so is its display selection.
    assert tuple(channel.name for channel in result.channels) == ("stim_readback",)
    assert result.display_channels == ("stim_readback",)
    warning, = [record.getMessage() for record in caplog.records
                if "drops the stored channel" in record.getMessage()]
    assert "'tone1'" in warning and "not set" in warning


@pytest.mark.parametrize(
    "role", ["tone1", "tone2", "tone3_r", "tone3_l", "cam_frames", "barcode"])
def test_every_port_role_claims_its_name_when_cleared(role):
    stored = _stream_of(
        NidaqSignalChannelConfiguration(role, "InputCard/port0/line7", kind="digital"))

    result = build_nidaq_acquisition_configuration(
        stored, NidaqPortConfiguration(), LaserSystemConfiguration())

    assert result.channels == ()


# ------------------------------------------------------- cleared laser tabs


def _christielab10_laser1_only():
    lasers = _christielab10_lasers()
    return dataclasses.replace(lasers, channels=(lasers.get_channel(1),))


def test_a_laser_cleared_from_its_tab_drops_its_stored_channels(caplog):
    # Clearing a whole laser tab in Edit DAQ Ports takes that laser out of
    # laser.channels. Its names were claimed only for the lasers configured,
    # so christielab10's laser2_diode on ai4 and laser2_command_copy on ai5
    # stayed in the scan: recorded, hidden, and shown nowhere.
    stored = _christielab10_stream()

    with caplog.at_level("WARNING"):
        result = build_nidaq_acquisition_configuration(
            stored, CHRISTIELAB10_PORTS, _christielab10_laser1_only())

    assert tuple(channel.name for channel in result.channels) == (
        "cam_frames", "barcode", "tone1", "tone2",
        "laser1_diode", "laser1_command_copy",
        "laser1_trigger_readback", "laser2_trigger_readback",
    )
    assert {"PXI1Slot5/ai4", "PXI1Slot5/ai5"}.isdisjoint(
        channel.physical_channel for channel in result.channels)
    warnings = sorted(record.getMessage() for record in caplog.records
                      if "drops the stored channel" in record.getMessage())
    assert len(warnings) == 2
    assert "'laser2_command_copy' on PXI1Slot5/ai5" in warnings[0]
    assert "'laser2_diode' on PXI1Slot5/ai4" in warnings[1]
    assert all("not set" in warning for warning in warnings)


def test_a_disabled_backend_drops_the_stored_channels_of_a_laser_not_configured(caplog):
    # With the backend disabled no laser name was claimed, so a laser cleared
    # in Edit DAQ Ports kept its stored channels as custom inputs: recorded,
    # hidden from Analysis by name, and shown on no tab.
    lasers = dataclasses.replace(_christielab10_laser1_only(), backend="disabled")
    stored = _stream_of(
        NidaqSignalChannelConfiguration("laser1_diode", "PXI1Slot5/ai8"),
        NidaqSignalChannelConfiguration("laser2_diode", "PXI1Slot5/ai4"),
        NidaqSignalChannelConfiguration("stim_readback", "PXI1Slot5/ai6"),
    )

    with caplog.at_level("WARNING"):
        result = build_nidaq_acquisition_configuration(
            stored, NidaqPortConfiguration(), lasers)

    pins = {channel.name: channel.physical_channel for channel in result.channels}
    # A configured laser's stays a custom input, on the pin its tab finds.
    assert pins["laser1_diode"] == lasers.get_channel(1).diode_input
    assert "laser2_diode" not in pins
    assert "PXI1Slot5/ai4" not in pins.values()
    assert pins["stim_readback"] == "PXI1Slot5/ai6"
    warning, = [record.getMessage() for record in caplog.records
                if "drops the stored channel" in record.getMessage()]
    assert "'laser2_diode' on PXI1Slot5/ai4" in warning and "not set" in warning


@pytest.mark.parametrize(
    "name", ["laser2_diode", "laser3_command_copy", "laser4_trigger"])
def test_every_laser_claims_its_names_while_the_backend_is_enabled(name):
    stored = _stream_of(NidaqSignalChannelConfiguration(name, "InputCard/ai7"))

    result = build_nidaq_acquisition_configuration(
        stored, NidaqPortConfiguration(), _laser_with())

    assert name not in {channel.name for channel in result.channels}
    assert "InputCard/ai7" not in {
        channel.physical_channel for channel in result.channels}


# ------------------------------------------------ christielab10, from YAML


#: christielab10's own nidaqPorts and nidaqStream blocks, deviceIdentities
#: and timing included, copied read-only from the rig's
#: ~/Autotrainer/system_configuration.yaml on 2026-09-25.
_CHRISTIELAB10_BLOCKS = Path(__file__).with_name("christielab10_nidaq_blocks.yaml")


def _christielab10_from_yaml():
    import io

    from autotrainer.core import SystemConfiguration

    return SystemConfiguration.load_yaml(io.StringIO(
        f"!SystemConfiguration\nversion: {SystemConfiguration.version}\n"
        + _CHRISTIELAB10_BLOCKS.read_text(encoding="utf-8")))


def test_christielab10s_yaml_blocks_load_unchanged():
    # The fixtures above are Python objects; this goes through the parser,
    # camelCase and all, as a load does, on the rig's own blocks. The plan
    # they build is the one they built at 1b13e15a, which is the stored
    # stream itself: every channel, in order, with its settings and the
    # display selection.
    loaded = _christielab10_from_yaml()
    ports, stored = loaded.nidaq_ports, loaded.nidaq_stream

    for attribute in ("device_name", "tone1", "tone2", "tone3_r", "tone3_l",
                      "cam_frames", "barcode"):
        assert getattr(ports, attribute) == getattr(CHRISTIELAB10_PORTS, attribute)
    assert [(identity.runtime_name, identity.product_type, identity.serial_number)
            for identity in ports.device_identities] == [
        ("PXI1Slot5", "PXI-6221", 21803707), ("PXI1Slot4", "PXI-6713", 27056752)]
    assert ports.timing.sync_mode == "auto"
    assert stored.channels == _christielab10_stream().channels
    assert stored.display_channels == _christielab10_stream().display_channels

    result = build_nidaq_acquisition_configuration(
        stored, ports, _christielab10_lasers())

    assert result == stored
    assert tuple(channel.name for channel in result.channels) == (
        "cam_frames", "barcode", "tone1", "tone2",
        "laser1_diode", "laser1_command_copy", "laser2_diode",
        "laser2_command_copy", "laser1_trigger_readback", "laser2_trigger_readback",
    )
    assert result.display_channels == (
        "laser1_command_copy", "cam_frames", "barcode", "tone2", "tone1")


def _with_readbacks(lasers, **inputs):
    """christielab10's lasers with trigger readbacks, laser1="PXI1Slot5/ai9"."""
    return dataclasses.replace(lasers, channels=tuple(
        dataclasses.replace(
            channel,
            trigger_monitor_input=inputs.get(f"laser{int(channel.channel_id)}"),
        )
        for channel in lasers.channels
    ))


def test_christielab10s_readbacks_take_over_its_custom_channels_in_place():
    # With triggerMonitorInput set on ai9 and ai10, the two custom
    # *_trigger_readback channels become the laser trigger roles where they
    # stand, last in the scan, with their own V / 1.0 / 0.0.
    loaded = _christielab10_from_yaml()
    lasers = _with_readbacks(
        _christielab10_lasers(), laser1="PXI1Slot5/ai9", laser2="PXI1Slot5/ai10")

    result = build_nidaq_acquisition_configuration(
        loaded.nidaq_stream, loaded.nidaq_ports, lasers)

    names = tuple(channel.name for channel in result.channels)
    pins = tuple(channel.physical_channel for channel in result.channels)
    assert len(names) == len(set(names)) == 10
    assert len(pins) == len(set(pins)) == 10
    assert "laser1_trigger_readback" not in names
    assert "laser2_trigger_readback" not in names
    # Positions 9 and 10 of the scan, as the custom channels were.
    ninth, tenth = result.channels[8], result.channels[9]
    assert (ninth.name, ninth.physical_channel) == ("laser1_trigger", "PXI1Slot5/ai9")
    assert (tenth.name, tenth.physical_channel) == ("laser2_trigger", "PXI1Slot5/ai10")
    for trigger in (ninth, tenth):
        assert (trigger.unit, trigger.scale, trigger.offset) == ("V", 1.0, 0.0)
    assert result.channels[:8] == loaded.nidaq_stream.channels[:8]
    assert result.display_channels == loaded.nidaq_stream.display_channels


def test_the_claimed_names_need_the_configured_lasers():
    # Defaulting to none configured, a caller that left them out had every
    # disabled laser's stored channels dropped, the configured ones too.
    from tools.acquisition.model.nidaq_channel_plan import _claimed_role_names

    with pytest.raises(TypeError):
        _claimed_role_names(False)
    assert "laser1_diode" not in _claimed_role_names(False, (1,))
    assert "laser2_diode" in _claimed_role_names(False, (1,))



# ------------------------------------------------ the final fix round


@pytest.mark.parametrize(("channel", "kind"), [
    ("PXI1Slot5/ai8", "analog"),
    ("Portable1/ai3", "analog"),
    ("Linear2/ai0", "analog"),
    ("PXI1Slot5/port0/line2", "digital"),
    ("Portable1/port0/line1", "digital"),
])
def test_a_channels_kind_follows_its_own_name_not_its_devices(channel, kind):
    # By substring, a device whose name holds "port" or "line" made every
    # analog input on it digital.
    assert nidaq_channel_kind(channel) == kind


#: christielab10's laser block, from the rig's configuration as the hardware
#: checks read it (hardware-check-report, phase 2) and the plan's ledger.
_CHRISTIELAB10_LASER_BLOCK = Path(__file__).with_name("christielab10_laser_block.yaml")


def test_christielab10s_laser_block_loads_and_keeps_its_lines_apart():
    import io

    from autotrainer.core import SystemConfiguration
    from autotrainer.core.configuration.laser_configuration import clock_line_clashes

    loaded = SystemConfiguration.load_yaml(io.StringIO(
        f"!SystemConfiguration\nversion: {SystemConfiguration.version}\n"
        + _CHRISTIELAB10_LASER_BLOCK.read_text(encoding="utf-8")))
    laser = loaded.laser

    assert laser == _christielab10_lasers()
    assert (laser.backplane_clock_line, laser.pulse_clock_line) == ("PXI_Trig1", "PXI_Trig3")
    assert [(channel.board_stim_line, channel.board_trigger_pulse_us)
            for channel in laser.channels] == [(3, 1000), (2, 1000)]
    assert not clock_line_clashes(
        laser.backplane_clock_line, laser.pulse_clock_line, laser.channels,
        laser.trigger_listener_inputs)
