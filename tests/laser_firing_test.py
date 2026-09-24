import pytest
from types import SimpleNamespace

from autotrainer.device import LaserChannelConfiguration, LaserSystemConfiguration
from tools.acquisition.model.laser_firing import (
    LaserFiring,
    amplitude_refusal,
    resolve_laser_firing,
)
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


def configuration(**laser_2):
    values = dict(
        channel_id=2,
        analog_output="Dev1/ao1",
        diode_input="Dev1/ai4",
        shutter_output="Dev1/port0/line5",
        trigger_source="/Dev1/PXI_Trig2",
        board_stim_line=2,
        board_trigger_pulse_us=1500,
    )
    values.update(laser_2)
    return LaserSystemConfiguration.from_channels((LaserChannelConfiguration(**values),))


def test_a_board_firing_takes_the_lasers_terminal_line_and_pulse():
    firing = resolve_laser_firing(configuration(), 2, "hardware_stim3")

    assert firing == LaserFiring(
        channel_id=2,
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PXI_Trig2",
        stim_line=2,
        trigger_pulse_us=1500,
    )
    assert firing.is_board_trigger


def test_a_software_firing_needs_no_terminal_or_line():
    firing = resolve_laser_firing(
        configuration(trigger_source=None, board_stim_line=None), 2,
        LaserTriggerRoute.DIRECT_NI_SOFTWARE)

    assert firing.trigger_terminal == ""
    assert firing.stim_line is None
    assert not firing.is_board_trigger


def test_an_unconfigured_laser_is_refused():
    with pytest.raises(ValueError, match="Laser 1 is not configured"):
        resolve_laser_firing(configuration(), 1, "hardware_stim3")


def test_no_laser_is_refused():
    with pytest.raises(ValueError, match="Laser 0 is not configured"):
        resolve_laser_firing(configuration(), 0, "hardware_stim3")


def test_a_board_firing_without_a_terminal_is_refused():
    with pytest.raises(ValueError, match="no trigger terminal"):
        resolve_laser_firing(configuration(trigger_source=None), 2, "hardware_stim3")


def test_a_board_firing_without_a_board_line_is_refused():
    with pytest.raises(ValueError, match="boardStimLine"):
        resolve_laser_firing(configuration(board_stim_line=None), 2, "hardware_stim3")


def test_no_route_cannot_fire():
    with pytest.raises(ValueError, match="Laser 2 trigger route 'none' cannot fire"):
        resolve_laser_firing(configuration(), 2, "none")


def test_the_record_names_the_route_by_value():
    record = resolve_laser_firing(configuration(), 2, "hardware_stim3").to_record()

    assert record == {
        "channel_id": 2,
        "trigger_route": "hardware_stim3",
        "trigger_terminal": "/Dev1/PXI_Trig2",
        "stim_line": 2,
        "trigger_pulse_us": 1500,
    }


def test_an_amplitude_outside_the_lasers_range_is_refused():
    channel = configuration().get_channel(2)
    profile = SimpleNamespace(profile_id="hot", amplitude_volts=6.0)

    assert "accepts 0..5 V" in amplitude_refusal(profile, channel)
    assert amplitude_refusal(SimpleNamespace(profile_id="ok", amplitude_volts=1.0), channel) == ""
