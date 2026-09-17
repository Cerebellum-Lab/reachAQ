"""Host readiness for a firmware that can pulse STIM2 as well as STIM3."""

import pytest

from autotrainer.device.device_interface import (
    BOARD_STIM_LINE_OUTPUTS,
    DigitalOutputs,
)
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


def make_profile(**overrides):
    values = dict(
        profile_id="stim-a",
        revision=1,
        channel_id=1,
        amplitude_volts=2.0,
        pulse_duration_ms=5.0,
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
    )
    values.update(overrides)
    return LaserPulseProfile(**values)


def test_the_board_line_map_records_the_off_by_one():
    # The board device tree names its outputs STIM0..STIM3 while DigitalOutputs
    # counts from one, so board STIM2 is STIMULUS_3.
    assert BOARD_STIM_LINE_OUTPUTS[2] is DigitalOutputs.STIMULUS_3
    assert BOARD_STIM_LINE_OUTPUTS[3] is DigitalOutputs.STIMULUS_4


def test_the_tone_confirmation_lines_are_not_in_the_map():
    assert 0 not in BOARD_STIM_LINE_OUTPUTS
    assert 1 not in BOARD_STIM_LINE_OUTPUTS


def test_a_profile_defaults_to_stim3():
    assert make_profile().stim_line == 3


def test_a_profile_can_select_stim2():
    assert make_profile(stim_line=2).stim_line == 2


def test_a_profile_rejects_a_tone_confirmation_line():
    with pytest.raises(ValueError, match="STIM2 or STIM3"):
        make_profile(stim_line=1)


def test_a_profile_rejects_an_unknown_line():
    with pytest.raises(ValueError, match="STIM2 or STIM3"):
        make_profile(stim_line=9)


def test_a_profile_round_trips_its_line():
    record = make_profile(stim_line=2).to_record()

    assert record["stim_line"] == 2
    assert LaserPulseProfile(**record).stim_line == 2


def test_a_stored_profile_without_a_line_loads_as_stim3():
    record = make_profile().to_record()
    del record["stim_line"]

    assert LaserPulseProfile(**record).stim_line == 3
