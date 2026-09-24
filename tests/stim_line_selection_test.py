"""Host readiness for a firmware that can pulse STIM2 as well as STIM3."""

from autotrainer.device.device_interface import (
    BOARD_STIM_LINE_OUTPUTS,
    DigitalOutputs,
)


def test_the_board_line_map_records_the_off_by_one():
    # The board device tree names its outputs STIM0..STIM3 while DigitalOutputs
    # counts from one, so board STIM2 is STIMULUS_3.
    assert BOARD_STIM_LINE_OUTPUTS[2] is DigitalOutputs.STIMULUS_3
    assert BOARD_STIM_LINE_OUTPUTS[3] is DigitalOutputs.STIMULUS_4


def test_the_tone_confirmation_lines_are_not_in_the_map():
    assert 0 not in BOARD_STIM_LINE_OUTPUTS
    assert 1 not in BOARD_STIM_LINE_OUTPUTS
