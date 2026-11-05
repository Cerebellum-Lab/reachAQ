import json

import pytest

from tools.acquisition.model.laser_kill_matrix import (
    HELD_THROUGH_RESET, HELD_UNTIL_REOPEN, HELD_UNTIL_RESET, REFUSED_BY_DESIGN,
    SAFE_ON_DEATH, TRAIN_RAN_ON, UNKNOWN, KillCase, KillObservation,
    LevelReading, LevelSample, format_table, from_record, judge, kill_cases,
    level_at, level_between, to_record,
)

def _reading(mean, minimum=None, maximum=None):
    return LevelReading(mean, mean if minimum is None else minimum,
                        mean if maximum is None else maximum, 500, "PXI1Slot5/ai3")

def _observation(case=KillCase("sigkill", "software", "dc"), **fields):
    values = dict(before=_reading(2.0), after_kill=_reading(2.0), hold=_reading(2.0),
                  after_hold=_reading(2.0), after_reopen=_reading(0.0),
                  after_reset=_reading(0.0), reopen_error=None, reset_error=None,
                  victim_exit="signal 9")
    values.update(fields)
    return KillObservation(case, **values)

def test_pulse_rows_repeat_three_times_and_dc_rows_once():
    cases = kill_cases()
    assert len(cases) == 6 + 1 + 6 + 6 * 3
    assert [c.case_id for c in cases if c.mode == "sigkill"] == [
        "sigkill-software-dc", "sigkill-hardware-dc",
        "sigkill-hardware-pulse-r1", "sigkill-hardware-pulse-r2", "sigkill-hardware-pulse-r3"]

def test_the_software_pulse_row_is_refused_by_design():
    refused = [c for c in kill_cases() if c.refusal]
    assert [c.case_id for c in refused] == ["clean-software-pulse"]
    assert judge(_observation(refused[0], after_kill=None), minimum_volts=0.0,
                 level_volts=2.0) == REFUSED_BY_DESIGN

def test_level_at_takes_only_the_window_before_the_instant():
    samples = [LevelSample(t / 100, t / 100, 0.0, 3.0) for t in range(0, 100)]
    # (0.455, 0.505]: the five chunks 0.46..0.50; the instant sits between
    # chunks so no boundary is decided by floating point.
    reading = level_at(samples, 0.505, window_s=0.05, instrument="ai3")
    assert (reading.sample_count, reading.minimum, reading.maximum) == (5, 0.0, 3.0)
    assert abs(reading.mean - 0.48) < 1e-9
    assert level_at(samples, 5.0, window_s=0.05, instrument="ai3") is None

def test_a_row_with_no_sample_after_the_kill_is_unknown():
    assert judge(_observation(after_kill=None, hold=None, after_hold=None),
                 minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_train_that_keeps_generating_after_the_kill_is_not_safe():
    landed_low = _observation(KillCase("sigkill", "hardware", "pulse", 2),
                              after_kill=_reading(0.0), hold=_reading(1.0, 0.0, 2.0))
    assert judge(landed_low, minimum_volts=0.0, level_volts=2.0) == TRAIN_RAN_ON

def test_an_output_at_its_minimum_after_the_kill_is_safe():
    assert judge(_observation(after_kill=_reading(0.01), hold=_reading(0.01)),
                 minimum_volts=0.0, level_volts=2.0) == SAFE_ON_DEATH

def test_a_held_output_is_graded_by_what_reset_it():
    assert judge(_observation(), minimum_volts=0.0, level_volts=2.0) == HELD_UNTIL_REOPEN
    assert judge(_observation(after_reopen=None, reopen_error="DAQmx -50103"),
                 minimum_volts=0.0, level_volts=2.0) == HELD_UNTIL_RESET
    assert judge(_observation(after_reopen=_reading(2.0), after_reset=_reading(2.0)),
                 minimum_volts=0.0, level_volts=2.0) == HELD_THROUGH_RESET

def test_the_table_names_every_instant_and_the_verdict():
    text = format_table([_observation(dmm={"after_kill": 1.98})],
                        minimum_volts=0.0, level_volts=2.0)
    header = text.splitlines()[0]
    for column in ("before", "after kill", "hold min..max", "after 5 s",
                   "after reopen", "after reset", "victim exit", "instrument", "verdict"):
        assert column in header
    assert "sigkill-software-dc" in text and "held_until_reopen" in text
    assert "dmm=1.98" in text

def test_the_record_round_trips_through_json():
    observations = (_observation(dmm={"after_kill": 1.98}),
                    _observation(KillCase("hang", "hardware", "pulse", 3),
                                 after_kill=None, notes=("reader timed out",)))
    record = to_record(observations, host="christielab10", level_volts=2.0)
    assert record["schema"] == 1 and record["host"] == "christielab10"
    # JSON turns tuples into lists; from_record puts them back.
    assert from_record(json.loads(json.dumps(record))) == observations

# Fix round 1: a stalled reader must read unknown, never safe.

def test_a_pulse_row_at_its_minimum_with_no_hold_reading_is_unknown():
    # The output was low at +0.10 s, but nothing shows it stayed low: a train
    # killed on its low phase reads exactly like this.
    pulse = KillCase("sigkill", "hardware", "pulse", 1)
    assert judge(_observation(pulse, after_kill=_reading(0.0), hold=None),
                 minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_hold_that_left_the_minimum_without_a_full_train_is_not_safe():
    pulse = KillCase("sigkill", "hardware", "pulse", 1)
    risen = _observation(pulse, after_kill=_reading(0.0), hold=_reading(0.5, 0.0, 1.0))
    assert judge(risen, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_dc_row_at_its_minimum_with_no_hold_reading_is_unknown():
    assert judge(_observation(after_kill=_reading(0.0), hold=None),
                 minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_dc_row_whose_hold_shows_both_phases_is_unknown_not_a_train():
    both = _observation(KillCase("sigkill", "software", "dc"),
                        after_kill=_reading(0.0), hold=_reading(1.0, 0.0, 2.0))
    assert judge(both, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_held_row_with_no_reset_reading_is_unknown():
    stalled = _observation(after_reopen=None, after_reset=None,
                           reopen_error="DAQmx -50103", reset_error="reset failed")
    assert judge(stalled, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_refused_reopen_with_a_high_reset_reading_is_still_held_through_reset():
    held = _observation(after_reopen=None, after_reset=_reading(2.0),
                        reopen_error="DAQmx -50103")
    assert judge(held, minimum_volts=0.0, level_volts=2.0) == HELD_THROUGH_RESET

def test_a_dmm_value_that_is_none_is_skipped_by_the_table():
    text = format_table([_observation(dmm={"before": None, "after_kill": 1.98})],
                        minimum_volts=0.0, level_volts=2.0)
    assert text.count("dmm=") == 1 and "dmm=1.98" in text
    assert "dmm=None" not in text


# Fix round 2 (Task 3 review): a verdict needs a before reading at the level, the
# exit the mode should give, and levels the judge can tell apart.

_EXITS = {
    "clean": "exit 0", "sigterm": "signal 15", "sigkill": "signal 9",
    "exception": "exit 1", "abort": "signal 6",
    "hang": "killed after hang (sigkill by pid)",
}

def test_a_row_with_no_before_reading_is_unknown():
    assert judge(_observation(before=None), minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_row_whose_output_was_already_down_before_the_kill_is_unknown_not_safe():
    # A train that ended by itself during a slow prompt, then a kill, then
    # nothing to see: every later reading is at the minimum.
    ended = _observation(before=_reading(0.0), after_kill=_reading(0.0), hold=_reading(0.0))
    assert judge(ended, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_before_window_that_reaches_the_level_counts_even_when_its_mean_does_not():
    pulse = KillCase("sigkill", "hardware", "pulse", 1)
    running = _observation(pulse, before=_reading(1.0, 0.0, 2.0),
                           after_kill=_reading(0.0), hold=_reading(1.0, 0.0, 2.0))
    assert judge(running, minimum_volts=0.0, level_volts=2.0) == TRAIN_RAN_ON

@pytest.mark.parametrize("mode", sorted(_EXITS))
def test_each_mode_has_to_end_the_way_it_should(mode):
    case = KillCase(mode, "software", "dc")
    assert judge(_observation(case, victim_exit=_EXITS[mode]),
                 minimum_volts=0.0, level_volts=2.0) == HELD_UNTIL_REOPEN
    # A victim that closed itself at its own cue timeout, or was ended some
    # other way, did not die the way the row says.
    assert judge(_observation(case, victim_exit="exit 3"),
                 minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_safe_looking_row_with_the_wrong_exit_is_unknown():
    zombie = _observation(after_kill=_reading(0.0), hold=_reading(0.0), victim_exit="exit 3")
    assert judge(zombie, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_refused_row_stays_refused_whatever_its_exit_and_before():
    refused = KillCase("clean", "software", "pulse")
    assert judge(_observation(refused, before=None, victim_exit="exit 2"),
                 minimum_volts=0.0, level_volts=2.0) == REFUSED_BY_DESIGN

def test_levels_too_close_to_tell_apart_are_unknown():
    assert judge(_observation(), minimum_volts=0.0, level_volts=0.08) == UNKNOWN

def test_a_refusal_is_in_its_cell_even_when_the_reader_saw_a_level():
    text = format_table(
        [_observation(after_reopen=_reading(2.0), reopen_error="DaqError: -50103 reserved",
                      after_reset=_reading(0.0), reset_error="DaqError: reset refused")],
        minimum_volts=0.0, level_volts=2.0)
    assert "2.000 (DaqError: -50103 reserved)" in text
    assert "0.000 (DaqError: reset refused)" in text

def test_notes_are_in_the_table_beside_the_exit():
    text = format_table([_observation(notes=("reopen skipped", "reset skipped"))],
                        minimum_volts=0.0, level_volts=2.0)
    assert "signal 9<br>reopen skipped; reset skipped" in text
