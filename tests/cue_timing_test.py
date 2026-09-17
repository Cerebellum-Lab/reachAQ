import pytest

from tools.acquisition.model.cue_timing import (
    MAX_POST_CLEAR_DELAY_MS,
    CueCancelReason,
    CueDecision,
    CueGateState,
    CueTimingConfiguration,
    CueTimingPolicy,
)


def _clear(at):
    return CueGateState(observed_at=at, pellet_present=True, reach_active=False)


def _policy(**changes):
    return CueTimingPolicy(CueTimingConfiguration(**changes))


def test_lock_timing_is_on_by_default():
    configuration = CueTimingConfiguration()
    assert configuration.lock_timing is True
    assert configuration.post_clear_delay_ms == 0


def test_post_clear_delay_requires_lock_timing_off():
    with pytest.raises(ValueError, match="only when Lock Timing is off"):
        CueTimingConfiguration(lock_timing=True, post_clear_delay_ms=250)


@pytest.mark.parametrize("delay", [-1, MAX_POST_CLEAR_DELAY_MS + 1])
def test_post_clear_delay_bounds_are_enforced(delay):
    with pytest.raises(ValueError, match="Post-clear delay must be"):
        CueTimingConfiguration(lock_timing=False, post_clear_delay_ms=delay)


@pytest.mark.parametrize("staleness", [0, 10_001])
def test_gate_staleness_bounds_are_enforced(staleness):
    with pytest.raises(ValueError, match="Gate staleness must be"):
        CueTimingConfiguration(gate_staleness_ms=staleness)


def test_start_returns_the_deadline():
    policy = _policy()
    assert policy.start(100.0, 900) == pytest.approx(100.9)
    assert policy.is_started is True


@pytest.mark.parametrize("interval", [0, -1])
def test_a_non_positive_interval_is_rejected(interval):
    with pytest.raises(ValueError, match="Cue interval must be positive"):
        _policy().start(100.0, interval)


def test_evaluating_before_start_is_an_error():
    with pytest.raises(RuntimeError, match="was not started"):
        _policy().evaluate(100.0, _clear(100.0))


def test_a_clear_gate_waits_then_fires_on_the_deadline():
    policy = _policy()
    policy.start(100.0, 900)
    waiting = policy.evaluate(100.5, _clear(100.5))
    assert waiting.decision is CueDecision.WAIT
    assert waiting.remaining_seconds == pytest.approx(0.4)
    fired = policy.evaluate(100.9, _clear(100.9))
    assert fired.decision is CueDecision.FIRE
    assert fired.reason is CueCancelReason.NONE
    assert fired.should_fire is True


def test_locked_deadline_with_a_missing_pellet_skips_and_resets():
    policy = _policy()
    policy.start(100.0, 900)
    gate = CueGateState(observed_at=100.9, pellet_present=False)
    evaluation = policy.evaluate(100.9, gate)
    assert evaluation.decision is CueDecision.SKIP_AND_RESET
    assert evaluation.reason is CueCancelReason.LOCK_DEADLINE_PELLET_MISSING
    assert evaluation.should_reset is True


def test_locked_deadline_with_an_active_reach_skips_and_resets():
    policy = _policy()
    policy.start(100.0, 900)
    gate = CueGateState(observed_at=100.9, pellet_present=True, reach_active=True)
    evaluation = policy.evaluate(100.9, gate)
    assert evaluation.reason is CueCancelReason.LOCK_DEADLINE_EARLY_REACH


def test_locked_deadline_with_stale_evidence_skips_and_resets():
    policy = _policy(gate_staleness_ms=100)
    policy.start(100.0, 900)
    stale = CueGateState(observed_at=100.5, pellet_present=True)
    evaluation = policy.evaluate(100.9, stale)
    assert evaluation.reason is CueCancelReason.LOCK_DEADLINE_GATE_STATE_STALE


def test_unknown_presence_is_treated_as_stale_not_clear():
    policy = _policy()
    policy.start(100.0, 900)
    unknown = CueGateState(observed_at=100.9, pellet_present=None)
    evaluation = policy.evaluate(100.9, unknown)
    assert evaluation.reason is CueCancelReason.LOCK_DEADLINE_GATE_STATE_STALE


def test_a_locked_trial_never_fires_late():
    policy = _policy()
    policy.start(100.0, 900)
    blocked = CueGateState(observed_at=100.9, pellet_present=False)
    assert policy.evaluate(100.9, blocked).should_reset is True
    # Even well past the deadline the locked trial resets rather than firing.
    policy.start(100.0, 900)
    assert policy.evaluate(105.0, CueGateState(observed_at=105.0, pellet_present=False)).should_reset


def test_unlocked_trial_waits_out_a_block_past_the_deadline():
    policy = _policy(lock_timing=False)
    policy.start(100.0, 900)
    blocked = CueGateState(observed_at=101.0, pellet_present=False)
    evaluation = policy.evaluate(101.0, blocked)
    assert evaluation.decision is CueDecision.WAIT
    assert evaluation.was_blocked is True


def test_unlocked_trial_applies_the_post_clear_delay_after_a_block():
    policy = _policy(lock_timing=False, post_clear_delay_ms=300)
    policy.start(100.0, 900)
    policy.evaluate(101.0, CueGateState(observed_at=101.0, pellet_present=False))
    waiting = policy.evaluate(101.1, _clear(101.1))
    assert waiting.decision is CueDecision.WAIT
    assert waiting.remaining_seconds == pytest.approx(0.3)
    still_waiting = policy.evaluate(101.3, _clear(101.3))
    assert still_waiting.decision is CueDecision.WAIT
    fired = policy.evaluate(101.45, _clear(101.45))
    assert fired.decision is CueDecision.FIRE


def test_a_re_block_restarts_the_post_clear_delay():
    policy = _policy(lock_timing=False, post_clear_delay_ms=300)
    policy.start(100.0, 900)
    policy.evaluate(101.0, CueGateState(observed_at=101.0, pellet_present=False))
    policy.evaluate(101.1, _clear(101.1))
    # Re-entry before the delay elapses restarts it.
    policy.evaluate(101.2, CueGateState(observed_at=101.2, pellet_present=False))
    # The clear run restarts at 101.45, so the delay now ends at 101.75.
    still_waiting = policy.evaluate(101.6, _clear(101.45))
    assert still_waiting.decision is CueDecision.WAIT
    assert still_waiting.remaining_seconds == pytest.approx(0.15)
    fired = policy.evaluate(101.76, _clear(101.7))
    assert fired.decision is CueDecision.FIRE


def test_an_unblocked_unlocked_trial_receives_no_extension():
    policy = _policy(lock_timing=False, post_clear_delay_ms=300)
    policy.start(100.0, 900)
    policy.evaluate(100.5, _clear(100.5))
    fired = policy.evaluate(100.9, _clear(100.9))
    assert fired.decision is CueDecision.FIRE
    assert fired.was_blocked is False


def test_cancel_reports_an_external_reason_and_clears_state():
    policy = _policy()
    policy.start(100.0, 900)
    evaluation = policy.cancel(CueCancelReason.HOST_REQUEST)
    assert evaluation.decision is CueDecision.SKIP_AND_RESET
    assert evaluation.reason is CueCancelReason.HOST_REQUEST
    assert policy.is_started is False


def test_reset_clears_the_deadline_and_block_history():
    policy = _policy(lock_timing=False, post_clear_delay_ms=300)
    policy.start(100.0, 900)
    policy.evaluate(101.0, CueGateState(observed_at=101.0, pellet_present=False))
    policy.reset()
    policy.start(200.0, 900)
    fired = policy.evaluate(200.9, _clear(200.9))
    assert fired.was_blocked is False
    assert fired.decision is CueDecision.FIRE


def test_evaluation_records_are_plain_and_typed():
    policy = _policy()
    policy.start(100.0, 900)
    record = policy.evaluate(
        100.9, CueGateState(observed_at=100.9, pellet_present=False)
    ).to_record()
    assert record["decision"] == "skip_and_reset"
    assert record["reason"] == "lock_deadline_pellet_missing"
    assert record["lock_timing"] is True


def test_non_finite_times_are_rejected():
    policy = _policy()
    with pytest.raises(ValueError, match="Tone 1 time must be finite"):
        policy.start(float("nan"), 900)
    with pytest.raises(ValueError, match="observation time must be finite"):
        CueGateState(observed_at=float("inf"))
    policy.start(100.0, 900)
    with pytest.raises(ValueError, match="Evaluation time must be finite"):
        policy.evaluate(float("nan"), _clear(100.0))


def test_lock_timing_is_protocol_persisted_and_defaults_on():
    from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow

    row = TrialProtocolRow(trial_id=1)
    assert row.cue_lock_timing is True
    assert row.cue_post_clear_delay_ms == 0


def test_row_rejects_a_post_clear_delay_while_locked():
    from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow

    with pytest.raises(ValueError, match="only when lock timing is off"):
        TrialProtocolRow(trial_id=1).with_updates({
            "tone_profile_id": "tone-1",
            "tone_phase": "pellet_presentation",
            "cue_tone_profile_id": "tone-2",
            "cue_interval_fixed_ms": 900,
            "cue_post_clear_delay_ms": 250,
        })


def test_row_accepts_a_post_clear_delay_when_unlocked():
    from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow

    row = TrialProtocolRow(trial_id=1).with_updates({
        "tone_profile_id": "tone-1",
        "tone_phase": "pellet_presentation",
        "cue_tone_profile_id": "tone-2",
        "cue_interval_fixed_ms": 900,
        "cue_lock_timing": False,
        "cue_post_clear_delay_ms": 250,
    })
    assert row.cue_post_clear_delay_ms == 250


def test_row_rejects_a_post_clear_delay_without_a_cue_tone():
    from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow

    with pytest.raises(ValueError, match="requires a cue tone profile"):
        TrialProtocolRow(trial_id=1).with_updates({
            "cue_lock_timing": False,
            "cue_post_clear_delay_ms": 250,
        })


def test_row_post_clear_delay_bounds_are_enforced():
    from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow

    with pytest.raises(ValueError, match="cue_post_clear_delay_ms must be"):
        TrialProtocolRow(trial_id=1).with_updates({
            "cue_lock_timing": False,
            "cue_post_clear_delay_ms": MAX_POST_CLEAR_DELAY_MS + 1,
        })
