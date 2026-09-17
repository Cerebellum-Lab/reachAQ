import pytest

from autotrainer.behavior.pellet_trial import TrialOutcome
from tools.acquisition.model.pellet_precheck import (
    DEFAULT_AUTHORIZATION_TIMEOUT_MS,
    MAX_AUTHORIZATION_TIMEOUT_MS,
    MAX_PRESENCE_FRESHNESS_MS,
    PELLET_PRECHECK_CAPABILITY,
    PelletPrecheckConfiguration,
    PelletPrecheckEvidence,
    PelletPrecheckMethod,
    PelletPrecheckOutcome,
    capability_available,
    evaluate_precheck,
)


CAPABLE = (PELLET_PRECHECK_CAPABILITY, "time_sync")


def _presence(**changes):
    values = {"method": PelletPrecheckMethod.PELLET_PRESENCE}
    values.update(changes)
    return PelletPrecheckConfiguration(**values)


def _before_send(**changes):
    values = {"method": PelletPrecheckMethod.BEFORE_SEND}
    values.update(changes)
    return PelletPrecheckConfiguration(**values)


def test_precheck_defaults_to_disabled():
    configuration = PelletPrecheckConfiguration()
    assert configuration.method is PelletPrecheckMethod.DISABLED
    assert configuration.authorization_timeout_ms == DEFAULT_AUTHORIZATION_TIMEOUT_MS


def test_disabled_precheck_authorizes_without_firmware():
    result = evaluate_precheck(
        PelletPrecheckConfiguration(), now_perf_time=100.0
    )
    assert result.outcome is PelletPrecheckOutcome.NOT_REQUIRED
    assert result.authorized is True
    assert result.trial_outcome is None


def test_a_selected_method_without_the_capability_is_unsupported():
    result = evaluate_precheck(
        _presence(),
        now_perf_time=100.0,
        evidence=PelletPrecheckEvidence(observed_at=100.0, pellet_present=True),
        reported_capabilities=("time_sync",),
    )
    assert result.outcome is PelletPrecheckOutcome.UNSUPPORTED
    assert result.authorized is False
    assert PELLET_PRECHECK_CAPABILITY in result.detail


def test_capability_availability_is_reported_directly():
    assert capability_available(CAPABLE) is True
    assert capability_available(()) is False
    assert capability_available(None) is False


def test_fresh_present_evidence_authorizes():
    result = evaluate_precheck(
        _presence(),
        now_perf_time=100.2,
        evidence=PelletPrecheckEvidence(observed_at=100.0, pellet_present=True),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.AUTHORIZED
    assert result.evidence_age_ms == pytest.approx(200.0)


def test_absent_pellet_fails_the_trial():
    result = evaluate_precheck(
        _presence(),
        now_perf_time=100.1,
        evidence=PelletPrecheckEvidence(observed_at=100.0, pellet_present=False),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.FAILED
    assert result.trial_outcome == TrialOutcome.PELLET_PRECHECK_FAILED.value


def test_stale_evidence_does_not_authorize():
    result = evaluate_precheck(
        _presence(presence_freshness_ms=100),
        now_perf_time=101.0,
        evidence=PelletPrecheckEvidence(observed_at=100.0, pellet_present=True),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.EVIDENCE_STALE
    assert result.authorized is False
    assert result.trial_outcome == TrialOutcome.PELLET_PRECHECK_FAILED.value


def test_missing_evidence_does_not_authorize():
    result = evaluate_precheck(
        _presence(), now_perf_time=100.0, reported_capabilities=CAPABLE
    )
    assert result.outcome is PelletPrecheckOutcome.EVIDENCE_STALE


def test_before_send_authorizes_within_the_timeout():
    result = evaluate_precheck(
        _before_send(),
        now_perf_time=100.5,
        evidence=PelletPrecheckEvidence(
            observed_at=100.0,
            authorized=True,
            responded_at=100.2,
            firmware_response="PRECHECKAUTH",
        ),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.AUTHORIZED
    assert result.latency_ms == pytest.approx(200.0)
    assert result.firmware_response == "PRECHECKAUTH"


def test_before_send_refusal_fails_the_trial():
    result = evaluate_precheck(
        _before_send(),
        now_perf_time=100.3,
        evidence=PelletPrecheckEvidence(
            observed_at=100.0, authorized=False, responded_at=100.2
        ),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.FAILED
    assert result.trial_outcome == TrialOutcome.PELLET_PRECHECK_FAILED.value


def test_before_send_answer_after_the_timeout_is_a_timeout():
    result = evaluate_precheck(
        _before_send(authorization_timeout_ms=100),
        now_perf_time=100.5,
        evidence=PelletPrecheckEvidence(
            observed_at=100.0, authorized=True, responded_at=100.4
        ),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.TIMED_OUT
    assert result.trial_outcome == (
        TrialOutcome.PELLET_PRECHECK_AUTHORIZATION_TIMEOUT.value
    )


def test_before_send_silence_past_the_timeout_fails_rather_than_blocking():
    result = evaluate_precheck(
        _before_send(authorization_timeout_ms=100),
        now_perf_time=100.5,
        evidence=PelletPrecheckEvidence(observed_at=100.0),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.TIMED_OUT
    assert result.is_pending is False


def test_before_send_inside_the_window_is_pending_not_a_timeout():
    result = evaluate_precheck(
        _before_send(authorization_timeout_ms=750),
        now_perf_time=100.2,
        evidence=PelletPrecheckEvidence(observed_at=100.0),
        reported_capabilities=CAPABLE,
    )
    assert result.outcome is PelletPrecheckOutcome.PENDING
    assert result.is_pending is True
    assert result.authorized is False
    # A pending answer must never score the trial.
    assert result.trial_outcome is None


def test_before_send_requires_a_request_time():
    with pytest.raises(ValueError, match="requires the request time"):
        evaluate_precheck(
            _before_send(), now_perf_time=100.0, reported_capabilities=CAPABLE
        )


@pytest.mark.parametrize("timeout", [0, MAX_AUTHORIZATION_TIMEOUT_MS + 1])
def test_authorization_timeout_bounds_are_enforced(timeout):
    with pytest.raises(ValueError, match="Authorization timeout must be"):
        PelletPrecheckConfiguration(authorization_timeout_ms=timeout)


@pytest.mark.parametrize("freshness", [0, MAX_PRESENCE_FRESHNESS_MS + 1])
def test_presence_freshness_bounds_are_enforced(freshness):
    with pytest.raises(ValueError, match="Presence freshness must be"):
        PelletPrecheckConfiguration(presence_freshness_ms=freshness)


def test_evidence_times_must_be_finite():
    with pytest.raises(ValueError, match="observed_at must be finite"):
        PelletPrecheckEvidence(observed_at=float("nan"))


def test_new_trial_outcomes_are_scored():
    assert TrialOutcome.PELLET_PRECHECK_FAILED.is_scored is True
    assert TrialOutcome.PELLET_PRECHECK_AUTHORIZATION_TIMEOUT.is_scored is True


def test_result_records_are_plain_and_typed():
    record = evaluate_precheck(
        _presence(),
        now_perf_time=100.1,
        evidence=PelletPrecheckEvidence(observed_at=100.0, pellet_present=False),
        reported_capabilities=CAPABLE,
    ).to_record()
    assert record["outcome"] == "failed"
    assert record["method"] == "pellet_presence"
    assert record["trial_outcome"] == "pellet_precheck_failed"
