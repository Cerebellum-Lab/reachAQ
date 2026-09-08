import pytest

from tools.acquisition.model.intertrial_timing import (
    MAX_TARGET_SECONDS,
    InterTrialDecision,
    InterTrialReason,
    InterTrialTimingConfiguration,
    InterTrialTimingPolicy,
)


def _policy(**changes):
    values = {"enabled": True, "target_seconds": 2.5}
    values.update(changes)
    return InterTrialTimingPolicy(InterTrialTimingConfiguration(**values))


def test_timing_defaults_off():
    configuration = InterTrialTimingConfiguration()
    assert configuration.enabled is False
    assert configuration.target_seconds is None
    assert configuration.apply_to_retries is False


def test_disabled_policy_always_sends_now():
    policy = InterTrialTimingPolicy()
    policy.arm(100.0)
    evaluation = policy.evaluate(100.1)
    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.reason is InterTrialReason.DISABLED
    assert evaluation.should_wait is False


def test_enabling_without_a_target_is_rejected():
    with pytest.raises(ValueError, match="requires a target"):
        InterTrialTimingConfiguration(enabled=True)


@pytest.mark.parametrize("target", [0.0, -1.0, float("inf"), float("nan")])
def test_non_positive_targets_are_rejected(target):
    with pytest.raises(ValueError, match="must be positive"):
        InterTrialTimingConfiguration(enabled=True, target_seconds=target)


def test_targets_beyond_the_maximum_are_rejected():
    with pytest.raises(ValueError, match="at most"):
        InterTrialTimingConfiguration(
            enabled=True, target_seconds=MAX_TARGET_SECONDS + 1
        )


def test_target_is_stored_at_millisecond_resolution():
    configuration = InterTrialTimingConfiguration(
        enabled=True, target_seconds=2.50049
    )
    assert configuration.target_seconds == 2.5


def test_early_readiness_waits_for_the_remaining_time():
    policy = _policy()
    policy.arm(100.0)
    evaluation = policy.evaluate(101.0)
    assert evaluation.decision is InterTrialDecision.WAIT
    assert evaluation.reason is InterTrialReason.TARGET_NOT_REACHED
    assert evaluation.remaining_seconds == 1.5
    assert evaluation.elapsed_seconds == 1.0
    assert evaluation.lateness_seconds is None
    assert evaluation.should_wait is True


def test_readiness_exactly_on_target_sends_now_with_no_lateness():
    policy = _policy()
    policy.arm(100.0)
    evaluation = policy.evaluate(102.5)
    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.reason is InterTrialReason.TARGET_REACHED
    assert evaluation.lateness_seconds == 0.0
    assert evaluation.was_late is False


def test_late_readiness_sends_immediately_and_reports_lateness():
    policy = _policy()
    policy.arm(100.0)
    evaluation = policy.evaluate(103.25)
    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.reason is InterTrialReason.READY_LATE
    assert evaluation.lateness_seconds == 0.75
    assert evaluation.remaining_seconds == 0.0
    assert evaluation.was_late is True


def test_lateness_is_reported_at_millisecond_resolution():
    policy = _policy(target_seconds=1.0)
    policy.arm(0.0)
    assert policy.evaluate(1.0004).lateness_seconds == 0.0
    policy.arm(0.0)
    assert policy.evaluate(1.002).lateness_seconds == 0.002


def test_an_unarmed_policy_never_invents_a_wait():
    policy = _policy()
    evaluation = policy.evaluate(100.0)
    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.reason is InterTrialReason.NOT_ARMED


def test_disarming_clears_the_measurement():
    policy = _policy()
    policy.arm(100.0)
    policy.disarm()
    assert policy.is_armed is False
    assert policy.evaluate(100.1).reason is InterTrialReason.NOT_ARMED


def test_retries_bypass_the_target_by_default():
    policy = _policy()
    policy.arm(100.0)
    evaluation = policy.evaluate(100.1, is_retry=True)
    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.reason is InterTrialReason.RETRY_BYPASS


def test_retries_can_opt_into_the_target():
    policy = _policy(apply_to_retries=True)
    policy.arm(100.0)
    evaluation = policy.evaluate(100.1, is_retry=True)
    assert evaluation.decision is InterTrialDecision.WAIT


def test_rearming_restarts_the_measurement():
    policy = _policy()
    policy.arm(100.0)
    assert policy.evaluate(102.5).reason is InterTrialReason.TARGET_REACHED
    policy.arm(200.0)
    assert policy.evaluate(201.0).decision is InterTrialDecision.WAIT


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_times_are_rejected(value):
    policy = _policy()
    with pytest.raises(ValueError, match="must be finite"):
        policy.arm(value)
    policy.arm(100.0)
    with pytest.raises(ValueError, match="must be finite"):
        policy.evaluate(value)


def test_records_are_plain_and_carry_the_reason():
    policy = _policy()
    policy.arm(100.0)
    record = policy.evaluate(101.0).to_record()
    assert record["decision"] == "wait"
    assert record["reason"] == "target_not_reached"
    assert record["remaining_seconds"] == 1.5
    assert policy.configuration.to_record()["target_seconds"] == 2.5
