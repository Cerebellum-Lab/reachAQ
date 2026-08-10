import pytest

from tools.acquisition.model.session_stop_policy import (
    SessionStopConfiguration,
    SessionStopDecision,
    SessionStopPolicy,
    SessionStopReason,
)


def test_duration_threshold_finishes_active_trial_without_error():
    policy = SessionStopPolicy(
        SessionStopConfiguration(duration_seconds=60, drain_timeout_seconds=15),
    )
    policy.start(100.0)

    draining = policy.evaluate(
        160.0,
        trial_count=3,
        protocol_complete=False,
        trial_active=True,
    )
    stopped = policy.evaluate(
        164.0,
        trial_count=4,
        protocol_complete=False,
        trial_active=False,
    )

    assert draining.decision is SessionStopDecision.FINISH_ACTIVE_TRIAL
    assert draining.reason is SessionStopReason.DURATION_LIMIT
    assert not draining.is_error
    assert stopped.decision is SessionStopDecision.STOP
    assert not stopped.is_error


def test_only_drain_timeout_is_an_error():
    policy = SessionStopPolicy(
        SessionStopConfiguration(trial_limit=2, drain_timeout_seconds=15),
    )
    policy.start(10.0)
    policy.evaluate(
        20.0,
        trial_count=2,
        protocol_complete=False,
        trial_active=True,
    )

    before_timeout = policy.evaluate(
        34.999,
        trial_count=2,
        protocol_complete=False,
        trial_active=True,
    )
    timed_out = policy.evaluate(
        35.0,
        trial_count=2,
        protocol_complete=False,
        trial_active=True,
    )

    assert before_timeout.decision is SessionStopDecision.FINISH_ACTIVE_TRIAL
    assert not before_timeout.is_error
    assert timed_out.decision is SessionStopDecision.TIMEOUT_ERROR
    assert timed_out.is_error
    assert timed_out.timeout_seconds == 15.0


def test_protocol_completion_stops_immediately_without_active_trial():
    policy = SessionStopPolicy(
        SessionStopConfiguration(stop_on_protocol_complete=True),
    )
    policy.start(1.0)

    result = policy.evaluate(
        2.0,
        trial_count=0,
        protocol_complete=True,
        trial_active=False,
    )

    assert result.decision is SessionStopDecision.STOP
    assert result.reason is SessionStopReason.PROTOCOL_COMPLETE


def test_simultaneous_conditions_are_persisted_with_stable_primary_reason():
    policy = SessionStopPolicy(
        SessionStopConfiguration(
            duration_seconds=10,
            trial_limit=5,
            stop_on_protocol_complete=True,
        ),
    )
    policy.start(100.0)

    result = policy.evaluate(
        110.0,
        trial_count=5,
        protocol_complete=True,
        trial_active=False,
    )

    assert result.reason is SessionStopReason.DURATION_LIMIT
    assert result.triggered_reasons == (
        SessionStopReason.DURATION_LIMIT,
        SessionStopReason.TRIAL_LIMIT,
        SessionStopReason.PROTOCOL_COMPLETE,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"duration_seconds": 0}, "duration"),
        ({"trial_limit": 0}, "Trial target"),
        ({"drain_timeout_seconds": 0}, "drain timeout"),
    ),
)
def test_invalid_limits_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SessionStopConfiguration(**kwargs)


def test_evaluation_requires_started_session():
    with pytest.raises(RuntimeError, match="has not been started"):
        SessionStopPolicy().evaluate(
            1.0,
            trial_count=0,
            protocol_complete=False,
            trial_active=False,
        )
