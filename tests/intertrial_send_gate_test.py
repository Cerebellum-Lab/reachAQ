"""Tests for holding the next automatic send until the inter-trial target.

Three failures matter more than the arithmetic:

  * a held send that is never released would stall the session outright, so
    every path out of the hold is exercised, including a failing re-check;
  * an already-late trial must not be delayed further, or the drift compounds;
  * releasing this hold must not release a hold another subsystem is keeping.
"""

import pytest

from tools.acquisition.model.intertrial_send_gate import BLOCK_NAME, InterTrialSendGate
from tools.acquisition.model.intertrial_timing import (
    InterTrialDecision,
    InterTrialReason,
    InterTrialTimingConfiguration,
    InterTrialTimingPolicy,
)
from tools.acquisition.model.send_block_reasons import SendBlockReasons


class _ManualTimer:
    def __init__(self):
        self._callback = None
        self.deadline = None
        self.cancelled = False

    def schedule(self, deadline_perf_time, callback):
        self.deadline = deadline_perf_time
        self._callback = callback

    def cancel(self):
        self.cancelled = True
        self._callback = None
        return True

    def fire(self):
        callback, self._callback = self._callback, None
        assert callback is not None, "timer was not scheduled"
        callback(self.deadline, 0.0)


class _Harness:
    def __init__(self, enabled=True, target_seconds=2.0, apply_to_retries=False):
        self.now = 100.0
        self.timers = []
        self.observed = []
        self.published = []
        self.reasons = SendBlockReasons(publish=self.published.append)
        configuration = InterTrialTimingConfiguration(
            enabled=enabled,
            target_seconds=target_seconds if enabled else None,
            apply_to_retries=apply_to_retries,
        )
        self.gate = InterTrialSendGate(
            InterTrialTimingPolicy(configuration),
            self.reasons,
            timer_factory=self._make_timer,
            clock=lambda: self.now,
            observe=self.observed.append,
        )

    def _make_timer(self):
        timer = _ManualTimer()
        self.timers.append(timer)
        return timer

    @property
    def timer(self):
        return self.timers[-1]


def test_disabled_timing_never_holds_a_send():
    """Wiring this unconditionally must change nothing until an operator opts in."""
    harness = _Harness(enabled=False)
    evaluation = harness.gate.arm(100.0)

    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.reason is InterTrialReason.DISABLED
    assert harness.gate.is_holding is False
    assert harness.timers == []


def test_arming_early_holds_the_send_and_schedules_the_release():
    harness = _Harness(target_seconds=2.0)
    evaluation = harness.gate.arm(100.0)

    assert evaluation.decision is InterTrialDecision.WAIT
    assert harness.gate.is_holding is True
    assert "inter-trial target" in harness.reasons.render()
    assert harness.timer.deadline == pytest.approx(102.0)


def test_the_hold_is_released_when_the_target_elapses():
    harness = _Harness(target_seconds=2.0)
    harness.gate.arm(100.0)

    harness.now = 102.0
    harness.timer.fire()

    assert harness.gate.is_holding is False
    assert harness.reasons.render() == ""


def test_a_late_trial_is_released_immediately_and_not_delayed_further():
    """Padding an already-late trial compounds the drift instead of fixing it."""
    harness = _Harness(target_seconds=2.0)
    harness.now = 105.0  # already 5 s past the retrieval
    evaluation = harness.gate.arm(100.0)

    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.was_late is True
    assert harness.gate.is_holding is False
    assert harness.timers == []


def test_the_lateness_is_reported_as_evidence():
    harness = _Harness(target_seconds=2.0)
    harness.now = 105.0
    harness.gate.arm(100.0)

    assert harness.observed, "the evaluation must be reported"
    assert harness.observed[-1].lateness_seconds == pytest.approx(3.0)


def test_releasing_this_hold_leaves_another_subsystem_holding():
    """The reason SendBlockReasons exists, asserted end to end."""
    harness = _Harness(target_seconds=2.0)
    harness.reasons.set("analysis", "waiting for retried pellet trial analysis")
    harness.gate.arm(100.0)
    assert harness.reasons.names == ("analysis", BLOCK_NAME)

    harness.now = 102.0
    harness.timer.fire()

    assert harness.gate.is_holding is False
    assert harness.reasons.is_blocked is True
    assert harness.reasons.render() == "waiting for retried pellet trial analysis"


def test_release_clears_the_hold_and_cancels_the_timer():
    harness = _Harness(target_seconds=2.0)
    harness.gate.arm(100.0)
    timer = harness.timer

    harness.gate.release()

    assert harness.gate.is_holding is False
    assert timer.cancelled is True
    assert harness.gate.policy.is_armed is False


def test_re_arming_cancels_the_previous_timer():
    """A stale timer must not release a hold that belongs to the next interval."""
    harness = _Harness(target_seconds=2.0)
    harness.gate.arm(100.0)
    first = harness.timer

    harness.now = 100.5
    harness.gate.arm(100.5)

    assert first.cancelled is True
    assert harness.timer is not first
    assert harness.timer.deadline == pytest.approx(102.5)


def test_a_failing_recheck_releases_rather_than_stalling_the_session():
    """A permanently held send is worse than an inaccurate interval."""
    harness = _Harness(target_seconds=2.0)
    harness.gate.arm(100.0)

    def explode():
        raise RuntimeError("clock source died")

    harness.gate._clock = explode
    harness.timer.fire()

    assert harness.gate.is_holding is False


def test_a_retry_bypasses_the_target_by_default():
    harness = _Harness(target_seconds=2.0)
    harness.gate.arm(100.0)

    evaluation = harness.gate.evaluate(is_retry=True)

    assert evaluation.decision is InterTrialDecision.SEND_NOW
    assert evaluation.reason is InterTrialReason.RETRY_BYPASS


def test_a_retry_can_be_made_subject_to_the_target():
    harness = _Harness(target_seconds=2.0, apply_to_retries=True)
    harness.gate.arm(100.0)

    evaluation = harness.gate.evaluate(is_retry=True)

    assert evaluation.decision is InterTrialDecision.WAIT
