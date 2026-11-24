from types import SimpleNamespace
from unittest import mock

from reachaq.behavior.pellet_trial import TrialOutcome
from reachaq.behavior.training import TrainingPlan, TrainingProgressState, parse_plan

from reachaq.app.model.trial_protocol_runner import TrialProtocolRunner


def make_plan(*, advance_after=None, fallback_after=None, phases=2, pause=None):
    def phase(index):
        document = {"phaseId": f"p{index}", "name": f"Phase {index}"}
        if advance_after is not None:
            document["advance"] = {"metric": "trials", "op": ">=", "value": advance_after}
        if fallback_after is not None:
            document["fallback"] = {"metric": "consecutiveFailures", "op": ">=", "value": fallback_after}
        if pause is not None:
            document["trialActions"] = [{"action": "pause", "seconds": pause}]
        return document

    return TrainingPlan(parse_plan({
        "schemaVersion": 1,
        "planId": "runner-plan",
        "phases": [phase(i) for i in range(phases)],
    }))


def make_algorithm():
    return SimpleNamespace(
        pellet_delivery_enabled=True,
        pellet_cover_enabled=False,
        intersession_pellet_shift_enabled=True,
    )


def attached(plan, **kwargs):
    runner = TrialProtocolRunner(**kwargs)
    runner.attach(plan, make_algorithm(), object())
    return runner


def test_completed_pellet_trial_advances_progress_once():
    plan = make_plan()
    updates = []
    plan.progress_updated += lambda: updates.append(1)
    runner = attached(plan)

    assert runner.begin_trial("10.1")
    assert not runner.begin_trial("10.1")
    assert runner.finish_trial("10.1", TrialOutcome.SUCCESS, reach_count=2, success_count=1,
                               consumption_count=1)
    assert not runner.finish_trial("10.1", TrialOutcome.SUCCESS)
    progress = plan.current_phase.progress
    assert (progress.session_count, progress.successes, progress.total_reaches) == (1, 1, 2)
    assert progress.pellets_consumed == 1
    assert updates == [1]


def test_hardware_error_never_counts_or_evaluates_protocol_trial():
    plan = make_plan(advance_after=1)
    runner = attached(plan, automatic_advance=True)
    runner.begin_trial("3.2")

    assert not runner.finish_trial("3.2", TrialOutcome.HARDWARE_ERROR)
    assert plan.current_phase.progress.session_count == 0
    assert plan.current_phase.phase_id == "p0"


def test_pellet_missing_counts_a_trial_without_a_pellet():
    plan = make_plan()
    runner = attached(plan)
    assert runner.record_trial_outcome("4.1", TrialOutcome.PELLET_MISSING)
    progress = plan.current_phase.progress
    assert (progress.session_count, progress.pellets_presented) == (1, 0)


def test_automatic_advance_and_fallback_change_phase():
    plan = make_plan(advance_after=1, fallback_after=1)
    runner = attached(plan, automatic_advance=True)
    assert runner.record_trial_outcome("1.1", TrialOutcome.FAILURE)  # advance is checked first
    assert plan.current_phase.phase_id == "p1"


def test_manual_mode_only_reports():
    plan = make_plan(advance_after=1)
    runner = attached(plan)
    runner.record_trial_outcome("1.1", TrialOutcome.SUCCESS)
    assert plan.current_phase.phase_id == "p0"


def test_terminal_advance_marks_protocol_complete():
    completed = mock.Mock()
    plan = make_plan(advance_after=1, phases=1)
    runner = attached(plan, automatic_advance=True, on_protocol_complete=completed)
    runner.begin_trial("7.1")

    assert runner.finish_trial("7.1", TrialOutcome.SUCCESS)
    assert runner.protocol_complete
    assert plan.progress_state is TrainingProgressState.Complete
    completed.assert_called_once_with()


def test_trial_recorded_hook_sees_the_new_phase():
    seen = []
    plan = make_plan(advance_after=1, pause=2.5)
    runner = attached(plan, automatic_advance=True,
                      on_trial_recorded=lambda label: seen.append((label, plan.current_phase.phase_id)))
    runner.record_trial_outcome("2.1", TrialOutcome.SUCCESS)
    runner.record_trial_outcome("2.2", TrialOutcome.ABORTED)
    assert seen == [("2.1", "p1")]
    assert runner.next_trial_directive("k").pause_seconds == 2.5


def test_detach_drops_the_plan_and_the_active_trial():
    plan = make_plan()
    algorithm = make_algorithm()
    runner = TrialProtocolRunner()
    runner.attach(plan, algorithm, object())
    runner.begin_trial("1.1")

    runner.detach()

    assert runner.plan is None and runner.active_attempt_label is None
    assert runner.next_trial_directive("k") is None
