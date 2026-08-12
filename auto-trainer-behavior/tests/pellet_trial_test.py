import pytest
from types import SimpleNamespace

from autotrainer.core.reach_event import (
    ReachEvent,
    ReachEventMethod,
    ReachEventOutcome,
)

from autotrainer.behavior.pellet_trial import (
    AttemptAssignmentPolicy,
    HardwareErrorKind,
    PelletTrialLedger,
    TrialAccountingConfiguration,
    TrialCountBasis,
    TrialOutcome,
)


def test_user_facing_policy_names_are_plain_language():
    assert AttemptAssignmentPolicy.RETRY_WITHIN_TRIAL.display_name == (
        "Retry within the same trial"
    )
    assert AttemptAssignmentPolicy.EVERY_ATTEMPT_IS_TRIAL.display_name == (
        "Count every attempt as a new trial"
    )
    assert AttemptAssignmentPolicy.SUCCESSFUL_PRESENTATIONS_ONLY.display_name == (
        "Count only successful pellet presentations"
    )
    assert TrialCountBasis.PRESENTED.display_name == "Pellets presented"


def test_hardware_error_retries_same_logical_trial_and_never_counts():
    ledger = PelletTrialLedger("session001")

    first = ledger.begin_send(10.0, 100.0, operation_id="send-1")
    assert first.attempt_label == "1.1"
    failed = ledger.finalize_hardware_error(
        HardwareErrorKind.MOTOR_FAILURE,
        10.1,
        100.1,
        error="Y motor stalled",
    )

    assert failed.outcome is TrialOutcome.HARDWARE_ERROR
    assert ledger.planned_trial_id == 1
    assert ledger.summary() == {
        "physical_attempts": 1,
        "hardware_errors": 1,
        "incomplete_attempts": 0,
        "pending_analysis_attempts": 0,
        "trials_started": 0,
        "pellets_presented": 0,
        "trials_completed": 0,
        "scored_trials": 0,
        "configured_trial_count": 0,
        "reaches": 0,
        "successful_reaches": 0,
        "pellets_consumed": 0,
        "unassigned_reaches": 0,
        "unassigned_successful_reaches": 0,
        "unassigned_pellets_consumed": 0,
    }

    retry = ledger.begin_send(10.2, 100.2, operation_id="send-2")
    assert retry.attempt_label == "1.2"
    ledger.acknowledge_presentation(10.3, 100.3)
    completed = ledger.finalize(TrialOutcome.SUCCESS, 11.0, 101.0)

    assert completed.logical_trial_complete
    assert ledger.count(TrialCountBasis.STARTED) == 1
    assert ledger.count(TrialCountBasis.PRESENTED) == 1
    assert ledger.count(TrialCountBasis.COMPLETED) == 1
    assert ledger.count(TrialCountBasis.SCORED) == 1


def test_behavioral_retry_stays_within_logical_trial_by_default():
    ledger = PelletTrialLedger("session001")
    ledger.begin_send(1.0, 11.0)
    ledger.acknowledge_presentation(1.1, 11.1)
    first = ledger.finalize(
        TrialOutcome.PELLET_MISSING,
        2.0,
        12.0,
        retry=True,
    )
    second = ledger.begin_send(2.1, 12.1)

    assert first.attempt_label == "1.1"
    assert not first.logical_trial_complete
    assert second.attempt_label == "1.2"

    ledger.acknowledge_presentation(2.2, 12.2)
    ledger.finalize(TrialOutcome.FAILURE, 3.0, 13.0)
    assert ledger.summary()["physical_attempts"] == 2
    assert ledger.summary()["trials_completed"] == 1


def test_every_behavioral_attempt_can_be_a_new_trial():
    ledger = PelletTrialLedger(
        "session001",
        TrialAccountingConfiguration(
            assignment_policy=AttemptAssignmentPolicy.EVERY_ATTEMPT_IS_TRIAL,
        ),
    )
    ledger.begin_send(1.0, 11.0)
    ledger.acknowledge_presentation(1.1, 11.1)
    first = ledger.finalize(TrialOutcome.FAILURE, 2.0, 12.0, retry=True)
    second = ledger.begin_send(2.1, 12.1)

    assert first.logical_trial_complete
    assert first.attempt_label == "1.1"
    assert second.attempt_label == "2.1"


def test_successful_presentation_only_leaves_failed_attempt_unindexed():
    ledger = PelletTrialLedger(
        "session001",
        TrialAccountingConfiguration(
            assignment_policy=(
                AttemptAssignmentPolicy.SUCCESSFUL_PRESENTATIONS_ONLY
            ),
        ),
    )
    first = ledger.begin_send(1.0, 11.0)
    assert first.trial_id is None
    ledger.finalize(TrialOutcome.INCOMPLETE, 1.5, 11.5, retry=True)

    second = ledger.begin_send(2.0, 12.0)
    assert second.trial_id is None
    presented = ledger.acknowledge_presentation(2.1, 12.1)
    assert presented.attempt_label == "1.1"
    ledger.finalize(TrialOutcome.SUCCESS, 3.0, 13.0)

    records = ledger.to_records()
    assert records[0]["trial_id"] is None
    assert records[0]["attempt_label"] == "unindexed.1"
    assert ledger.count() == 1


def test_hardware_errors_cannot_be_configured_as_counted_outcomes():
    with pytest.raises(ValueError, match="Hardware errors"):
        TrialAccountingConfiguration(
            counted_outcomes=frozenset({TrialOutcome.HARDWARE_ERROR}),
        )


def test_attempt_can_only_be_acknowledged_and_finalized_once():
    ledger = PelletTrialLedger("session001")
    ledger.begin_send(1.0, 11.0)
    ledger.acknowledge_presentation(1.1, 11.1)
    with pytest.raises(RuntimeError, match="already presented"):
        ledger.acknowledge_presentation(1.2, 11.2)
    ledger.finalize(TrialOutcome.SUCCESS, 2.0, 12.0)
    with pytest.raises(RuntimeError, match="No pellet-send attempt"):
        ledger.finalize(TrialOutcome.SUCCESS, 2.1, 12.1)


def test_capture_window_can_close_before_offline_outcome_is_known():
    ledger = PelletTrialLedger("session001")
    ledger.begin_send(1.0, 11.0)
    ledger.acknowledge_presentation(1.1, 11.1)

    pending = ledger.close_active_for_analysis(2.0, 12.0)

    assert pending.outcome is TrialOutcome.PENDING_ANALYSIS
    assert not pending.is_finalized
    assert ledger.summary()["pending_analysis_attempts"] == 1
    assert ledger.count(TrialCountBasis.PRESENTED) == 1
    assert ledger.count(TrialCountBasis.COMPLETED) == 1

    final = ledger.finalize_pending(
        1,
        1,
        TrialOutcome.SUCCESS,
        2.0,
        12.0,
    )
    assert final.is_finalized
    assert ledger.summary()["pending_analysis_attempts"] == 0
    assert ledger.count(TrialCountBasis.COMPLETED) == 1
    with pytest.raises(RuntimeError, match="not pending analysis"):
        ledger.finalize_pending(
            1,
            1,
            TrialOutcome.FAILURE,
            2.1,
            12.1,
        )


def test_runtime_behavioral_retry_reserves_next_attempt_label_before_analysis():
    ledger = PelletTrialLedger("session001")
    ledger.begin_send(1.0, 11.0)
    ledger.acknowledge_presentation(1.1, 11.1)
    first = ledger.close_active_for_analysis(2.0, 12.0, retry=True)
    second = ledger.begin_send(2.1, 12.1)

    assert first.attempt_label == "1.1"
    assert not first.logical_trial_complete
    assert second.attempt_label == "1.2"


def test_analysis_fields_are_persisted_on_pending_attempt():
    ledger = PelletTrialLedger("session001")
    ledger.begin_send(
        1.0,
        11.0,
        pellet_position={"x": 1.0, "y": 2.0, "z": 3.0},
        planned_shift={"x": 0.1, "y": 0.2, "z": 0.3},
        applied_shift={"x": 0.0, "y": 0.2, "z": 0.0},
        protocol_context={"protocol_id": "reach", "phase_id": "phase-1"},
    )
    ledger.acknowledge_presentation(1.1, 11.1)
    ledger.close_active_for_analysis(2.0, 12.0)

    final = ledger.finalize_pending(
        1,
        1,
        TrialOutcome.SUCCESS,
        2.5,
        12.5,
        reach_count=2,
        success_count=1,
        consumption_count=1,
        reach_event_indices=(3, 4),
        tone_references=({"context": "tone-1"},),
        laser_references=({"channel": "left"},),
        pellet_presence="present",
        pellet_misplacement="not_misplaced",
        analysis_window_start_perf=1.2,
        analysis_window_end_perf=2.0,
        tracking_coverage=0.98,
        tracking_missing_frame_ids=(8,),
        tracking_interpolated_points=1,
        analysis_duration_seconds=0.03,
        recommended_shift={"x": 0.1, "y": 0.0, "z": -0.1},
    )

    assert final.reach_count == 2
    assert final.success_count == 1
    assert final.consumption_count == 1
    record = ledger.to_records()[0]
    assert record["pellet_position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert record["protocol_context"]["phase_id"] == "phase-1"
    assert record["reach_event_indices"] == (3, 4)
    assert record["pellet_presence"] == "present"
    assert record["tracking_missing_frame_ids"] == (8,)
    assert record["recommended_shift"]["z"] == -0.1


def test_presented_attempt_without_reach_is_not_classified_as_missing_pellet():
    ledger = PelletTrialLedger("session001")
    ledger.begin_send(101.0, 1001.0)
    ledger.acknowledge_presentation(101.1, 1001.1)
    ledger.close_active_for_analysis(102.0, 1002.0)
    result = SimpleNamespace(
        reach_events=(),
        other_events=(),
        total_reaches=0,
        successful_reaches=0,
        food_consumed=0,
    )
    finalized = ledger.reconcile_analysis(
        result,
        recording_start_perf_time=100.0,
        frame_rate=150.0,
        finalized_perf_time=103.0,
        finalized_wall_time=1003.0,
    )
    assert finalized[0].outcome is TrialOutcome.NO_REACH
    assert finalized[0].outcome is not TrialOutcome.PELLET_MISSING


def test_session_frame_analysis_is_reconciled_to_attempt_windows_once():
    ledger = PelletTrialLedger("session001")
    ledger.begin_send(101.0, 1001.0, operation_id="send-1")
    ledger.acknowledge_presentation(101.1, 1001.1)
    ledger.close_active_for_analysis(103.0, 1003.0, retry=True)
    ledger.begin_send(103.0, 1003.0, operation_id="send-2")
    ledger.acknowledge_presentation(103.1, 1003.1)
    ledger.close_active_for_analysis(105.0, 1005.0)
    result = SimpleNamespace(
        reach_events=(
            ReachEvent(
                init=150,
                end=165,
                method=ReachEventMethod.RIGHT_HAND,
                outcome=ReachEventOutcome.MISSED,
            ),
            ReachEvent(
                init=600,
                end=620,
                method=ReachEventMethod.RIGHT_HAND,
                outcome=ReachEventOutcome.EATEN,
            ),
        ),
        other_events=(),
        total_reaches=2,
        successful_reaches=1,
        food_consumed=1,
    )

    finalized = ledger.reconcile_analysis(
        result,
        recording_start_perf_time=100.0,
        frame_rate=150.0,
        finalized_perf_time=106.0,
        finalized_wall_time=1006.0,
        tone_references=({"eventPerfTime": 101.5, "channel": "tone1"},),
        laser_references=({"perf_time": 103.5, "channel": "left"},),
    )

    assert [attempt.attempt_label for attempt in finalized] == ["1.1", "1.2"]
    assert finalized[0].outcome is TrialOutcome.FAILURE
    assert finalized[0].reach_event_indices == (0,)
    assert finalized[0].tone_references[0]["channel"] == "tone1"
    assert finalized[1].outcome is TrialOutcome.SUCCESS
    assert finalized[1].reach_event_indices == (1,)
    assert finalized[1].laser_references[0]["channel"] == "left"
    assert ledger.summary()["scored_trials"] == 1
    assert ledger.summary()["reaches"] == 2
    assert ledger.summary()["unassigned_reaches"] == 0
    with pytest.raises(RuntimeError, match="not pending analysis"):
        ledger.finalize_pending(1, 1, TrialOutcome.SUCCESS, 107.0, 1007.0)
