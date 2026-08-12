import threading

import pytest

from autotrainer.behavior import TrialOutcome
from autotrainer.core.pose_elements import SceneElement
from tools.acquisition.model.intertrial_analysis import (
    IntertrialAnalysisCoordinator,
    IntertrialAnalysisRequest,
    PelletMisplacement,
    PelletPresence,
    analyze_tracking_window,
    classify_pellet_state,
    tracking_request_from_record,
    tracking_request_record,
)
from tools.acquisition.model.live_tracking_buffer import (
    LiveTrackingSample,
    TrackingLocation,
    TrackingOffset,
    TrackingWindow,
)


def sample(frame_id, hand_x, *, pellet_seen=True, perf=None, triangle_distance=5.0):
    perf = frame_id / 100.0 if perf is None else perf
    return LiveTrackingSample(
        sequence=frame_id,
        primary_frame_ids=(frame_id,),
        primary_frame_perf_times=(perf,),
        source_start_perf=perf,
        source_end_perf=perf,
        processing_perf=perf + 0.01,
        pellet_seen=pellet_seen,
        locations_3d=(
            TrackingLocation(str(SceneElement.Pellet), 0.0, 0.0, 0.0),
            TrackingLocation(str(SceneElement.R_Hand), hand_x, 0.0, 0.0),
        ),
        offsets_3d=(
            TrackingOffset(
                str(SceneElement.Triangle),
                str(SceneElement.Pellet),
                triangle_distance,
                0.0,
                0.0,
            ),
        ),
    )


def window(samples):
    return TrackingWindow(
        start_perf=samples[0].source_perf,
        end_perf=samples[-1].source_perf,
        samples=tuple(samples),
        expected_frames=len(samples),
        observed_frames=len(samples),
        missing_frame_ids=(),
        duplicate_frame_ids=(),
        monotonic=True,
    )


def request(samples, *, presence=None, generation=1):
    tracking_window = window(samples)
    evidence = presence or classify_pellet_state(
        tracking_window,
        expected_triangle_pellet_distance=5.0,
        misplacement_threshold=1.0,
    )
    return IntertrialAnalysisRequest(
        generation=generation,
        session_id="session001",
        trial_id=1,
        attempt_id=1,
        operation_id="send-1",
        window=tracking_window,
        pellet_state=evidence,
    )


def test_live_pellet_state_is_finalized_without_trajectory_analysis():
    evidence = classify_pellet_state(
        window([sample(index, 30, triangle_distance=7.0) for index in range(5)]),
        expected_triangle_pellet_distance=5.0,
        misplacement_threshold=1.0,
    )
    assert evidence.presence is PelletPresence.PRESENT
    assert evidence.misplacement is PelletMisplacement.MISPLACED

    missing = classify_pellet_state(
        window([sample(index, 30, pellet_seen=False) for index in range(5)]),
        expected_triangle_pellet_distance=5.0,
        misplacement_threshold=1.0,
    )
    assert missing.presence is PelletPresence.MISSING


def test_analysis_classifies_no_reach_without_calling_it_missing():
    result = analyze_tracking_window(
        request([sample(index, 30) for index in range(10)])
    )
    assert result.outcome is TrialOutcome.NO_REACH
    assert result.reach_count == 0


def test_analysis_uses_existing_tracking_for_reach_and_consumption():
    distances = (30, 24, 18, 12, 5, 9, 18, 25)
    samples = [
        sample(index, distance, pellet_seen=index < 5)
        for index, distance in enumerate(distances)
    ]
    result = analyze_tracking_window(request(samples))
    assert result.outcome is TrialOutcome.SUCCESS
    assert result.reach_count == 1
    assert result.success_count == 1
    assert result.consumption_count == 1


def test_failed_reach_recommendation_has_one_average_per_axis():
    distances = (30, 24, 18, 12, 5, 9, 18, 25)
    result = analyze_tracking_window(
        request([sample(index, distance) for index, distance in enumerate(distances)])
    )

    assert result.outcome is TrialOutcome.FAILURE
    assert result.recommended_shift is not None
    assert len(result.recommended_shift) == 3
    assert result.recommended_shift == pytest.approx((5.0, 0.0, 0.0))


def test_short_internal_gaps_are_interpolated_but_long_gaps_are_reported():
    samples = [sample(0, 30, perf=0.0), sample(3, 10, perf=0.03), sample(20, 30, perf=0.2)]
    result = analyze_tracking_window(request(samples))
    assert result.interpolated_points == 2
    assert result.long_gap_count == 1


def test_coordinator_rejects_stale_results_after_abort():
    entered = threading.Event()
    release = threading.Event()
    received = []

    def slow_analyzer(value):
        entered.set()
        assert release.wait(2)
        return analyze_tracking_window(value)

    coordinator = IntertrialAnalysisCoordinator(received.append, analyzer=slow_analyzer)
    coordinator.begin_session(1, "session001")
    assert coordinator.submit(request([sample(index, 30) for index in range(5)]))
    assert entered.wait(1)
    coordinator.cancel_session()
    release.set()
    assert coordinator.wait_for_idle(2)
    assert received == []
    assert coordinator.close()


def test_coordinator_queue_is_bounded_and_timing_starts_as_estimating():
    entered = threading.Event()
    release = threading.Event()

    def slow_analyzer(value):
        entered.set()
        release.wait(2)
        return analyze_tracking_window(value)

    coordinator = IntertrialAnalysisCoordinator(
        lambda result: None,
        queue_capacity=1,
        analyzer=slow_analyzer,
    )
    coordinator.begin_session(1, "session001")
    first = request([sample(index, 30) for index in range(5)])
    assert coordinator.timing_estimate(2).display_text == "estimating"
    assert coordinator.submit(first)
    assert entered.wait(1)
    assert coordinator.submit(first)
    assert not coordinator.submit(first)
    release.set()
    assert coordinator.wait_for_idle(2)
    assert coordinator.timing_estimate(2).projected_seconds is not None
    assert coordinator.close()


def test_tracking_record_round_trip_retains_generation_and_frame_identity():
    original = request([sample(index, 30) for index in range(5)])
    restored = tracking_request_from_record(tracking_request_record(original))
    assert restored.generation == original.generation
    assert restored.operation_id == original.operation_id
    assert restored.window.samples[2].primary_frame_ids == (2,)
    assert restored.pellet_state.presence is PelletPresence.PRESENT
