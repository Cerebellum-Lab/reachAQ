import json
import math

import pytest

from tools.acquisition.model.session_telemetry import SessionTelemetry


@pytest.fixture
def telemetry():
    item = SessionTelemetry()
    item.begin(started_perf=100.0)
    return item


def test_counters_start_at_zero(telemetry):
    assert telemetry.frames_acquired == 0
    assert telemetry.dropped_frames == 0
    assert telemetry.frames_inferenced == 0
    assert telemetry.has_dropped_frames is False
    assert math.isnan(telemetry.inference_mean_ms)


def test_acquired_is_the_max_across_cameras_not_the_sum(telemetry):
    """Two synchronised cameras deliver the same frames; one pose consumes both.

    Summing them would double the denominator and halve every percentage.
    """
    telemetry.record_capture(0, acquired=1000, dropped=0)
    telemetry.record_capture(1, acquired=998, dropped=0)
    assert telemetry.frames_acquired == 1000


def test_dropped_is_summed_across_cameras(telemetry):
    """A drop on either camera is a hole in that camera's recording."""
    telemetry.record_capture(0, acquired=1000, dropped=3)
    telemetry.record_capture(1, acquired=1000, dropped=2)
    assert telemetry.dropped_frames == 5
    assert telemetry.has_dropped_frames is True
    assert telemetry.dropped_by_camera == {0: 3, 1: 2}


def test_capture_totals_are_absolute_not_cumulative(telemetry):
    """Reports are absolute totals, so a repeat must not double-count.

    The capture process emits periodically; a duplicated or reordered message
    would silently corrupt a running sum.
    """
    telemetry.record_capture(0, acquired=500, dropped=2)
    telemetry.record_capture(0, acquired=500, dropped=2)
    assert telemetry.frames_acquired == 500
    assert telemetry.dropped_frames == 2

    telemetry.record_capture(0, acquired=900, dropped=4)
    assert telemetry.frames_acquired == 900
    assert telemetry.dropped_frames == 4


def test_inferenced_percentage(telemetry):
    telemetry.record_capture(0, acquired=1000, dropped=0)
    telemetry.record_inference(count=640, mean_ms=4.0, max_ms=9.0)
    assert telemetry.frames_inferenced == 640
    assert telemetry.inferenced_percent == pytest.approx(64.0)


def test_percentage_is_zero_before_any_frames(telemetry):
    telemetry.record_inference(count=10, mean_ms=4.0, max_ms=4.0)
    assert telemetry.inferenced_percent == 0.0


def test_mean_is_weighted_by_calls_in_each_window(telemetry):
    """A slow burst must not be averaged away by a later quiet window.

    900 calls at 4 ms then 100 at 20 ms is 5.6 ms, not the 12 ms an unweighted
    mean of the two windows would report.
    """
    telemetry.record_capture(0, acquired=1000, dropped=0)
    telemetry.record_inference(count=900, mean_ms=4.0, max_ms=5.0)
    telemetry.record_inference(count=1000, mean_ms=20.0, max_ms=31.0)
    assert telemetry.inference_mean_ms == pytest.approx(5.6)


def test_max_is_kept_across_windows(telemetry):
    telemetry.record_capture(0, acquired=100, dropped=0)
    telemetry.record_inference(count=50, mean_ms=4.0, max_ms=31.0)
    telemetry.record_inference(count=100, mean_ms=4.0, max_ms=7.0)
    assert telemetry.inference_max_ms == pytest.approx(31.0)


def test_nothing_is_recorded_outside_a_session():
    item = SessionTelemetry()
    item.record_capture(0, acquired=500, dropped=9)
    item.record_inference(count=100, mean_ms=4.0, max_ms=4.0)
    assert item.frames_acquired == 0
    assert item.dropped_frames == 0
    assert item.frames_inferenced == 0


def test_begin_clears_the_previous_session(telemetry):
    telemetry.record_capture(0, acquired=500, dropped=7)
    telemetry.record_inference(count=100, mean_ms=9.0, max_ms=40.0)
    telemetry.end(ended_perf=110.0)

    telemetry.begin(started_perf=200.0)
    assert telemetry.frames_acquired == 0
    assert telemetry.dropped_frames == 0
    assert telemetry.has_dropped_frames is False
    assert math.isnan(telemetry.inference_max_ms)


def test_elapsed_freezes_when_the_session_ends(telemetry):
    telemetry.end(ended_perf=137.5)
    assert telemetry.elapsed_seconds == pytest.approx(37.5)
    assert telemetry.elapsed_seconds == pytest.approx(37.5)


def test_end_is_idempotent(telemetry):
    telemetry.end(ended_perf=110.0)
    telemetry.end(ended_perf=999.0)
    assert telemetry.elapsed_seconds == pytest.approx(10.0)


def test_summary_is_json_safe_with_no_inference(telemetry):
    """NaN is not valid JSON; the summary must carry None instead."""
    telemetry.record_capture(0, acquired=10, dropped=0)
    telemetry.end(ended_perf=105.0)
    summary = telemetry.summary()
    assert summary["inferenceCallMeanMs"] is None
    assert summary["inferenceCallMaxMs"] is None
    assert summary["sensorToResultMeanMs"] is None
    assert summary["sensorToResultMaxMs"] is None
    json.dumps(summary)


def test_summary_contents(telemetry):
    telemetry.record_capture(0, acquired=1000, dropped=2)
    telemetry.record_capture(1, acquired=1000, dropped=1)
    telemetry.record_inference(count=500, mean_ms=4.0, max_ms=11.0)
    telemetry.end(ended_perf=160.0)
    summary = telemetry.summary()
    assert summary["elapsedSeconds"] == pytest.approx(60.0)
    assert summary["framesAcquired"] == 1000
    assert summary["framesDropped"] == 3
    assert summary["framesDroppedByCamera"] == {"0": 2, "1": 1}
    assert summary["framesInferenced"] == 500
    assert summary["inferencedPercent"] == pytest.approx(50.0)
    assert summary["inferenceCallMeanMs"] == pytest.approx(4.0)
    assert summary["inferenceCallMaxMs"] == pytest.approx(11.0)
    json.dumps(summary)


def test_observers_are_notified_when_counters_move(telemetry):
    seen = []
    telemetry.property_changed += lambda name, new, old: seen.append(name)
    telemetry.record_capture(0, acquired=100, dropped=1)
    assert SessionTelemetry.ACQUIRED_PROP in seen
    assert SessionTelemetry.DROPPED_PROP in seen


def test_no_notification_when_nothing_changed(telemetry):
    telemetry.record_capture(0, acquired=100, dropped=0)
    seen = []
    telemetry.property_changed += lambda name, new, old: seen.append(name)
    telemetry.record_capture(0, acquired=100, dropped=0)
    assert seen == []


def test_call_time_and_sensor_to_result_are_tracked_separately(telemetry):
    """The call is what the model costs; sensor-to-result adds the queue wait.

    A model can look fast on the first and miss the deadline on the second, so
    conflating them would hide exactly the failure the 5 ms target guards
    against.
    """
    telemetry.record_capture(0, acquired=1000, dropped=0)
    telemetry.record_inference(
        count=500, mean_ms=3.6, max_ms=4.3,
        sensor_to_result_mean_ms=8.1, sensor_to_result_max_ms=19.2)
    assert telemetry.inference_mean_ms == pytest.approx(3.6)
    assert telemetry.inference_max_ms == pytest.approx(4.3)
    assert telemetry.sensor_to_result_mean_ms == pytest.approx(8.1)
    assert telemetry.sensor_to_result_max_ms == pytest.approx(19.2)


def test_sensor_to_result_absent_leaves_the_call_figures_intact(telemetry):
    """An older pose process sends no end-to-end figure; that must not poison
    the call figures or produce a fabricated zero."""
    telemetry.record_capture(0, acquired=100, dropped=0)
    telemetry.record_inference(count=50, mean_ms=4.0, max_ms=5.0)
    assert telemetry.inference_mean_ms == pytest.approx(4.0)
    assert math.isnan(telemetry.sensor_to_result_mean_ms)
    assert math.isnan(telemetry.sensor_to_result_max_ms)


def test_sensor_to_result_mean_is_weighted_like_the_call_mean(telemetry):
    telemetry.record_capture(0, acquired=1000, dropped=0)
    telemetry.record_inference(count=900, mean_ms=4.0, max_ms=5.0,
                               sensor_to_result_mean_ms=6.0,
                               sensor_to_result_max_ms=9.0)
    telemetry.record_inference(count=1000, mean_ms=4.0, max_ms=5.0,
                               sensor_to_result_mean_ms=24.0,
                               sensor_to_result_max_ms=40.0)
    assert telemetry.sensor_to_result_mean_ms == pytest.approx(7.8)
    assert telemetry.sensor_to_result_max_ms == pytest.approx(40.0)


def test_summary_separates_the_two_timings(telemetry):
    telemetry.record_capture(0, acquired=100, dropped=0)
    telemetry.record_inference(count=100, mean_ms=3.5, max_ms=4.0,
                               sensor_to_result_mean_ms=7.5,
                               sensor_to_result_max_ms=12.0)
    telemetry.end(ended_perf=101.0)
    summary = telemetry.summary()
    assert summary["inferenceCallMeanMs"] == pytest.approx(3.5)
    assert summary["sensorToResultMeanMs"] == pytest.approx(7.5)
    assert summary["sensorToResultMaxMs"] == pytest.approx(12.0)
    json.dumps(summary)
def test_counts_are_relative_to_the_session_not_the_capture_process():
    """Capture starts at Running, recording starts later; only the gap counts.

    The processes report totals accumulated since they started, which on a real
    rig is tens of seconds before Record is pressed. Taking them whole made a
    45 s session claim 7820 frames from a pair of 150 fps cameras - 174 fps.
    """
    item = SessionTelemetry()
    # Eight seconds of streaming while the operator sets up.
    item.record_capture(0, acquired=1200, dropped=0)
    item.record_capture(1, acquired=1200, dropped=0)
    item.record_inference(count=1100, mean_ms=4.0, max_ms=9.0)

    item.begin(started_perf=100.0)
    item.record_capture(0, acquired=1200 + 6750, dropped=0)
    item.record_capture(1, acquired=1200 + 6748, dropped=0)
    item.record_inference(count=1100 + 6300, mean_ms=4.0, max_ms=9.0)

    assert item.frames_acquired == 6750
    assert item.frames_inferenced == 6300
    assert item.inferenced_percent == pytest.approx(93.33, abs=0.01)


def test_drops_before_the_session_are_not_blamed_on_it():
    """A drop while idling in Running is not a hole in this recording."""
    item = SessionTelemetry()
    item.record_capture(0, acquired=1000, dropped=4)
    item.begin(started_perf=100.0)
    assert item.dropped_frames == 0
    assert item.has_dropped_frames is False

    item.record_capture(0, acquired=2000, dropped=6)
    assert item.dropped_frames == 2
    assert item.dropped_by_camera == {0: 2}


def test_counts_freeze_when_the_session_ends(telemetry):
    """Acquisition continues in Running, so the totals keep climbing.

    Subtracting the baseline live would make a finished session grow after the
    operator stopped it.
    """
    telemetry.record_capture(0, acquired=1000, dropped=2)
    telemetry.record_inference(count=900, mean_ms=4.0, max_ms=5.0)
    telemetry.end(ended_perf=110.0)

    telemetry.record_capture(0, acquired=5000, dropped=9)
    telemetry.record_inference(count=4800, mean_ms=4.0, max_ms=5.0)
    assert telemetry.frames_acquired == 1000
    assert telemetry.dropped_frames == 2
    assert telemetry.frames_inferenced == 900
    assert telemetry.summary()["framesAcquired"] == 1000


def test_a_restarted_source_counts_from_zero_rather_than_going_negative():
    """A capture process that restarts reports a total below the baseline."""
    item = SessionTelemetry()
    item.record_capture(0, acquired=5000, dropped=3)
    item.begin(started_perf=100.0)
    item.record_capture(0, acquired=120, dropped=1)
    assert item.frames_acquired == 120
    assert item.dropped_frames == 1


def test_work_finished_after_the_session_is_not_counted_against_it(telemetry):
    """The pose backlog drains after capture stops; it is not session work.

    Once the model was fast enough to keep up, a session that ran at 99.8% of
    frames inferenced reported 125% because the drain landed before the
    counters were frozen.
    """
    telemetry.record_capture(0, acquired=6764, dropped=0)
    telemetry.record_inference(count=6749, mean_ms=4.3, max_ms=10.2)
    telemetry.end(ended_perf=145.0)

    telemetry.record_inference(count=8474, mean_ms=4.3, max_ms=10.2)
    assert telemetry.frames_inferenced == 6749
    assert telemetry.inferenced_percent <= 100.0


def test_poses_never_outnumber_frames(telemetry):
    """The two counters are sampled independently, half a second apart.

    A pose is computed from an acquired frame, so more poses than frames cannot
    happen; the excess is the fresher counter running ahead. A real session
    reported 101.4% this way, from 4575 poses against 4512 frames.
    """
    telemetry.record_capture(0, acquired=4512, dropped=0)
    telemetry.record_inference(count=4575, mean_ms=4.3, max_ms=5.3)
    assert telemetry.frames_inferenced == 4512
    assert telemetry.inferenced_percent == pytest.approx(100.0)


def test_the_cap_lifts_once_the_frame_count_catches_up(telemetry):
    """Capping the readout must not lose the poses themselves."""
    telemetry.record_capture(0, acquired=4512, dropped=0)
    telemetry.record_inference(count=4575, mean_ms=4.3, max_ms=5.3)
    assert telemetry.frames_inferenced == 4512

    telemetry.record_capture(0, acquired=4600, dropped=0)
    assert telemetry.frames_inferenced == 4575


def test_a_genuine_shortfall_is_still_reported(telemetry):
    """The cap is an upper bound only; under-inferencing must stay visible."""
    telemetry.record_capture(0, acquired=1000, dropped=0)
    telemetry.record_inference(count=400, mean_ms=4.0, max_ms=5.0)
    assert telemetry.frames_inferenced == 400
    assert telemetry.inferenced_percent == pytest.approx(40.0)
