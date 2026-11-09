from types import SimpleNamespace

import pytest

from autotrainer.core import Offset3DTuple
from autotrainer.core.pose_elements import SceneElement
from tools.acquisition.model.live_tracking_buffer import (
    FrameTimelineAnchor,
    LiveTrackingBuffer,
    LiveTrackingSample,
)


def response(sequence, frame_ids, *, processing_perf=99.0, pellet_seen=True):
    return SimpleNamespace(
        sequence=sequence,
        source_frame_ids=(tuple(frame_ids), tuple(value + 100 for value in frame_ids)),
        perf_c=processing_perf,
        pellet_seen=pellet_seen,
        locations_3d={SceneElement.Pellet: Offset3DTuple(1, 2, 3)},
        parts_3d_offsets={
            SceneElement.Triangle: {
                SceneElement.Pellet: Offset3DTuple(4, 5, 6),
            }
        },
    )


def test_sample_uses_camera_timeline_not_processing_completion_time():
    anchor = FrameTimelineAnchor(100, 10.0, 150.0)
    sample = LiveTrackingSample.from_pose_response(
        response(3, (100, 101, 102), processing_perf=88.0), anchor
    )
    assert sample.source_start_perf == 10.0
    assert sample.source_end_perf == pytest.approx(10.0 + 2 / 150)
    assert sample.processing_perf == 88.0
    assert sample.location(str(SceneElement.Pellet)).z == 3


def test_sample_takes_each_frames_own_exposure_time_over_the_anchor():
    # The anchor would say 10 + n/150; the camera says otherwise, and the
    # camera's own time does not drift with the rate error.
    anchor = FrameTimelineAnchor(0, 10.0, 150.0)
    rsp = response(3, (1, 2, 3))
    rsp.source_frame_perf_times = ((10.0067, 10.0134, 10.0201), (10.0067, 10.0134, 10.0201))
    sample = LiveTrackingSample.from_pose_response(rsp, anchor)
    assert sample.primary_frame_perf_times == (10.0067, 10.0134, 10.0201)
    assert (sample.source_start_perf, sample.source_end_perf) == (10.0067, 10.0201)


def test_sample_falls_back_to_the_anchor_without_usable_exposure_times():
    anchor = FrameTimelineAnchor(0, 10.0, 150.0)
    rsp = response(3, (1, 2, 3))
    rsp.source_frame_perf_times = ((10.0067, float("nan"), 10.0201), ())
    sample = LiveTrackingSample.from_pose_response(rsp, anchor)
    assert sample.primary_frame_perf_times == pytest.approx(
        (10.0 + 1 / 150, 10.0 + 2 / 150, 10.0 + 3 / 150)
    )


def test_window_reports_missing_and_duplicate_frame_ids():
    anchor = FrameTimelineAnchor(100, 10.0, 100.0)
    buffer = LiveTrackingBuffer(capacity=10)
    for sequence, ids in enumerate(((100, 101), (101, 103))):
        buffer.append(LiveTrackingSample.from_pose_response(response(sequence, ids), anchor))
    window = buffer.window(9.0, 12.0)
    assert window.observed_frames == 3
    assert window.expected_frames == 4
    assert window.missing_frame_ids == (102,)
    assert window.duplicate_frame_ids == (101,)
    assert not window.complete


def test_window_reports_missing_leading_and_trailing_requested_frames():
    anchor = FrameTimelineAnchor(100, 10.0, 100.0)
    buffer = LiveTrackingBuffer(capacity=10)
    buffer.append(
        LiveTrackingSample.from_pose_response(response(1, (102, 103)), anchor)
    )

    window = buffer.window(
        10.0,
        10.05,
        expected_start_frame_id=100,
        expected_end_frame_id=105,
    )

    assert window.expected_frames == 6
    assert window.missing_frame_ids == (100, 101, 104, 105)
    assert not window.complete


def test_ring_is_bounded_and_reports_eviction():
    anchor = FrameTimelineAnchor(0, 0.0, 10.0)
    buffer = LiveTrackingBuffer(capacity=2)
    for frame_id in range(3):
        buffer.append(
            LiveTrackingSample.from_pose_response(
                response(frame_id, (frame_id,)), anchor
            )
        )
    assert buffer.dropped_samples == 1
    assert [sample.sequence for sample in buffer.snapshot()] == [1, 2]


def test_missing_source_identity_is_not_fabricated():
    anchor = FrameTimelineAnchor(0, 0.0, 10.0)
    rsp = response(1, ())
    assert LiveTrackingSample.from_pose_response(rsp, anchor) is None


def test_frame_id_bounds_are_the_frames_captured_inside_the_interval():
    anchor = FrameTimelineAnchor(0, 500.0, 150.0)
    assert anchor.frame_id_bounds(500.0 + 100 / 150, 500.0 + 109 / 150) == (100, 109)
    # A frame half a period inside either edge was captured outside it.
    assert anchor.frame_id_bounds(500.0 + 100.5 / 150, 500.0 + 109.5 / 150) == (101, 109)
    # Nothing before the anchor's first frame exists to be expected.
    assert anchor.frame_id_bounds(499.0, 499.5) == (0, 0)
