"""Live single-frame inference with a padded model batch.

DeepLabCut fixes its graph batch size at construction, so live and offline must
feed the same number of rows to one model. The live queue was changed to release
a batch after one frame per camera instead of three, and `PoseProcess` pads the
batch up to the model size before calling predict.

These tests exercise the real `FixedArrayMultiQueue` rather than a stub, because
the release semantics and the write-into-a-view behaviour are the parts that
actually carry risk.
"""

import numpy
import pytest

from autotrainer.core import FixedArrayMultiQueue
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.inference.pose_process import DEFAULT_MODEL_FRAMES_PER_CAMERA


SHAPE = (4, 6)  # rows, cols -- small enough to assert on directly
CAMERAS = 2


def _queue(frames_per_camera, name):
    return FixedArrayMultiQueue(
        1,
        CAMERAS,
        frames_per_camera,
        shape=SHAPE,
        primary=0,
        name=name,
        mp_ctx=get_mp_ctx(),
    )


def _frame(value):
    return numpy.full(SHAPE, value, dtype="uint8")


def _buffers(queue, model_frames_per_camera=DEFAULT_MODEL_FRAMES_PER_CAMERA):
    """Allocate exactly as PoseProcess._process does."""
    model_batch_size = queue.camera_count * model_frames_per_camera
    live_batch_size = queue.batch_size
    predict_buffer = numpy.zeros((model_batch_size, *queue.shape, 3))
    live_view = predict_buffer[:live_batch_size]
    frames_indices = numpy.ndarray(
        (queue.camera_count, queue.frames_per_camera), dtype="int64")
    return predict_buffer, live_view, frames_indices, model_batch_size, live_batch_size


def test_model_batch_size_ignores_the_live_frame_count():
    # The whole point: the graph size is pinned to the model frame count, so the
    # live queue can shrink without rebuilding the DeepLabCut graph.
    live = _queue(1, "batch_indep_live")
    assert live.batch_size == CAMERAS * 1
    model_batch_size = live.camera_count * DEFAULT_MODEL_FRAMES_PER_CAMERA
    assert model_batch_size == CAMERAS * 3
    assert model_batch_size != live.batch_size


def test_single_frame_queue_releases_after_one_frame_per_camera():
    queue = _queue(1, "release_1")
    _, live_view, frames_indices, _, _ = _buffers(queue)

    # Nothing queued yet.
    assert queue.get_output(live_view, frames_indices, timeout=0.01) is False

    for camera in range(CAMERAS):
        queue.put(_frame(camera + 1), camera, camera + 1)

    # One frame per camera is enough: this is the latency win.
    assert queue.get_output(live_view, frames_indices, timeout=0.5) is True


def test_three_frame_queue_still_waits_for_three():
    queue = _queue(3, "release_3")
    predict_buffer = numpy.zeros((CAMERAS * 3, *queue.shape, 3))
    frames_indices = numpy.ndarray((CAMERAS, 3), dtype="int64")

    for frame in range(2):
        for camera in range(CAMERAS):
            queue.put(_frame(frame + 1), camera, frame + 1)
        # Two frames per camera must not release a batch.
        assert queue.get_output(predict_buffer, frames_indices, timeout=0.01) is False

    for camera in range(CAMERAS):
        queue.put(_frame(3), camera, 3)
    assert queue.get_output(predict_buffer, frames_indices, timeout=0.5) is True


def test_live_queue_writes_through_the_view_and_padding_stays_zero():
    queue = _queue(1, "padding")
    predict_buffer, live_view, frames_indices, model_batch, live_batch = _buffers(queue)

    values = (7, 9)
    for camera in range(CAMERAS):
        queue.put(_frame(values[camera]), camera, camera + 1)
    assert queue.get_output(live_view, frames_indices, timeout=0.5) is True

    # The queue wrote through the view straight into the predict buffer.
    for camera in range(CAMERAS):
        assert (predict_buffer[camera, :, :, 0] == values[camera]).all()
        # get_output replicates gray into all three planes.
        for plane in (1, 2):
            assert (predict_buffer[camera, :, :, plane] == values[camera]).all()

    # Everything past the real rows is still zero padding.
    assert live_batch == CAMERAS
    assert model_batch == CAMERAS * DEFAULT_MODEL_FRAMES_PER_CAMERA
    assert (predict_buffer[live_batch:] == 0).all()


def test_padding_rows_are_reused_without_contaminating_later_batches():
    queue = _queue(1, "reuse")
    predict_buffer, live_view, frames_indices, _, live_batch = _buffers(queue)

    for round_index, value in enumerate((11, 22), start=1):
        for camera in range(CAMERAS):
            queue.put(_frame(value + camera), camera, round_index)
        assert queue.get_output(live_view, frames_indices, timeout=0.5) is True
        for camera in range(CAMERAS):
            assert (predict_buffer[camera, :, :, 0] == value + camera).all()
        # Padding must never pick up stale frame content.
        assert (predict_buffer[live_batch:] == 0).all()


def test_predict_output_slice_keeps_the_real_camera_rows():
    """The batch is interleaved by camera, so the leading rows are frame 0."""
    queue = _queue(1, "slice")
    _, _, _, model_batch, live_batch = _buffers(queue)

    # DlcPoseModel.predict returns one entry per batch row.
    predicted = [numpy.full((3, 3), row, dtype=float) for row in range(model_batch)]
    live_pose = predicted[:live_batch]

    assert len(live_pose) == CAMERAS
    # Rows 0..CAMERAS-1 are frame 0 of each camera, which is what the queue filled.
    for camera in range(CAMERAS):
        assert (live_pose[camera] == camera).all()


def test_zero_pose_lengths_match_each_mode():
    """Live and offline no longer share a frame count."""
    queue = _queue(1, "zero_pose")
    body_parts = 14
    model_batch = queue.camera_count * DEFAULT_MODEL_FRAMES_PER_CAMERA
    frames_indices = numpy.ndarray(
        (queue.camera_count, queue.frames_per_camera), dtype="int64")

    zero_row = numpy.asarray([0] * 3 * body_parts)
    empty_live = [zero_row] * frames_indices.size
    empty_offline = [zero_row] * model_batch

    assert len(empty_live) == CAMERAS
    assert len(empty_offline) == CAMERAS * DEFAULT_MODEL_FRAMES_PER_CAMERA
    # The pre-change code sized both from the live indices, which would now be
    # too short for an offline batch.
    assert len(empty_live) != len(empty_offline)


def test_default_model_frames_per_camera_matches_the_offline_vote():
    # The offline three-frame confidence vote depends on this staying 3.
    assert DEFAULT_MODEL_FRAMES_PER_CAMERA == 3
