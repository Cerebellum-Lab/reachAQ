"""The live pose path must work on the newest frame, never a queued one.

`FixedArrayMultiQueue.get_output` frees the buffer slot as soon as it has
copied the frame out, which is at the start of predict rather than the end. At
depth 1 the producer therefore refills that single slot immediately and has
nowhere to put anything that arrives afterwards, so the batch waiting at the
next call is already a whole predict-period old. Measured on the rig at 150 fps
that cost cspnext_m 3.8 ms of its 17.6 ms end-to-end, and the capture loop
could not hand over 35.2% of its batches at all.

Depth 2 plus draining to the newest fixes both: 13.854 ms p50 and 0.7%.

What must not regress:

  * acquisition is never blocked or slowed by inference. The capture loop puts
    with block=False onto a queue separate from the recording queue, so a slow
    pose process costs inference work and never a recorded frame.
  * the drain hands back the newest batch, not the oldest.
  * the drain does not run at depth 1, where nothing can be behind the current
    batch and the probe would pay a full frame copy to discover that.
"""

import inspect

import numpy
import pytest

from autotrainer.core.fixed_array_multiqueue import BufferResult
from autotrainer.core.fixed_array_multiqueue import FixedArrayMultiQueue

SHAPE = (8, 8)
CAMERAS = 2


def _queue(depth):
    return FixedArrayMultiQueue(depth, CAMERAS, 1, SHAPE, name="test")


def _frame(value):
    return numpy.full(SHAPE, value, dtype=numpy.uint8)


def _put(queue, value, index):
    return [queue.put(_frame(value), camera, index, block=False)
            for camera in range(CAMERAS)]


def _take(queue):
    output = numpy.zeros((CAMERAS, *SHAPE, 3), dtype=numpy.uint8)
    indices = numpy.zeros((CAMERAS, 1), dtype="int64")
    if not queue.get_output(output, indices, timeout=0):
        return None
    return int(output[0, 0, 0, 0]), int(indices[0, 0])


# --- what depth 1 could not do ----------------------------------------------


def test_depth_one_cannot_hold_a_second_batch():
    """The reason depth 1 inflates latency: the producer has nowhere to put a
    newer frame, so the consumer is served the older one."""
    queue = _queue(1)
    assert all(r == BufferResult.Ok for r in _put(queue, 10, 0))
    assert any(r != BufferResult.Ok for r in _put(queue, 20, 1))
    assert _take(queue) == (10, 0)


def test_depth_two_accepts_a_newer_batch_while_one_is_unread():
    queue = _queue(2)
    assert all(r == BufferResult.Ok for r in _put(queue, 10, 0))
    assert all(r == BufferResult.Ok for r in _put(queue, 20, 1))


# --- draining ----------------------------------------------------------------


def test_draining_yields_the_newest_batch():
    """What get_live_input does: take one, then keep taking while more are
    ready, and predict on the last."""
    queue = _queue(2)
    _put(queue, 10, 0)
    _put(queue, 20, 1)

    taken = _take(queue)
    assert taken == (10, 0)
    while True:
        newer = _take(queue)
        if newer is None:
            break
        taken = newer
    assert taken == (20, 1), "the consumer kept a stale batch"


def test_draining_an_empty_queue_returns_nothing():
    queue = _queue(2)
    _put(queue, 10, 0)
    assert _take(queue) == (10, 0)
    assert _take(queue) is None


# --- the wiring --------------------------------------------------------------


def test_pose_process_drains_and_only_above_depth_one():
    """A revert would be silent: the result is identical, only staler."""
    from autotrainer.inference import pose_process

    source = inspect.getsource(pose_process.PoseProcess)
    assert "live_drain = live_input.depth > 1" in source, (
        "the drain must be gated on depth; at depth 1 the probe costs a full "
        "frame copy to discover there is nothing behind the current batch")
    assert "while live_drain and live_input.get_output(" in source, (
        "the live path takes the oldest batch again")


def test_the_live_queue_is_not_depth_one():
    """app_model sizes the queue the pose path depends on."""
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1].joinpath(
        "tools", "acquisition", "model", "app_model.py").read_text()
    start = source.index("self._inference_queue = FixedArrayMultiQueue(")
    block = source[start:start + 2000]
    assert "Depth 2, with PoseProcess draining to the newest batch." in block


def test_capture_offers_frames_without_blocking():
    """Acquisition must never wait on inference. This is the line that
    guarantees a slow pose process cannot cost a recorded frame."""
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1].joinpath(
        "auto-trainer-video", "src", "autotrainer", "video",
        "video_capture.py").read_text()
    assert "net_q_put(frame, net_q_idx, frame_idx_cat, block=False)" in source
