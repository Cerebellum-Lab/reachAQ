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

import ast

import numpy
import pytest

import source_contract

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


def test_the_pose_process_uses_the_drain():
    """A revert would be silent: the result is identical, only staler.

    The drain's behaviour is covered directly in live_drain_eof_test. What is
    left to check here is that the live path still goes through it, and that
    queue depth still decides whether it runs.
    """
    from autotrainer.inference import pose_process

    call = source_contract.one_call(pose_process, "take_newest_live_batch")
    assert source_contract.keyword_name(call, "drain") == "live_drain", (
        "the live path no longer drains to the newest batch")

    gate = source_contract.assigned(pose_process, "live_drain")
    assert gate is not None, "live_drain is no longer derived at all"
    assert "depth" in gate and "Gt" in gate, (
        "the drain must stay gated on queue depth; at depth 1 the probe costs "
        "a full frame copy to discover there is nothing behind the batch "
        f"(found {gate!r})")


def test_the_live_queue_is_not_depth_one():
    """app_model sizes the queue the pose path depends on.

    Read as the argument rather than as a comment beside it: the comment was
    what used to be asserted, which a correct rewording would have broken and
    a silent change from 2 to 1 would not.
    """
    call = source_contract.one_call(
        "tools/acquisition/model/app_model.py", "FixedArrayMultiQueue")
    depth = call.args[0] if call.args else source_contract.keyword(call, "depth")
    assert isinstance(depth, ast.Constant), (
        "the live queue depth is no longer a literal; check it by hand")
    assert depth.value > 1, (
        "depth 1 cannot hold a newer batch, so the pose path is served a stale "
        "one and the capture loop cannot hand over at all")


def test_capture_offers_frames_without_blocking():
    """Acquisition must never wait on inference. This is the guarantee that a
    slow pose process cannot cost a recorded frame.

    Asserted on the call's arguments rather than on one formatting of it: the
    guarantee is that this particular put is non-blocking, and pinning the
    source line made an unrelated reformat look like a regression while a
    genuine change from block=False to block=True inside a rewritten line
    would still have slipped through.
    """
    call = source_contract.one_call(
        "auto-trainer-video/src/autotrainer/video/video_capture.py",
        "net_q_put")
    assert source_contract.keyword_equals(call, "block", False), (
        "the capture loop must offer frames to inference without blocking")
