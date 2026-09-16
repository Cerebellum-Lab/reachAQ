"""The drain-to-newest must not swallow the end-of-recording marker.

Live inference skips queued frames to take the freshest one, which costs
inference work and never acquisition. But one batch in the stream is not a
frame: EOF_RECORDING is the marker that tells the pose writers to close their
files. Capture goes straight back to streaming after sending it, so anything
draining past it overwrote it in place.

The effect was invisible during the recording and fatal afterwards: the pose
process never saw the stop, the live pose files were never closed, the wait for
them timed out after 10 s and every session finalized with dataComplete false.
Whether the marker survived was a race against the next frame - it survived
once in eleven sessions on the rig.

These drive the real function rather than a copy of its loop. It was extracted
from a closure inside PoseProcess.__do_run for exactly that reason: starting a
pose process needs a GPU, a trained model and live queues, and a test that
reimplements the loop only proves the reimplementation works.
"""

import numpy as np

from autotrainer.core.frame_index import FrameIndexCategory
from autotrainer.inference.pose_process import (
    _is_end_of_recording,
    take_newest_live_batch,
)


def _batch(*values):
    """One (cameras, frames_per_camera) index batch."""
    return np.array([[v] for v in values], dtype="int64")


EOF = FrameIndexCategory.EOF_RECORDING
IDLE = FrameIndexCategory.ONLINE_NO_RECORDING


class _Queue:
    """A live queue that keeps streaming after the recording ends.

    Which is what the real one does: capture sends the marker and immediately
    sets itself back to RUNNING, so frames keep arriving behind it.
    """

    def __init__(self, batches):
        self._batches = list(batches)
        self.reads = 0

    def get_output(self, frame_buffer, frames_indices, timeout=0.1,
                   frames_perf_c=None):
        if not self._batches:
            return False
        batch = self._batches.pop(0)
        frames_indices[:, :] = batch
        if frame_buffer is not None:
            frame_buffer[...] = self.reads + 1
        self.reads += 1
        return True

    @property
    def remaining(self):
        return len(self._batches)


def _take(queue, *, drain=True):
    indices = np.zeros((2, 1), dtype="int64")
    buffer = np.zeros((2, 4, 4, 3), dtype="uint8")
    took = take_newest_live_batch(queue, buffer, indices, None, drain=drain)
    return took, indices, buffer


# --- the marker itself -------------------------------------------------------


def test_a_normal_batch_is_not_the_marker():
    assert _is_end_of_recording(_batch(120, 120)) is False


def test_a_not_recording_batch_is_not_the_marker():
    """Streaming outside a recording is ordinary traffic, not a stop."""
    assert _is_end_of_recording(_batch(IDLE, IDLE)) is False


def test_the_marker_is_recognised():
    assert _is_end_of_recording(_batch(EOF, EOF)) is True


def test_one_camera_is_enough():
    """Capture sends one per camera; a partial batch still ends the recording."""
    assert _is_end_of_recording(_batch(EOF, 120)) is True


# --- what the live path actually does ----------------------------------------


def test_an_empty_queue_yields_nothing():
    took, _indices, _buffer = _take(_Queue([]))
    assert took is False


def test_the_newest_batch_wins():
    """The whole reason the drain exists: never predict on a stale frame."""
    queue = _Queue([_batch(10, 10), _batch(11, 11), _batch(12, 12)])
    took, indices, _buffer = _take(queue)
    assert took is True
    assert indices.tolist() == [[12], [12]]
    assert queue.reads == 3


def test_the_frame_buffer_matches_the_batch_that_was_kept():
    """Indices and pixels must come from the same read, not different ones."""
    queue = _Queue([_batch(10, 10), _batch(11, 11)])
    _took, _indices, buffer = _take(queue)
    assert int(buffer[0, 0, 0, 0]) == 2, "the buffer holds the second read"


def test_the_drain_stops_on_the_marker_and_hands_it_over():
    queue = _Queue([_batch(10, 10), _batch(EOF, EOF), _batch(11, 11)])
    took, indices, _buffer = _take(queue)
    assert took is True
    assert _is_end_of_recording(indices) is True
    # The frame behind it is left queued rather than dropped.
    assert queue.reads == 2
    assert queue.remaining == 1


def test_the_batch_left_behind_is_served_next():
    """Stopping early costs one iteration of staleness, not a lost frame."""
    queue = _Queue([_batch(EOF, EOF), _batch(11, 11)])
    _take(queue)
    _took, indices, _buffer = _take(queue)
    assert indices.tolist() == [[11], [11]]


def test_a_marker_arriving_first_is_returned_alone():
    queue = _Queue([_batch(EOF, EOF)])
    took, indices, _buffer = _take(queue)
    assert took is True
    assert _is_end_of_recording(indices) is True
    assert queue.reads == 1


def test_without_draining_the_oldest_batch_is_served():
    """Depth 1 does not drain: nothing can be behind the current batch, and
    the probe would pay a full frame copy to discover that."""
    queue = _Queue([_batch(10, 10), _batch(11, 11)])
    took, indices, _buffer = _take(queue, drain=False)
    assert took is True
    assert indices.tolist() == [[10], [10]]
    assert queue.reads == 1


def test_not_draining_still_surfaces_a_marker():
    queue = _Queue([_batch(EOF, EOF)])
    _took, indices, _buffer = _take(queue, drain=False)
    assert _is_end_of_recording(indices) is True
