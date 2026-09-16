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
"""

import numpy as np

from autotrainer.core.frame_index import FrameIndexCategory
from autotrainer.inference.pose_process import _is_end_of_recording


def _batch(*values):
    """One (cameras, frames_per_camera) index batch."""
    return np.array([[v] for v in values], dtype="int64")


def test_a_normal_batch_is_not_the_marker():
    assert _is_end_of_recording(_batch(120, 120)) is False


def test_a_not_recording_batch_is_not_the_marker():
    """Streaming outside a recording is ordinary traffic, not a stop."""
    assert _is_end_of_recording(
        _batch(FrameIndexCategory.ONLINE_NO_RECORDING,
               FrameIndexCategory.ONLINE_NO_RECORDING)) is False


def test_the_marker_is_recognised():
    assert _is_end_of_recording(
        _batch(FrameIndexCategory.EOF_RECORDING,
               FrameIndexCategory.EOF_RECORDING)) is True


def test_one_camera_is_enough():
    """Capture sends one per camera; a partial batch still ends the recording."""
    assert _is_end_of_recording(
        _batch(FrameIndexCategory.EOF_RECORDING, 120)) is True


class _Queue:
    """A live queue that keeps streaming after the recording ends.

    Which is what the real one does: capture sends the marker and immediately
    sets itself back to RUNNING, so frames keep arriving behind it.
    """

    depth = 2

    def __init__(self, batches):
        self._batches = list(batches)
        self.reads = 0

    def get_output(self, frame_buffer, frames_indices, timeout=0.1,
                   frames_perf_c=None):
        if not self._batches:
            return False
        frames_indices[:, :] = self._batches.pop(0)
        self.reads += 1
        return True


def _drain(queue, stop_on_marker):
    """The drain loop, with and without the guard under test."""
    indices = np.zeros((2, 1), dtype="int64")
    if not queue.get_output(None, indices):
        return None
    while True:
        if stop_on_marker and _is_end_of_recording(indices):
            break
        if not queue.get_output(None, indices, timeout=0):
            break
    return indices


def test_draining_past_the_marker_loses_it():
    """The bug, kept as a test so the fix cannot be quietly undone."""
    queue = _Queue([_batch(10, 10),
                    _batch(FrameIndexCategory.EOF_RECORDING,
                           FrameIndexCategory.EOF_RECORDING),
                    _batch(11, 11)])
    indices = _drain(queue, stop_on_marker=False)
    assert _is_end_of_recording(indices) is False, "this is what used to happen"


def test_the_drain_stops_on_the_marker_and_hands_it_over():
    queue = _Queue([_batch(10, 10),
                    _batch(FrameIndexCategory.EOF_RECORDING,
                           FrameIndexCategory.EOF_RECORDING),
                    _batch(11, 11)])
    indices = _drain(queue, stop_on_marker=True)
    assert _is_end_of_recording(indices) is True
    # The frame behind it is left queued rather than dropped.
    assert queue.reads == 2


def test_frames_are_still_skipped_when_no_marker_is_present():
    """The drain exists to cut queue latency; that must keep working."""
    queue = _Queue([_batch(10, 10), _batch(11, 11), _batch(12, 12)])
    indices = _drain(queue, stop_on_marker=True)
    assert indices.tolist() == [[12], [12]]
    assert queue.reads == 3


def test_the_pose_process_guards_its_drain():
    """Asserted on the source: the loop is inside a nested closure."""
    import inspect
    from autotrainer.inference import pose_process
    source = inspect.getsource(pose_process)
    drain = source.split("while (live_drain")[1][:220]
    assert "_is_end_of_recording(frames_indices1)" in drain
