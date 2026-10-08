"""A queue whose reader has gone does not make its writers wait for space.

Live inference reads the pose queue and the cameras write it: frames with
block=False, but the end-of-recording marker, and any padding before it, with a
blocking put. With the pose process stopped nothing frees a slot. On the rig
every Stop after a pose crash waited 5 s per camera for the marker's slot -
right at the camera watchdog's 5 s - and then raised out of the capture loop,
taking the camera with it.
"""

import threading
import time

import numpy
import pytest

from autotrainer.core.fixed_array_multiqueue import FixedArrayMultiQueue
from autotrainer.core.frame_index import FrameIndexCategory

_FRAME = numpy.zeros((4, 4), dtype="uint8")


def _full_queue(cam_count=1):
    """Every slot written and none read, as the pose queue is once its reader dies."""
    queue = FixedArrayMultiQueue(depth=2, cam_count=cam_count, frames_per_camera=1,
                                 shape=(4, 4), name="consumer-test")
    for camera in range(cam_count):
        for _ in range(queue.depth):
            queue.put_block(_FRAME, camera, FrameIndexCategory.ONLINE_NO_RECORDING, timeout=1)
    return queue


def test_a_queue_starts_with_its_reader():
    assert not _full_queue().consumer_closed


def test_with_a_reader_the_marker_still_waits_for_its_slot():
    """Unchanged while inference runs: the marker must not be dropped then."""
    queue = _full_queue()

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="Timeout waiting space"):
        queue.put_frame_index_category(
            _FRAME, FrameIndexCategory.EOF_RECORDING, cam_idx=0, timeout=0.2)
    assert time.perf_counter() - started >= 0.2


def test_a_marker_for_a_reader_that_has_gone_is_not_waited_for():
    queue = _full_queue()
    queue.close_consumer()

    started = time.perf_counter()
    queue.put_frame_index_category(_FRAME, FrameIndexCategory.EOF_RECORDING, cam_idx=0, timeout=5)

    assert time.perf_counter() - started < 0.5
    assert queue.consumer_closed


def test_padding_for_a_reader_that_has_gone_is_not_waited_for():
    """The writers still meet at the barrier, so neither is left waiting there;
    only the padding puts, which need a free slot, are skipped."""
    queue = _full_queue(cam_count=2)
    queue.close_consumer()
    errors = []

    def pad(camera, puts):
        try:
            queue.pad_to_batch_size(camera, _FRAME, puts, timeout=5)
        except Exception as err:  # noqa: BLE001 - reported below
            errors.append(err)

    # Camera 1 is two frames behind camera 0, so it has two to pad.
    writers = [threading.Thread(target=pad, args=(0, 4)),
               threading.Thread(target=pad, args=(1, 2))]
    started = time.perf_counter()
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(10)

    assert errors == []
    assert time.perf_counter() - started < 1.0
