import math
import time

import numpy

from autotrainer.core.fixed_array_multiqueue import FixedArrayMultiQueue
from autotrainer.core.fixed_array_queue import BufferResult


def _queue():
    return FixedArrayMultiQueue(depth=2, cam_count=2, frames_per_camera=1, shape=(4, 4),
                                name="latency-test")


def _buffers(queue):
    output = numpy.zeros((queue.batch_size, 4, 4, 3), dtype=numpy.uint8)
    indices = numpy.zeros((2, 1), dtype=numpy.int64)
    ids = numpy.zeros((2, 1), dtype=numpy.int64)
    put_perf = numpy.zeros((2, 1), dtype=numpy.float64)
    return output, indices, ids, put_perf


def test_a_batch_carries_each_cameras_own_frame_id_and_put_time():
    queue = _queue()
    frame = numpy.zeros((4, 4), dtype=numpy.uint8)
    before = time.perf_counter()
    assert queue.put(frame, 0, 7, block=False, frame_perf_c=1.0, cam_frame_id=1007) == BufferResult.Ok
    assert queue.put(frame, 1, 7, block=False, frame_perf_c=1.0, cam_frame_id=2007) == BufferResult.Ok
    after = time.perf_counter()
    output, indices, ids, put_perf = _buffers(queue)
    frames_perf = numpy.zeros((2, 1), dtype=numpy.float64)

    assert queue.get_output(output, indices, cam_frame_ids=ids, put_perf_c=put_perf,
                            frames_perf_c=frames_perf)

    assert ids[:, 0].tolist() == [1007, 2007]
    assert indices[:, 0].tolist() == [7, 7]
    # Slot-write times must be captured during the puts, not zero-initialized defaults.
    assert before <= put_perf[0, 0] <= put_perf[1, 0] <= after
    # Frame exposure times (frame_perf_c=1.0) stay independent: guards against array aliasing.
    assert frames_perf[:, 0].tolist() == [1.0, 1.0]


def test_a_put_without_an_id_reads_back_as_minus_one():
    queue = _queue()
    frame = numpy.zeros((4, 4), dtype=numpy.uint8)
    queue.put(frame, 0, -1, block=False)
    queue.put(frame, 1, -1, block=False)
    output, indices, ids, put_perf = _buffers(queue)

    assert queue.get_output(output, indices, cam_frame_ids=ids, put_perf_c=put_perf)

    assert ids[:, 0].tolist() == [-1, -1]
