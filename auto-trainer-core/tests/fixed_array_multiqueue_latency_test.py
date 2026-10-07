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
    # Triggered cameras give one exposure one id, and the queue pairs on it.
    assert queue.put(frame, 0, 7, block=False, frame_perf_c=1.0, frame_id=1007) == BufferResult.Ok
    assert queue.put(frame, 1, 7, block=False, frame_perf_c=1.0, frame_id=1007) == BufferResult.Ok
    after = time.perf_counter()
    output, indices, ids, put_perf = _buffers(queue)
    frames_perf = numpy.zeros((2, 1), dtype=numpy.float64)

    assert queue.get_output(output, indices, cam_frame_ids=ids, put_perf_c=put_perf,
                            frames_perf_c=frames_perf)

    assert ids[:, 0].tolist() == [1007, 1007]
    assert indices[:, 0].tolist() == [7, 7]
    # Slot-write times must be captured during the puts, not zero-initialized defaults.
    assert before <= put_perf[0, 0] <= put_perf[1, 0] <= after
    # Frame exposure times (frame_perf_c=1.0) stay independent: guards against array aliasing.
    assert frames_perf[:, 0].tolist() == [1.0, 1.0]


def test_a_put_without_an_id_reads_back_as_minus_one():
    queue = _queue()
    frame = numpy.zeros((4, 4), dtype=numpy.uint8)
    # One camera without an id leaves the pairing positional, so the other's id
    # still reads back on its own camera's row.
    queue.put(frame, 0, -1, block=False, frame_id=5)
    queue.put(frame, 1, -1, block=False)
    output, indices, ids, put_perf = _buffers(queue)

    assert queue.get_output(output, indices, cam_frame_ids=ids, put_perf_c=put_perf)

    assert ids[:, 0].tolist() == [5, -1]


def test_a_pairing_skip_leaves_each_cameras_row_from_its_own_slot():
    queue = _queue()
    # Camera 0 holds 14 and 15, camera 1 only 15: the reader skips camera 0's 14,
    # so the two cameras' read positions differ (slot 1 against slot 0) when the
    # batch is copied out. Every latency output has to follow its camera's slot.
    frame = numpy.zeros((4, 4), dtype=numpy.uint8)
    queue.put(frame, 0, 4, block=False, frame_perf_c=114.0, frame_id=14)
    time.sleep(0.002)
    between = time.perf_counter()
    queue.put(frame, 0, 5, block=False, frame_perf_c=115.0, frame_id=15)
    queue.put(frame, 1, 5, block=False, frame_perf_c=215.0, frame_id=15)
    after = time.perf_counter()
    output, indices, ids, put_perf = _buffers(queue)
    frames_perf = numpy.zeros((2, 1), dtype=numpy.float64)

    assert queue.get_output(output, indices, cam_frame_ids=ids, put_perf_c=put_perf,
                            frames_perf_c=frames_perf)

    assert ids[:, 0].tolist() == [15, 15]
    assert indices[:, 0].tolist() == [5, 5]
    assert frames_perf[:, 0].tolist() == [115.0, 215.0]
    # Camera 0's stamp is its put of 15, which came after the put of 14 that was
    # skipped; camera 1's slot 0 holds its only put.
    assert between <= put_perf[0, 0] <= put_perf[1, 0] <= after
