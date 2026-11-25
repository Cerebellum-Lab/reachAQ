import threading
import time
from functools import partial

import numpy

from reachaq.core import FixedArrayMultiQueue


def increase_buffer(buffer: FixedArrayMultiQueue, shape: (int, int), frames_per_camera: int, offset: int):
    content = numpy.zeros(shape, dtype=numpy.uint8)

    for idx in range(frames_per_camera):
        buffer.put(content + idx + offset, 1, 0)
        buffer.put(content + idx + offset, 0, 0)


def check_buffer(buffer: FixedArrayMultiQueue, shape: (int, int), frames_per_camera: int, expected_index: int):
    output = numpy.zeros((frames_per_camera * 2, *shape, 3), dtype=numpy.uint8)

    result = buffer.get_output(output)

    assert result is True

    assert buffer.buffer_index == expected_index

    assert buffer.get_output(output) is False


def test_buffer():
    shape = (200, 300)

    frames_per_camera = 5

    buffer = FixedArrayMultiQueue(3, 2, frames_per_camera, shape)

    assert buffer.buffer_index == 0

    increase_buffer(buffer, shape, frames_per_camera, 1)

    check_buffer(buffer, shape, frames_per_camera, 1)

    increase_buffer(buffer, shape, frames_per_camera, 6)

    check_buffer(buffer, shape, frames_per_camera, 2)

    increase_buffer(buffer, shape, frames_per_camera, 11)

    check_buffer(buffer, shape, frames_per_camera, 0)


def _consume_queue(buffer: FixedArrayMultiQueue):
    frames = numpy.ndarray((buffer.batch_size, *buffer.shape, 3), dtype=numpy.uint8)
    frame_indices = numpy.ndarray((buffer.camera_count, buffer.frames_per_camera), dtype=numpy.int64)
    while True:
        res = buffer.get_output(frames, frames_indices=frame_indices)
        if res:
            pass
            # print(f"got: {frame_indices.tolist()}", )
        else:
            time.sleep(0.002)


def test_get_cam_missing_frames():
    shape = (100, 100)
    frames_per_batch_per_cam = 3
    queue_batch_depth = 16  # need big enough for all below puts,
    #
    buffer = FixedArrayMultiQueue(queue_batch_depth, 2, frames_per_batch_per_cam, shape)
    content = numpy.zeros(shape, dtype=numpy.uint8)
    #
    check_missing = buffer.get_cam_missing_frames

    tot_put_cam0 = tot_put_cam1 = 0
    def put_cam_0():
        nonlocal tot_put_cam0
        buffer.put(content, 0, 0)
        tot_put_cam0 += 1
        buffer.set_cam_tot_frames(0, tot_put_cam0)

    def put_cam_1():
        nonlocal tot_put_cam1
        buffer.put(content, 1, 0)
        tot_put_cam1 += 1
        buffer.set_cam_tot_frames(1, tot_put_cam1)

    def inc_buffer(buff: FixedArrayMultiQueue, shape_, frames_per_camera: int, offset: int):
        increase_buffer(buff, shape_, frames_per_camera, offset)
        nonlocal tot_put_cam0, tot_put_cam1
        tot_put_cam0 += frames_per_camera
        tot_put_cam1 += frames_per_camera
        buffer.set_cam_tot_frames(0, tot_put_cam0)
        buffer.set_cam_tot_frames(1, tot_put_cam0)

    consumer = threading.Thread(target=_consume_queue, args=(buffer,), daemon=True)
    consumer.start()

    for outer_loop_idx in range(1024):
        # this "big" loop allows to stress test a bit the implementation,
        # even though here we don't use 2 cams writer threads/procs,
        # but the code doing it anyway also does sync_barrier between the 2 cams, so it's ~equivalent.

        for _ in range(2):
            put_cam_0()

        assert check_missing(0) == 1
        assert check_missing(1) == frames_per_batch_per_cam

        put_cam_0()

        assert check_missing(0) == 0
        assert check_missing(1) == frames_per_batch_per_cam

        put_cam_0()

        assert check_missing(0) == 2
        assert check_missing(1) == 2 * frames_per_batch_per_cam

        put_cam_0()

        assert check_missing(0) == 1
        assert check_missing(1) == frames_per_batch_per_cam * 2

        put_cam_0()

        assert check_missing(0) == 0
        assert check_missing(1) == frames_per_batch_per_cam * 2

        for x in range(2 * frames_per_batch_per_cam):
            assert check_missing(1) == frames_per_batch_per_cam * 2 - x
            assert check_missing(0) == 0
            put_cam_1()
            assert check_missing(0) == 0
            assert check_missing(1) == frames_per_batch_per_cam * 2 - x - 1

        #

        inc_buffer(buffer, shape, 2, 0)

        assert check_missing(0) == frames_per_batch_per_cam - 2  # 1
        assert check_missing(1) == frames_per_batch_per_cam - 2  # 1
        inc_buffer(buffer, shape, 2, 0)
        assert check_missing(0) == frames_per_batch_per_cam - 1  # 2
        assert check_missing(1) == frames_per_batch_per_cam - 1  # 2

        put_cam_0()
        put_cam_0()
        assert check_missing(0) == 0
        assert check_missing(1) == frames_per_batch_per_cam - 1

        put_cam_0()
        assert check_missing(0) == 2
        assert check_missing(1) == 2 * frames_per_batch_per_cam - 1  # 5

        put_cam_1()
        assert check_missing(0) == 2  # still
        assert check_missing(1) == frames_per_batch_per_cam + 1  # 4

        put_cam_0()
        assert check_missing(0) == 1  # now
        assert check_missing(1) == frames_per_batch_per_cam + 1  # 4 still

        put_cam_0()
        assert check_missing(0) == 0  # now
        assert check_missing(1) == frames_per_batch_per_cam + 1  # 4 still

        put_cam_1()
        assert check_missing(0) == 0  # now
        assert check_missing(1) == frames_per_batch_per_cam  # 3

        put_cam_1()
        assert check_missing(0) == 0  # now
        assert check_missing(1) == frames_per_batch_per_cam - 1  # 2

        put_cam_1()
        put_cam_1()

        assert check_missing(0) == 0  # now
        assert check_missing(1) == 0
