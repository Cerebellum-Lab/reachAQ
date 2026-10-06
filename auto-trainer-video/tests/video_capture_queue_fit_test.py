import ctypes
import multiprocessing
import time

import numpy

from autotrainer.core import FixedArrayMultiQueue, FixedArrayQueue
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.video import (
    CaptureAttrs,
    CaptureCameraAttrs,
    CaptureCommandKind,
    CaptureInferenceAttrs,
    VideoCapture,
    VideoRecordProperties,
)


def _start(url, project_info):
    ctx = get_mp_ctx()
    net_q = FixedArrayMultiQueue(2, 1, 1, shape=(256, 256), primary=0, name="fit_net_q", mp_ctx=ctx)
    img_q = FixedArrayQueue(3, (256, 256), name="fit_img_q", mp_ctx=ctx)
    attrs = CaptureAttrs(
        command_queue=multiprocessing.Queue(),
        status=multiprocessing.Value("i", CaptureProcessStatus.UNKNOWN),
        image_queue=img_q,
        camera=CaptureCameraAttrs(name="fit", url=url),
        frame=multiprocessing.Value(ctypes.c_int64, -1),
        errors=multiprocessing.Array(ctypes.c_char, bytes(512)),
        inference=CaptureInferenceAttrs(queue=net_q, index=0),
        fps_image_queue=30,
        record_stop_sema=multiprocessing.Semaphore(0),
    )
    # The recorder thread exits without a valid project, and the capture process
    # shuts itself down when it sees that, so every capture here needs one.
    properties = VideoRecordProperties(project_info=project_info)
    process = VideoCapture(attrs, record_properties=properties)
    process.start()
    attrs.command_queue.put((CaptureCommandKind.ENABLE_CAPTURE, None))
    return process, attrs, net_q, img_q


def _wait(predicate, timeout):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _stop(process, attrs):
    attrs.command_queue.put((CaptureCommandKind.TERMINATE, None))
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(3)


def test_a_larger_camera_frame_is_averaged_into_both_consumer_queues(project_info):
    process, attrs, net_q, img_q = _start("random://fit?width=512&height=512&fps=30", project_info)
    try:
        assert _wait(lambda: attrs.frame.value >= 10, 15), attrs.errors.value
        out = numpy.zeros((1, 256, 256, 3))
        assert net_q.get_output(out, timeout=2)
        shown = img_q.get(timeout=2)
        # RandomCam draws uniform noise, std ~73.6. A 2x2 average of it has
        # half that. An unaveraged frame would sit near 73.6, and a size
        # mismatch would have failed the put instead.
        assert 25 < out[0, :, :, 0].std() < 50
        assert shown is not None and shown.shape == (256, 256)
        assert 25 < shown.std() < 50
        assert attrs.status.value == CaptureProcessStatus.RUNNING
    finally:
        _stop(process, attrs)


def test_a_frame_that_is_not_an_integer_multiple_of_the_queues_fails_the_start(project_info):
    process, attrs, _, _ = _start("random://fit?width=500&height=500&fps=30", project_info)
    try:
        assert _wait(lambda: attrs.status.value == CaptureProcessStatus.FAILED, 15)
        assert b"one integer factor" in attrs.errors.value
    finally:
        _stop(process, attrs)
