"""The capture loop latches the camera clock when a latency stream opens and closes.

The loop runs here on a thread of the test process, with the same random
camera, recorder and command thread the capture process gives it, so the
camera's latch can be replaced and the thread that calls it observed.
"""

import multiprocessing
import queue
import threading
import time

import h5py
import numpy

from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.latency import latency_stream_path
from autotrainer.video import (
    CaptureAttrs,
    CaptureCameraAttrs,
    CaptureCommandKind,
    VideoCapture,
    VideoRecordMode,
    VideoRecordProperties,
)

LOOP_THREAD = "CaptureLoopUnderTest"


class _Latch:
    """Stands in for a camera's latch_clock, recording the thread of each call."""

    def __init__(self, result="latch"):
        self.result = result
        self.threads = []

    def __call__(self):
        self.threads.append(threading.current_thread().name)
        if self.result == "raise":
            raise RuntimeError("latch failed")
        if self.result is None:
            return None
        before = time.perf_counter()
        return before, time.perf_counter_ns(), time.perf_counter()


def _record(project_info, latch, seconds=1.0):
    commands = queue.Queue()
    status = multiprocessing.Value("i", CaptureProcessStatus.UNKNOWN)
    attrs = CaptureAttrs(
        command_queue=commands, status=status, image_queue=None,
        camera=CaptureCameraAttrs(name="latch", url="random://0"),
        frame=multiprocessing.Value("i", 0), errors=None,
    )
    record = VideoRecordProperties(project_info=project_info,
                                   record_mode=VideoRecordMode.TRIGGER,
                                   video_rotate_interval=0)
    capture = VideoCapture(attrs, record_properties=record, project_info=project_info)
    assert capture._prepare_to_run()
    capture._camera.latch_clock = latch
    loop = threading.Thread(target=capture._run_capture_loop, args=(capture._camera,),
                            name=LOOP_THREAD, daemon=True)
    loop.start()
    try:
        commands.put((CaptureCommandKind.ENABLE_CAPTURE, None))
        commands.put((CaptureCommandKind.ENABLE_RECORDING, None))
        time.sleep(seconds)
        commands.put((CaptureCommandKind.DISABLE_RECORDING, None))
        time.sleep(0.7)
        assert status.value == CaptureProcessStatus.RUNNING
    finally:
        commands.put((CaptureCommandKind.TERMINATE, None))
        loop.join(10)
        capture._terminate_capture_loop(None)
    assert not loop.is_alive()
    with h5py.File(latency_stream_path(project_info, "camera_latch"), "r") as store:
        return {name: store[name][()] for name in store}


def test_latches_follow_the_streams_first_frame_and_precede_its_close(project_info):
    latch = _Latch()

    stream = _record(project_info, latch)

    rows, frames = stream["clock_latches"], stream["frames"]
    assert latch.threads == [LOOP_THREAD] * 6
    assert len(rows) == 6
    assert (rows["perf_before"] <= rows["perf_after"]).all()
    opening, closing = rows[:3], rows[3:]
    # After the first frame row, before the next frame was polled.
    assert (opening["perf_before"] > frames["capture_return_perf"][0]).all()
    assert (opening["perf_after"] < frames["poll_perf"][1]).all()
    # After the last frame row: the stream closed right after them.
    assert (closing["perf_before"] > frames["capture_return_perf"][-1]).all()


def _assert_capture_unaffected(frames, caplog):
    assert len(frames) > 20  # about 50 at 30 fps
    assert (numpy.diff(frames["frame_id"]) == 1).all()
    assert "Error during capture loop" not in caplog.text


def test_a_camera_without_a_clock_to_latch_costs_one_call_per_mark(project_info, caplog):
    latch = _Latch(result=None)

    stream = _record(project_info, latch)

    assert latch.threads == [LOOP_THREAD] * 2
    assert len(stream["clock_latches"]) == 0
    _assert_capture_unaffected(stream["frames"], caplog)


def test_a_latch_that_raises_stays_out_of_the_capture_loop(project_info, caplog):
    latch = _Latch(result="raise")

    stream = _record(project_info, latch)

    # A raise that reached the loop would also have skipped the close it precedes,
    # and the loop would have latched again on the next frame.
    assert latch.threads == [LOOP_THREAD] * 2
    assert len(stream["clock_latches"]) == 0
    _assert_capture_unaffected(stream["frames"], caplog)
    assert "camera clock latch failed" in caplog.text
