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
    # What the stub may need to act as the command thread, and which streams were closed.
    latch.capture, finished = capture, []
    latch.finished, finish = finished, capture._finish_latency
    capture._finish_latency = lambda writer: (finished.append(writer), finish(writer))
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
        latch.open_at_exit = capture._latency
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


class _RecordDuringStopMark(_Latch):
    """On the first latch of the stop mark, starts the next stream as an ENABLE would."""

    closing = opened = None

    def __call__(self):
        if len(self.threads) == 3:  # the opening mark took three
            self.closing = self.capture._latency
            self.capture._start_latency()
            self.opened = self.capture._latency
        return super().__call__()


def test_a_record_inside_the_stop_mark_keeps_the_stream_it_opened(project_info):
    latch = _RecordDuringStopMark()

    stream = _record(project_info, latch)

    assert latch.closing is not None and latch.opened is not None
    assert latch.opened is not latch.closing
    # The old stream's close still happened, and the new one lived on to exit.
    assert latch.closing in latch.finished
    assert latch.open_at_exit is latch.opened
    assert latch.finished.index(latch.closing) < latch.finished.index(latch.opened)
    # The file now holds the new stream: its frames and its own opening latches.
    assert len(stream["clock_latches"]) == 3
    assert len(stream["frames"]) > 5


class _Camera:
    """latch_clock stub: brackets of ``bracket`` seconds, each call taking ``takes``."""

    def __init__(self, bracket, takes=0.0):
        self.bracket, self.takes, self.calls = bracket, takes, 0

    def latch_clock(self):
        self.calls += 1
        before = time.perf_counter()
        time.sleep(self.takes)
        return before, self.calls, before + self.bracket


class _Rows:
    def __init__(self):
        self.rows = []

    def append(self, dataset, row):
        self.rows.append((dataset, row))


def _mark(camera):
    capture = object.__new__(VideoCapture)
    capture._name = "budget"
    rows = _Rows()
    started = time.perf_counter()
    capture._append_clock_latches(camera, rows)
    return rows.rows, time.perf_counter() - started


def test_three_quick_latches_make_a_mark():
    rows, _ = _mark(_Camera(bracket=0.0002))
    assert [dataset for dataset, _ in rows] == ["clock_latches"] * 3


def test_a_latch_wider_than_the_finalizer_uses_ends_the_mark_after_its_row():
    rows, _ = _mark(_Camera(bracket=0.003))
    assert len(rows) == 1


def test_a_slow_mark_stops_at_its_time_budget():
    # Tight brackets, but each call stalls 3 ms after the latch (the value read).
    camera = _Camera(bracket=0.0002, takes=0.003)

    rows, elapsed = _mark(camera)

    assert camera.calls == 2 and len(rows) == 2
    assert elapsed < 0.009
