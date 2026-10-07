import threading
import time

import h5py

from autotrainer.core.latency import latency_stream_path
from autotrainer.video import VideoCapture


class _Properties:
    def should_record(self, active, *, is_from_start=False):
        return active


class _SlowWriter:
    """Stands in for a LatencyStreamWriter whose close() takes as long as a busy disk does."""

    def __init__(self, delay):
        self.delay = delay
        self.closed_on = None
        self.finished = threading.Event()

    def close(self, timeout=10.0):
        time.sleep(self.delay)
        self.closed_on = threading.current_thread()
        self.finished.set()
        return {"failed": False, "rowsDropped": {}, "rowsRejected": 0}


def _capture(project_info):
    """A VideoCapture with only what the latency lifecycle touches (as stim_camera_test does)."""
    capture = object.__new__(VideoCapture)
    capture._name = "lifecycle"
    capture._camera_idx = 0
    capture._project_info = project_info
    capture._attrs = type("Attrs", (), {"is_primary": True})()
    capture._record_properties = _Properties()
    capture._is_record_active = False
    capture._stim_detector = None
    capture._latency = None
    capture._latency_close_requested = False
    capture._latency_closers = []
    return capture


def test_a_disable_with_no_stream_open_does_not_end_the_next_recordings_stream(project_info):
    capture = _capture(project_info)
    # Routine: the app sends DISABLE_RECORDING to every camera on entering a calibration step.
    capture._disable_record()
    assert capture._latency_close_requested is False

    capture._enable_record()
    try:
        assert capture._latency is not None
        assert capture._latency_close_requested is False
        for frame_id in range(5):
            # What the capture loop does per frame: append, then close only if asked to.
            capture._latency.append("frames", (frame_id, 0, 0.0, 0.0, 0.0, -1))
            assert not capture._latency_close_requested
        assert capture._latency is not None
    finally:
        capture._close_latency()

    with h5py.File(latency_stream_path(project_info, "camera_lifecycle"), "r") as store:
        assert len(store["frames"]) == 5


def test_starting_a_stream_clears_a_request_left_standing(project_info):
    capture = _capture(project_info)
    capture._latency_close_requested = True  # e.g. a DISABLE that raced with the loop's own close
    capture._enable_record()
    try:
        assert capture._latency is not None
        assert capture._latency_close_requested is False
    finally:
        capture._close_latency()


def test_a_disable_with_a_stream_open_still_asks_for_it_to_be_closed(project_info):
    capture = _capture(project_info)
    capture._enable_record()
    try:
        capture._disable_record()
        assert capture._latency_close_requested is True
    finally:
        capture._close_latency()


def test_the_capture_loops_close_does_not_wait_for_the_writer(project_info):
    capture = _capture(project_info)
    writer = capture._latency = _SlowWriter(delay=1.0)
    capture._latency_close_requested = True

    started = time.perf_counter()
    capture._close_latency(wait=False)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, f"the loop waited {elapsed:.2f} s for a writer that needs 1 s"
    assert capture._latency is None
    assert capture._latency_close_requested is False
    assert not writer.finished.is_set(), "the close should still be running in the background"
    assert writer.finished.wait(5)
    assert writer.closed_on is not threading.current_thread()
    assert writer.closed_on.name == "LatencyClose-lifecycle"


def test_a_waiting_close_returns_once_the_stream_is_closed(project_info):
    capture = _capture(project_info)
    writer = capture._latency = _SlowWriter(delay=0.3)

    capture._close_latency()

    assert writer.finished.is_set()
    assert writer.closed_on is threading.current_thread()


def test_a_waiting_close_also_waits_for_an_earlier_asynchronous_one(project_info):
    capture = _capture(project_info)
    writer = capture._latency = _SlowWriter(delay=0.5)
    capture._close_latency(wait=False)
    assert not writer.finished.is_set()

    capture._close_latency()  # nothing open now: this is what process exit calls

    assert writer.finished.is_set()


def test_a_new_stream_waits_for_the_previous_one_to_finish_closing(project_info):
    capture = _capture(project_info)
    writer = capture._latency = _SlowWriter(delay=0.5)
    capture._close_latency(wait=False)
    assert not writer.finished.is_set()

    # Without a project nothing is opened, so this isolates the wait: with one,
    # the new writer would truncate the file the old one is still writing.
    capture._project_info = None
    capture._start_latency()

    assert writer.finished.is_set()


def test_a_close_thread_that_cannot_start_falls_back_to_closing_in_place(project_info, monkeypatch):
    class _NoThreads:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr("autotrainer.video.video_capture.threading.Thread", _NoThreads)
    capture = _capture(project_info)
    writer = capture._latency = _SlowWriter(delay=0.05)

    capture._close_latency(wait=False)

    assert writer.finished.is_set()
    assert capture._latency is None
