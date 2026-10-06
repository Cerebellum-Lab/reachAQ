import threading

import numpy as np

from autotrainer.core.fixed_array_multiqueue import FixedArrayMultiQueue
from autotrainer.inference.pose_process import take_newest_live_batch
from autotrainer.inference.pose_result_process import (
    close_in_background,
    split_pose_item,
    wait_for_latency_closers,
)


def _put_pair(queue, cam_frame_id, index):
    frame = np.zeros((4, 4), dtype=np.uint8)
    for camera in range(2):
        queue.put(frame, camera, index, block=False, frame_perf_c=1.0,
                  cam_frame_id=cam_frame_id + camera * 1000)


def test_the_drain_reports_skips_and_the_newest_batches_ids():
    queue = FixedArrayMultiQueue(depth=3, cam_count=2, frames_per_camera=1, shape=(4, 4),
                                 name="pose-latency")
    for number in range(3):
        _put_pair(queue, 10 + number, number)
    buffer = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    indices = np.zeros((2, 1), dtype=np.int64)
    ids = np.zeros((2, 1), dtype=np.int64)
    put_perf = np.zeros((2, 1))
    stats = {}

    took = take_newest_live_batch(queue, buffer, indices, None, drain=True,
                                  cam_frame_ids=ids, put_perf_c=put_perf, stats=stats)

    assert took is True
    assert stats["skipped"] == 2
    assert ids[:, 0].tolist() == [12, 1012]


def test_a_three_element_item_has_no_latency():
    assert split_pose_item(("pose", "live", "idx")) == ("pose", "live", "idx", None)


def test_a_four_element_item_keeps_its_latency():
    assert split_pose_item(("pose", "live", "idx", (1,)))[3] == (1,)


class _BlockedWriter:
    """A writer whose close() waits to be released, like a stalled disk."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed_on = None

    def close(self):
        self.entered.set()
        self.release.wait(3)
        self.closed_on = threading.current_thread()
        return {"failed": False, "rowsDropped": {}, "rowsRejected": 0}


def test_closing_in_the_background_does_not_block_the_caller():
    writer = _BlockedWriter()

    closer = close_in_background(writer, "PoseLatencyClose")

    try:
        # close() is under way and still blocked, yet the call has returned.
        assert writer.entered.wait(5)
        assert closer.is_alive()
        assert writer.closed_on is None
    finally:
        writer.release.set()
    closer.join(5)
    assert not closer.is_alive()
    assert writer.closed_on is closer
    assert closer is not threading.current_thread()
    assert closer.name == "PoseLatencyClose"
    assert closer.daemon


def test_waiting_for_closers_returns_once_they_have_finished():
    writer = _BlockedWriter()
    closers = [close_in_background(writer, "PoseLatencyClose")]
    assert writer.entered.wait(5)
    threading.Timer(0.2, writer.release.set).start()

    wait_for_latency_closers(closers)

    assert writer.closed_on is not None
    assert closers == []


def test_a_failing_close_does_not_escape_the_closer_thread(monkeypatch):
    escaped = []
    monkeypatch.setattr(threading, "excepthook", escaped.append)

    class Broken:
        def close(self):
            raise OSError("disk gone")

    closer = close_in_background(Broken(), "PoseLatencyClose")
    closer.join(5)

    assert not closer.is_alive()
    assert escaped == []


def test_when_no_thread_can_start_the_close_still_happens(monkeypatch):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    writer = _BlockedWriter()
    writer.release.set()

    assert close_in_background(writer, "PoseLatencyClose") is None
    assert writer.closed_on is threading.current_thread()
