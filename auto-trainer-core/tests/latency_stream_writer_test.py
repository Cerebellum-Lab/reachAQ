import json
import threading
import time

import h5py
import numpy
import pytest

from autotrainer.core.latency.schema import (
    CAMERA_FRAME_DTYPE,
    CLOCK_PAIRS,
    LATENCY_SCHEMA_VERSION,
)
from autotrainer.core.latency.stream_writer import LatencyStreamWriter, latency_stream_path


def _row(frame_id):
    return (frame_id, frame_id * 1000, 1.0 + frame_id, 1.001 + frame_id, 1.002 + frame_id, 0)


def test_rows_land_in_order_with_schema_and_stats(tmp_path):
    path = tmp_path / "streams" / "latency" / "camera_left.h5"
    writer = LatencyStreamWriter(path, {"frames": CAMERA_FRAME_DTYPE}, batch_rows=4,
                                 attrs={"camera": "left"})
    for frame_id in range(10):
        writer.append("frames", _row(frame_id))
    stats = writer.close()

    assert stats["rowsWritten"]["frames"] == 10
    assert stats["rowsDropped"]["frames"] == 0
    with h5py.File(path, "r") as store:
        assert store.attrs["schema_version"] == LATENCY_SCHEMA_VERSION
        assert store.attrs["camera"] == "left"
        frames = store["frames"][:]
        assert frames["frame_id"].tolist() == list(range(10))
        assert frames["arrival_perf"][3] == pytest.approx(4.001)
        assert json.loads(store.attrs["writer_stats"])["rowsWritten"]["frames"] == 10


def test_every_stream_carries_clock_pairs(tmp_path):
    path = tmp_path / "pairs.h5"
    writer = LatencyStreamWriter(path, {"frames": CAMERA_FRAME_DTYPE}, clock_pair_period=0.01)
    time.sleep(0.05)
    writer.close()
    with h5py.File(path, "r") as store:
        pairs = store[CLOCK_PAIRS][:]
    assert len(pairs) >= 2
    assert numpy.all(numpy.diff(pairs["perf"]) > 0)
    # Wall and perf advance together between consecutive pairs.
    assert abs(numpy.diff(pairs["wall"])[-1] - numpy.diff(pairs["perf"])[-1]) < 0.01


def test_a_full_queue_drops_and_counts_instead_of_blocking(tmp_path, monkeypatch):
    gate = threading.Event()
    writer = LatencyStreamWriter(tmp_path / "slow.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=1, queue_batches=1)
    original = writer._write

    def slow_write(datasets, name, batch):
        if name == "frames":
            gate.wait(5)
        original(datasets, name, batch)

    monkeypatch.setattr(writer, "_write", slow_write)
    started = time.perf_counter()
    for frame_id in range(50):
        writer.append("frames", _row(frame_id))
    elapsed = time.perf_counter() - started
    gate.set()
    stats = writer.close()

    assert elapsed < 0.5, "append must never wait for the writer thread"
    assert stats["rowsDropped"]["frames"] > 0
    assert stats["rowsWritten"]["frames"] + stats["rowsDropped"]["frames"] == 50


def test_an_unwritable_path_fails_without_raising_into_the_caller(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    writer = LatencyStreamWriter(blocker / "sub" / "x.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=2)
    for frame_id in range(6):
        writer.append("frames", _row(frame_id))
    stats = writer.close()

    assert stats["failed"] is True
    assert stats["firstError"]
    assert stats["rowsWritten"]["frames"] == 0
    assert stats["rowsDropped"]["frames"] == 6


def test_append_after_close_is_ignored(tmp_path):
    writer = LatencyStreamWriter(tmp_path / "x.h5", {"frames": CAMERA_FRAME_DTYPE})
    writer.close()
    writer.append("frames", _row(1))
    assert writer.close()["rowsWritten"]["frames"] == 0


def test_stream_path_follows_the_session(project_info):
    path = latency_stream_path(project_info, "pose")
    assert path.parts[-3:] == ("streams", "latency", "pose.h5")
    assert str(path).startswith(project_info.get_session_path().location)


def test_the_record_can_be_switched_off(monkeypatch):
    from autotrainer.core.latency import latency_recording_enabled

    monkeypatch.setenv("REACHAQ_LATENCY_RECORD", "0")
    assert latency_recording_enabled() is False
    monkeypatch.setenv("REACHAQ_LATENCY_RECORD", "1")
    assert latency_recording_enabled() is True
    monkeypatch.delenv("REACHAQ_LATENCY_RECORD")
    assert latency_recording_enabled() is True


def test_append_never_raises_on_unknown_dataset_or_bad_row(tmp_path):
    """Unknown dataset names, wrong arity, type errors (list, None), and overflow are silently rejected."""
    writer = LatencyStreamWriter(tmp_path / "x.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=4)
    # Unknown dataset: should not raise
    writer.append("no_such_dataset", _row(0))
    # Wrong arity tuple: should not raise
    writer.append("frames", (1,))
    # List row instead of tuple: should not raise (TypeError from numpy)
    writer.append("frames", [1, 1000, 1.0, 1.001, 1.002, 0])
    # None row: should not raise (TypeError from numpy)
    writer.append("frames", None)
    # Good rows: should work
    writer.append("frames", _row(1))
    writer.append("frames", _row(2))
    stats = writer.close()

    # All 4 bad appends are counted in the single rowsRejected counter
    assert stats["rowsRejected"] == 4
    assert stats["rowsWritten"]["frames"] == 2
    with h5py.File(tmp_path / "x.h5", "r") as store:
        frames = store["frames"][:]
        # Only the good rows landed
        assert frames["frame_id"].tolist() == [1, 2]


def test_close_returns_quickly_even_if_the_final_write_fails(tmp_path, monkeypatch):
    """If the closing clock-pair write fails, close() still stops the thread and closes the file."""
    writer = LatencyStreamWriter(tmp_path / "x.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=2, clock_pair_period=10.0)
    original_write = writer._write

    def write_that_fails_on_closing_clock_pair(datasets, name, batch):
        if name == CLOCK_PAIRS and writer._stop_event.is_set():
            raise RuntimeError("simulated closing clock-pair write failure")
        original_write(datasets, name, batch)

    monkeypatch.setattr(writer, "_write", write_that_fails_on_closing_clock_pair)
    writer.append("frames", _row(0))
    # Wait for the opening clock pair to be written, so the failure is the closing one.
    started_wait = time.perf_counter()
    while time.perf_counter() - started_wait < 2.0:
        if writer.stats["rowsWritten"][CLOCK_PAIRS] >= 1:
            break
        time.sleep(0.01)

    started = time.perf_counter()
    stats = writer.close(timeout=2.0)
    elapsed = time.perf_counter() - started

    # Should return quickly, not wait the full timeout.
    assert elapsed < 1.0, f"close took {elapsed} s, expected < 1 s"
    assert stats["failed"] is True
    assert "simulated closing clock-pair write failure" in stats["firstError"]
    # The file should have been closed by the finally block.
    with h5py.File(tmp_path / "x.h5", "r") as store:
        assert store.attrs["schema_version"] == LATENCY_SCHEMA_VERSION


def test_close_with_blocked_writer_and_full_queue_respects_deadline(tmp_path, monkeypatch):
    """When the writer is blocked and the queue is full, close() respects a single deadline."""
    gate = threading.Event()
    writer = LatencyStreamWriter(tmp_path / "slow.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=1, queue_batches=1)
    original = writer._write

    def slow_write(datasets, name, batch):
        if name == "frames":
            gate.wait(10)  # Wait for signal or timeout
        original(datasets, name, batch)

    monkeypatch.setattr(writer, "_write", slow_write)
    # The writer is now blocked on the gate. Append one row to fill the batch and queue it.
    writer.append("frames", _row(0))
    time.sleep(0.05)  # Give the writer thread time to start processing the first batch.
    # Now the writer is blocked on the gate, holding one batch. Append one more row
    # to fill a second batch. The queue has 1 slot and already has the first batch queued,
    # so this second batch will be dropped (queue is full).
    writer.append("frames", _row(1))
    time.sleep(0.05)  # Give time for the second append to be processed.
    # Append more rows that will be dropped due to queue saturation.
    for frame_id in range(2, 10):
        writer.append("frames", _row(frame_id))

    started = time.perf_counter()
    stats = writer.close(timeout=0.5)
    elapsed = time.perf_counter() - started

    # With a single deadline, close should not wait longer than deadline + some slack.
    # If two independent timeouts were used, it could wait up to 2x timeout.
    assert elapsed < 1.0, f"close took {elapsed} s with 0.5 s timeout; expected < 1 s (one deadline)"
    # Some rows should have been dropped due to queue overflow.
    assert stats["rowsDropped"]["frames"] > 0
    # After close returns, release the gate and verify the thread exits promptly.
    gate.set()
    writer._thread.join(timeout=5.0)
    # The thread should have exited.
    assert not writer._thread.is_alive(), "writer thread should exit after gate opens"


def test_queue_full_at_close_with_closing_write_failure(tmp_path, monkeypatch):
    """Queue full at close + closing write fails: thread exits, file closed, no spurious TimeoutError."""
    gate = threading.Event()
    writer = LatencyStreamWriter(tmp_path / "full_queue_fail.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=1, queue_batches=1, clock_pair_period=0.05)
    original_write = writer._write

    def slow_write_frames(datasets, name, batch):
        if name == "frames":
            gate.wait(10)
        original_write(datasets, name, batch)

    def write_fails_on_closing(datasets, name, batch):
        slow_write_frames(datasets, name, batch)
        # Fail only the closing clock-pair write.
        if name == CLOCK_PAIRS and writer._stop_event.is_set():
            raise RuntimeError("closing write failure with full queue")

    monkeypatch.setattr(writer, "_write", write_fails_on_closing)
    # Wait for the opening clock pair to be written.
    started_wait = time.perf_counter()
    while time.perf_counter() - started_wait < 2.0:
        if writer.stats["rowsWritten"][CLOCK_PAIRS] >= 1:
            break
        time.sleep(0.01)
    # Fill the queue: writer is blocked on batch 0, append batch 1 to fill the single slot.
    writer.append("frames", _row(0))
    time.sleep(0.05)
    writer.append("frames", _row(1))
    time.sleep(0.05)
    # Release gate after 0.2s so close() can proceed but the closing write will fail.
    release_timer = threading.Timer(0.2, gate.set)
    release_timer.start()

    started = time.perf_counter()
    stats = writer.close(timeout=3.0)
    elapsed = time.perf_counter() - started
    release_timer.cancel()

    # close() should return well under 3s despite queue full and closing write failure.
    assert elapsed < 1.5, f"close took {elapsed} s; expected < 1.5 s"
    assert stats["failed"] is True
    # The failure should be from the closing write, not from thread timeout.
    assert "closing write failure with full queue" in stats["firstError"]
    assert "did not stop" not in stats["firstError"]
    # The file should have been closed by the finally block.
    with h5py.File(tmp_path / "full_queue_fail.h5", "r") as store:
        assert store.attrs["schema_version"] == LATENCY_SCHEMA_VERSION
    # Verify thread exited after close().
    writer._thread.join(2.0)
    assert not writer._thread.is_alive()
