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
    """Unknown dataset names, wrong arity, and type errors are silently rejected."""
    writer = LatencyStreamWriter(tmp_path / "x.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=4)
    # Unknown dataset: should not raise
    writer.append("no_such_dataset", _row(0))
    # Wrong arity: should not raise
    writer.append("frames", (1,))
    # Good row: should work
    writer.append("frames", _row(1))
    writer.append("frames", _row(2))
    stats = writer.close()

    assert stats["rowsRejected"]["frames"] == 1
    assert stats["rowsRejected"]["no_such_dataset"] == 1
    assert stats["rowsWritten"]["frames"] == 2
    with h5py.File(tmp_path / "x.h5", "r") as store:
        frames = store["frames"][:]
        # Only the good rows landed
        assert frames["frame_id"].tolist() == [1, 2]


def test_close_returns_quickly_even_if_the_final_write_fails(tmp_path, monkeypatch):
    """If the closing clock-pair write fails, close() still stops the thread and closes the file."""
    writer = LatencyStreamWriter(tmp_path / "x.h5", {"frames": CAMERA_FRAME_DTYPE},
                                 batch_rows=2)
    original_write = writer._write

    def write_that_fails_on_clock_pairs(datasets, name, batch):
        if name == CLOCK_PAIRS:
            raise RuntimeError("simulated final write failure")
        original_write(datasets, name, batch)

    monkeypatch.setattr(writer, "_write", write_that_fails_on_clock_pairs)
    writer.append("frames", _row(0))
    started = time.perf_counter()
    stats = writer.close(timeout=2.0)
    elapsed = time.perf_counter() - started

    # Should return quickly, not wait the full timeout.
    assert elapsed < 1.0, f"close took {elapsed} s, expected < 1 s"
    assert stats["failed"] is True
    assert "RuntimeError" in stats["firstError"]
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
    # Fill the queue with one pending batch
    writer.append("frames", _row(0))
    # This append will block on the full queue check, but since the lock is brief,
    # we can trigger it and then attempt close before the gate opens.
    time.sleep(0.05)  # Give the writer thread time to start processing.

    started = time.perf_counter()
    close_stats = writer.close(timeout=0.5)
    elapsed = time.perf_counter() - started

    # With a single deadline, close should not wait longer than deadline + some slack.
    # If two independent timeouts were used, it could wait up to 2x timeout.
    assert elapsed < 1.0, f"close took {elapsed} s with 0.5 s timeout; expected < 1 s (one deadline)"
    # After close returns, release the gate and give the thread time to exit.
    gate.set()
    time.sleep(0.1)
    # The thread should have exited (or exited after gate opens).
    assert not writer._thread.is_alive(), "writer thread should exit shortly after gate opens"
