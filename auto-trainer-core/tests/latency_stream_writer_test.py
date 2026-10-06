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
