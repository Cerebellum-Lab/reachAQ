import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime

import h5py
import numpy as np
import pytest

from autotrainer.core import ProjectInfo
from autotrainer.core.latency import LatencyStreamWriter, latency_stream_path
from autotrainer.core.latency.schema import CAMERA_FRAME_DTYPE, RECORD_BATCH_DTYPE
from tools.acquisition.model.session_data_recorder import SessionDataRecorder
from tools.latency import finalize


def _nidaq_chunk():
    return (
        np.arange(4, dtype=np.int64),
        np.array([9.9, 10.0, 11.0, 12.1]),
        np.array([99.9, 100.0, 101.0, 102.1]),
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        ("force",),
        1000.0,
        1,
        0,
        0,
    )


def test_finalization_builds_latency_and_lists_it_in_the_manifest(tmp_path):
    project = _latency_project(tmp_path)

    result = SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),), latency_events={},
    )

    session_dir = tmp_path / "20260102" / "test" / "session001"
    # The recording loop has batches but no recorder rows, so it is partial.
    assert result["latency"]["status"] == "partial"
    assert result["latency"]["output"] == "latency.h5"
    assert (session_dir / "streams" / "latency.h5").is_file()
    with h5py.File(session_dir / "streams" / "latency.h5", "r") as store:
        assert store.attrs["session_id"] == project.short_id
    manifest = json.loads((session_dir / "streams" / "stream_manifest.json").read_text())
    listed = json.dumps(manifest["files"])
    assert "latency.h5" in listed
    assert "camera_left.h5" in listed
    assert "events.h5" in listed


def test_a_session_without_latency_streams_still_publishes(tmp_path):
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=1)

    result = SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),),
    )

    assert result["latency"]["status"] == "absent"
    assert result["metadataGenerationId"]


def _camera_stream(project, name):
    writer = LatencyStreamWriter(latency_stream_path(project, f"camera_{name}"),
                                 {"frames": CAMERA_FRAME_DTYPE,
                                  "record_batches": RECORD_BATCH_DTYPE})
    for frame_id in range(5):
        writer.append("frames", (frame_id, frame_id * 6_666_667, 10.0 + frame_id / 150,
                                 10.002 + frame_id / 150, 10.0021 + frame_id / 150, 0))
    writer.append("record_batches", (0, 4, 5, 10.1, 10.1001, 1, False))
    writer.close()


def _latency_project(tmp_path):
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=1)
    _camera_stream(project, "left")
    return project


class _Holder:
    """A process that keeps a raw stream open for writing, as a capture process does while closing it."""

    def __init__(self, path):
        code = ("import sys, h5py\n"
                "store = h5py.File(sys.argv[1], 'a')\n"
                "print('ready', flush=True)\n"
                "sys.stdin.read()\n"
                "store.close()\n")
        self.process = subprocess.Popen([sys.executable, "-c", code, str(path)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert self.process.stdout.readline().strip() == "ready"

    def release(self):
        self.process.stdin.close()
        self.process.wait(timeout=20)


def _hold_open(path):
    holder = _Holder(path)
    try:
        with h5py.File(path, "r"):
            pass
    except OSError:
        return holder
    holder.release()
    pytest.skip("HDF5 file locking is not in effect here, so a held file stays readable")


def test_listed_latency_files_match_what_is_on_disk(tmp_path):
    # session.manifest compares each listed file's size and sha256 with the
    # file, so the analysis has to be finished before the manifest is built.
    project = _latency_project(tmp_path)
    SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),), latency_events={},
    )

    session_dir = tmp_path / "20260102" / "test" / "session001"
    manifest = json.loads((session_dir / "streams" / "stream_manifest.json").read_text())
    listed = {item["path"]: item for item in manifest["files"]}
    for relative in ("streams/latency.h5", "streams/latency/camera_left.h5",
                     "streams/latency/events.h5"):
        assert relative in listed
        data = (session_dir / relative).read_bytes()
        assert listed[relative]["sizeBytes"] == len(data)
        assert listed[relative]["sha256"] == hashlib.sha256(data).hexdigest()


def test_the_narrower_manifest_rewrite_keeps_the_latency_files_listed(tmp_path):
    project = _latency_project(tmp_path)
    record = {
        "session_id": project.short_id,
        "operation_id": "send-1",
        "trial_id": 1,
        "attempt_id": 1,
        "attempt_label": "1.1",
        "send_perf_time": 10.25,
        "outcome": "pending_analysis",
    }
    SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),), latency_events={},
        trial_records=(record,), trial_summary={"pending_analysis_attempts": 1},
    )

    SessionDataRecorder.update_persisted_trial_ledger(
        project, ({**record, "outcome": "success"},), {"pending_analysis_attempts": 0},
    )

    session_dir = tmp_path / "20260102" / "test" / "session001"
    manifest = json.loads((session_dir / "streams" / "stream_manifest.json").read_text())
    listed = {item["path"] for item in manifest["files"]}
    assert {"streams/latency.h5", "streams/latency/camera_left.h5",
            "streams/latency/events.h5"} <= listed


def _write_with_a_held_stream(tmp_path, monkeypatch):
    """Finalize a session whose camera_right stream is still held open by its writer."""
    project = _latency_project(tmp_path)
    _camera_stream(project, "right")
    held = latency_stream_path(project, "camera_right")
    monkeypatch.setattr(finalize, "OPEN_RETRY_SECONDS", 0.3)
    holder = _hold_open(held)
    try:
        result = SessionDataRecorder._write_session(
            project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),), latency_events={},
        )
    finally:
        holder.release()
    return project, result


def _manifest(tmp_path):
    session_dir = tmp_path / "20260102" / "test" / "session001"
    manifest = json.loads((session_dir / "streams" / "stream_manifest.json").read_text())
    return session_dir, {item["path"]: item for item in manifest["files"]}


def test_a_raw_stream_still_being_closed_is_left_out_of_the_manifest(tmp_path, monkeypatch, caplog):
    # A capture or pose process closes its stream with no handshake. If that is slower
    # than the finalizer's retry, hashing the file into the manifest records bytes that
    # are still changing, and session.manifest then fails on a sha256 mismatch.
    with caplog.at_level(logging.WARNING):
        _, result = _write_with_a_held_stream(tmp_path, monkeypatch)

    session_dir, listed = _manifest(tmp_path)
    assert result["latency"]["status"] == "partial"
    assert [item["path"] for item in result["latency"]["unreadable"]] == [
        "streams/latency/camera_right.h5"]
    assert "streams/latency/camera_right.h5" not in listed
    assert (session_dir / "streams" / "latency" / "camera_right.h5").is_file()
    assert any("camera_right.h5" in record.getMessage() and "manifest" in record.getMessage()
               for record in caplog.records)
    for relative in ("streams/latency.h5", "streams/latency/camera_left.h5",
                     "streams/latency/events.h5"):
        data = (session_dir / relative).read_bytes()
        assert listed[relative]["sizeBytes"] == len(data)
        assert listed[relative]["sha256"] == hashlib.sha256(data).hexdigest()


def test_the_narrower_rewrite_does_not_list_what_finalization_left_out(tmp_path, monkeypatch):
    # By the time the trial ledger is rewritten the held stream is closed and readable,
    # but nothing there can say it was final when finalization looked, so a latency
    # stream is listed only if the manifest already listed it.
    project, _ = _write_with_a_held_stream(tmp_path, monkeypatch)
    record = {"session_id": project.short_id, "operation_id": "send-1", "trial_id": 1,
              "attempt_id": 1, "attempt_label": "1.1", "send_perf_time": 10.25,
              "outcome": "pending_analysis"}

    SessionDataRecorder.update_persisted_trial_ledger(
        project, ({**record, "outcome": "success"},), {"pending_analysis_attempts": 0},
    )

    _, listed = _manifest(tmp_path)
    assert "streams/latency/camera_right.h5" not in listed
    assert {"streams/latency.h5", "streams/latency/camera_left.h5",
            "streams/latency/events.h5"} <= set(listed)
