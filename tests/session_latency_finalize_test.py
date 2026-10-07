import hashlib
import json
from datetime import datetime

import numpy as np

from autotrainer.core import ProjectInfo
from autotrainer.core.latency import LatencyStreamWriter, latency_stream_path
from autotrainer.core.latency.schema import CAMERA_FRAME_DTYPE
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


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
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=1)
    writer = LatencyStreamWriter(latency_stream_path(project, "camera_left"),
                                 {"frames": CAMERA_FRAME_DTYPE})
    for frame_id in range(5):
        writer.append("frames", (frame_id, frame_id * 6_666_667, 10.0 + frame_id / 150,
                                 10.002 + frame_id / 150, 10.0021 + frame_id / 150, 0))
    writer.close()

    result = SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),), latency_events={},
    )

    session_dir = tmp_path / "20260102" / "test" / "session001"
    assert result["latency"]["status"] in {"complete", "partial"}
    assert (session_dir / "streams" / "latency.h5").is_file()
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


def _latency_project(tmp_path):
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=1)
    writer = LatencyStreamWriter(latency_stream_path(project, "camera_left"),
                                 {"frames": CAMERA_FRAME_DTYPE})
    for frame_id in range(5):
        writer.append("frames", (frame_id, frame_id * 6_666_667, 10.0 + frame_id / 150,
                                 10.002 + frame_id / 150, 10.0021 + frame_id / 150, 0))
    writer.close()
    return project


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
