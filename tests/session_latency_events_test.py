from datetime import datetime
from types import SimpleNamespace

import h5py
import numpy as np

from autotrainer.core import ObservableObject, ProjectInfo
from autotrainer.core.latency.schema import LIVE_POSE_DTYPE
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


class _EventSource(ObservableObject):
    def __init__(self, *event_names):
        super().__init__(event_names)


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


def test_write_session_writes_the_gui_latency_rows(tmp_path):
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=1)
    tables = {"live_poses": np.array([(1, 1.0, 1.1, 1.2, 1.3)], dtype=LIVE_POSE_DTYPE)}

    SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),),
        latency_events=tables,
    )

    path = tmp_path / "20260102" / "test" / "session001" / "streams" / "latency" / "events.h5"
    with h5py.File(path, "r") as store:
        assert store["live_poses"]["live_sequence"].tolist() == [1]


def test_write_session_without_latency_rows_creates_no_latency_files(tmp_path):
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=1)

    SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),),
        latency_events=None,
    )

    streams = tmp_path / "20260102" / "test" / "session001" / "streams"
    assert (streams / "nidaq.h5").exists()  # the session itself was written
    assert not (streams / "latency").exists()


def _stopped_snapshot(monkeypatch, tmp_path):
    """Arm, record one pose and stop; return what stop() handed to _write_session."""
    laser = _EventSource("trace_received")
    laser.configuration = SimpleNamespace(backend="disabled")
    recorder = SessionDataRecorder(SimpleNamespace(timing_plan=None), laser)
    seen = {}
    monkeypatch.setattr(recorder, "_poll_nidaq", lambda: None)
    monkeypatch.setattr(recorder, "_open_nidaq_spool_locked", lambda: None)
    monkeypatch.setattr(recorder, "_snapshot_nidaq_locked", lambda: ())
    monkeypatch.setattr(recorder, "_write_session",
                        lambda **snapshot: seen.update(snapshot) or {})
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=10)
    try:
        recorder.arm(project)
        recorder.commit_start(10.0, 100.0)
        recorder.latency_events.record_live_pose(
            SimpleNamespace(sequence=3, perf_c=2.0, live_recv_perf_c=1.5, live_put_perf_c=2.1),
            2.2,
        )
        recorder.stop(12.0)
        assert not recorder.latency_events.is_active
    finally:
        recorder.close()
    return seen


def test_stop_hands_the_recorded_rows_to_the_session_write(monkeypatch, tmp_path):
    monkeypatch.delenv("REACHAQ_LATENCY_RECORD", raising=False)

    seen = _stopped_snapshot(monkeypatch, tmp_path)

    assert seen["latency_events"]["live_poses"]["live_sequence"].tolist() == [3]


def test_stop_hands_over_no_rows_when_the_record_was_off_at_arm(monkeypatch, tmp_path):
    # An empty events.h5 would make an "off" session look recorded: the
    # finalizer reports "absent" only when streams/latency holds no stream file.
    monkeypatch.setenv("REACHAQ_LATENCY_RECORD", "0")

    seen = _stopped_snapshot(monkeypatch, tmp_path)

    assert "latency_events" in seen
    assert seen["latency_events"] is None
