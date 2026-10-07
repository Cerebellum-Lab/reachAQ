import logging
import threading
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


def _stopped_snapshot(monkeypatch, tmp_path, *, end_raises=False, probe=None):
    """Arm, record one pose and stop; return what stop() handed to _write_session.

    end_raises makes the log's first end() stop the log and then raise, which is
    what a MemoryError while copying the tables does; the later end() in close()
    behaves normally. probe(recorder), if given, runs once the recorder exists."""
    laser = _EventSource("trace_received")
    laser.configuration = SimpleNamespace(backend="disabled")
    recorder = SessionDataRecorder(SimpleNamespace(timing_plan=None), laser)
    seen = {}
    monkeypatch.setattr(recorder, "_poll_nidaq", lambda: None)
    monkeypatch.setattr(recorder, "_open_nidaq_spool_locked", lambda: None)
    monkeypatch.setattr(recorder, "_snapshot_nidaq_locked", lambda: ())
    monkeypatch.setattr(recorder, "_write_session",
                        lambda **snapshot: seen.update(snapshot) or {})
    if end_raises:
        real_end = recorder.latency_events.end
        calls = []

        def end():
            calls.append(None)
            result = real_end()
            if len(calls) == 1:
                raise MemoryError("copying the latency tables")
            return result

        monkeypatch.setattr(recorder.latency_events, "end", end)
    if probe is not None:
        probe(recorder)
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


def test_a_failing_log_end_still_hands_the_session_to_finalization(monkeypatch, tmp_path):
    # The latency log is a diagnostic. If ending it raises, the session's own
    # finalization must still be set up, just without the events file.
    monkeypatch.delenv("REACHAQ_LATENCY_RECORD", raising=False)

    seen = _stopped_snapshot(monkeypatch, tmp_path, end_raises=True)

    assert "latency_events" in seen  # _write_session was reached
    assert seen["latency_events"] is None
    assert seen["end_perf"] == 12.0


def test_a_failing_log_end_is_logged_outside_the_recorder_lock(monkeypatch, tmp_path):
    # The session log handler takes the recorder's lock, so a log call made
    # while stop() holds it could wait on a thread that is mid-emit and holds
    # the handler lock: a lock-order inversion. Probe from another thread.
    monkeypatch.delenv("REACHAQ_LATENCY_RECORD", raising=False)
    records = []
    blocked = []

    class _Probe(logging.Handler):
        def __init__(self, recorder):
            super().__init__(logging.NOTSET)
            self._recorder = recorder

        def emit(self, record):
            if "Latency event log" not in record.getMessage():
                return
            records.append(record)

            def take():
                got = self._recorder._lock.acquire(timeout=2)
                if got:
                    self._recorder._lock.release()
                blocked.append(not got)

            thread = threading.Thread(target=take)
            thread.start()
            thread.join()

    handlers = []

    def probe(recorder):
        handlers.append(_Probe(recorder))
        logging.getLogger().addHandler(handlers[0])

    try:
        _stopped_snapshot(monkeypatch, tmp_path, end_raises=True, probe=probe)
    finally:
        for handler in handlers:
            logging.getLogger().removeHandler(handler)

    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None  # the failure itself is in the log
    assert blocked == [False]
