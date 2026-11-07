import logging
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from autotrainer.core import ObservableObject, ProjectInfo
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


class _EventSource(ObservableObject):
    def __init__(self, *event_names):
        super().__init__(event_names)


def _recorder(monkeypatch):
    laser = _EventSource("trace_received")
    laser.configuration = SimpleNamespace(backend="disabled")
    recorder = SessionDataRecorder(SimpleNamespace(timing_plan=None), laser)
    monkeypatch.setattr(recorder, "_poll_nidaq", lambda: None)
    monkeypatch.setattr(recorder, "_open_nidaq_spool_locked", lambda: None)
    monkeypatch.setattr(recorder, "_snapshot_nidaq_locked", lambda: ())
    return recorder


def _pending(project, send_perf_time=10.25):
    return {
        "session_id": project.short_id, "operation_id": "send-1", "trial_id": 1,
        "attempt_id": 1, "attempt_label": "1.1", "send_perf_time": send_perf_time,
        "outcome": "pending_analysis",
    }


def _published_session(tmp_path, session):
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=session)
    SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (),
        trial_records=(_pending(project),), trial_summary={"pending_analysis_attempts": 1},
    )
    return project


def _bytes(session_dir):
    return {path.relative_to(session_dir).as_posix(): path.read_bytes()
            for path in session_dir.rglob("*") if path.is_file()}


def test_a_sealed_session_refuses_late_ledger_and_tracking_writes(monkeypatch, tmp_path, caplog):
    recorder = _recorder(monkeypatch)
    project = _published_session(tmp_path, 7)
    session_dir = Path(project.get_session_path().location)
    before = _bytes(session_dir)
    try:
        recorder.seal_session(project.short_id)
        with caplog.at_level(logging.ERROR):
            recorder.persist_trial_ledger(
                project, ({**_pending(project), "outcome": "success"},),
                {"pending_analysis_attempts": 0, "scored_trials": 1})
            recorder.persist_trial_tracking(project, SimpleNamespace(trial_id=1, attempt_id=1))
    finally:
        recorder.close()

    assert _bytes(session_dir) == before
    assert not (session_dir / "streams" / "tracking").exists()
    refused = [record.getMessage() for record in caplog.records
               if record.levelno == logging.ERROR and "refused" in record.getMessage()]
    assert refused == [
        f"Late trial ledger write for {project.short_id} refused: "
        "the session's files were published and sealed",
        f"Late trial tracking write for {project.short_id} refused: "
        "the session's files were published and sealed",
    ]


def test_a_seal_waits_for_a_write_in_flight_and_refuses_the_next(monkeypatch, tmp_path, caplog):
    recorder = _recorder(monkeypatch)
    project = _published_session(tmp_path, 7)
    session_dir = Path(project.get_session_path().location)
    trials = session_dir / "streams" / "trials.jsonl"
    real_update = SessionDataRecorder.update_persisted_trial_ledger
    in_flight = threading.Event()
    release = threading.Event()
    order = []

    def blocked_update(*args):
        # The ledger write has passed the seal check and is now reading and
        # rewriting the session's files.
        in_flight.set()
        release.wait(5)
        real_update(*args)
        order.append("write finished")

    def seal():
        recorder.seal_session(project.short_id)
        order.append("sealed")

    monkeypatch.setattr(recorder, "update_persisted_trial_ledger", blocked_update)
    writer = threading.Thread(target=recorder.persist_trial_ledger, args=(
        project, ({**_pending(project), "outcome": "success"},),
        {"pending_analysis_attempts": 0, "scored_trials": 1}))
    sealer = threading.Thread(target=seal)
    try:
        try:
            writer.start()
            assert in_flight.wait(5)
            sealer.start()
            sealer.join(0.3)
            assert sealer.is_alive(), "the seal returned while a ledger write was in flight"
        finally:
            release.set()
            writer.join(5)
            if sealer.ident is not None:
                sealer.join(5)
        assert order == ["write finished", "sealed"]
        assert b'"success"' in trials.read_bytes()
        before = _bytes(session_dir)
        with caplog.at_level(logging.ERROR):
            recorder.persist_trial_ledger(
                project, ({**_pending(project), "outcome": "failure"},),
                {"pending_analysis_attempts": 0, "scored_trials": 1})
    finally:
        recorder.close()

    assert _bytes(session_dir) == before
    assert [record.getMessage() for record in caplog.records
            if record.levelno == logging.ERROR and "refused" in record.getMessage()] == [
        f"Late trial ledger write for {project.short_id} refused: "
        "the session's files were published and sealed",
    ]


def test_a_seal_names_one_session_only(monkeypatch, tmp_path):
    recorder = _recorder(monkeypatch)
    sealed = _published_session(tmp_path, 7)
    other = _published_session(tmp_path, 8)
    trials = Path(other.get_session_path().location) / "streams" / "trials.jsonl"
    before = trials.read_bytes()
    try:
        recorder.seal_session(sealed.short_id)
        recorder.persist_trial_ledger(
            other, ({**_pending(other), "outcome": "success"},),
            {"pending_analysis_attempts": 0, "scored_trials": 1})
    finally:
        recorder.close()

    assert trials.read_bytes() != before
    assert b'"success"' in trials.read_bytes()


def test_arming_the_next_session_lifts_the_seal(monkeypatch, tmp_path):
    recorder = _recorder(monkeypatch)
    sealed = _published_session(tmp_path, 7)
    following = ProjectInfo(root=str(tmp_path), device_id="test",
                            when=datetime(2026, 1, 2, 3, 4, 5), session=8)
    try:
        recorder.seal_session(sealed.short_id)
        recorder.arm(following)
        recorder.commit_start(20.0, 200.0)
        # The seal is per recorder, not per project: after arm() even the sealed
        # id would write again, and the next session certainly does.
        recorder.persist_trial_ledger(
            following, (_pending(following, 20.25),), {"pending_analysis_attempts": 1})
        trials = Path(following.get_session_path().location) / "streams" / "trials.jsonl"
        assert trials.is_file()
        assert recorder._sealed_session_id is None
    finally:
        recorder.close()
