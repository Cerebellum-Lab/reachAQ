import json
import multiprocessing
import threading

import pytest

from tools.acquisition.model.session_validation_controller import (
    SessionValidationController,
)


def test_validation_controller_publishes_automatic_quick_report(tmp_path, monkeypatch):
    session = tmp_path / "session001"
    session.mkdir()

    class Report:
        counts = {"pass": 1, "warning": 0, "fail": 0, "tool_error": 0, "not_applicable": 0}
        exit_code = 0
        def to_record(self):
            return {"profile": "quick", "counts": self.counts}

    monkeypatch.setattr(
        "tools.acquisition.model.session_validation_controller.validate_session",
        lambda *_args, **_kwargs: Report(),
    )
    controller = SessionValidationController(mp_ctx=multiprocessing.get_context("fork"))
    output = controller.start(session, "quick", automatic=True, reduced_priority=False)
    state = controller.wait(5)

    assert state.status == "pass"
    assert output == session / "validation/quick.json"
    assert json.loads(output.read_text(encoding="utf-8"))["profile"] == "quick"


def test_validation_controller_uses_unique_manual_reports(tmp_path, monkeypatch):
    session = tmp_path / "session001"
    session.mkdir()

    class Report:
        counts = {"pass": 1, "warning": 0, "fail": 0, "tool_error": 0, "not_applicable": 0}
        exit_code = 0
        def to_record(self):
            return {"profile": "fast"}

    monkeypatch.setattr(
        "tools.acquisition.model.session_validation_controller.validate_session",
        lambda *_args, **_kwargs: Report(),
    )
    controller = SessionValidationController(mp_ctx=multiprocessing.get_context("fork"))
    first = controller.start(session, "fast", automatic=False, reduced_priority=False)
    controller.wait(5)
    second = controller.start(session, "fast", automatic=False, reduced_priority=False)
    controller.wait(5)
    assert first != second


def test_validation_controller_delivers_every_state_outside_its_lock(tmp_path, monkeypatch):
    session = tmp_path / "session001"
    session.mkdir()

    class Report:
        counts = {"pass": 1, "warning": 0, "fail": 0, "tool_error": 0, "not_applicable": 0}
        exit_code = 0
        def to_record(self):
            return {"profile": "quick", "counts": self.counts}

    monkeypatch.setattr(
        "tools.acquisition.model.session_validation_controller.validate_session",
        lambda *_args, **_kwargs: Report(),
    )
    taken = []

    def callback(state):
        # From another thread: an RLock answers its own holder, not the question.
        def probe():
            got = controller._lock.acquire(timeout=1)
            if got:
                controller._lock.release()
            taken.append((state.status, got))
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()

    controller = SessionValidationController(callback, mp_ctx=multiprocessing.get_context("fork"))
    controller.start(session, "quick", automatic=True, reduced_priority=False)
    controller.wait(5)

    assert [status for status, _ in taken] == ["pending", "running", "pass"]
    assert all(got for _, got in taken), taken


def test_validation_controller_final_state_cannot_overtake_start_states(tmp_path, monkeypatch):
    session = tmp_path / "session001"
    session.mkdir()

    class Report:
        counts = {"pass": 1, "warning": 0, "fail": 0, "tool_error": 0, "not_applicable": 0}
        exit_code = 0
        def to_record(self):
            return {"profile": "quick", "counts": self.counts}

    monkeypatch.setattr(
        "tools.acquisition.model.session_validation_controller.validate_session",
        lambda *_args, **_kwargs: Report(),
    )
    delivered = []
    final_delivered = threading.Event()

    def callback(state):
        if state.status == "pending":
            # A monitor started before delivery would deliver "pass" during this wait.
            final_delivered.wait(0.5)
        delivered.append(state.status)
        if state.status == "pass":
            final_delivered.set()

    controller = SessionValidationController(callback, mp_ctx=multiprocessing.get_context("fork"))
    controller.start(session, "quick", automatic=True, reduced_priority=False)
    controller.wait(5)

    assert delivered == ["pending", "running", "pass"]


def test_validation_controller_monitors_the_process_when_a_callback_raises(tmp_path, monkeypatch):
    session = tmp_path / "session001"
    session.mkdir()

    class Report:
        counts = {"pass": 1, "warning": 0, "fail": 0, "tool_error": 0, "not_applicable": 0}
        exit_code = 0
        def to_record(self):
            return {"profile": "quick", "counts": self.counts}

    monkeypatch.setattr(
        "tools.acquisition.model.session_validation_controller.validate_session",
        lambda *_args, **_kwargs: Report(),
    )
    delivered = []

    def callback(state):
        delivered.append(state.status)
        if state.status == "pending":
            raise ValueError("callback failed")

    controller = SessionValidationController(callback, mp_ctx=multiprocessing.get_context("fork"))
    with pytest.raises(ValueError):
        controller.start(session, "quick", automatic=True, reduced_priority=False)
    state = controller.wait(5)

    assert state.status == "pass"
    assert delivered[-1] == "pass"
