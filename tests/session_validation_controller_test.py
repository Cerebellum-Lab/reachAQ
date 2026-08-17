import json
import multiprocessing

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
