from types import SimpleNamespace

from tools.acquisition.model.app_model import AppModel


def test_stopped_session_notes_use_the_snapshotted_project(monkeypatch):
    model = object.__new__(AppModel)
    project = SimpleNamespace(short_id="animal_session005")
    model._editable_notes_project = project
    calls = []
    monkeypatch.setattr(
        model,
        "_save_project_metadata",
        lambda saved_project, **kwargs: calls.append((saved_project, kwargs)),
    )

    assert model.persist_stopped_session_notes()
    assert calls == [(project, {"caller": "session_notes_updated"})]


def test_notes_save_is_a_noop_without_a_stopped_session(monkeypatch):
    model = object.__new__(AppModel)
    model._editable_notes_project = None
    monkeypatch.setattr(
        model,
        "_save_project_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    assert not model.persist_stopped_session_notes()
