import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.view import preferences_content as preferences_module
from tools.acquisition.view.preferences_content import PreferencesContent


def _qapp():
    return QApplication.instance() or QApplication([])


def test_open_preferences_disable_when_session_leaves_ready(app_model):
    app = _qapp()
    previous_inference = app_model._inference
    app_model._inference = SimpleNamespace(is_enabled=False, model_location="")
    content = PreferencesContent(app_model.preferences, app_model)
    try:
        assert content._tabs.isEnabled()

        app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)
        app.processEvents()
        assert not content._tabs.isEnabled()

        app_model._set_session_recording_status(SessionRecordingStatus.READY)
        app.processEvents()
        assert content._tabs.isEnabled()
    finally:
        content.close()
        app_model._inference = previous_inference


def test_data_browser_starts_from_data_location(app_model, monkeypatch, tmp_path):
    _qapp()
    previous_inference = app_model._inference
    app_model._inference = SimpleNamespace(is_enabled=False, model_location="")
    content = PreferencesContent(app_model.preferences, app_model)
    selected = tmp_path / "selected"
    initial_output_location = app_model.output_location
    calls = []

    def choose_directory(_parent, _title, initial_location):
        calls.append(initial_location)
        return selected.as_posix()

    monkeypatch.setattr(
        preferences_module.QFileDialog,
        "getExistingDirectory",
        choose_directory,
    )
    try:
        content._browse_for_location("data")
        assert calls == [initial_output_location]
        assert content._data_location_edit.text() == selected.as_posix()
    finally:
        content.close()
        app_model._inference = previous_inference
