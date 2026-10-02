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


def test_a_closed_preferences_window_does_not_stop_a_session_starting(app_model):
    # christielab10, 2026-10-02: Preferences was opened and closed, then
    # Record stuck at "arming". The closed window's handler was still
    # subscribed, so ready -> arming called setEnabled on its deleted tabs
    # ("Internal C++ object (PySide6.QtWidgets.QTabWidget) already deleted"),
    # and the error left start_recording after ARMING.
    import shiboken6
    from tools.acquisition.view.preferences_dialog import PreferencesDialog

    app = _qapp()
    previous_inference = app_model._inference
    app_model._inference = SimpleNamespace(is_enabled=False, model_location="")
    try:
        # As MainWindow opens it: a local dialog that is freed after exec().
        dialog = PreferencesDialog(app_model.preferences, app_model)
        content = dialog.findChild(PreferencesContent)
        assert content is not None
        shiboken6.delete(dialog)
        app.processEvents()
        assert not shiboken6.isValid(content)

        app_model._set_session_recording_status(SessionRecordingStatus.ARMING)
        app.processEvents()
        app_model._set_session_recording_status(SessionRecordingStatus.READY)
        app.processEvents()
    finally:
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
