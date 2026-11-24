import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import zipfile

import pytest
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

from operator_alerts_test import FakeModel  # noqa: E402
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState
from tools.acquisition.view import operator_alerts_view
from tools.acquisition.view.operator_alerts_view import install_operator_alerts


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class Preferences:
    log_location = ""

    def __init__(self, path):
        self.settings_file_name = str(path)


def _install(qapp, tmp_path):
    window = QMainWindow()
    window.statusBar()
    model = FakeModel()
    model.output_location = str(tmp_path / "data")
    model.status = "idle"
    model.acquisition_started = False
    model.recording_blockers = ()
    model.get_config_location = lambda: tmp_path / "system_configuration.yaml"
    ui = install_operator_alerts(window, model, Preferences(tmp_path / "prefs.conf"))
    window.show()
    qapp.processEvents()
    return window, model, ui


def test_banner_follows_the_most_severe_visible_alert(qapp, tmp_path):
    window, model, ui = _install(qapp, tmp_path)
    assert not ui.toolbar.isVisible()
    model.set_status(SubsystemId.LASER, SubsystemState.FAILED, error="controller fault")
    model.on_error("Storage below 30 minutes", "29 min left")
    qapp.processEvents()
    assert ui.toolbar.isVisible()
    assert ui.banner.title_label.text() == "Error: Laser failed"
    assert ui.banner.remedy_label.full_text().startswith("What to do: Check the laser")
    assert ui.banner.more_label.text() == "+1 more"
    ui.banner.dismiss_button.click()
    qapp.processEvents()
    assert ui.banner.title_label.text() == "Warning: Storage below 30 minutes"
    ui.banner.dismiss_button.click()
    qapp.processEvents()
    assert not ui.toolbar.isVisible()
    ui.show_alert_log()
    assert ui._log_dialog.table.rowCount() == 3
    ui.close()


def test_data_at_risk_opens_one_window_modal_dialog(qapp, tmp_path):
    window, model, ui = _install(qapp, tmp_path)
    model.on_error("Camera writer close timeout", "left did not close")
    model.on_error("Session data incomplete", "trial_summary.json")
    qapp.processEvents()
    boxes = [box for box in window.findChildren(QMessageBox) if box.isVisible()]
    assert len(boxes) == 1
    assert boxes[0].text() == "Camera writer close timeout"
    assert "What to do:" in boxes[0].informativeText()
    boxes[0].close()
    ui.close()


def test_help_menu_and_export(qapp, tmp_path, monkeypatch):
    window, model, ui = _install(qapp, tmp_path)
    titles = [action.text() for action in ui.help_menu.actions()]
    assert titles == ["Alerts…", "Export Diagnostic Bundle…"]
    target = tmp_path / "bundle.zip"
    monkeypatch.setattr(operator_alerts_view.QFileDialog, "getSaveFileName",
                        lambda *args, **kwargs: (str(target), ""))
    shown = []
    monkeypatch.setattr(operator_alerts_view.QMessageBox, "information",
                        lambda *args: shown.append(args[1]))
    monkeypatch.setattr(operator_alerts_view.QMessageBox, "warning",
                        lambda *args: shown.append(args[1:]))
    model.set_status(SubsystemId.LASER, SubsystemState.FAILED, error="fault")
    qapp.processEvents()
    # Built in a separate process, the way the app does it.
    ui.export_bundle()
    ui._export_thread.join(180)
    qapp.processEvents()
    assert shown == ["Diagnostic bundle written"]
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        assert {"operator_alerts.json", "subsystem_statuses.json", "app_state.json"} <= names
        assert b"Laser failed" in archive.read("operator_alerts.json")
    ui.close()


def test_export_skips_the_gpu_query_while_recording(qapp, tmp_path):
    from tools.acquisition.model.app_model_status import SessionRecordingStatus

    window, model, ui = _install(qapp, tmp_path)
    sources = ui.bundle_sources()
    command, environment, _ = ui.export_command(tmp_path / "b.zip", sources, tmp_path)
    assert "--no-gpu-query" not in command
    model.session_recording_status = SessionRecordingStatus.RECORDING
    command, environment, _ = ui.export_command(tmp_path / "b.zip", sources, tmp_path)
    assert "--no-gpu-query" in command
    assert command[1:3] == ["-m", "tools.acquisition.model.diagnostic_bundle"]
    ui.close()


def test_reconcile_runs_on_the_gui_thread(qapp, tmp_path):
    import threading

    window, model, ui = _install(qapp, tmp_path)
    threads = []
    original = ui.bridge._reconcile_subsystems
    ui.bridge._reconcile_subsystems = lambda: (threads.append(threading.current_thread()), original())
    worker = threading.Thread(target=lambda: model.set_status(
        SubsystemId.LASER, SubsystemState.FAILED, error="fault"))
    worker.start()
    worker.join()
    qapp.processEvents()  # the reconcile
    qapp.processEvents()  # the banner refresh it queues
    assert threads == [threading.main_thread()]
    assert ui.banner.title_label.text() == "Error: Laser failed"
    ui.close()
