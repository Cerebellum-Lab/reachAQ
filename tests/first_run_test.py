import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from reachaq.core.configuration.system_configuration import SystemConfiguration
from reachaq.app.model import first_run
from reachaq.app.model.first_run import HardwareCheck
from reachaq.app.view import first_run_dialog
from reachaq.app.view.first_run_dialog import FirstRunDialog


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _no_hardware():
    return HardwareCheck("Cameras", False, "No FLIR camera found", "Connect the cameras")


def test_a_new_install_needs_the_first_run_step(tmp_path):
    assessment = first_run.assess(tmp_path / "reachAQ" / "system_configuration.yaml")
    assert not assessment.configuration_exists
    assert assessment.needed


def test_an_existing_configuration_with_a_writable_folder_needs_nothing(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    config = first_run.create_configuration(tmp_path / "reachAQ" / "system_configuration.yaml", data)
    assessment = first_run.assess(config)
    assert assessment.configuration_exists and assessment.data_check.ok
    assert not assessment.needed
    loaded = SystemConfiguration.load_yaml_file(config)
    assert loaded.persistence.output_location == data.as_posix()


def test_an_unmounted_data_folder_needs_the_step(tmp_path):
    config = first_run.create_configuration(
        tmp_path / "system_configuration.yaml", tmp_path / "unmounted" / "data")
    assessment = first_run.assess(config)
    assert assessment.needed
    assert "does not exist" in assessment.data_check.problem


def test_set_data_folder_changes_only_that_line_and_keeps_a_backup(tmp_path):
    config = first_run.create_configuration(tmp_path / "system_configuration.yaml", tmp_path / "old")
    before = config.read_text(encoding="utf-8")
    new = tmp_path / "it's new"
    backup = first_run.set_data_folder(config, new)
    after = config.read_text(encoding="utf-8")
    assert backup.read_text(encoding="utf-8") == before
    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert len(changed) == 1 and "outputLocation" in changed[0][1]
    assert SystemConfiguration.load_yaml_file(config).persistence.output_location == new.as_posix()


def test_set_data_folder_writes_nothing_when_unchanged_and_never_overwrites_a_backup(tmp_path):
    config = first_run.create_configuration(tmp_path / "system_configuration.yaml", tmp_path / "a")
    assert first_run.set_data_folder(config, tmp_path / "a") is None
    assert not list(tmp_path.glob("*.before-first-run*"))
    first = first_run.set_data_folder(config, tmp_path / "b")
    second = first_run.set_data_folder(config, tmp_path / "c")
    assert first != second and first.is_file() and second.is_file()
    assert "/a" in first.read_text() and "/b" in second.read_text()


@pytest.mark.parametrize("text, expected", [
    ("persistence:\n  outputLocation: /data/rig # moved 2026\n", "/data/rig"),
    ("  outputLocation: 'it''s here'\n", "it's here"),
    ('  outputLocation: "/q d"\n', "/q d"),
    ("  outputLocation:\nnext: 1\n", None),
])
def test_output_location_from_text(text, expected):
    assert first_run.output_location_from_text(text) == expected


def test_an_unmounted_configured_folder_is_never_created(qapp, tmp_path):
    missing = tmp_path / "mnt" / "data"
    config = first_run.create_configuration(tmp_path / "system_configuration.yaml", missing)
    dialog = FirstRunDialog(first_run.assess(config), hardware_checks=(_no_hardware,))
    assert not dialog.accept_button.isEnabled()
    assert "mounted" in dialog.data_status.text()
    dialog.accept_button.click()  # disabled: does nothing
    assert not missing.exists() and dialog.result() != dialog.DialogCode.Accepted
    dialog.start_unchanged_button.click()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert not missing.exists()
    assert first_run.output_location_from_text(config.read_text()) == missing.as_posix()


def test_an_unreadable_folder_is_reported(tmp_path):
    a_file = tmp_path / "file"
    a_file.write_text("x")
    assert not first_run.check_folder(a_file).ok


def test_hardware_checks_report_a_failing_probe():
    def broken():
        raise RuntimeError("boom")

    (result,) = first_run.run_hardware_checks((broken,))
    assert not result.found and "boom" in result.detail


def test_camera_check_reads_lsusb(monkeypatch):
    monkeypatch.setattr(first_run.sys, "platform", "linux")
    monkeypatch.setattr(first_run, "_module_available", lambda name: True)
    result = first_run.check_cameras(run=lambda command: (0, "Bus 002 Device 003: ID 1e10:4000 FLIR\n"))
    assert result.found and "1 FLIR" in result.detail


def test_dialog_creates_the_configuration(qapp, tmp_path):
    config = tmp_path / "reachAQ" / "system_configuration.yaml"
    dialog = FirstRunDialog(first_run.assess(config), hardware_checks=(_no_hardware,))
    dialog.data_edit.setText(str(tmp_path / "new data"))
    dialog._check_data_folder()
    assert dialog.data_status.text() == "The folder will be created."
    dialog._accept()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert config.is_file() and (tmp_path / "new data").is_dir()
    assert SystemConfiguration.load_yaml_file(config).persistence.output_location == (
        tmp_path / "new data").as_posix()


def test_dialog_shows_missing_hardware(qapp, tmp_path):
    dialog = FirstRunDialog(first_run.assess(tmp_path / "c.yaml"), hardware_checks=(_no_hardware,))
    dialog._check_thread.join(5)
    qapp.processEvents()
    assert [result.name for result in dialog.hardware_results] == ["Cameras"]


def test_quit_writes_nothing(qapp, tmp_path):
    config = tmp_path / "c.yaml"
    dialog = FirstRunDialog(first_run.assess(config), hardware_checks=(_no_hardware,))
    dialog.reject()
    assert not config.exists()


def test_offscreen_launch_skips_the_dialog(qapp, tmp_path, monkeypatch):
    monkeypatch.delenv(first_run_dialog.SKIP_ENV, raising=False)
    monkeypatch.setattr(FirstRunDialog, "exec", lambda self: pytest.fail("no dialog offscreen"))
    assert first_run_dialog.run_first_run_if_needed(tmp_path / "c.yaml")
