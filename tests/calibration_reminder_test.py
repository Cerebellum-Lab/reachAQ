"""The monthly reminder to make a new 3D calibration.

The reminder only suggests: it never starts a calibration. It is checked once per run, on the
first configuration load, against the calibration the app actually loaded.
"""

import inspect
import os
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QMessageBox

from autotrainer.inference.multires_calibration.store import CALIBRATION_FORMAT, INDEX_FILE
from tools.acquisition.model.calibration_reminder import calibration_due, reminder_text
from tools.acquisition.model.user_preferences import UserPreferences
from tools.acquisition.view import main_window as main_window_module
from tools.acquisition.view.main_window import MainWindow

NOW = datetime(2026, 10, 23, 14, 5, 6)
MENU_PATH = "Tools → Make 3D calibration"


def test_due_after_30_days():
    assert calibration_due(NOW - timedelta(days=30), NOW, 30) is True
    assert calibration_due(NOW - timedelta(days=45), NOW, 30) is True


def test_not_due_at_29_days():
    assert calibration_due(NOW - timedelta(days=29, hours=23), NOW, 30) is False


def test_missing_calibration_is_due():
    assert calibration_due(None, NOW, 30) is True


def test_the_threshold_is_the_days_given():
    created = NOW - timedelta(days=40)
    assert calibration_due(created, NOW, 60) is False
    assert calibration_due(created, NOW, 40) is True


def test_preference_default_30(tmp_path):
    preferences = UserPreferences(settings_file_path=tmp_path / "settings.ini")

    assert preferences.calibration_reminder_days == 30


def test_the_reminder_days_are_persisted_under_their_key(tmp_path):
    settings_path = tmp_path / "settings.ini"
    preferences = UserPreferences(settings_file_path=settings_path)
    seen = []
    preferences.property_changed += lambda name, value, previous: seen.append((name, value, previous))

    preferences.calibration_reminder_days = 14
    preferences.save()

    assert seen == [(UserPreferences.CALIBRATION_REMINDER_DAYS, 14, 30)]
    assert "reminder_days=14" in settings_path.read_text()
    assert UserPreferences(settings_file_path=settings_path).calibration_reminder_days == 14


def test_reminder_text_names_the_date_and_where_to_calibrate():
    text = reminder_text(datetime(2026, 9, 20, 14, 5))

    assert "3D calibration" in text
    assert "2026-09-20" in text
    assert MENU_PATH in text


def test_reminder_text_for_no_calibration_says_never():
    text = reminder_text(None)

    assert "3D calibration" in text
    assert "never" in text
    assert MENU_PATH in text


def test_the_menu_path_in_the_text_is_where_the_action_is():
    # The text sends the operator to a menu; it must not drift from the real one.
    source = inspect.getsource(MainWindow._configure_menubar)
    tools = source.index('menu_bar.addMenu("Tools")')
    assert source.index("tools_menu.addAction(self.make_3d_calib_action)") > tools
    assert 'menu_bar.addMenu(' not in source[tools + 1:source.index("tools_menu.addAction(self.make_3d_calib_action)")]
    assert '"Make 3D calibration"' in inspect.getsource(MainWindow._create_actions)


# The main window's reminder, driven through the real configuration-load handler with a stub
# window: every other call that handler makes lands on a MagicMock and is not under test.

def _window(monkeypatch, *, active_dir, created=None, days=30):
    """A window whose reminder, message box and calibration lookup are recorded.

    `created` is what calibration_created returns for the active folder: a datetime, or an
    exception to raise. None leaves the real calibration_created in place, for tests that
    give it a real folder.
    """
    seen = SimpleNamespace(boxes=[], looked_up=[])
    window = MagicMock()
    window._calibration_reminder_shown = False
    window._app_model.active_calibration_dir = active_dir
    window._preferences.calibration_reminder_days = days
    window._show_msg_box = lambda title, text, icon: seen.boxes.append((title, text, icon))
    window._remind_if_calibration_due = partial(MainWindow._remind_if_calibration_due, window)

    if created is not None:
        def fake_created(folder):
            seen.looked_up.append(folder)
            if isinstance(created, Exception):
                raise created
            return created

        monkeypatch.setattr(main_window_module, "calibration_created", fake_created)
    return window, seen


def _load_configuration(window):
    MainWindow._on_app_model_configuration_loaded(window, SimpleNamespace())


def test_reminder_shown_once_per_run(monkeypatch, tmp_path):
    created = datetime.now() - timedelta(days=45)
    window, seen = _window(monkeypatch, active_dir=tmp_path, created=created)

    _load_configuration(window)
    _load_configuration(window)

    assert seen.boxes == [("Calibration due", reminder_text(created), QMessageBox.Icon.Information)]
    assert seen.looked_up == [tmp_path]
    assert window._calibration_reminder_shown is True


def test_reminder_never_starts_a_calibration(monkeypatch, tmp_path):
    window, seen = _window(monkeypatch, active_dir=tmp_path, created=datetime.now() - timedelta(days=90))

    _load_configuration(window)

    assert len(seen.boxes) == 1
    window.on_3d_calibrate.assert_not_called()
    window.make_3d_calib_action.trigger.assert_not_called()


def test_a_recent_calibration_is_not_reminded_and_is_not_checked_again(monkeypatch, tmp_path):
    window, seen = _window(monkeypatch, active_dir=tmp_path, created=datetime.now() - timedelta(days=5))

    _load_configuration(window)
    assert seen.boxes == []
    assert window._calibration_reminder_shown is True

    # Checked once per run: a later load does not look again, even at an old calibration.
    window._app_model.active_calibration_dir = tmp_path / "other"
    monkeypatch.setattr(main_window_module, "calibration_created", lambda folder: datetime.now() - timedelta(days=90))
    _load_configuration(window)
    assert seen.boxes == []


def test_no_active_calibration_is_reminded_as_never(monkeypatch):
    window, seen = _window(monkeypatch, active_dir=None, created=RuntimeError("must not be read"))

    _load_configuration(window)

    assert [title for title, _, _ in seen.boxes] == ["Calibration due"]
    assert seen.boxes[0][1] == reminder_text(None)
    assert "never" in seen.boxes[0][1]
    assert seen.looked_up == []


def test_the_preference_sets_how_old_is_due(monkeypatch, tmp_path):
    window, seen = _window(monkeypatch, active_dir=tmp_path, created=datetime.now() - timedelta(days=40), days=60)

    _load_configuration(window)

    assert seen.boxes == []


@pytest.mark.parametrize("age_days, due", [(40, True), (5, False)])
def test_a_timezone_aware_creation_time_is_compared_like_with_like(monkeypatch, tmp_path, age_days, due):
    # A fixed -06:00 offset, as an index written with one would carry. A naive now cannot be compared with it.
    created = datetime.now(timezone(timedelta(hours=-6))) - timedelta(days=age_days)
    window, seen = _window(monkeypatch, active_dir=tmp_path, created=created)

    _load_configuration(window)

    assert len(seen.boxes) == (1 if due else 0)


def test_a_calibration_that_cannot_be_read_is_due_and_does_not_break_the_load(monkeypatch, tmp_path, caplog):
    window, seen = _window(monkeypatch, active_dir=tmp_path, created=ValueError("calibration.yaml is corrupt"))

    with caplog.at_level("WARNING"):
        _load_configuration(window)

    assert [title for title, _, _ in seen.boxes] == ["Calibration due"]
    assert "calibration.yaml is corrupt" in caplog.text
    # The rest of the handler still ran.
    window._refresh_ui_availability.assert_called_once()


def test_an_empty_folder_and_a_corrupt_index_are_due_with_the_real_lookup(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / INDEX_FILE).write_text(f"format: {CALIBRATION_FORMAT}\n")

    for folder in (empty, corrupt):
        window, seen = _window(monkeypatch, active_dir=folder)
        _load_configuration(window)
        assert [title for title, _, _ in seen.boxes] == ["Calibration due"], folder


@pytest.mark.parametrize("age_days, due", [(40, True), (10, False)])
def test_a_legacy_folder_is_dated_by_its_stereo_pickle(monkeypatch, tmp_path, age_days, due):
    stereo = tmp_path / "camera_matrix" / "stereo_params.pickle"
    stereo.parent.mkdir()
    stereo.write_bytes(b"not read: only its time is")
    then = (datetime.now() - timedelta(days=age_days)).timestamp()
    os.utime(stereo, (then, then))
    window, seen = _window(monkeypatch, active_dir=tmp_path)

    _load_configuration(window)

    assert len(seen.boxes) == (1 if due else 0)
    if due:
        assert datetime.fromtimestamp(then).strftime("%Y-%m-%d") in seen.boxes[0][1]
