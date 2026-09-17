import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.model.user_preferences import UserPreferences  # noqa: E402


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def preferences(qapp, tmp_path):
    return UserPreferences(settings_file_path=tmp_path / "prefs.ini")


def test_right_panel_state_defaults_to_docked(preferences):
    assert preferences.right_panel_state == "docked"


def test_right_panel_state_round_trips(preferences, tmp_path):
    preferences.right_panel_state = "detached"
    preferences.save()

    reloaded = UserPreferences(settings_file_path=tmp_path / "prefs.ini")

    assert reloaded.right_panel_state == "detached"


def test_right_panel_detached_geometry_defaults_to_an_invalid_rectangle(preferences):
    assert not preferences.right_panel_detached_geometry.isValid()


def test_right_panel_detached_geometry_round_trips(preferences, tmp_path):
    preferences.right_panel_detached_geometry = QRect(20, 30, 800, 600)
    preferences.save()

    reloaded = UserPreferences(settings_file_path=tmp_path / "prefs.ini")

    assert reloaded.right_panel_detached_geometry == QRect(20, 30, 800, 600)


def test_right_panel_detached_geometry_rejects_a_degenerate_rectangle(preferences):
    preferences.right_panel_detached_geometry = QRect(20, 30, 800, 600)

    preferences.right_panel_detached_geometry = QRect(0, 0, 0, 0)

    assert preferences.right_panel_detached_geometry == QRect(20, 30, 800, 600)
