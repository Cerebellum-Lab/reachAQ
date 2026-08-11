from PySide6.QtCore import QRect

from tools.acquisition.model.animal_metadata_sync import (
    DEFAULT_SOFTMOUSE_MANIFEST_PATH,
)
from tools.acquisition.model.user_preferences import UserPreferences


def test_softmouse_manifest_uses_shared_isilon_default(tmp_path):
    preferences = UserPreferences(settings_file_path=tmp_path / "settings.ini")

    assert preferences.softmouse_manifest_path == (
        DEFAULT_SOFTMOUSE_MANIFEST_PATH.as_posix()
    )


def test_empty_legacy_manifest_setting_migrates_to_shared_default(tmp_path):
    settings_path = tmp_path / "settings.ini"
    preferences = UserPreferences(settings_file_path=settings_path)
    preferences.softmouse_manifest_path = ""
    preferences.save()

    reloaded = UserPreferences(settings_file_path=settings_path)

    assert reloaded.softmouse_manifest_path == (
        DEFAULT_SOFTMOUSE_MANIFEST_PATH.as_posix()
    )


def test_normal_window_geometry_is_persisted(tmp_path):
    settings_path = tmp_path / "settings.ini"
    preferences = UserPreferences(settings_file_path=settings_path)
    preferences.window_normal_geometry = QRect(30, 40, 1200, 800)
    preferences.save()

    reloaded = UserPreferences(settings_file_path=settings_path)

    assert reloaded.window_normal_geometry == QRect(30, 40, 1200, 800)
