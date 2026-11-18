from PySide6.QtCore import QRect

from tools.acquisition.model.user_preferences import UserPreferences


def test_a_fresh_install_has_no_softmouse_publication(tmp_path):
    # A lab's publication path is its own: none is built in.
    preferences = UserPreferences(settings_file_path=tmp_path / "settings.ini")

    assert preferences.softmouse_manifest_path == ""
    assert preferences.softmouse_publication_mount == ""


def test_softmouse_publication_settings_persist(tmp_path):
    settings_path = tmp_path / "settings.ini"
    preferences = UserPreferences(settings_file_path=settings_path)
    preferences.softmouse_manifest_path = " /srv/lab/SoftMouse/current.manifest.json "
    preferences.softmouse_publication_mount = "/srv/lab"
    preferences.save()

    reloaded = UserPreferences(settings_file_path=settings_path)

    assert reloaded.softmouse_manifest_path == "/srv/lab/SoftMouse/current.manifest.json"
    assert reloaded.softmouse_publication_mount == "/srv/lab"

    reloaded.softmouse_manifest_path = ""
    reloaded.save()
    assert UserPreferences(settings_file_path=settings_path).softmouse_manifest_path == ""


def test_normal_window_geometry_is_persisted(tmp_path):
    settings_path = tmp_path / "settings.ini"
    preferences = UserPreferences(settings_file_path=settings_path)
    preferences.window_normal_geometry = QRect(30, 40, 1200, 800)
    preferences.save()

    reloaded = UserPreferences(settings_file_path=settings_path)

    assert reloaded.window_normal_geometry == QRect(30, 40, 1200, 800)
