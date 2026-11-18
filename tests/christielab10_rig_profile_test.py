"""christielab10's values left shared code for configuration; its configuration
must resolve to exactly what was built in before, and a fresh install to none of it.

The before values are spelled out here as they stood in shared code at
feature-dev 95375bb0: the SoftMouse publication under /mnt/isilon, the mount
guard on /mnt/isilon, and every settings file under ~/Autotrainer.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools.acquisition.model.app_model_status import SessionRecordingStatus  # noqa: E402
from tools.acquisition.model.user_preferences import UserPreferences  # noqa: E402
from tools.acquisition.view.softmouse_publication_controller import (  # noqa: E402
    SoftMousePublicationController,
)
from tools.softmouse_sync import cli  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "tests" / "christielab10_preferences.ini"

# As built into shared code before the move.
BEFORE_PUBLICATION_DIRECTORY = Path("/mnt/isilon/Data/ReachingData/SoftMouse")
BEFORE_MANIFEST = BEFORE_PUBLICATION_DIRECTORY / "SoftMouse-AnimalList-current.manifest.json"
BEFORE_MOUNT = Path("/mnt/isilon")
BEFORE_CONFIG_DIR = Path("~/Autotrainer")


@pytest.fixture
def christielab10_preferences(tmp_path):
    path = tmp_path / "Auto Trainer.conf"
    shutil.copy(PROFILE, path)
    return path


@pytest.fixture
def fresh_preferences(tmp_path):
    return tmp_path / "fresh.conf"


def _use_preferences(monkeypatch, path):
    """Point the publisher's own preference lookup at a given file."""
    monkeypatch.setattr(
        "tools.acquisition.model.user_preferences.UserPreferences",
        lambda: UserPreferences(settings_file_path=path),
    )


def _default_paths(env):
    """The settings paths shared code resolves at import, in a clean interpreter."""
    script = (
        "import json\n"
        "from autotrainer.core.config_directory import DEFAULT_CONFIG_DIR\n"
        "from autotrainer.core.configuration import SystemConfiguration\n"
        "from autotrainer.core.diamond_triangle_config import DEFAULT_DIAMOND_TRIANGLE_CONFIG_PATH\n"
        "from autotrainer.device import CompoundMovements, MotorConfigurationFile\n"
        "from tools.acquisition.model.user_preferences import get_default_configuration_location\n"
        "print(json.dumps({\n"
        "    'dir': str(DEFAULT_CONFIG_DIR),\n"
        "    'system': str(SystemConfiguration.DEFAULT_PATH),\n"
        "    'motor': str(MotorConfigurationFile.DEFAULT_LOCATION),\n"
        "    'move': str(CompoundMovements.DEFAULT_LOCATION),\n"
        "    'diamond': str(DEFAULT_DIAMOND_TRIANGLE_CONFIG_PATH),\n"
        "    'location': str(get_default_configuration_location()),\n"
        "}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("REACHAQ_CONFIG_DIR", "AUTOTRAINER_DIAMOND_TRIANGLE_CONFIG")}
    env.update(extra)
    return env


def test_the_settings_directory_is_still_autotrainer_by_default():
    paths = _default_paths(_clean_env())

    assert paths == {
        "dir": str(BEFORE_CONFIG_DIR),
        "system": str(BEFORE_CONFIG_DIR / "system_configuration.yaml"),
        "motor": str(BEFORE_CONFIG_DIR / "motor_config.yaml"),
        "move": str(BEFORE_CONFIG_DIR / "move_config.yaml"),
        "diamond": str((BEFORE_CONFIG_DIR / "diamond_triangle_offset.yaml").expanduser()),
        "location": str(BEFORE_CONFIG_DIR.expanduser()),
    }


def test_reachaq_config_dir_moves_every_settings_file(tmp_path):
    moved = tmp_path / "lab-settings"
    paths = _default_paths(_clean_env(REACHAQ_CONFIG_DIR=str(moved)))

    assert paths == {
        "dir": str(moved),
        "system": str(moved / "system_configuration.yaml"),
        "motor": str(moved / "motor_config.yaml"),
        "move": str(moved / "move_config.yaml"),
        "diamond": str(moved / "diamond_triangle_offset.yaml"),
        "location": str(moved),
    }


def test_christielab10_preferences_resolve_to_the_built_in_publication(christielab10_preferences):
    preferences = UserPreferences(settings_file_path=christielab10_preferences)

    assert Path(preferences.softmouse_manifest_path) == BEFORE_MANIFEST
    assert Path(preferences.softmouse_publication_mount) == BEFORE_MOUNT


def test_christielab10s_scheduled_publisher_writes_where_it_did(monkeypatch, christielab10_preferences):
    _use_preferences(monkeypatch, christielab10_preferences)
    monkeypatch.setattr(cli, "_credentials_from_keyring", lambda: object())
    calls = []

    def ensure(destination, *, mount_root=None):
        calls.append((Path(destination), mount_root and Path(mount_root)))
        raise RuntimeError("stop before publishing")

    monkeypatch.setattr(cli, "_ensure_publication_directory", ensure)

    # The systemd unit runs the publisher with no arguments.
    assert cli.main([]) == 1
    assert calls == [(BEFORE_PUBLICATION_DIRECTORY, BEFORE_MOUNT)]


def test_christielab10s_sync_button_passes_the_same_publication(christielab10_preferences):
    app_model = SimpleNamespace(
        session_recording_status=SessionRecordingStatus.READY,
        preferences=UserPreferences(settings_file_path=christielab10_preferences),
    )
    controller = SoftMousePublicationController(app_model)

    assert controller._publisher_arguments() == (
        "-m", "tools.softmouse_sync.cli",
        "--destination", BEFORE_PUBLICATION_DIRECTORY.as_posix(),
        "--require-mount", BEFORE_MOUNT.as_posix(),
    )


def test_the_mount_guard_still_refuses_an_unmounted_isilon(tmp_path):
    with pytest.raises(RuntimeError, match="not mounted"):
        cli._ensure_publication_directory(
            BEFORE_PUBLICATION_DIRECTORY, mount_root=BEFORE_MOUNT,
            is_mount=lambda _path: False,
        )


def test_a_fresh_install_publishes_nowhere(monkeypatch, capsys, fresh_preferences):
    _use_preferences(monkeypatch, fresh_preferences)
    monkeypatch.setattr(cli, "_credentials_from_keyring", lambda: object())

    assert cli.main([]) == 1
    assert "No SoftMouse publication directory is configured" in capsys.readouterr().err

    app_model = SimpleNamespace(preferences=UserPreferences(settings_file_path=fresh_preferences))
    assert SoftMousePublicationController(app_model)._publisher_arguments() == (
        "-m", "tools.softmouse_sync.cli",
    )


def test_an_explicit_destination_needs_no_preferences(monkeypatch, tmp_path, fresh_preferences):
    _use_preferences(monkeypatch, fresh_preferences)
    monkeypatch.setattr(cli, "_credentials_from_keyring", lambda: object())
    calls = []

    def ensure(destination, *, mount_root=None):
        calls.append((Path(destination), mount_root))
        raise RuntimeError("stop before publishing")

    monkeypatch.setattr(cli, "_ensure_publication_directory", ensure)

    assert cli.main(["--destination", str(tmp_path / "pub")]) == 1
    assert calls == [(tmp_path / "pub", None)]


def test_shipped_files_name_no_christie_rig_mount():
    unit = REPO_ROOT / "tools" / "softmouse_sync" / "systemd" / "reachaq-softmouse-publisher.service"
    assert "/mnt/isilon" not in unit.read_text()
