import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reachaq.core import settings_migration as sm


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def home(tmp_path):
    """A v3 home laid out like christielab10's on 2026-11-23."""
    home = tmp_path / "home"
    old = home / "Autotrainer"
    prefix = (home / "Autotrainer").as_posix()
    files = {
        "system_configuration.yaml": "inference:\n  pose_model_location: ''\n",
        "demo/demo_sources.yaml": f"sources:\n  left: {prefix}/demo/trial_reel_left.mp4\n"
                                  f"  right: {prefix}/demo/trial_reel_right.mp4\n",
        "mouseGYM/config.yaml": f"project_path: {prefix}/mouseGYM\r\nother: ~/Autotrainer/mouseGYM\r\n",
        # Another machine's home, and a longer name: neither is this home's settings directory.
        "JetNet1/config.yaml": "project_path: /home/agx004/Autotrainer/JetNet1\nx: ~/Autotrainer2/y\n",
        # Not live files: a backup, a non-yaml file, and an immutable library entry.
        "demo/demo_sources.yaml.bak": f"left: {prefix}/demo/old.mp4\n",
        "notes.json": f'{{"p": "{prefix}/x"}}\n',
        "calibrations/0123/index.yaml": f"source: {prefix}/4mm\n",
    }
    for rel, text in files.items():
        path = old / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    colorado = home / ".config" / "Colorado"
    colorado.mkdir(parents=True)
    (colorado / "Auto Trainer.conf").write_bytes(
        f"[system]\nconfiguration_location={prefix}\nserial_number=christielab10\n".encode())
    (colorado / "PelletDeliveryUI.conf").write_bytes(b"[General]\nport=/dev/ttyACM0\n")
    (colorado / "autotrainer_running_status.env").write_bytes(b"RUNNING=0\n")
    (colorado / "Auto Trainer.conf.bak-before-phase1").write_bytes(b"[system]\n")
    return home


def _run(home, *args):
    return sm.main(["--home", str(home), *args])


def test_dry_run_changes_nothing(home, capsys):
    before = _tree(home)
    assert _run(home, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "would rename" in out and "demo_sources.yaml" in out and "reachAQ.conf" in out
    assert _tree(home) == before


def test_migrate_moves_rewrites_and_copies(home):
    prefix = (home / "Autotrainer").as_posix().encode()
    new_prefix = (home / "reachAQ").as_posix().encode()
    original = _tree(home / "Autotrainer")
    demo_before = (home / "Autotrainer" / "demo" / "demo_sources.yaml").read_bytes()
    assert _run(home) == 0

    new = home / "reachAQ"
    assert not (home / "Autotrainer").exists()
    demo = (new / "demo" / "demo_sources.yaml").read_bytes()
    assert prefix not in demo and demo == demo_before.replace(prefix, new_prefix)
    assert (new / "demo" / f"demo_sources.yaml{sm.BACKUP_SUFFIX}").read_bytes() == demo_before
    # The ~ form keeps its form, and the CRLF line ends survive.
    assert (new / "mouseGYM" / "config.yaml").read_bytes() == (
        b"project_path: " + new_prefix + b"/mouseGYM\r\nother: ~/reachAQ/mouseGYM\r\n")
    # Untouched: another home, a longer name, backups, non-yaml, the library, and files naming nothing.
    for rel in ("JetNet1/config.yaml", "demo/demo_sources.yaml.bak", "notes.json",
                "calibrations/0123/index.yaml", "system_configuration.yaml"):
        assert _tree(new)[rel] == original[rel], rel
        assert not (new / f"{rel}{sm.BACKUP_SUFFIX}").exists()

    target = home / ".config" / "reachAQ"
    assert sorted(p.name for p in target.iterdir()) == ["PelletDeliveryUI.conf", "reachAQ.conf"]
    assert b"configuration_location=" + new_prefix + b"\n" in (target / "reachAQ.conf").read_bytes()
    assert (home / ".config" / "Colorado" / "Auto Trainer.conf").read_bytes().count(prefix) == 1  # kept for v3


def test_reverse_restores_every_byte(home):
    before = _tree(home)
    assert _run(home) == 0
    assert _run(home, "--reverse") == 0
    assert _tree(home) == before
    assert not (home / ".config" / "reachAQ").exists()
    assert not (home / "reachAQ").exists()


def test_reverse_keeps_preferences_changed_since(home, capsys):
    assert _run(home) == 0
    conf = home / ".config" / "reachAQ" / "reachAQ.conf"
    conf.write_bytes(conf.read_bytes() + b"[ui]\nchanged=1\n")
    assert _run(home, "--reverse") == 0
    assert conf.exists() and "kept" in capsys.readouterr().err
    assert (home / "Autotrainer").is_dir()


@pytest.mark.parametrize("blocker", ["reachAQ", ".config/reachAQ/reachAQ.conf",
                                     f"Autotrainer/demo/demo_sources.yaml{sm.BACKUP_SUFFIX}"])
def test_refuses_without_changing_anything(home, blocker, capsys):
    path = home / blocker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"") if "." in path.name else path.mkdir()
    before = _tree(home)
    assert _run(home) == 2
    assert "exist" in capsys.readouterr().err
    assert _tree(home) == before


def test_nothing_to_migrate(tmp_path, capsys):
    assert _run(tmp_path) == 0
    assert "nothing to migrate" in capsys.readouterr().out


def test_reverse_needs_a_record(home, capsys):
    (home / "reachAQ").mkdir()
    (home / "Autotrainer").rename(home / "elsewhere")
    assert _run(home, "--reverse") == 2
    assert sm.RECORD_NAME in capsys.readouterr().err


def test_runs_as_a_script_without_reachaq_installed(home):
    """The installer runs it with the host's python3 before the environment exists."""
    script = Path(sm.__file__)
    done = subprocess.run([sys.executable, "-I", str(script), "--home", str(home), "--dry-run"],
                          capture_output=True, text=True, timeout=60, cwd=home)
    assert done.returncode == 0, done.stderr
    assert "would rename" in done.stdout


def test_start_is_refused_only_with_just_the_v3_directory(home):
    assert "reachaq-migrate-settings" in sm.unmigrated_settings_message(home, environ={})
    assert sm.unmigrated_settings_message(home, environ={"REACHAQ_CONFIG_DIR": "/srv/lab"}) is None
    (home / "reachAQ").mkdir()
    assert sm.unmigrated_settings_message(home, environ={}) is None
    assert sm.unmigrated_settings_message(home.parent / "nobody", environ={}) is None


@pytest.mark.skipif(os.name != "posix", reason="HOME decides Path.home() on posix only")
def test_headless_refuses_to_start_on_an_unmigrated_home(home):
    env = os.environ.copy()
    env["HOME"] = home.as_posix()
    env.pop("REACHAQ_CONFIG_DIR", None)
    done = subprocess.run([sys.executable, "-m", "reachaq.app.headless"],
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == sm.UNMIGRATED_SETTINGS_EXIT, done.stderr
    assert "reachaq-migrate-settings" in done.stderr
    assert not (home / "reachAQ").exists()
