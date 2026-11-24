"""The YOLO plugin as an installed distribution.

These run with reachaq-pose-yolo installed: discovery has to find it
through its entry point, not through anything core still ships.
"""

import pathlib
import subprocess
import sys

import pytest

from reachaq.inference import pose_plugins
from reachaq.inference.pose_plugins import discover_plugins

PLUGIN_MODULE = "reachaq.pose_plugins.yolo.plugin"


def _yolo():
    found = {plugin.name: plugin for plugin in discover_plugins()}
    assert "yolo" in found, (
        f"the plugin was not discovered; skipped: {pose_plugins.broken_plugins()}")
    return found["yolo"]


def test_the_plugin_is_discovered_through_its_entry_point():
    assert "yolo" in [point.name for point in pose_plugins._entry_points()]
    plugin = _yolo()
    assert type(plugin).__module__ == PLUGIN_MODULE
    assert plugin.distribution == "reachaq-pose-yolo"
    assert plugin.requires == ("ultralytics", "torch")


def test_the_plugin_claims_a_project_with_yolo_pose_yaml(tmp_path):
    plugin = _yolo()
    assert not plugin.claims(str(tmp_path))
    (tmp_path / "config.yaml").write_text("bodyparts: [a]\n")
    assert not plugin.claims(str(tmp_path))
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]\n")
    assert plugin.claims(str(tmp_path))
    assert not plugin.claims("")


def test_the_model_class_is_exported_lazily():
    import reachaq.pose_plugins.yolo as package
    from reachaq.pose_plugins.yolo.yolo_pose_model import YoloPoseModel

    assert package.YoloPoseModel is YoloPoseModel
    with pytest.raises(AttributeError):
        package.NoSuchModel


def test_discovering_the_plugin_loads_no_engine():
    """Discovery runs on every rig; only build() may pay for an engine.

    Checked in a fresh interpreter, because this one has imported plenty.
    """
    # .../src/reachaq/inference/pose_plugins.py
    inference_src = pathlib.Path(pose_plugins.__file__).resolve().parents[2]
    repo_root = inference_src.parents[1]
    source_dirs = [str(inference_src), str(repo_root / "reachaq" / "src")]
    script = (
        "import sys\n"
        f"sys.path[:0] = {source_dirs!r}\n"
        "import reachaq.inference.pose_plugins as pose_plugins\n"
        "import reachaq.pose_plugins.yolo\n"
        "names = [plugin.name for plugin in pose_plugins.discover_plugins()]\n"
        "assert 'yolo' in names, (names, pose_plugins.broken_plugins())\n"
        "leaked = [name for name in ('deeplabcut', 'tensorflow', 'torch', 'ultralytics')\n"
        "          if name in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"imported at module scope: {completed.stdout.strip()}"
