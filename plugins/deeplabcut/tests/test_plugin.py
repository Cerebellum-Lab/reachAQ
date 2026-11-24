"""The DeepLabCut plugin as an installed distribution.

These run with reachaq-pose-deeplabcut installed: discovery has to find it
through its entry point, not through anything core still ships.
"""

import pathlib
import subprocess
import sys

import pytest

from reachaq.inference import pose_plugins
from reachaq.inference.pose_plugins import discover_plugins

PLUGIN_MODULE = "reachaq.pose_plugins.deeplabcut.plugin"


def _deeplabcut():
    found = {plugin.name: plugin for plugin in discover_plugins()}
    assert "deeplabcut" in found, (
        f"the plugin was not discovered; skipped: {pose_plugins.broken_plugins()}")
    return found["deeplabcut"]


def test_the_plugin_is_discovered_through_its_entry_point():
    assert "deeplabcut" in [point.name for point in pose_plugins._entry_points()]
    plugin = _deeplabcut()
    assert type(plugin).__module__ == PLUGIN_MODULE
    assert plugin.distribution == "reachaq-pose-deeplabcut"


def test_the_plugin_claims_a_project_with_config_yaml(tmp_path):
    plugin = _deeplabcut()
    assert not plugin.claims(str(tmp_path))
    (tmp_path / "config.yaml").write_text("bodyparts: [a]\n")
    assert plugin.claims(str(tmp_path))
    assert not plugin.claims("")


def test_the_model_classes_are_exported_lazily():
    import reachaq.pose_plugins.deeplabcut as package
    from reachaq.pose_plugins.deeplabcut.dlc_pose_model import DlcPoseModel
    from reachaq.pose_plugins.deeplabcut.dlc_torch_pose_model import DlcTorchPoseModel

    assert package.DlcPoseModel is DlcPoseModel
    assert package.DlcTorchPoseModel is DlcTorchPoseModel
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
        "import reachaq.pose_plugins.deeplabcut\n"
        "names = [plugin.name for plugin in pose_plugins.discover_plugins()]\n"
        "assert 'deeplabcut' in names, (names, pose_plugins.broken_plugins())\n"
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
