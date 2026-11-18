"""Discovery and contract of the pose backend plugins.

Fake entry points are injected through pose_plugins._entry_points, so none of
this needs an installed plugin distribution.
"""

import pathlib
import subprocess
import sys

import pytest

from autotrainer.inference import pose_plugins
from autotrainer.inference.pose_plugins import (
    PoseBackendPlugin,
    broken_plugins,
    discover_plugins,
    reset_discovery,
)


class _FakeEntryPoint:
    def __init__(self, name, loaded=None, error=None):
        self.name = name
        self._loaded = loaded
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._loaded


def _plugin(name, **attributes):
    cls = type(f"{name}Plugin", (PoseBackendPlugin,),
               {"name": name, "distribution": f"dist-{name}", **attributes})
    return cls()


@pytest.fixture
def entry_points(monkeypatch):
    """Install a list of fake entry points and start from a clean cache."""
    installed = []
    monkeypatch.setattr(pose_plugins, "_entry_points", lambda: list(installed))
    reset_discovery()
    yield installed
    reset_discovery()


def test_keypoint_plugin_is_always_first(entry_points):
    entry_points.append(_FakeEntryPoint("zeta", _plugin("zeta")))
    entry_points.append(_FakeEntryPoint("alpha", _plugin("alpha")))
    assert [p.name for p in discover_plugins()] == ["keypoint", "alpha", "zeta"]


def test_a_plugin_that_raises_on_load_is_skipped_and_reported(entry_points):
    entry_points.append(_FakeEntryPoint("bad", error=ImportError("boom")))
    assert [p.name for p in discover_plugins()] == ["keypoint"]
    assert broken_plugins() == (("bad", "ImportError: boom"),)


def test_a_plugin_with_another_api_version_is_skipped(entry_points):
    entry_points.append(_FakeEntryPoint("future", _plugin("future", api_version=2)))
    assert [p.name for p in discover_plugins()] == ["keypoint"]
    ((name, reason),) = broken_plugins()
    assert name == "future"
    assert "api_version 2" in reason


def test_an_entry_point_that_is_not_a_plugin_is_reported(entry_points):
    entry_points.append(_FakeEntryPoint("odd", object()))
    assert [p.name for p in discover_plugins()] == ["keypoint"]
    assert [name for name, _ in broken_plugins()] == ["odd"]


def test_a_factory_entry_point_is_called(entry_points):
    cls = type("FactoryPlugin", (PoseBackendPlugin,),
               {"name": "factory", "distribution": "dist-factory"})
    entry_points.append(_FakeEntryPoint("factory", cls))
    found = {p.name: p for p in discover_plugins()}
    assert isinstance(found["factory"], cls)


def test_missing_requirements_uses_find_spec_only():
    plugin = _plugin("needy", requires=("no_such_module_xyz", "sys"))
    assert plugin.missing_requirements() == ("no_such_module_xyz",)


def test_keypoint_plugin_claims_only_its_sidecar(tmp_path):
    plugin = pose_plugins.KEYPOINT_PLUGIN
    assert not plugin.claims(str(tmp_path))
    (tmp_path / "keypoint_pose.yaml").write_text("keypoints: []\n")
    assert plugin.claims(str(tmp_path))
    assert not plugin.claims("")


def test_importing_pose_plugins_loads_no_engine():
    """Discovery must not import an engine, or a torch-only rig pays for TensorFlow.

    Checked in a fresh interpreter, because this one has imported plenty.
    """
    # .../auto-trainer-inference/src/autotrainer/inference/pose_plugins.py
    inference_src = pathlib.Path(pose_plugins.__file__).resolve().parents[2]
    repo_root = inference_src.parents[1]
    source_dirs = [str(inference_src), str(repo_root / "auto-trainer-core" / "src")]
    script = (
        "import sys\n"
        f"sys.path[:0] = {source_dirs!r}\n"
        "import autotrainer.inference.pose_plugins\n"
        "autotrainer.inference.pose_plugins.discover_plugins()\n"
        "leaked = [name for name in ('deeplabcut', 'ultralytics', 'tensorflow') if name in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"imported at module scope: {completed.stdout.strip()}"
