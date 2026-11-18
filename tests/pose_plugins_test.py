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
    AmbiguousPoseModel,
    KEYPOINT_PLUGIN,
    PoseBackendNotInstalled,
    PoseBackendPlugin,
    broken_plugins,
    discover_plugins,
    reset_discovery,
    resolve,
)

# DeepLabCut and YOLO are still in core, registered after the entry points.
IN_TREE = ["deeplabcut", "yolo"]


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
    assert [p.name for p in discover_plugins()] == ["keypoint", "alpha", "zeta", *IN_TREE]


def test_a_plugin_that_raises_on_load_is_skipped_and_reported(entry_points):
    entry_points.append(_FakeEntryPoint("bad", error=ImportError("boom")))
    assert [p.name for p in discover_plugins()] == ["keypoint", *IN_TREE]
    assert broken_plugins() == (("bad", "ImportError: boom"),)


def test_a_plugin_with_another_api_version_is_skipped(entry_points):
    entry_points.append(_FakeEntryPoint("future", _plugin("future", api_version=2)))
    assert [p.name for p in discover_plugins()] == ["keypoint", *IN_TREE]
    ((name, reason),) = broken_plugins()
    assert name == "future"
    assert "api_version 2" in reason


def test_an_entry_point_that_is_not_a_plugin_is_reported(entry_points):
    entry_points.append(_FakeEntryPoint("odd", object()))
    assert [p.name for p in discover_plugins()] == ["keypoint", *IN_TREE]
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


def test_an_entry_point_reusing_a_plugin_name_is_skipped_and_reported(entry_points):
    """Two plugins under one name would make routing depend on install order."""
    first, second = _plugin("dup"), _plugin("dup")
    entry_points.append(_FakeEntryPoint("dup-b", second))
    entry_points.append(_FakeEntryPoint("dup-a", first))
    entry_points.append(_FakeEntryPoint("impostor", _plugin("keypoint")))
    found = discover_plugins()
    assert [p.name for p in found] == ["keypoint", "dup", *IN_TREE]
    assert found[0] is KEYPOINT_PLUGIN
    assert found[1] is first
    assert sorted(name for name, _ in broken_plugins()) == ["dup-b", "impostor"]


def test_a_failing_entry_point_lookup_keeps_the_built_ins(monkeypatch):
    """Broken distribution metadata must not take the built-in backends with it."""
    def explode():
        raise RuntimeError("corrupt metadata")
    monkeypatch.setattr(pose_plugins, "_entry_points", explode)
    reset_discovery()
    try:
        assert [p.name for p in discover_plugins()] == ["keypoint", *IN_TREE]
        assert any("corrupt metadata" in reason for _, reason in broken_plugins())
    finally:
        reset_discovery()


def test_an_entry_point_replaces_the_in_tree_plugin_of_its_name(entry_points):
    """Once a plugin distribution is installed, it wins over the copy in core."""
    external = _plugin("yolo")
    entry_points.append(_FakeEntryPoint("yolo", external))
    found = discover_plugins()
    assert [p.name for p in found] == ["keypoint", "yolo", "deeplabcut"]
    assert found[1] is external
    assert broken_plugins() == ()


# --- routing ------------------------------------------------------------------


def test_keypoint_sidecar_beats_a_deeplabcut_config(tmp_path, entry_points):
    """The sidecar is an explicit declaration; config.yaml is only a guess."""
    (tmp_path / "keypoint_pose.yaml").write_text("keypoints: []\n")
    (tmp_path / "config.yaml").write_text("bodyparts: [a]\n")
    assert resolve(str(tmp_path)) is KEYPOINT_PLUGIN


def test_two_claiming_plugins_is_ambiguous(tmp_path, entry_points):
    claim = {"claims": lambda self, model_path: True}
    entry_points.append(_FakeEntryPoint("alpha", _plugin("alpha", **claim)))
    entry_points.append(_FakeEntryPoint("beta", _plugin("beta", **claim)))
    with pytest.raises(AmbiguousPoseModel) as raised:
        resolve(str(tmp_path))
    assert "alpha" in str(raised.value)
    assert "beta" in str(raised.value)


def test_a_known_layout_without_its_plugin_names_the_install_command(
        tmp_path, entry_points, monkeypatch):
    monkeypatch.setattr(pose_plugins, "_in_tree_plugins", lambda: ())
    reset_discovery()
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]\n")
    with pytest.raises(PoseBackendNotInstalled) as raised:
        resolve(str(tmp_path))
    message = str(raised.value)
    assert "auto-trainer-pose-yolo" in message
    assert "reachaq-plugin-install.sh yolo" in message
    assert raised.value.distribution == "auto-trainer-pose-yolo"
    assert raised.value.model_path == str(tmp_path)


def test_a_claiming_plugin_with_a_missing_engine_names_it(
        tmp_path, entry_points, monkeypatch):
    monkeypatch.setattr(pose_plugins, "_installed",
                        lambda module: module != "deeplabcut")
    (tmp_path / "config.yaml").write_text("bodyparts: [a]\n")
    with pytest.raises(PoseBackendNotInstalled) as raised:
        resolve(str(tmp_path))
    assert raised.value.missing == ("deeplabcut",)
    assert "reachaq-plugin-install.sh deeplabcut" in str(raised.value)


def test_a_claiming_plugin_with_its_engine_is_returned(
        tmp_path, entry_points, monkeypatch):
    monkeypatch.setattr(pose_plugins, "_installed", lambda module: True)
    (tmp_path / "config.yaml").write_text("bodyparts: [a]\n")
    assert resolve(str(tmp_path)).name == "deeplabcut"


def test_an_unrecognised_directory_lists_the_installed_plugins(tmp_path, entry_points):
    with pytest.raises(PoseBackendNotInstalled) as raised:
        resolve(str(tmp_path))
    assert raised.value.engine is None
    message = str(raised.value)
    for name in ("keypoint", *IN_TREE):
        assert name in message
