"""Routing a DeepLabCut project to the installed plugin.

Moved from core's tests/pose_plugins_test.py and
tests/pose_prevalidate_routing_test.py with the plugin. Discovery is real here:
the plugin is found through its entry point, not a fake one.
"""

import pytest

from reachaq.inference import pose_plugins
from reachaq.inference.backend_selection import (
    POSE_BACKEND_ENV_VAR,
    pre_validate_model,
)
from reachaq.inference.pose_plugins import PoseBackendNotInstalled, resolve


def test_a_claiming_plugin_with_a_missing_engine_names_it(
        tmp_path, monkeypatch):
    monkeypatch.setattr(pose_plugins, "_installed",
                        lambda module: module != "deeplabcut")
    (tmp_path / "config.yaml").write_text("bodyparts: [a]\n")
    with pytest.raises(PoseBackendNotInstalled) as raised:
        resolve(str(tmp_path))
    assert raised.value.missing == ("deeplabcut",)
    assert "reachaq-plugin-install.sh deeplabcut" in str(raised.value)


def test_a_claiming_plugin_with_its_engine_is_returned(
        tmp_path, monkeypatch):
    monkeypatch.setattr(pose_plugins, "_installed", lambda module: True)
    (tmp_path / "config.yaml").write_text("bodyparts: [a]\n")
    assert resolve(str(tmp_path)).name == "deeplabcut"


def test_a_deeplabcut_project_is_still_checked_as_one(tmp_path, monkeypatch):
    """The DeepLabCut path keeps the check it always had."""
    from reachaq.pose_plugins.deeplabcut import DlcPoseModel

    monkeypatch.setattr(pose_plugins, "_installed", lambda module: True)
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "tensorflow")
    checked = []
    monkeypatch.setattr(DlcPoseModel, "pre_validate",
                        classmethod(lambda cls, location: checked.append(location)))
    (tmp_path / "config.yaml").write_text("bodyparts: [a]")
    pre_validate_model(str(tmp_path))
    assert checked == [str(tmp_path)]
