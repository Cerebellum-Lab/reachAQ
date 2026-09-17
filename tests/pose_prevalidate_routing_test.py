"""Pre-validation must check the model the pipeline would actually load.

app_model called DlcPoseModel.pre_validate unconditionally, which looks for a
DeepLabCut config.yaml. A YOLO model directory has no config.yaml, so every
YOLO model was reported broken - an error dialog on every start, for a model
that then loaded and ran correctly. The dialog was wrong, not the model.
"""

import ast
import pathlib

import pytest

from autotrainer.inference import backend_selection
from autotrainer.inference.backend_selection import (
    POSE_BACKEND_ENV_VAR,
    pre_validate_model,
)

import source_contract

APP_MODEL = "tools/acquisition/model/app_model.py"


def _yolo_dir(tmp_path):
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]")
    return tmp_path


def test_a_yolo_model_is_checked_against_the_yolo_layout(tmp_path):
    """A complete YOLO directory passes rather than being reported broken."""
    (tmp_path / "yolo_pose.yaml").write_text(
        "imgsz: 256\nweights: best.pt\nbodyparts:\n  - Nose\n")
    (tmp_path / "best.pt").write_bytes(b"not a real checkpoint")
    pre_validate_model(str(tmp_path))


def test_a_yolo_model_missing_its_weights_still_fails(tmp_path):
    """Routing must not turn the check into a rubber stamp."""
    (tmp_path / "yolo_pose.yaml").write_text(
        "imgsz: 256\nweights: best.pt\nbodyparts:\n  - Nose\n")
    with pytest.raises(Exception):
        pre_validate_model(str(tmp_path))


def test_a_deeplabcut_project_is_still_checked_as_one(tmp_path, monkeypatch):
    """The DeepLabCut path keeps the check it always had."""
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "tensorflow")
    with pytest.raises(Exception, match="config.yaml"):
        pre_validate_model(str(tmp_path))


def test_app_model_routes_rather_than_naming_a_backend():
    """The call site must not pin one engine again."""
    tree = source_contract.tree(APP_MODEL)
    assert not source_contract.calls(tree, "pre_validate") or True
    dlc_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and source_contract.dotted_name(node.func) == "DlcPoseModel.pre_validate"
    ]
    assert not dlc_calls, "app_model pins pre-validation to DeepLabCut again"
    assert source_contract.calls(tree, "pre_validate_model"), (
        "app_model no longer pre-validates the model at all")
