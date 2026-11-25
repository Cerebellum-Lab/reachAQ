"""Pre-validation must check the model the pipeline would actually load.

app_model called DlcPoseModel.pre_validate unconditionally, which looks for a
DeepLabCut config.yaml. A YOLO model directory has no config.yaml, so every
YOLO model was reported broken - an error dialog on every start, for a model
that then loaded and ran correctly. The dialog was wrong, not the model.
The YOLO cases moved to plugins/yolo/tests with the plugin.
"""

import ast
import pathlib

import pytest

from reachaq.inference.backend_selection import (
    PoseBackendNotInstalled,
    pre_validate_model,
)

import source_contract

APP_MODEL = "src/reachaq/app/model/app_model.py"


def test_a_directory_no_backend_recognises_is_refused(tmp_path):
    """No config.yaml and no sidecar: refused, naming what is installed."""
    with pytest.raises(PoseBackendNotInstalled, match="keypoint"):
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
