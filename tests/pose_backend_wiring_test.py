"""The selected backend has to reach both the GPU probe and the pose process.

These are the two places where a half-wired switch is dangerous rather than
merely wrong:

  * check_live_inference_runtime used to hardcode "tensorflow". On the reachAQ
    rig that probe passes while every torch convolution aborts in cuDNN, so a
    torch deployment would clear its preflight and then crash the pose process.

  * PoseProcess constructs the model. If it kept constructing DlcPoseModel the
    environment variable would appear to work - the preflight would report
    torch - while inference still ran on TensorFlow.

The pose process is checked by reading the wiring rather than by starting it: a
real start needs a GPU, a trained model and a multiprocessing queue.
"""

import ast

import source_contract

from autotrainer.inference import pose_process
from autotrainer.inference.backend_selection import (
    POSE_BACKEND_ENV_VAR,
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
)


INFERENCE_MODEL = "tools/acquisition/model/inference_model.py"


def test_the_runtime_probe_is_no_longer_pinned_to_tensorflow():
    """The probe must ask the selector, not name an engine."""
    call = source_contract.one_call(INFERENCE_MODEL, "detect_gpu_runtime")
    required = source_contract.keyword(call, "required_backend")
    assert required is not None, "the probe no longer states which backend it wants"
    assert isinstance(required, ast.Call), (
        "required_backend is a literal again; on this rig TensorFlow sees the "
        "GPU while every torch convolution aborts in cuDNN, so naming one "
        "engine passes the preflight and then crashes the pose process")
    assert source_contract._called_name(required) == "selected_backend"


def test_the_runtime_probe_tells_the_selector_which_model_is_configured():
    """A YOLO model is torch whatever the environment defaults to.

    The model is chosen in the configuration and the backend only through the
    environment, so a probe that does not pass the model can demand a
    TensorFlow runtime for torch weights and refuse to start.
    """
    call = source_contract.one_call(INFERENCE_MODEL, "detect_gpu_runtime")
    selector = source_contract.keyword(call, "required_backend")
    assert source_contract.keyword_name(selector, "model_path") == (
        "self._model_location")


def test_the_model_check_asks_the_selector_about_the_same_model():
    """The preflight and the loader must agree on the engine."""
    checks = source_contract.calls(INFERENCE_MODEL, "selected_backend")
    assert checks, "the model check no longer consults the selector"
    assert all(
        source_contract.keyword_name(call, "model_path") == "self._model_location"
        for call in checks), (
        "a selected_backend() call in inference_model is not told the model")


def test_the_runtime_probe_imports_the_selector():
    assert source_contract.imports(INFERENCE_MODEL, "selected_backend")


def test_can_start_live_inference_also_validates_the_model():
    """A bad model path must fail in the parent, not inside the child process."""
    tree = source_contract.tree(INFERENCE_MODEL)
    source_contract.function(tree, "can_start_live_inference")
    source_contract.function(tree, "_can_load_pose_model")
    assert source_contract.calls(
        source_contract.function(tree, "can_start_live_inference"),
        "_can_load_pose_model"), "the model is no longer validated up front"


def test_an_empty_model_location_stays_valid():
    """An empty location selects MemoryPoseModel for rig checkout."""
    check = source_contract.function(INFERENCE_MODEL, "_can_load_pose_model")
    guards = [node for node in ast.walk(check) if isinstance(node, ast.If)]
    assert any(
        isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and source_contract.dotted_name(node.test.operand) == "self._model_location"
        and any(isinstance(inner, ast.Return) for inner in node.body)
        for node in guards), (
        "an empty model location must return early rather than be validated")


def test_the_pose_process_builds_through_the_selector():
    assert source_contract.calls(pose_process, "build_pose_model")
    # The direct construction is what the selector replaces.
    assert not source_contract.constructs(pose_process, "DlcPoseModel")


def test_the_pose_process_no_longer_imports_the_tensorflow_model():
    """A torch-only deployment cannot import DlcPoseModel at all."""
    assert not source_contract.imports(pose_process, "DlcPoseModel")
    assert source_contract.calls(pose_process, "MemoryPoseModel"), (
        "the in-memory model is still needed")


def test_the_pose_process_selects_from_the_model_it_was_given():
    """The child has to reach the same answer as the parent's probe."""
    call = source_contract.one_call(pose_process, "selected_backend")
    assert source_contract.keyword_name(call, "model_path") == "model_path"


def test_the_pose_process_selects_before_it_builds():
    """Building first would construct the model for the wrong engine."""
    order = source_contract.call_order(
        pose_process, ["selected_backend", "build_pose_model"])
    assert order == ["selected_backend", "build_pose_model"], order


def test_the_env_var_name_is_stable():
    """Operators put this in the rig's service environment; renaming it breaks them."""
    assert POSE_BACKEND_ENV_VAR == "REACHAQ_POSE_BACKEND"
    assert TENSORFLOW_BACKEND == "tensorflow"
    assert TORCH_BACKEND == "torch"
