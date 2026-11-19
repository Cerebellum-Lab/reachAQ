"""Tests for choosing the pose runtime.

The DeepLabCut engine choice is tested with its plugin
(plugins/deeplabcut/tests/test_backend_selection.py).

Two properties matter more than the parsing:

  1. An unset or malformed variable must leave a deployed rig on TensorFlow.
     Every trained snapshot in the field was produced by that engine, so a
     silent switch would change results, and a hard failure over a typo would
     ground the rig.

  2. The name returned here must be the same string gpu_runtime probes with.
     If they drift, a torch deployment ends up asserting that TensorFlow can
     see the GPU - which, on the reachAQ rig, is exactly the combination that
     passes the check and then aborts in cuDNN.
"""

import pathlib
import subprocess
import sys

import pytest

from autotrainer.inference import backend_selection
from autotrainer.inference import pose_plugins
from autotrainer.inference.backend_selection import (
    DEFAULT_POSE_BACKEND,
    POSE_BACKEND_ENV_VAR,
    POSE_BACKENDS,
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
    PoseBackendNotInstalled,
    build_pose_model,
    pre_validate_model,
    selected_backend,
)
from autotrainer.inference import gpu_runtime


def _only(monkeypatch, *installed):
    """Pin which engines are importable, so the test does not read this env."""
    monkeypatch.setattr(pose_plugins, "_installed",
                        lambda module: module in installed)


@pytest.mark.parametrize("backend", POSE_BACKENDS)
def test_backend_names_match_the_gpu_runtime_probe_names(backend, monkeypatch):
    """These strings are passed to detect_gpu_runtime(required_backend=...).

    detect_gpu_runtime raises ValueError on a name it has no detector for, so a
    drift between the two name sets grounds the rig on start. The framework
    probes are stubbed here: the assertion is about names, not about what this
    host happens to have installed.
    """
    available = gpu_runtime.GpuRuntimeStatus(True, backend="nvidia-driver", devices=("FakeGPU",))
    monkeypatch.setattr(gpu_runtime, "_detect_nvidia_driver", lambda: available)
    monkeypatch.setattr(
        gpu_runtime, "_detect_tensorflow_gpu",
        lambda: gpu_runtime.GpuRuntimeStatus(True, backend="tensorflow", devices=("FakeGPU",)),
    )
    monkeypatch.setattr(
        gpu_runtime, "_detect_torch_cuda",
        lambda: gpu_runtime.GpuRuntimeStatus(True, backend="torch", devices=("FakeGPU",)),
    )

    status = gpu_runtime.detect_gpu_runtime(required_backend=backend)
    assert status.is_available is True
    assert status.backend == backend

    with pytest.raises(ValueError, match="Unsupported GPU backend"):
        gpu_runtime.detect_gpu_runtime(required_backend=backend + "-nonexistent")


def test_a_yolo_model_selects_torch_whatever_is_installed(monkeypatch, tmp_path):
    """YOLO weights are torch and cannot be anything else.

    The model is chosen in the configuration while the backend can only be
    chosen through the environment, so without this a YOLO model on a rig with
    TensorFlow installed probes the wrong runtime.
    """
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR, raising=False)
    _only(monkeypatch, "ultralytics", "tensorflow", "torch")
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]")
    assert backend_selection.selected_backend(
        model_path=str(tmp_path)) == TORCH_BACKEND


def test_a_yolo_model_keeps_torch_over_an_explicit_variable(monkeypatch, tmp_path):
    """The variable chooses DeepLabCut's engine; YOLO weights have only one.

    This used to answer TensorFlow, and the GPU probe then checked a runtime
    the model could never run on.
    """
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "tensorflow")
    _only(monkeypatch, "ultralytics", "tensorflow", "torch")
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]")
    assert backend_selection.selected_backend(
        model_path=str(tmp_path)) == TORCH_BACKEND


def test_with_nothing_resolvable_or_installed_the_answer_is_torch(monkeypatch, tmp_path):
    """No model, or one no installed plugin can run, and no TensorFlow.

    Never raises - pre-validation is where the operator is told what to
    install - so the GPU probe still has a name to ask about.
    """
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR, raising=False)
    _only(monkeypatch)
    assert backend_selection.selected_backend() == TORCH_BACKEND
    assert backend_selection.selected_backend({}) == TORCH_BACKEND
    (tmp_path / "config.yaml").write_text("bodyparts: [a]")
    assert backend_selection.selected_backend(model_path=str(tmp_path)) == TORCH_BACKEND
    assert backend_selection.available_backends() == ()


# --- no model, or one that does not resolve -----------------------------------
#
# An empty model location is rig checkout, and its GPU probe still runs. On the
# reachAQ rig every torch convolution aborts in cuDNN, so answering torch there
# fails a preflight that passes on TensorFlow - and the variable has to be able
# to steer it.


def test_no_model_follows_an_explicit_tensorflow(monkeypatch):
    _only(monkeypatch, "torch")
    assert selected_backend({POSE_BACKEND_ENV_VAR: "tensorflow"}) == TENSORFLOW_BACKEND
    assert selected_backend({POSE_BACKEND_ENV_VAR: "tf"}, model_path="") == TENSORFLOW_BACKEND


def test_no_model_unset_prefers_an_installed_tensorflow(monkeypatch):
    _only(monkeypatch, "tensorflow", "torch")
    assert selected_backend({}) == TENSORFLOW_BACKEND
    assert selected_backend({}, model_path="") == TENSORFLOW_BACKEND


def test_no_model_unset_without_tensorflow_is_torch(monkeypatch):
    _only(monkeypatch, "torch")
    assert selected_backend({}) == TORCH_BACKEND


def test_an_unresolvable_model_follows_an_explicit_torch(monkeypatch, tmp_path):
    _only(monkeypatch, "tensorflow", "torch")
    assert selected_backend({POSE_BACKEND_ENV_VAR: "torch"},
                            model_path=str(tmp_path)) == TORCH_BACKEND


@pytest.mark.parametrize("value", ["torch", "pytorch", "pt", "PyTorch", "  TORCH  "])
def test_torch_spellings_are_accepted_without_a_model(value, monkeypatch):
    _only(monkeypatch, "tensorflow", "torch")
    assert selected_backend({POSE_BACKEND_ENV_VAR: value}) == TORCH_BACKEND


@pytest.mark.parametrize("value", ["tensorflow", "tf", "tf1", "TensorFlow"])
def test_tensorflow_spellings_are_accepted_without_a_model(value, monkeypatch):
    _only(monkeypatch, "torch")
    assert selected_backend({POSE_BACKEND_ENV_VAR: value}) == TENSORFLOW_BACKEND


@pytest.mark.parametrize("value", ["", "   ", "onnx", "tensorrt", "true"])
def test_an_unusable_value_without_a_model_falls_back(value, monkeypatch):
    _only(monkeypatch, "tensorflow", "torch")
    assert selected_backend({POSE_BACKEND_ENV_VAR: value}) == DEFAULT_POSE_BACKEND


def test_without_a_model_the_process_environment_is_the_default_source(monkeypatch):
    _only(monkeypatch, "tensorflow", "torch")
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "pytorch")
    assert selected_backend() == TORCH_BACKEND
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR)
    assert selected_backend() == TENSORFLOW_BACKEND


def test_a_typo_without_a_model_falls_through_to_detection(monkeypatch):
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "tensorflwo")
    _only(monkeypatch, "torch")
    assert selected_backend() == TORCH_BACKEND


def test_a_deeplabcut_project_without_its_plugin_fails_where_the_operator_sees_it(
        tmp_path, monkeypatch):
    """DeepLabCut left core; a project of it names the plugin to install.

    Real discovery and the real in-tree list, so this fails if anything core
    still ships claims the project. Only the plugin's own entry point is taken
    away, so it means the same whether or not auto-trainer-pose-deeplabcut is
    installed here.
    """
    installed = list(pose_plugins._entry_points())
    monkeypatch.setattr(pose_plugins, "_entry_points", lambda: [
        point for point in installed if point.name != "deeplabcut"])
    (tmp_path / "config.yaml").write_text("bodyparts: [a]")
    (tmp_path / "dlc-models-pytorch").mkdir()
    pose_plugins.reset_discovery()
    try:
        with pytest.raises(PoseBackendNotInstalled) as raised:
            pre_validate_model(str(tmp_path))
        assert raised.value.distribution == "auto-trainer-pose-deeplabcut"
        assert raised.value.missing == ()
        assert "reachaq-plugin-install.sh deeplabcut" in str(raised.value)
        with pytest.raises(PoseBackendNotInstalled, match="auto-trainer-pose-deeplabcut"):
            build_pose_model(str(tmp_path))
    finally:
        pose_plugins.reset_discovery()


def test_available_backends_names_each_usable_plugin_with_its_runtimes(monkeypatch):
    """The preflight message: what this rig could run, and on which runtime.

    DeepLabCut's entry is checked in its plugin's tests.
    """
    monkeypatch.setattr(pose_plugins, "_entry_points", lambda: [])
    pose_plugins.reset_discovery()
    try:
        _only(monkeypatch, "ultralytics", "torch")
        assert backend_selection.available_backends() == (
            "keypoint (torch)", "yolo (torch)")
        _only(monkeypatch, "torch")
        assert backend_selection.available_backends() == ("keypoint (torch)",)
    finally:
        pose_plugins.reset_discovery()


def test_build_pose_model_rejects_an_explicit_unknown_backend():
    """An explicit argument is a programming error, unlike a typo in the env."""
    with pytest.raises(ValueError, match="Unsupported pose backend"):
        build_pose_model("/some/project", backend="onnx")


def test_importing_the_selector_pulls_in_neither_framework():
    """A torch-only deployment cannot import the TensorFlow model at all.

    So the framework imports have to stay inside the selected branch of
    build_pose_model. Checked in a fresh interpreter, because by the time this
    test runs the current one has imported plenty on its own account.
    """
    # .../auto-trainer-inference/src/autotrainer/inference/backend_selection.py
    inference_src = pathlib.Path(backend_selection.__file__).resolve().parents[2]
    repo_root = inference_src.parents[1]
    source_dirs = [str(inference_src), str(repo_root / "auto-trainer-core" / "src")]
    script = (
        "import sys\n"
        f"sys.path[:0] = {source_dirs!r}\n"
        "import autotrainer.inference.backend_selection\n"
        "leaked = [name for name in ('tensorflow', 'torch', 'deeplabcut') if name in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"imported at module scope: {completed.stdout.strip()}"
