"""Tests for choosing the DeepLabCut engine at runtime.

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
from autotrainer.inference.backend_selection import (
    DEFAULT_POSE_BACKEND,
    POSE_BACKEND_ENV_VAR,
    POSE_BACKENDS,
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
    build_pose_model,
    selected_backend,
)
from autotrainer.inference import gpu_runtime


def test_an_unset_variable_keeps_the_tensorflow_engine():
    assert selected_backend({}) == TENSORFLOW_BACKEND
    assert DEFAULT_POSE_BACKEND == TENSORFLOW_BACKEND


def test_the_variable_selects_the_torch_engine():
    assert selected_backend({POSE_BACKEND_ENV_VAR: "torch"}) == TORCH_BACKEND


@pytest.mark.parametrize("value", ["torch", "pytorch", "pt", "PyTorch", "  TORCH  "])
def test_torch_spellings_are_accepted(value):
    assert selected_backend({POSE_BACKEND_ENV_VAR: value}) == TORCH_BACKEND


@pytest.mark.parametrize("value", ["tensorflow", "tf", "tf1", "TensorFlow"])
def test_tensorflow_spellings_are_accepted(value):
    assert selected_backend({POSE_BACKEND_ENV_VAR: value}) == TENSORFLOW_BACKEND


@pytest.mark.parametrize("value", ["", "   ", "onnx", "tensorrt", "true"])
def test_an_unusable_value_falls_back_rather_than_raising(value):
    """A typo must not be able to stop live inference from starting."""
    assert selected_backend({POSE_BACKEND_ENV_VAR: value}) == DEFAULT_POSE_BACKEND


def test_the_process_environment_is_the_default_source(monkeypatch):
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "pytorch")
    assert selected_backend() == TORCH_BACKEND
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR)
    assert selected_backend() == TENSORFLOW_BACKEND


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


def test_build_pose_model_returns_the_torch_backend():
    model = build_pose_model("/some/project", backend=TORCH_BACKEND)
    assert type(model).__name__ == "DlcTorchPoseModel"


def test_build_pose_model_returns_the_tensorflow_backend():
    model = build_pose_model("/some/project", backend=TENSORFLOW_BACKEND)
    assert type(model).__name__ == "DlcPoseModel"


def test_build_pose_model_passes_the_batch_size_through():
    """The live batch is sized by the caller; a default would silently pad."""
    model = build_pose_model("/some/project", 2, 1, 6, backend=TORCH_BACKEND)
    assert model._shuffle_index == 2
    assert model._training_index == 1
    assert model._model_batch_size == 6


def test_build_pose_model_follows_the_environment(monkeypatch):
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "torch")
    assert type(build_pose_model("/some/project")).__name__ == "DlcTorchPoseModel"
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR)
    assert type(build_pose_model("/some/project")).__name__ == "DlcPoseModel"


def test_build_pose_model_rejects_an_explicit_unknown_backend():
    """An explicit argument is a programming error, unlike a typo in the env."""
    with pytest.raises(ValueError, match="Unsupported pose backend"):
        build_pose_model("/some/project", backend="onnx")


def test_both_backends_satisfy_the_pose_model_contract():
    from autotrainer.inference.pose_model import PoseModel

    for backend in POSE_BACKENDS:
        model = build_pose_model("/some/project", backend=backend)
        assert isinstance(model, PoseModel)
        assert callable(model.load)
        assert callable(model.predict)


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
