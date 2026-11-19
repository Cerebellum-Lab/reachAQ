"""Choosing DeepLabCut's engine at runtime, through the installed plugin.

Moved from core's tests/pose_backend_selection_test.py with the plugin. Two
properties matter more than the parsing:

  1. An unset or malformed variable must leave a deployed rig on TensorFlow.
     Every trained snapshot in the field was produced by that engine, so a
     silent switch would change results, and a hard failure over a typo would
     ground the rig.

  2. The name returned here must be the same string gpu_runtime probes with;
     core's tests check the names against the probes.
"""

import pytest

from autotrainer.inference import backend_selection
from autotrainer.inference import pose_plugins
from autotrainer.inference.backend_selection import (
    DEFAULT_POSE_BACKEND,
    POSE_BACKEND_ENV_VAR,
    POSE_BACKENDS,
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
    AmbiguousPoseModel,
    PoseBackendNotInstalled,
    build_pose_model,
    pre_validate_model,
    selected_backend,
)


def _only(monkeypatch, *installed):
    """Pin which engines are importable, so the test does not read this env."""
    monkeypatch.setattr(pose_plugins, "_installed",
                        lambda module: module in installed)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A DeepLabCut project on a rig with DeepLabCut and both engines.

    DeepLabCut claims a directory by its config.yaml, so the engine choice
    below is only reached for one.
    """
    _only(monkeypatch, "deeplabcut", "tensorflow", "torch")
    (tmp_path / "config.yaml").write_text("bodyparts: [a]")
    return str(tmp_path)


def test_an_unset_variable_keeps_the_tensorflow_engine(project):
    """Preferred when present: deployed rigs run trained TensorFlow snapshots.

    Pinned rather than read from the ambient interpreter, because the answer
    now depends on what is installed and the test would otherwise pass on a
    rig with TensorFlow and fail on a torch-only one.
    """
    assert selected_backend({}, model_path=project) == TENSORFLOW_BACKEND
    assert DEFAULT_POSE_BACKEND == TENSORFLOW_BACKEND


def test_the_variable_selects_the_torch_engine(project):
    assert selected_backend({POSE_BACKEND_ENV_VAR: "torch"},
                            model_path=project) == TORCH_BACKEND


@pytest.mark.parametrize("value", ["torch", "pytorch", "pt", "PyTorch", "  TORCH  "])
def test_torch_spellings_are_accepted(value, project):
    assert selected_backend({POSE_BACKEND_ENV_VAR: value},
                            model_path=project) == TORCH_BACKEND


@pytest.mark.parametrize("value", ["tensorflow", "tf", "tf1", "TensorFlow"])
def test_tensorflow_spellings_are_accepted(value, project):
    assert selected_backend({POSE_BACKEND_ENV_VAR: value},
                            model_path=project) == TENSORFLOW_BACKEND


@pytest.mark.parametrize("value", ["", "   ", "onnx", "tensorrt", "true"])
def test_an_unusable_value_falls_back_rather_than_raising(value, project):
    """A typo must not be able to stop live inference from starting."""
    assert selected_backend({POSE_BACKEND_ENV_VAR: value},
                            model_path=project) == DEFAULT_POSE_BACKEND


def test_the_process_environment_is_the_default_source(monkeypatch, project):
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "pytorch")
    assert selected_backend(model_path=project) == TORCH_BACKEND
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR)
    assert selected_backend(model_path=project) == TENSORFLOW_BACKEND


def test_build_pose_model_returns_the_torch_backend(project):
    model = build_pose_model(project, backend=TORCH_BACKEND)
    assert type(model).__name__ == "DlcTorchPoseModel"


def test_build_pose_model_returns_the_tensorflow_backend(project):
    model = build_pose_model(project, backend=TENSORFLOW_BACKEND)
    assert type(model).__name__ == "DlcPoseModel"


def test_build_pose_model_passes_the_batch_size_through(project):
    """The live batch is sized by the caller; a default would silently pad."""
    model = build_pose_model(project, 2, 1, 6, backend=TORCH_BACKEND)
    assert model._shuffle_index == 2
    assert model._training_index == 1
    assert model._model_batch_size == 6


def test_build_pose_model_follows_the_environment(monkeypatch, project):
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "torch")
    assert type(build_pose_model(project)).__name__ == "DlcTorchPoseModel"
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "tensorflow")
    assert type(build_pose_model(project)).__name__ == "DlcPoseModel"


def test_an_unset_variable_prefers_tensorflow_when_it_is_installed(monkeypatch, project):
    """Deployed rigs run trained TensorFlow snapshots; leave them alone."""
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR, raising=False)
    assert backend_selection.selected_backend(model_path=project) == TENSORFLOW_BACKEND


def test_an_unset_variable_uses_torch_where_tensorflow_is_absent(monkeypatch, project):
    """A torch-only deployment must not be sent to look for TensorFlow.

    Answering TensorFlow there makes the GPU probe demand a runtime that was
    never installed, refuses live inference, and reports a missing TensorFlow
    runtime without mentioning that torch is present and is what the model
    needs.
    """
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR, raising=False)
    _only(monkeypatch, "deeplabcut", "torch")
    assert backend_selection.selected_backend(model_path=project) == TORCH_BACKEND


def test_a_deeplabcut_project_keeps_its_own_engine_choice(monkeypatch, project):
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR, raising=False)
    assert backend_selection.selected_backend(
        model_path=project) == TENSORFLOW_BACKEND


def test_a_typo_falls_through_to_detection_rather_than_a_blind_default(monkeypatch, project):
    """Refusing to start over a typo is worse than picking what is installed."""
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "tensorflwo")
    _only(monkeypatch, "deeplabcut", "torch")
    assert backend_selection.selected_backend(model_path=project) == TORCH_BACKEND


def test_an_ambiguous_model_selects_a_runtime_but_fails_pre_validation(monkeypatch, tmp_path):
    """A second plugin claiming the same directory, here a fake one, so the test
    does not depend on any other engine's distribution being installed."""
    class _Claimant(pose_plugins.PoseBackendPlugin):
        name = "claimant"
        distribution = "dist-claimant"

        def claims(self, model_path):
            return True

    class _Point:
        name = "claimant"

        def load(self):
            return _Claimant()

    installed = list(pose_plugins._entry_points())
    monkeypatch.setattr(pose_plugins, "_entry_points",
                        lambda: [*installed, _Point()])
    pose_plugins.reset_discovery()
    _only(monkeypatch, "deeplabcut", "tensorflow", "torch")
    (tmp_path / "config.yaml").write_text("bodyparts: [a]")
    try:
        assert backend_selection.selected_backend(
            {}, model_path=str(tmp_path)) == TENSORFLOW_BACKEND
        with pytest.raises(AmbiguousPoseModel, match="deeplabcut"):
            pre_validate_model(str(tmp_path))
        with pytest.raises(AmbiguousPoseModel, match="claimant"):
            build_pose_model(str(tmp_path))
    finally:
        pose_plugins.reset_discovery()


def test_a_model_without_its_engine_fails_where_the_operator_sees_it(monkeypatch, tmp_path):
    _only(monkeypatch, "tensorflow", "torch")
    (tmp_path / "config.yaml").write_text("bodyparts: [a]")
    with pytest.raises(PoseBackendNotInstalled, match="reachaq-plugin-install.sh deeplabcut"):
        pre_validate_model(str(tmp_path))
    with pytest.raises(PoseBackendNotInstalled, match="deeplabcut"):
        build_pose_model(str(tmp_path))


def test_available_backends_names_each_usable_plugin_with_its_runtimes(monkeypatch):
    """The preflight message: what this rig could run, and on which runtime."""
    _only(monkeypatch, "deeplabcut", "tensorflow", "torch")
    assert "deeplabcut (tensorflow, torch)" in backend_selection.available_backends()
    _only(monkeypatch, "deeplabcut", "torch")
    assert "deeplabcut (torch)" in backend_selection.available_backends()


def test_both_backends_satisfy_the_pose_model_contract(project):
    from autotrainer.inference.pose_model import PoseModel

    for backend in POSE_BACKENDS:
        model = build_pose_model(project, backend=backend)
        assert isinstance(model, PoseModel)
        assert callable(model.load)
        assert callable(model.predict)
