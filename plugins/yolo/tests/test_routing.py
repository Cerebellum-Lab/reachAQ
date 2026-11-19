"""Routing a YOLO model to the installed plugin.

Moved from core's tests/pose_backend_selection_test.py,
tests/pose_prevalidate_routing_test.py and tests/overlay_part_coverage_test.py
with the plugin. Discovery is real here: the plugin is found through its entry
point, not a fake one.
"""

import pytest

from autotrainer.inference import backend_selection
from autotrainer.inference import pose_plugins
from autotrainer.inference.backend_selection import (
    POSE_BACKEND_ENV_VAR,
    TORCH_BACKEND,
    confidence_threshold,
    pre_validate_model,
    selected_backend,
)
from autotrainer.inference.pose_plugins import PoseBackendNotInstalled, resolve
from autotrainer.pose_plugins.yolo.plugin import YOLO_CONFIDENCE_THRESHOLD


def _only(monkeypatch, *installed):
    """Pin which engines are importable, so the test does not read this env."""
    monkeypatch.setattr(pose_plugins, "_installed",
                        lambda module: module in installed)


@pytest.fixture
def engines(monkeypatch):
    """Every engine importable, as on a site that installed ultralytics."""
    monkeypatch.setattr(pose_plugins, "_installed", lambda module: True)


def test_a_yolo_model_selects_torch_whatever_is_installed(monkeypatch, tmp_path):
    """YOLO weights are torch and cannot be anything else.

    The model is chosen in the configuration while the backend can only be
    chosen through the environment, so without this a YOLO model on a rig with
    TensorFlow installed probes the wrong runtime.
    """
    monkeypatch.delenv(POSE_BACKEND_ENV_VAR, raising=False)
    _only(monkeypatch, "ultralytics", "tensorflow", "torch")
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]")
    assert selected_backend(model_path=str(tmp_path)) == TORCH_BACKEND


def test_a_yolo_model_keeps_torch_over_an_explicit_variable(monkeypatch, tmp_path):
    """The variable chooses DeepLabCut's engine; YOLO weights have only one.

    This used to answer TensorFlow, and the GPU probe then checked a runtime
    the model could never run on.
    """
    monkeypatch.setenv(POSE_BACKEND_ENV_VAR, "tensorflow")
    _only(monkeypatch, "ultralytics", "tensorflow", "torch")
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]")
    assert selected_backend(model_path=str(tmp_path)) == TORCH_BACKEND


def test_the_installed_plugin_is_listed_with_its_runtime(monkeypatch):
    _only(monkeypatch, "ultralytics", "torch")
    assert "yolo (torch)" in backend_selection.available_backends()
    _only(monkeypatch, "torch")
    assert "yolo (torch)" not in backend_selection.available_backends()


def test_a_claiming_plugin_with_a_missing_engine_names_it_and_the_alternative(
        tmp_path, monkeypatch):
    _only(monkeypatch, "torch")
    (tmp_path / "yolo_pose.yaml").write_text("bodyparts: [a]\n")
    with pytest.raises(PoseBackendNotInstalled) as raised:
        resolve(str(tmp_path))
    assert raised.value.missing == ("ultralytics",)
    assert "reachaq-plugin-install.sh yolo" in str(raised.value)
    assert "autotrainer.inference.keypoint.train" in str(raised.value)


def test_a_yolo_model_is_checked_against_the_yolo_layout(tmp_path, engines):
    """A complete YOLO directory passes rather than being reported broken."""
    (tmp_path / "yolo_pose.yaml").write_text(
        "imgsz: 256\nweights: best.pt\nbodyparts:\n  - Nose\n")
    (tmp_path / "best.pt").write_bytes(b"not a real checkpoint")
    pre_validate_model(str(tmp_path))


def test_a_yolo_model_missing_its_weights_still_fails(tmp_path, engines):
    """Routing must not turn the check into a rubber stamp."""
    (tmp_path / "yolo_pose.yaml").write_text(
        "imgsz: 256\nweights: best.pt\nbodyparts:\n  - Nose\n")
    with pytest.raises(FileNotFoundError, match="best.pt"):
        pre_validate_model(str(tmp_path))


def test_a_yolo_model_is_gated_on_its_own_measured_scale(tmp_path, engines):
    """Measured at 0.10: 99.1% coverage, 2.04 px median on 33 held-out frames.

    The backend default of 0.6 costs 15 points of coverage and takes both
    tongue points to zero; TensorFlow's 0.9 takes Pellet to 8%.
    """
    model = tmp_path / "N_full_p36"
    model.mkdir()
    (model / "yolo_pose.yaml").write_text("bodyparts: [Pellet]\n")

    assert selected_backend(model_path=str(model)) == TORCH_BACKEND
    assert confidence_threshold(
        selected_backend(model_path=str(model)),
        model_path=str(model),
    ) == pytest.approx(YOLO_CONFIDENCE_THRESHOLD)


def test_a_yolo_model_is_not_gated_on_the_tensorflow_scale(tmp_path, engines):
    """The live bug: the backend was picked without looking at the model."""
    model = tmp_path / "N_full_p36"
    model.mkdir()
    (model / "yolo_pose.yaml").write_text("bodyparts: [Pellet]\n")

    gate = confidence_threshold(
        selected_backend(model_path=str(model)),
        model_path=str(model),
    )
    # Pellet's highest confidence over 600 demo frames was 0.862, so anything
    # at or above that gate hides it on every single frame.
    assert gate < 0.862
