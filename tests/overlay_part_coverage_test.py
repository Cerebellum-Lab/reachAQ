"""Every part the model reports must be drawable, and the gate must fit it.

Two independent faults kept ten of this rig's fourteen keypoints off the live
overlay, and neither announced itself:

  * the painter held a fixed registry naming six elements, two of which were
    composites the pose algorithm never emits, so Mouth, both tongue points and
    all six hand parts had no slot at any confidence;
  * the backend was chosen without looking at the model, so a YOLO model was
    gated at TensorFlow's 0.9 - above the confidence Pellet ever reaches.

These pin both, because the failure mode of each is silence.
"""

import os

import pytest

from autotrainer.core.pose_elements import SceneElement
from autotrainer.inference.backend_selection import (
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
    YOLO_CONFIDENCE_THRESHOLD,
    confidence_threshold,
    selected_backend,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# What the live model emits, from its sidecar.
MODEL_PARTS = (
    "RH_flat", "RH_spread", "RH_grab", "LH_flat", "LH_spread", "LH_grab",
    "Pellet", "Star", "Triangle", "Diamond", "Nose", "Mouth",
    "Tongue_mid", "Tongue_tip",
)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _view(qapp):
    from autotrainer.pyside.capture.QtGLImageView import QGLImageView
    view = QGLImageView(width=256, height=256)
    view._width = view._height = 256.0
    return view


def _points(parts):
    from autotrainer.inference import PoseLocation
    return {part: PoseLocation(index, 10.0 + index, 20.0)
            for index, part in enumerate(parts)}


def _visible(view):
    return {name for name, item in view._points.items() if item.isVisible()}


def test_every_part_the_model_emits_is_drawn(qapp):
    """The registry used to name six; ten parts could not be drawn at all."""
    view = _view(qapp)
    view.set_points(_points(MODEL_PARTS))

    assert _visible(view) == set(MODEL_PARTS)


def test_the_parts_that_were_silently_undrawable_now_draw(qapp):
    view = _view(qapp)
    view.set_points(_points(MODEL_PARTS))

    previously_lost = {
        SceneElement.Mouth, SceneElement.Nose,
        SceneElement.Tongue_mid, SceneElement.Tongue_tip,
        SceneElement.RH_flat, SceneElement.RH_spread, SceneElement.RH_grab,
        SceneElement.LH_flat, SceneElement.LH_spread, SceneElement.LH_grab,
    }
    assert previously_lost <= _visible(view)


def test_a_part_absent_this_frame_is_hidden_not_left_stale(qapp):
    view = _view(qapp)
    view.set_points(_points(MODEL_PARTS))
    view.set_points(_points(("Pellet", "Star")))

    assert _visible(view) == {"Pellet", "Star"}


def test_configured_parts_narrow_the_overlay(qapp):
    view = _view(qapp)
    view.set_overlay_parts(("Pellet", "Nose"))
    view.set_points(_points(MODEL_PARTS))

    assert _visible(view) == {"Pellet", "Nose"}


def test_empty_configuration_means_every_part(qapp):
    """Empty is the default and must not be read as "draw nothing"."""
    view = _view(qapp)
    view.set_overlay_parts(())
    view.set_points(_points(MODEL_PARTS))

    assert _visible(view) == set(MODEL_PARTS)


def test_narrowing_hides_parts_already_on_screen(qapp):
    view = _view(qapp)
    view.set_points(_points(MODEL_PARTS))
    view.set_overlay_parts(("Pellet",))

    assert _visible(view) == {"Pellet"}


def test_a_part_outside_the_frame_is_not_drawn(qapp):
    from autotrainer.inference import PoseLocation
    view = _view(qapp)
    view.set_points({"Pellet": PoseLocation(0, 10.0, 10.0),
                     "Star": PoseLocation(1, -5.0, 10.0)})

    assert _visible(view) == {"Pellet"}


def test_a_yolo_model_is_gated_on_its_own_measured_scale(tmp_path):
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


def test_a_yolo_model_is_not_gated_on_the_tensorflow_scale(tmp_path):
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


def test_a_deeplabcut_project_keeps_its_own_threshold(tmp_path):
    """The YOLO rule must not capture DeepLabCut projects."""
    project = tmp_path / "dlc"
    project.mkdir()
    (project / "config.yaml").write_text("bodyparts: [Pellet]\n")

    gate = confidence_threshold(TENSORFLOW_BACKEND, model_path=str(project))
    assert gate == pytest.approx(0.9)
