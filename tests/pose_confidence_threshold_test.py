"""The confidence gate has to follow the backend, not a single constant.

`PoseAlgorithm` gates presence and 3D triangulation on a confidence threshold
that was calibrated for the DeepLabCut TensorFlow engine. Measured on this
project against its own labels, 157 keypoints from 24 labelled frames:

    engine        likelihood p50   spearman vs error   fraction >= 0.9
    TensorFlow        1.000            -0.048              100%
    PyTorch           0.773            -0.227                8.3%

TensorFlow's likelihood is a saturated constant, so 0.9 passes everything and
has never filtered on that engine. PyTorch's is a real distribution that ranks
error, so the same 0.9 would discard about 92% of correctly located keypoints.
One constant cannot serve both.

The threshold is therefore injected, with a per-backend default and an
environment override, because the scale belongs to the trained model rather
than to the engine.
"""

import pytest

from autotrainer.inference.backend_selection import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    POSE_BACKEND_ENV_VAR,
    POSE_CONFIDENCE_THRESHOLD_ENV_VAR,
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
    confidence_threshold,
)
from autotrainer.inference.pose_algorithm import PoseAlgorithm


def test_each_backend_has_its_own_default():
    assert confidence_threshold(TENSORFLOW_BACKEND, {}) == 0.9
    assert confidence_threshold(TORCH_BACKEND, {}) == 0.6


def test_the_tensorflow_default_is_unchanged():
    """Every deployed rig runs this engine; moving it changes live behaviour."""
    assert DEFAULT_CONFIDENCE_THRESHOLDS[TENSORFLOW_BACKEND] == 0.9
    assert PoseAlgorithm.MIN_CONFIDENCE_PRESENT_THRESHOLD == 0.9
    assert PoseAlgorithm.MIN_CONFIDENCE_PLOT_THRESHOLD == 0.9


def test_the_torch_default_would_not_discard_most_keypoints():
    """At 0.9 the PyTorch engine keeps 8.3% of keypoints; at 0.6 it keeps 91%."""
    assert DEFAULT_CONFIDENCE_THRESHOLDS[TORCH_BACKEND] < 0.9


def test_the_threshold_follows_the_selected_backend():
    assert confidence_threshold(None, {POSE_BACKEND_ENV_VAR: "torch"}) == 0.6
    assert confidence_threshold(None, {POSE_BACKEND_ENV_VAR: "tensorflow"}) == 0.9
    assert confidence_threshold(None, {}) == 0.9, "unset means TensorFlow"


def test_an_explicit_override_wins():
    """The scale belongs to the trained model, so a retrain can move it."""
    environ = {POSE_BACKEND_ENV_VAR: "torch",
               POSE_CONFIDENCE_THRESHOLD_ENV_VAR: "0.45"}
    assert confidence_threshold(None, environ) == pytest.approx(0.45)


@pytest.mark.parametrize("value", ["", "   ", "abc", "-0.1", "1.5", "None"])
def test_an_unusable_override_falls_back_to_the_backend_default(value):
    """Gating on a nonsense threshold is worse than ignoring the typo."""
    environ = {POSE_BACKEND_ENV_VAR: "torch",
               POSE_CONFIDENCE_THRESHOLD_ENV_VAR: value}
    assert confidence_threshold(None, environ) == 0.6


@pytest.mark.parametrize("value", ["0.0", "1.0"])
def test_the_bounds_are_accepted(value):
    environ = {POSE_CONFIDENCE_THRESHOLD_ENV_VAR: value}
    assert confidence_threshold(TENSORFLOW_BACKEND, environ) == pytest.approx(float(value))


def test_an_unknown_backend_uses_the_default_backend_threshold():
    assert confidence_threshold("onnx", {}) == 0.9


def test_pose_algorithm_defaults_to_the_class_constants():
    """Existing callers and tests must be unaffected by the new parameter."""
    algorithm = PoseAlgorithm()
    assert algorithm._present_threshold == PoseAlgorithm.MIN_CONFIDENCE_PRESENT_THRESHOLD
    assert algorithm._plot_threshold == PoseAlgorithm.MIN_CONFIDENCE_PLOT_THRESHOLD


def test_pose_algorithm_accepts_an_injected_threshold():
    algorithm = PoseAlgorithm(confidence_threshold=0.6)
    assert algorithm._present_threshold == pytest.approx(0.6)
    assert algorithm._plot_threshold == pytest.approx(0.6)


def test_no_consumer_still_reads_the_constant_directly():
    """A missed call site would silently keep gating at 0.9 on both engines."""
    import inspect

    source = inspect.getsource(PoseAlgorithm)
    body = source.split("MIN_CONFIDENCE_PRESENT_THRESHOLD = 0.9", 1)[1]
    for name in ("MIN_CONFIDENCE_PRESENT_THRESHOLD", "MIN_CONFIDENCE_PLOT_THRESHOLD"):
        # The only permitted mentions after the declarations are the two
        # fallbacks in __init__.
        assert body.count(name) <= 1, f"{name} is still read directly"


def test_the_three_dimensional_gate_uses_the_same_threshold():
    """p_thresh was a separate hardcoded 0.9 for triangulation."""
    import inspect

    source = inspect.getsource(PoseAlgorithm)
    assert "p_thresh = 0.9" not in source
    assert "p_thresh = self._present_threshold" in source


def test_the_env_var_name_is_stable():
    assert POSE_CONFIDENCE_THRESHOLD_ENV_VAR == "REACHAQ_POSE_CONFIDENCE_THRESHOLD"
