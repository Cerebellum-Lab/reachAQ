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
    TORCH_MODEL_CONFIDENCE_THRESHOLDS,
    confidence_threshold,
    trained_model_name,
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


# --- per-model thresholds -----------------------------------------------------
#
# The per-backend default was provisional: it was measured on 24 frames that
# were in the training set of both models, which compares the engines fairly
# but does not calibrate either. Re-measured on the 19 held-out frames of the
# curated project, scoring every labelled keypoint:
#
#   gate   resnet_50 pass / median   cspnext_s pass / median   cspnext RH_grab
#   0.10        97.5% / 1.56 px           93.4% / 3.73 px            97%
#   0.30        90.1% / 1.46 px           27.3% / 2.99 px            17%
#   0.60        76.0% / 1.30 px            1.7% / 1.30 px             0%
#
# cspnext_s never scores above 0.33, so the 0.6 default discards 98% of its
# output and holds rh_grab_seen permanently False - the flag a pose-driven
# reach gate is built on. resnet_50 at 0.6 is not free either: it drops 42% of
# the RH_grab detections that 0.3 keeps at 1.46 px.


def test_a_known_torch_model_refines_the_backend_default():
    """The score scale belongs to the trained model, not to the engine."""
    assert confidence_threshold(TORCH_BACKEND, {}, model_name="cspnext_s") == 0.1
    assert confidence_threshold(TORCH_BACKEND, {}, model_name="resnet_50") == 0.3


def test_the_cspnext_default_does_not_discard_the_whole_model():
    """At 0.6 cspnext_s passes 1.7% of keypoints and no RH_grab at all."""
    assert TORCH_MODEL_CONFIDENCE_THRESHOLDS["cspnext_s"] <= 0.15


def test_an_unknown_model_falls_back_to_the_backend_default():
    """A retrain on a net_type nobody has measured must not silently shift."""
    assert confidence_threshold(TORCH_BACKEND, {}, model_name="hrnet_w32") == 0.6
    assert confidence_threshold(TORCH_BACKEND, {}, model_name=None) == 0.6
    assert confidence_threshold(TORCH_BACKEND, {}, model_name="") == 0.6


def test_a_model_name_does_not_move_the_tensorflow_default():
    """These were measured under the PyTorch engine; TensorFlow saturates at 1.0."""
    assert confidence_threshold(TENSORFLOW_BACKEND, {}, model_name="resnet_50") == 0.9


def test_an_explicit_override_still_wins_over_a_per_model_default():
    environ = {POSE_CONFIDENCE_THRESHOLD_ENV_VAR: "0.25"}
    assert confidence_threshold(
        TORCH_BACKEND, environ, model_name="cspnext_s") == pytest.approx(0.25)


def test_model_name_is_keyword_only():
    """environ is the second positional argument in every existing call."""
    import inspect

    parameter = inspect.signature(confidence_threshold).parameters["model_name"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def _write_backbone(train_folder, model_name):
    """Write a pytorch_config.yaml naming this backbone, as DeepLabCut would.

    Serialised with yaml rather than hand-written text so the test exercises the
    real format trained_model_name has to read.
    """
    import yaml

    (train_folder / "pytorch_config.yaml").write_text(
        yaml.safe_dump({"model": {"backbone": {"model_name": model_name}}})
    )


def test_trained_model_name_reads_the_backbone_from_the_project(tmp_path):
    train = (tmp_path / "dlc-models-pytorch" / "iteration-2"
             / "projectSep18-trainset95shuffle3" / "train")
    train.mkdir(parents=True)
    (train / "snapshot-200.pt").write_bytes(b"")
    _write_backbone(train, "cspnext_s")
    assert trained_model_name(str(tmp_path)) == "cspnext_s"


def test_trained_model_name_is_none_when_there_is_nothing_to_read(tmp_path):
    """Never raises: an unreadable project must fall back, not stop inference."""
    assert trained_model_name(str(tmp_path)) is None
    assert trained_model_name(None) is None
    assert trained_model_name(str(tmp_path / "absent")) is None


def test_trained_model_name_ignores_an_untrained_shuffle(tmp_path):
    """A shuffle with a config but no snapshot is not what would be loaded."""
    base = tmp_path / "dlc-models-pytorch" / "iteration-2"
    untrained = base / "projectSep18-trainset95shuffle1" / "train"
    untrained.mkdir(parents=True)
    _write_backbone(untrained, "resnet_50")
    trained = base / "projectSep18-trainset95shuffle3" / "train"
    trained.mkdir(parents=True)
    (trained / "snapshot-200.pt").write_bytes(b"")
    _write_backbone(trained, "cspnext_s")
    assert trained_model_name(str(tmp_path)) == "cspnext_s"


# --- cspnext_m and rtmpose_s -------------------------------------------------
#
# Measured the same way, on the project's held-out frames.
#
# cspnext_m's score is discriminative - tightening the gate genuinely improves
# accuracy - so a threshold is a real control on it:
#
#   gate   pass    median      gate   pass    median
#   0.10  99.3%   1.44 px      0.30  84.2%   1.31 px
#   0.20  96.8%   1.43 px      0.60  51.6%   1.22 px
#
# 0.2 keeps 96.8% of keypoints and detects RH_grab in 95%, LH_grab 95%,
# Pellet 96% and Nose 99% of the frames where they are labelled.
#
# rtmpose_s is the opposite and the reason its entry exists at all. Its SimCC
# head reports 0.325 at the very lowest, so it detects 100% of every bodypart at
# any gate up to 0.30 - and its error does not move when the gate is tightened:
# 2.93 px at 0.05 against 2.87 px at 0.90. Rejecting predictions buys nothing,
# so the entry keeps coverage rather than discarding it for no gain.


def test_cspnext_m_uses_its_measured_threshold():
    assert confidence_threshold(TORCH_BACKEND, {}, model_name="cspnext_m") == 0.2


def test_rtmpose_keeps_coverage_because_its_gate_buys_nothing():
    """Its error is flat across thresholds, so rejecting only loses keypoints."""
    assert TORCH_MODEL_CONFIDENCE_THRESHOLDS["rtmpose_s"] <= 0.3
    assert confidence_threshold(TORCH_BACKEND, {}, model_name="rtmpose_s") == 0.3


def test_every_measured_threshold_is_a_usable_probability():
    for name, value in TORCH_MODEL_CONFIDENCE_THRESHOLDS.items():
        assert 0.0 < value < 1.0, f"{name} threshold {value} is not usable"


def test_the_measured_models_are_the_ones_we_have_trained():
    """A typo in a key would silently fall back to the backend default."""
    assert set(TORCH_MODEL_CONFIDENCE_THRESHOLDS) == {
        "resnet_50", "cspnext_s", "cspnext_m", "rtmpose_s"}
