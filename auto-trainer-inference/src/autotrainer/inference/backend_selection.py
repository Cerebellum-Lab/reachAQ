"""Which DeepLabCut engine live inference runs on.

The two engines cannot share one environment - nvidia-cudnn-cu11 and
nvidia-cudnn-cu12 both install libcudnn.so.8 - so a deployment installs exactly
one of the `tensorflow` or `torch` extras and this selects the matching backend.

Selection is an environment variable rather than a SystemConfiguration field on
purpose. SystemConfiguration.version is pinned at 57 and rejects any other
value, so a new field would make existing rigs fail to load their config. The
inference ROI is configured the same way and for the same reason. Once a version
bump is on the table both should move into the schema.

The returned name is also the name gpu_runtime uses for its probes, so it can be
passed straight to detect_gpu_runtime(required_backend=...). That coupling is
what stops a rig from checking TensorFlow's CUDA while running PyTorch.
"""

import os
import typing

from autotrainer.core.logging import get_verbose_logger

from .pose_model import PoseModel

logger = get_verbose_logger(__name__)

POSE_BACKEND_ENV_VAR = "REACHAQ_POSE_BACKEND"
POSE_CONFIDENCE_THRESHOLD_ENV_VAR = "REACHAQ_POSE_CONFIDENCE_THRESHOLD"


TENSORFLOW_BACKEND = "tensorflow"
TORCH_BACKEND = "torch"

# TensorFlow stays the default: it is what the trained snapshots on deployed
# rigs were produced with, so an unset variable must not change their behaviour.
DEFAULT_POSE_BACKEND = TENSORFLOW_BACKEND

# The confidence a keypoint needs before it counts as present and enters
# triangulation. This is not a property of the pipeline, it is a property of the
# model's output scale, and the two engines do not share one.
#
# Measured on this project, 157 keypoints from 24 labelled frames, scored
# against the project's own labels:
#
#                       likelihood p50   spearman vs error   fraction >= 0.9
#   TensorFlow              1.000            -0.048              100%
#   PyTorch                 0.773            -0.227                8.3%
#
# TensorFlow's likelihood is a saturated constant, so 0.9 has never filtered
# anything on that engine; it passes every keypoint. PyTorch's is a real
# distribution that ranks error: at >= 0.6 it keeps 91% of keypoints at
# 1.13 px RMSE and rejects the rest at 1.61 px.
#
# 0.9 is kept for TensorFlow because changing it would change live behaviour on
# every deployed rig. 0.6 for PyTorch is PROVISIONAL: those 24 frames were in
# the training set for both models, so the figures compare the engines fairly
# but are not a calibration. Re-derive it on held-out frames before relying on
# it, and override with the environment variable meanwhile.
DEFAULT_CONFIDENCE_THRESHOLDS = {
    TENSORFLOW_BACKEND: 0.9,
    TORCH_BACKEND: 0.6,
}

_MIN_CONFIDENCE_THRESHOLD = 0.0
_MAX_CONFIDENCE_THRESHOLD = 1.0

POSE_BACKENDS = (TENSORFLOW_BACKEND, TORCH_BACKEND)

# Spellings an operator is likely to type, mapped to the canonical name.
_BACKEND_ALIASES = {
    "tensorflow": TENSORFLOW_BACKEND,
    "tf": TENSORFLOW_BACKEND,
    "tf1": TENSORFLOW_BACKEND,
    "torch": TORCH_BACKEND,
    "pytorch": TORCH_BACKEND,
    "pt": TORCH_BACKEND,
}


def selected_backend(environ: typing.Optional[typing.Mapping[str, str]] = None) -> str:
    """
    Return the configured pose backend name.

    Never raises and never returns an unknown name: an unusable value falls back
    to the default with a warning, because refusing to start live inference over
    a typo in an environment variable is worse than running the engine the rig
    was already running.
    """
    source = os.environ if environ is None else environ
    raw = source.get(POSE_BACKEND_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_POSE_BACKEND

    backend = _BACKEND_ALIASES.get(raw.strip().lower())
    if backend is None:
        logger.warning("%s=%r is not one of %s; using %s",
                       POSE_BACKEND_ENV_VAR, raw, POSE_BACKENDS, DEFAULT_POSE_BACKEND)
        return DEFAULT_POSE_BACKEND

    logger.info("%s=%r selects the %s pose backend", POSE_BACKEND_ENV_VAR, raw, backend)
    return backend


def confidence_threshold(
    backend: typing.Optional[str] = None,
    environ: typing.Optional[typing.Mapping[str, str]] = None,
) -> float:
    """
    Return the confidence a keypoint needs to count as present.

    Defaults per backend, because the two engines report on different scales;
    see DEFAULT_CONFIDENCE_THRESHOLDS for the measurements. Overridable, because
    the scale is really a property of the trained model rather than the engine,
    so a different net_type or a retrain can move it and the default would go
    quietly stale.

    Never raises. An unusable value falls back to the backend default with a
    warning: refusing to start over a malformed number would be worse, and
    silently gating on a nonsense threshold would be worse still.
    """
    backend = selected_backend(environ) if backend is None else backend
    default = DEFAULT_CONFIDENCE_THRESHOLDS.get(
        backend, DEFAULT_CONFIDENCE_THRESHOLDS[DEFAULT_POSE_BACKEND]
    )

    source = os.environ if environ is None else environ
    raw = source.get(POSE_CONFIDENCE_THRESHOLD_ENV_VAR)
    if raw is None or not raw.strip():
        return default

    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not a number; using %.2f for the %s backend",
                       POSE_CONFIDENCE_THRESHOLD_ENV_VAR, raw, default, backend)
        return default
    if not _MIN_CONFIDENCE_THRESHOLD <= value <= _MAX_CONFIDENCE_THRESHOLD:
        logger.warning("%s=%r is outside %.1f..%.1f; using %.2f for the %s backend",
                       POSE_CONFIDENCE_THRESHOLD_ENV_VAR, raw,
                       _MIN_CONFIDENCE_THRESHOLD, _MAX_CONFIDENCE_THRESHOLD,
                       default, backend)
        return default

    logger.notice("%s=%.3f overrides the %s default of %.2f",
                  POSE_CONFIDENCE_THRESHOLD_ENV_VAR, value, backend, default)
    return value


def build_pose_model(model_path: str, shuffle_index=1, training_index=0, batch_size=1,
                     backend: typing.Optional[str] = None) -> PoseModel:
    """
    Construct the DeepLabCut pose model for the selected backend.

    Both models satisfy the same PoseModel contract, so callers do not branch on
    the backend. The imports are deferred to the selected branch: importing the
    TensorFlow model pulls in TensorFlow, and on a torch-only deployment that
    import does not resolve at all.
    """
    backend = selected_backend() if backend is None else backend

    if backend == TORCH_BACKEND:
        from .dlc import DlcTorchPoseModel
        return DlcTorchPoseModel(model_path, shuffle_index, training_index, batch_size)

    if backend == TENSORFLOW_BACKEND:
        from .dlc import DlcPoseModel
        return DlcPoseModel(model_path, shuffle_index, training_index, batch_size)

    raise ValueError(f"Unsupported pose backend: {backend!r}; expected one of {POSE_BACKENDS}")
