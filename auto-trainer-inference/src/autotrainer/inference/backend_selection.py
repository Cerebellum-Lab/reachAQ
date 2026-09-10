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

TENSORFLOW_BACKEND = "tensorflow"
TORCH_BACKEND = "torch"

# TensorFlow stays the default: it is what the trained snapshots on deployed
# rigs were produced with, so an unset variable must not change their behaviour.
DEFAULT_POSE_BACKEND = TENSORFLOW_BACKEND

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
