"""Which pose backend, and which GPU runtime, live inference runs a model on.

The model is routed to the pose backend plugin that claims it (see
pose_plugins.resolve): a keypoint or YOLO model always runs on torch, a
DeepLabCut model on REACHAQ_POSE_BACKEND. The functions here are the stable
face the rest of reachAQ calls; each one asks the resolved plugin.

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

from . import pose_plugins
from .pose_model import PoseModel
from .pose_plugins import AmbiguousPoseModel, PoseBackendNotInstalled, resolve

logger = get_verbose_logger(__name__)

POSE_BACKEND_ENV_VAR = "REACHAQ_POSE_BACKEND"
POSE_CONFIDENCE_THRESHOLD_ENV_VAR = "REACHAQ_POSE_CONFIDENCE_THRESHOLD"


TENSORFLOW_BACKEND = "tensorflow"
TORCH_BACKEND = "torch"

# TensorFlow stays the default: it is what the trained snapshots on deployed
# rigs were produced with, so an unset variable must not change their behaviour.
DEFAULT_POSE_BACKEND = TENSORFLOW_BACKEND

# The gate for a model no plugin can resolve: the DeepLabCut plugin's
# per-runtime defaults, kept here so a model that cannot run is still gated on
# the scale of the runtime it was reported under. See
# autotrainer.inference.dlc.plugin.DEFAULT_CONFIDENCE_THRESHOLDS for the
# measurements.
_FALLBACK_CONFIDENCE_THRESHOLDS = {
    TENSORFLOW_BACKEND: 0.9,
    TORCH_BACKEND: 0.6,
}

# The keypoint backend's confidence is the sigmoid of its heatmap peak, a scale
# of its own again. Measured the same way as YOLO above, on the same 33
# held-out frames (212 labelled keypoints), for K_full_s1 (2026-11-18):
#
#   gate   coverage   median err   RH_grab cov   Tongue_tip cov (n=5)
#   0.10      99.1%      0.87 px        100%            80%
#   0.30      95.8%      0.86 px         90%            40%
#   0.60      92.9%      0.82 px         70%            40%
#   0.90      63.2%      0.74 px         50%            20%
#
# The error barely moves with the gate, so a higher one buys nothing on these
# frames and costs coverage. On held-out recorded sessions the score drops when
# a part is out of view - a pellet already taken, or hidden by the paw - where
# YOLO went on reporting one, so this model's detection rate there is lower by
# design rather than by miss.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.1

_MIN_CONFIDENCE_THRESHOLD = 0.0
_MAX_CONFIDENCE_THRESHOLD = 1.0

POSE_BACKENDS = (TENSORFLOW_BACKEND, TORCH_BACKEND)


def is_yolo_model(model_path: typing.Optional[str]) -> bool:
    """Whether this location is a YOLO model directory, by its sidecar."""
    from .yolo.plugin import YOLO_PLUGIN
    return YOLO_PLUGIN.claims(model_path or "")


def is_keypoint_model(model_path: typing.Optional[str]) -> bool:
    """Whether this location is a keypoint model directory, by its sidecar."""
    return pose_plugins.KEYPOINT_PLUGIN.claims(model_path or "")


def available_backends() -> typing.Tuple[str, ...]:
    """The pose backends this deployment could actually run, each with the
    runtimes it would run on, in routing order."""
    return tuple(f"{plugin.name} ({', '.join(plugin.runtimes())})"
                 for plugin in pose_plugins.discover_plugins()
                 if not plugin.missing_requirements())


def selected_backend(environ: typing.Optional[typing.Mapping[str, str]] = None,
                     model_path: typing.Optional[str] = None) -> str:
    """
    Return the GPU runtime live inference should run this model on.

    The plugin that claims the model decides. A keypoint or YOLO model settles
    it on its own - those weights are torch and cannot be anything else - which
    matters because the model can be chosen in the configuration while the
    runtime can only be chosen through the environment. A DeepLabCut model
    follows REACHAQ_POSE_BACKEND, and unset, what is installed.

    Never raises and never returns an unknown name. A model no installed plugin
    can run, or no model at all, answers torch, the built-in keypoint
    backend's runtime: refusing here would stop the GPU probe from running, and
    pre_validate_model is where the operator is told what to install.
    """
    source = os.environ if environ is None else environ
    try:
        plugin = resolve(model_path)
    except (PoseBackendNotInstalled, AmbiguousPoseModel) as err:
        if model_path:
            logger.warning("using the %s runtime for now: %s", TORCH_BACKEND, err)
        return TORCH_BACKEND
    return plugin.runtime(model_path, source)


def confidence_threshold(
    backend: typing.Optional[str] = None,
    environ: typing.Optional[typing.Mapping[str, str]] = None,
    *,
    model_path: typing.Optional[str] = None,
) -> float:
    """
    Return the confidence a keypoint needs to count as present.

    The resolved plugin's default, because every engine reports on its own
    scale, and DeepLabCut's differs by runtime and by trained backbone too; see
    each plugin for the measurements. A model no plugin can resolve falls back
    to the per-runtime default.

    model_path is keyword-only because environ is the second positional
    argument in the existing call sites.

    Never raises. An unusable value falls back to the backend default with a
    warning: refusing to start over a malformed number would be worse, and
    silently gating on a nonsense threshold would be worse still.
    """
    backend = (selected_backend(environ, model_path=model_path)
               if backend is None else backend)
    try:
        plugin = resolve(model_path)
    except (PoseBackendNotInstalled, AmbiguousPoseModel) as err:
        default = _FALLBACK_CONFIDENCE_THRESHOLDS.get(
            backend, _FALLBACK_CONFIDENCE_THRESHOLDS[DEFAULT_POSE_BACKEND]
        )
        if model_path:
            logger.warning("gating at the %s default of %.2f for now: %s",
                           backend, default, err)
    else:
        default = plugin.confidence_threshold(model_path, backend)
        logger.info("%s pose backend gates %r at %.2f on the %s runtime",
                    plugin.name, model_path, default, backend)

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


def pre_validate_model(model_path: str,
                       backend: typing.Optional[str] = None) -> None:
    """Cheap filesystem check of a model location, by the plugin that runs it.

    Routed the same way build_pose_model routes, so a YOLO directory is checked
    against the YOLO layout. app_model used to call DlcPoseModel.pre_validate
    unconditionally, which looked for a DeepLabCut config.yaml and reported
    every YOLO model as broken - an error dialog on every start, for a model
    that then loaded and ran correctly.

    Raises PoseBackendNotInstalled, naming what to install, when no installed
    plugin can run the model, and otherwise whatever the plugin's own
    pre_validate raises, so the caller still reports one failure rather than
    guessing.
    """
    plugin = resolve(model_path)
    runtime = plugin.runtime(model_path, os.environ) if backend is None else backend
    plugin.pre_validate(model_path, runtime)


def build_pose_model(model_path: str, shuffle_index=1, training_index=0, batch_size=1,
                     backend: typing.Optional[str] = None) -> PoseModel:
    """
    Construct the pose model through the plugin that claims it.

    Every plugin's model satisfies the same PoseModel contract, so callers do
    not branch on the engine. backend is passed to the plugin as its runtime;
    a plugin with one runtime ignores it.
    """
    if backend is not None and backend not in POSE_BACKENDS:
        # An explicit argument is a programming error, unlike a typo in the
        # environment, so it is refused before the model is even looked at.
        raise ValueError(f"Unsupported pose backend: {backend!r}; expected one of {POSE_BACKENDS}")
    plugin = resolve(model_path)
    runtime = plugin.runtime(model_path, os.environ) if backend is None else backend
    return plugin.build(model_path, runtime, shuffle_index=shuffle_index,
                        training_index=training_index, batch_size=batch_size)
