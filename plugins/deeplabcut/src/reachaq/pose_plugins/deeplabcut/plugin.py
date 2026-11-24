"""DeepLabCut as a pose backend plugin.

Shipped as the reachaq-pose-deeplabcut distribution and found through its
entry point in the reachaq.pose_backends group. Nothing here imports DeepLabCut, TensorFlow or torch at module
scope: discovery imports this module on every rig, and only build() may pay for
an engine.
"""

import glob
import os
import typing

import yaml

from reachaq.core.logging import get_verbose_logger
from reachaq.inference import pose_plugins
from reachaq.inference.backend_selection import (
    DEFAULT_POSE_BACKEND,
    POSE_BACKEND_ENV_VAR,
    POSE_BACKENDS,
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
    parse_runtime,
)
from reachaq.inference.pose_model import PoseModel

logger = get_verbose_logger(__name__)

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

# Refinements of the per-backend default for the models whose scale has been
# measured on held-out frames. The per-backend figures above were taken on 24
# frames that were in the training set of both models: fair for comparing the
# engines, but not a calibration of either.
#
# Re-measured on the 19 held-out frames of the curated project, every labelled
# keypoint scored against its label:
#
#   gate   resnet_50 pass / median   cspnext_s pass / median   cspnext RH_grab
#   0.10        97.5% / 1.56 px           93.4% / 3.73 px            97%
#   0.15        96.7% / 1.55 px           78.5% / 3.57 px            83%
#   0.30        90.1% / 1.46 px           27.3% / 2.99 px            17%
#   0.60        76.0% / 1.30 px            1.7% / 1.30 px             0%
#
# cspnext_s never scores above 0.33, so 0.6 discards 98% of its output and
# holds rh_grab_seen permanently False - and that flag is what a pose-driven
# reach gate is built on. 0.1 keeps 93% of keypoints at 3.7 px and detects
# RH_grab in 97% of the frames where it is labelled.
#
# resnet_50 moves to 0.3 for the same reason, less dramatically: 0.6 drops 42%
# of the RH_grab detections that 0.3 keeps at 1.46 px. This is safe to change
# because the PyTorch engine has no deployment history - the TensorFlow default
# is what deployed rigs run, and it is untouched.
#
# Keyed by the backbone name DeepLabCut records in the trained shuffle's
# pytorch_config.yaml, which is what trained_model_name() reads.
#
# cspnext_m, measured the same way, has a score that is genuinely
# discriminative - tightening the gate improves accuracy, which is what a
# threshold is for:
#
#   gate   pass    median        gate   pass    median
#   0.10  99.3%   1.44 px        0.30  84.2%   1.31 px
#   0.20  96.8%   1.43 px        0.60  51.6%   1.22 px
#
# 0.2 keeps 96.8% of keypoints and detects RH_grab in 95%, LH_grab 95%,
# Pellet 96% and Nose 99% of the frames where each is labelled.
#
# rtmpose_s is the opposite case, and the reason its entry is here rather than
# left to fall through. Its SimCC head never scores below 0.325, so it "detects"
# 100% of every bodypart at any gate up to 0.30, and its error does not move
# when the gate is tightened: 2.93 px at 0.05 against 2.87 px at 0.90. Rejecting
# its predictions buys no accuracy, so the entry keeps coverage instead of
# discarding keypoints for nothing.
#
# That flatness is a warning, not a convenience. On a second camera this model
# reported a mean confidence of 0.847 while sitting about 48 px off the
# pipeline's own output, so its confidence cannot be used to reject a bad
# prediction. Do not put rtmpose_s behind a closed-loop gate on the strength of
# its score alone.
TORCH_MODEL_CONFIDENCE_THRESHOLDS = {
    "resnet_50": 0.3,
    "cspnext_s": 0.1,
    "cspnext_m": 0.2,
    "rtmpose_s": 0.3,
}

def _available_runtimes() -> typing.Tuple[str, ...]:
    """The engines DeepLabCut could run on here, in preference order."""
    return tuple(name for name, module in ((TENSORFLOW_BACKEND, "tensorflow"),
                                           (TORCH_BACKEND, "torch"))
                 if pose_plugins._installed(module))


def trained_model_name(
    model_path: typing.Optional[str],
    shuffle_index: typing.Optional[int] = None,
) -> typing.Optional[str]:
    """
    Return the backbone name of the project's trained PyTorch model, or None.

    Read off the filesystem rather than through DeepLabCut, for the same reason
    as DlcTorchPoseModel.has_trained_snapshot: this is called while deciding a
    threshold, and importing torch to answer it would be absurd.

    None whenever the answer is not unambiguous - no project, no trained
    shuffle, an unreadable config, or several trained shuffles that disagree
    about the backbone. The caller falls back to the per-backend default, which
    is the conservative direction: an unmeasured model keeps the threshold it
    has today rather than silently inheriting another model's calibration.
    """
    if not model_path:
        return None

    wanted = "*" if shuffle_index is None else str(shuffle_index)
    pattern = os.path.join(
        model_path, "dlc-models-pytorch", "*", f"*shuffle{wanted}", "train",
    )
    found = set()
    for train_folder in sorted(glob.glob(pattern)):
        # A config without a snapshot is not a model that could be loaded.
        if not glob.glob(os.path.join(train_folder, "snapshot-*.pt")):
            continue
        configuration_path = os.path.join(train_folder, "pytorch_config.yaml")
        try:
            with open(configuration_path) as handle:
                configuration = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as err:
            logger.debug("cannot read %r: %s", configuration_path, err)
            continue
        name = (configuration.get("model", {})
                .get("backbone", {})
                .get("model_name"))
        if name:
            found.add(str(name))

    if len(found) == 1:
        return found.pop()
    if found:
        logger.warning(
            "%r has trained shuffles with differing backbones (%s); using the "
            "per-backend confidence default",
            model_path, ", ".join(sorted(found)),
        )
    return None


class DeepLabCutPlugin(pose_plugins.PoseBackendPlugin):
    """A DeepLabCut project, on whichever of its two engines is selected."""

    name = "deeplabcut"
    distribution = "reachaq-pose-deeplabcut"
    requires = ("deeplabcut",)

    def claims(self, model_path: str) -> bool:
        # Only a directory with the project configuration. DeepLabCut used to
        # be the fallback for every directory without a sidecar; that narrowing
        # is deliberate, so an unrecognised directory says so.
        if not model_path:
            return False
        return os.path.isfile(os.path.join(model_path, "config.yaml"))

    def runtime(self, model_path: typing.Optional[str],
                environ: typing.Mapping[str, str]) -> str:
        """
        The engine REACHAQ_POSE_BACKEND names, else the one installed.

        An unusable value falls back with a warning, because refusing to start
        live inference over a typo in an environment variable is worse than
        running the engine the rig was already running.

        Unset, this used to answer TensorFlow unconditionally, which is right on
        a rig that has TensorFlow and a guaranteed failure anywhere else: the
        GPU probe would ask for a TensorFlow runtime that was never installed,
        live inference would be refused, and the operator was told a TensorFlow
        runtime was missing without being told that torch was present and was
        what the model needed. So it answers from what is installed.

        TensorFlow still wins when both are installed. Deployed rigs run trained
        TensorFlow snapshots, and an unset variable must not change what they do.
        """
        explicit = parse_runtime(environ)
        if explicit is not None:
            return explicit

        available = _available_runtimes()
        if not available:
            # Neither engine importable. Answering the historical default keeps
            # the message the operator sees pointing at the engine this rig was
            # set up for, rather than inventing a preference from nothing.
            logger.warning("neither TensorFlow nor torch is installed; reporting "
                           "the %s backend so the failure names it",
                           DEFAULT_POSE_BACKEND)
            return DEFAULT_POSE_BACKEND
        if DEFAULT_POSE_BACKEND in available:
            return DEFAULT_POSE_BACKEND
        backend = available[0]
        logger.notice("%s is not installed; using the %s pose backend. Set %s to "
                      "choose explicitly.",
                      DEFAULT_POSE_BACKEND, backend, POSE_BACKEND_ENV_VAR)
        return backend

    def runtimes(self) -> typing.Tuple[str, ...]:
        return _available_runtimes()

    def pre_validate(self, model_path: str, runtime: str) -> None:
        if runtime == TORCH_BACKEND:
            from .dlc_torch_pose_model import DlcTorchPoseModel
            DlcTorchPoseModel.pre_validate(model_path)
            return
        from .dlc_pose_model import DlcPoseModel
        DlcPoseModel.pre_validate(model_path)

    def build(self, model_path: str, runtime: str, *, shuffle_index: int,
              training_index: int, batch_size: int) -> PoseModel:
        # The imports are deferred to the selected branch: importing the
        # TensorFlow model pulls in TensorFlow, and on a torch-only deployment
        # that import does not resolve at all.
        if runtime == TORCH_BACKEND:
            from .dlc_torch_pose_model import DlcTorchPoseModel
            return DlcTorchPoseModel(model_path, shuffle_index, training_index, batch_size)
        if runtime == TENSORFLOW_BACKEND:
            from .dlc_pose_model import DlcPoseModel
            return DlcPoseModel(model_path, shuffle_index, training_index, batch_size)
        raise ValueError(f"Unsupported pose backend: {runtime!r}; expected one of {POSE_BACKENDS}")

    def confidence_threshold(self, model_path: typing.Optional[str],
                             runtime: str) -> float:
        """Per runtime, then per trained backbone.

        The scale belongs to the trained model rather than the engine:
        cspnext_s never scores above 0.33, so the runtime default of 0.6 would
        reject 98% of its keypoints. See TORCH_MODEL_CONFIDENCE_THRESHOLDS.
        """
        default = DEFAULT_CONFIDENCE_THRESHOLDS.get(
            runtime, DEFAULT_CONFIDENCE_THRESHOLDS[DEFAULT_POSE_BACKEND]
        )
        # Only the PyTorch engine: these were measured under it, and TensorFlow's
        # likelihood is a saturated constant that no per-model figure describes.
        if runtime != TORCH_BACKEND:
            return default
        model_name = trained_model_name(model_path)
        if not model_name:
            return default
        measured = TORCH_MODEL_CONFIDENCE_THRESHOLDS.get(model_name)
        if measured is None:
            logger.info("no measured confidence threshold for %r; using the %s "
                        "default of %.2f", model_name, runtime, default)
            return default
        logger.info("%r uses its measured confidence threshold of %.2f "
                    "rather than the %s default of %.2f",
                    model_name, measured, runtime, default)
        return measured


DLC_PLUGIN = DeepLabCutPlugin()
