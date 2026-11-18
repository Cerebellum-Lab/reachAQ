"""YOLO-pose as a pose backend plugin.

Still shipped inside core; it moves to its own distribution with the rest of
this package. YOLO-pose runs on the AGPL ultralytics package, which is not a
dependency of core, so this plugin is listed on every rig but is only usable
where ultralytics is installed. Nothing here imports ultralytics or torch at
module scope.
"""

import os
import typing

from autotrainer.inference import pose_plugins
from autotrainer.inference.backend_selection import TORCH_BACKEND
from autotrainer.inference.pose_model import PoseModel

# YOLO-pose reports keypoint confidence on its own scale again, so neither
# the DeepLabCut backbones above nor the torch backend default describes it.
# Measured the same way: every labelled keypoint of the 33 held-out frames of
# the merged project (DeepLabCut shuffle 30), scored against its label.
#
#   gate   coverage   median err   Pellet cov   Tongue_tip cov
#   0.10      99.1%      2.04 px        100%           100%
#   0.30      92.9%      1.98 px        100%            20%
#   0.60      84.0%      1.81 px         88%             0%
#   0.90      55.7%      1.69 px          8%             0%
#
# Raising the gate buys 0.35 px and costs 43 points of coverage. At the 0.9
# this model was gated at in production - the TensorFlow default, reached
# because the backend was chosen without looking at the model - Pellet fell to
# 8% and never appeared on the overlay at all. 0.05 measures identically to
# 0.10, so 0.10 is the same result with more margin.
#
# Thinly labelled parts: RH_spread, LH_spread and LH_grab have one labelled
# frame each and tongue 3-5, so those columns are indicative only. LH_spread
# and LH_grab score 0% at every gate including 0.05 - not learned, not gated.
YOLO_CONFIDENCE_THRESHOLD = 0.1


class YoloPlugin(pose_plugins.PoseBackendPlugin):
    """A YOLO-pose model directory, recognised by its sidecar."""

    name = "yolo"
    distribution = "auto-trainer-pose-yolo"
    requires = ("ultralytics", "torch")

    SIDECAR = "yolo_pose.yaml"

    def claims(self, model_path: str) -> bool:
        # The sidecar has to exist for the model to load at all - it carries
        # the keypoint names, which a .pt file does not - so its presence is an
        # unambiguous signal.
        if not model_path:
            return False
        return os.path.isfile(os.path.join(model_path, self.SIDECAR))

    def runtime(self, model_path: typing.Optional[str],
                environ: typing.Mapping[str, str]) -> str:
        # YOLO weights are torch and cannot be anything else.
        return TORCH_BACKEND

    def pre_validate(self, model_path: str, runtime: str) -> None:
        from .yolo_pose_model import YoloPoseModel
        YoloPoseModel.pre_validate(model_path)

    def build(self, model_path: str, runtime: str, *, shuffle_index: int,
              training_index: int, batch_size: int) -> PoseModel:
        from .yolo_pose_model import YoloPoseModel
        return YoloPoseModel(model_path, batch_size=batch_size)

    def confidence_threshold(self, model_path: typing.Optional[str],
                             runtime: str) -> float:
        return YOLO_CONFIDENCE_THRESHOLD


YOLO_PLUGIN = YoloPlugin()
