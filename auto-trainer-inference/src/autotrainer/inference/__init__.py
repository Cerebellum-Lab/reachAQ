# NB: this is to be back-compatible with h5 files possibly saved using pose_elements when they were defined
# in this sub-package (as autotrainer.inference.pose_elements)
from enum import Enum

from autotrainer.core import pose_elements
import sys
sys.modules[f"{__name__}.pose_elements"] = pose_elements
# this is/should be temporary. TODO: remove some when later


# must be before below other sub-imports
class InferenceStatus(str, Enum):
    stopped = "Stopped"
    loading = "Loading"
    waiting = "Waiting"
    live = "Live"
    intersession = "Intersession"
    stopping = "Stopping"


class InferenceMonitorDataMsg(str, Enum):

    SET_PROJECT_INFO = "set_project_info"
    SET_POSE_ALGO = "set_pose_algo"
    POSE_RESULT_READY = "pose_result_ready"
    LIVE_RECORDING_CLOSED = "live_recording_closed"
    INTERSESSION_SEGMENTATION_FINISHED = "intersession_segmentation_finished"
    SET_FEED_INTERSESSION_RESULT = "set_feed_intersession_result"

#

from .pose_algorithm import PoseAlgorithm, PoseResponse, PoseLocation, PoseTuple
from .pose_model import PoseModel
from .memory import MemoryPoseModel
from .dlc import DlcPoseModel
from .gpu_runtime import GpuRuntimeStatus, detect_gpu_runtime
from .pose_process import PoseProcess, InferenceCommandMessageKind, InferenceStatusMessageKind, InferenceMode
