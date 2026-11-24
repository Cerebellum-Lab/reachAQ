from enum import Enum


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
from .gpu_runtime import GpuRuntimeStatus, detect_gpu_runtime
from .pose_process import PoseProcess, InferenceCommandMessageKind, InferenceStatusMessageKind, InferenceMode
