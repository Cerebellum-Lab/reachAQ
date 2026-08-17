"""AutoTrainer Video"""

from .video_capture import VideoCapture, CaptureCommandKind
from .video_capture import CaptureAttrs, CaptureCameraAttrs, CaptureInferenceAttrs
from .video_manager import VideoManager
from .video_reader import VideoReader
from .video_record import VideoRecord, VideoRecordProperties, VideoRecordMode
from .stim_camera import (
    StimCameraDetectionConfiguration,
    StimCameraDetector,
    StimDetectorArm,
    StimEvidenceWriter,
    StimRoiDefinition,
    StimTriggerDecision,
)
