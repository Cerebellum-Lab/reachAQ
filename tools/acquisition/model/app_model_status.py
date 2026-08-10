import enum


class AppModelStatus(str, enum.Enum):
    IDLE = "idle"  # nothing running
    RUNNING = "running"  # enabled hardware is acquiring/previewing
    CALIBRATION_3D = "calibration_3d"  # executing calib 3d
    CALIBRATION_DCS = "calibration_dcs"  # executing calib dcs


class SessionRecordingStatus(str, enum.Enum):
    READY = "ready"
    ARMING = "arming"
    RECORDING = "recording"
    STOPPING = "stopping"
    ANALYZING = "analyzing"
    ABORTING = "aborting"
