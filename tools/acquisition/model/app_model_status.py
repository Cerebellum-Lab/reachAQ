import enum


class AppModelStatus(str, enum.Enum):
    IDLE = "idle"  # nothing running
    ACQUIRING = "acquiring"  # camera + system running, but without animal-in-device
    ANIMAL_IN_DEVICE = "animal_in_device"  # this is ACQUIRING with animal-in-device
    ANIMAL_IN_TRAINING = "animal_in_training"  # this is ANIMAL_IN_DEVICE with training behavior algo **enabled**
    CALIBRATION_3D = "calibration_3d"  # executing calib 3d
    CALIBRATION_DCS = "calibration_dcs"  # executing calib dcs


class SessionRecordingStatus(str, enum.Enum):
    READY = "ready"
    ARMING = "arming"
    RECORDING = "recording"
    STOPPING = "stopping"
    ANALYZING = "analyzing"
    ABORTING = "aborting"
