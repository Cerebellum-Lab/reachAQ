"""Event and state enumerations published by reachAQ.

Values that appear in recorded event files keep the id they were recorded with:
hourly ``*_events.csv`` files persist ``int(kind)`` and ``str(kind)``, payloads
persist ``repr()`` of any enum inside them, and session ``streams/events.csv``
persists the member name and id. Members marked "assigned" were never recorded
on a reachAQ rig, so their ids are reachAQ's own choice.
"""

import enum


class _ApiIntEnum(enum.IntEnum):
    # Pinned to the recorded forms so output is the same on every Python
    # version (3.11 changed str() of an IntEnum member to its bare value):
    # str() is "Class.member", repr() "<Class.member: id>", and f-string /
    # format() the bare id, as the rig's logs show ("..., 701, ApiEventKind.projectChanged").
    def __str__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__}.{self.name}: {int(self)}>"

    def __format__(self, format_spec: str) -> str:
        return format(int(self), format_spec)


class ApiEventKind(_ApiIntEnum):
    applicationLaunched = 201
    applicationTerminating = 202

    applicationModeChanged = 301

    systemStatus = 601

    projectChanged = 701
    projectSessionChanged = 702  # retired, last recorded 2026-07-29

    calibration3dCompleted = 801  # assigned
    calibration3dFailed = 802  # assigned
    calibrationDcsCompleted = 811  # assigned
    calibrationDcsFailed = 812  # assigned

    pelletLoadBegin = 1202
    pelletLoadEnd = 1203
    pelletSendBegin = 1205
    pelletSendEnd = 1206
    pelletCoverBegin = 1208
    pelletCoverEnd = 1209
    pelletReleaseBegin = 1211
    pelletReleaseEnd = 1212
    pelletHomeBegin = 1214
    pelletHomeEnd = 1215
    pelletRetractBegin = 1220
    pelletRetractEnd = 1221
    pelletHomeReset = 1251  # assigned
    pelletDriftReset = 1252  # assigned
    pelletExternalToken = 1253  # assigned

    trialStarted = 1301
    trialCaptureEnded = 1305
    trialEnded = 1311
    sessionStarted = 1331
    sessionEnded = 1332

    algorithmPause = 1401  # assigned
    algorithmResume = 1402  # assigned

    intertrialDetectionBegin = 1501  # assigned
    intertrialDetectionEnd = 1502  # assigned
    intertrialDetectionError = 1503  # assigned
    intertrialDetectionSave = 1504  # assigned
    intertrialDetectionSaveError = 1505  # assigned
    intertrialSegmentationBegin = 1511  # assigned
    intertrialSegmentationEnd = 1512  # assigned
    intertrialSegmentationError = 1513  # assigned
    intertrialSegmentationSave = 1514  # assigned
    intertrialSegmentationSaveError = 1515  # assigned
    intertrialPelletShift = 1521  # assigned

    pelletPresentedCountChanged = 1800
    pelletConsumedCountChanged = 1802
    reachCountChanged = 1803
    successfulReachesCountChanged = 1804

    deviceCommandSend = 2001
    deviceCommandAcknowledge = 2002

    trainingModeChanged = 5001
    trainingPlanLoad = 5101
    trainingPhaseEnter = 5151  # assigned
    trainingPhaseExit = 5152  # assigned
    trainingProgressUpdate = 5153  # assigned

    detectorChanged = 6001
    alarmChanged = 6501  # retired, last recorded 2026-08-10

    animalCreated = 7001
    animalSelected = 7003
    animalUpdated = 7051  # assigned


class ApiApplicationMode(_ApiIntEnum):
    IDLE = 0
    RUNNING = 100
    CALIBRATION_3D = 501  # assigned
    CALIBRATION_DCS = 502  # assigned


class ApiTrainingMode(_ApiIntEnum):
    UNDEFINED = 0  # assigned
    MANUAL = 100
    MANUAL_WITH_PROTOCOL = 150  # assigned
    AUTOMATIC = 200  # assigned


class ApiDetectorKind(_ApiIntEnum):
    pelletMisplaced = 301
    deviceAckTimeOut = 401
    pelletStatusMessageInterruption = 411


class ApiPelletShiftSource(_ApiIntEnum):
    TONGUE_EATEN = 1  # assigned
    REACH_FAILURES = 2  # assigned
