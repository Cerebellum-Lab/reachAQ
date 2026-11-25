import enum


class FrameIndexCategory(enum.IntEnum):

    SWITCH_TO_ONLINE = -5  # not anymore used, replaced by InferenceCommandMessageKind.SetOfflineToLive
    PADDING = -4
    EOF_OFFLINE_PROCESSING = -3
    EOF_RECORDING = -2

    ONLINE_NO_RECORDING = -1  # NB: DO NOT CHANGE VALUE OF THIS ONE

    RECORDING_OR_OFFLINE_PROCESSING = 1
    # NB: positive or zero frane index means recording online or offline reprocessing.

    @classmethod
    def is_signaling_index(cls, frame_idx) -> bool:
        return frame_idx < cls.ONLINE_NO_RECORDING

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = int(value)
        if isinstance(value, float):
            v = int(value)
            if v != value:
                return super()._missing_(value)
            value = v
        if value >= 0:
            return FrameIndexCategory.RECORDING_OR_OFFLINE_PROCESSING
        for k, m in cls._member_map_.items():
            if m.value == value:
                return m
        return super()._missing_(value)
