import dataclasses

from autotrainer.core.configuration.alarm_detector import AlarmDetectorConfig


@dataclasses.dataclass
class AnimalThrashAlarmConfig(AlarmDetectorConfig):

    use: bool = False
    is_emergency_condition: bool = True  # nb: is_emergency with use=False has no effect ...
    allow_autoresume_on_cleared: bool = False

    aggregate_delay: float = 5  # up to how long ago to look at previous results
    audio_thrash_count: int = 3  # spectrum thrash count greater than this (during last aggregate_delay)
    audio_thrash_percent_on: float = 50
