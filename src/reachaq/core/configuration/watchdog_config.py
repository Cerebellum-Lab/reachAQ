
import dataclasses

from reachaq.core.configuration.detector import DetectorConfig


@dataclasses.dataclass
class WatchdogConfig(DetectorConfig):

    timeout_trigger_delay: float = 5  # seconds
    # if watchdog perf counter older than this then trigger watchdog
