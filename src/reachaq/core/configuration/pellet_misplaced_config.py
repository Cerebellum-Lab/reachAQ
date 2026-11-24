import dataclasses

from reachaq.core.configuration.detector import DetectorConfig


@dataclasses.dataclass
class PelletMisplacedDetectorConfiguration(DetectorConfig):

    enabled: bool = True
    aggregate_duration: float = 1  # how long ago to check/look at results
    # previous results older than that are discarded before each check/update.

    use_dcs_y_low_limit: bool = True
    dcs_y_low_limit: float = 0  # lower than this -> error condition
