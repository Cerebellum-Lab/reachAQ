from typing import Optional

from autotrainer.api import ApiDetectorKind

from autotrainer.core.analysis.detector import BaseDetector
from autotrainer.core.analysis.headbar_pressure_monitor import HeadbarPressureMonitor
from autotrainer.core.configuration.autoclamp_evasion_config import AutoClampEvasionDetectorConfig


class AutoClampEvasionDetector(BaseDetector[AutoClampEvasionDetectorConfig]):

    PELLETS_CONSUMED = "pellets_consumed"  # under autoclamp evasion conditions

    config_cls = AutoClampEvasionDetectorConfig
    detector_api_kind = ApiDetectorKind.animalAutoClampEvasion

    def __init__(self, *, headbar_detector: HeadbarPressureMonitor):
        super().__init__()
        self._headbar_detector = headbar_detector
        self._pellets_consumed_count: int = 0
        self._autoclamp_enabled = False
        self._autoclamp_in_progress = False
        self._running = True  # always "active"

    def stop(self):
        pass  # keep always active

    def _check_state(self) -> Optional[float]:
        self.is_engaged = self._pellets_consumed_count >= self._config.pellets_consumed_trigger

    def increment_pellets_consumed(self, inc: int = 1):
        if (
                self._autoclamp_enabled
            and not self._headbar_detector.is_engaged
            and not self._autoclamp_in_progress
        ):
            self.pellets_consumed += inc
            self._logger.info("incremented pellets_consumed, now=%s", self._pellets_consumed_count)

    @property
    def pellets_consumed(self):
        return self._pellets_consumed_count

    @pellets_consumed.setter
    def pellets_consumed(self, value):
        prev, self._pellets_consumed_count = self._pellets_consumed_count, value
        if value != prev:
            self._logger.debug("set pellets_consumed=%s prev=%s", value, prev)
            self.property_changed(self.PELLETS_CONSUMED, value, prev)
            self.check_state()

    @property
    def autoclamp_enabled(self) -> bool:
        return self._autoclamp_enabled

    @autoclamp_enabled.setter
    def autoclamp_enabled(self, value: bool):
        prev, self._autoclamp_enabled = self._autoclamp_enabled, value
        if value != prev:
            self.check_state()

    @property
    def autoclamp_in_progress(self) -> bool:
        return self._autoclamp_in_progress

    @autoclamp_in_progress.setter
    def autoclamp_in_progress(self, value):
        prev, self._autoclamp_in_progress = self._autoclamp_in_progress, value
        if prev != value:
            self.check_state()
