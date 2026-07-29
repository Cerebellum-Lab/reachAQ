from math import floor
from datetime import datetime
from typing import List

import numpy

from .. import get_verbose_logger
from .detector import BaseDetector
from autotrainer.api import ApiEventKind
from ..configuration.headbar_pressure_config import HeadbarPressureConfiguration

logger = get_verbose_logger(__name__)


class HeadbarPressureMonitor(BaseDetector[HeadbarPressureConfiguration]):
    """
    Monitor the headbar pressure data stream and perform any required analysis.  The current implementation is used to
    determine if the headbar is considered sufficiently "engaged" or not.  At this time, this is specifically used
    downstream to allow for the tunnel auto-clamp behavior, when enabled.
    """

    config_cls = HeadbarPressureConfiguration

    def __init__(self):
        super().__init__()

        self._sample_rate = 100

        self.engaged_threshold: float = 30
        self.duration: float = 0.25

        self._values = numpy.empty((1, 0))

        self._buffer_length: float = 1.0
        self._retain_count: int = 1

        self._window_count: int = 3

        self._first_third = 1
        self._last_third = 1

        self._last_when: float = 0
        self._last_index: int = 0

        self._rebuild_buffers()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        # Odd number sample rates lead to mismatched indexing.  Can just handle here once by effectively doing the
        # round.
        self._sample_rate = value + (value % 2)
        self._rebuild_buffers()

    @property
    def engaged_threshold(self) -> float:
        return self._config.threshold

    @engaged_threshold.setter
    def engaged_threshold(self, value: float) -> None:
        cfg = self._config
        prev, cfg.threshold = cfg.threshold, value
        # self._on_property_changed("threshold", value, prev)  # unused

    @property
    def duration(self) -> float:
        return self._config.duration

    @duration.setter
    def duration(self, value: float) -> None:
        cfg = self.config
        prev, cfg.duration = cfg.duration, value
        # self._on_property_changed("duration", value, prev)  unused
        self._rebuild_buffers()

    def _custom_set_is_engaged(self):
        self._event_manager.post_event_content(
            ApiEventKind.headbarPressureEngagedChanged,
            data=dict(is_engaged=self._is_engaged),
            when=datetime.fromtimestamp(self._last_when), index=self._last_index)

    def update(self, values: List[float], when: float = 0.0, index: int = 0) -> bool:
        self._values = numpy.append(self._values, values)
        self._values = self._values[-self._retain_count:]

        cfg = self._config

        if len(self._values) < self._retain_count:
            return False

        is_engaged = False

        # Values are appended and dropped in batches.  Need to evaluate over the full window size as more than just one
        # sample will be gone the next evaluation.  We could just evaluate oldest last window_count samples, but there
        # would be greater latency in the response.  We are also evaluating the same window of samples multiple times
        # depending on the measurement batch size, but this is simple calculation and not worth optimizing out at the
        # moment.
        for idx in range(self._window_count * 3):
            new_start = idx + self._last_third
            new_end = idx + self._window_count
            old_start = idx
            old_end = idx + self._first_third

            if numpy.all(self._values[old_start:old_end] <= (
                    self._values[new_start:new_end] - cfg.threshold)):
                is_engaged = True
                break

        self._last_when = when
        self._last_index = index
        self.is_engaged = is_engaged
        return self._is_engaged

    def _rebuild_buffers(self):
        # Measurements are received in batches.  Depending on that batch size, it may not result in a continuous, moving
        # evaluation with data processed in batches.  Store a larger buffer than the window so we can apply the window
        # over each starting and ending element.
        cfg = self._config
        self._buffer_length = cfg.duration * 4
        self._retain_count = round(self._sample_rate * self._buffer_length)

        self._window_count = round(self._sample_rate * cfg.duration)

        self._first_third = floor(self._window_count / 3)
        self._last_third = self._window_count - self._first_third
