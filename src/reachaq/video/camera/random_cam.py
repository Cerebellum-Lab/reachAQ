import time
from typing import Tuple

import numpy

from .camera_base import CameraBase


class RandomCam(CameraBase):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self._rng = numpy.random.default_rng()

    def prepare_capture(self) -> None:
        super().prepare_capture()

    def capture(self) -> Tuple[numpy.ndarray, int]:
        now = time.time_ns()

        delta = self._frame_count / self._fps - 1e-9 * (now - self._capture_start)

        if delta > 0:
            time.sleep(delta)

        super().capture()

        return self._rng.integers(low=0, high=255, size=(self._height, self._width), dtype="uint8"), self._last_when
