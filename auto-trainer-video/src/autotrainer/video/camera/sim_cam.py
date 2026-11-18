"""Hardware-free camera on the simulated exposure grid (URL scheme ``sim``).

Exposure k starts at g_k = sim_epoch() + k / fps, the grid the simulated NI
stream draws cam_frames on, so its frames pair with those edges the way a
triggered camera's pair with the strobe. The camera clock is perf_counter in
ns, and every process reads the same epoch, so all simulated cameras give the
same exposure the same frame id without talking to each other.
"""

import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy

from autotrainer.core.latency.raw_clock import monotonic_raw_time
from autotrainer.core.simulation import sim_epoch

from .camera_base import CameraBase

# The same variables and defaults as autotrainer.device.nidaq_simulated, which
# this package cannot import: the camera and the NI stream must share a grid.
ENV_SIM_CAMERA_FPS = "REACHAQ_SIM_CAMERA_FPS"
ENV_SIM_CAMERA_EXPOSURE_S = "REACHAQ_SIM_CAMERA_EXPOSURE_S"
DEFAULT_CAMERA_FPS = 150.0
DEFAULT_CAMERA_EXPOSURE_S = 0.002

#: Readout and transfer from the end of exposure to the host.
TRANSFER_S = 0.001


class _SimDefaultParams:
    """default_params read on each access, so the environment a process has when
    it parses the URL is the one that counts, not the one it had at import."""

    def __get__(self, obj, owner) -> Dict[str, Any]:
        return {
            "width": 640,
            "height": 480,
            "fps": int(round(float(os.environ.get(ENV_SIM_CAMERA_FPS, DEFAULT_CAMERA_FPS)))),
            # In microseconds, as Spinnaker's ExposureTime and the app read it.
            "exposure": float(os.environ.get(ENV_SIM_CAMERA_EXPOSURE_S, DEFAULT_CAMERA_EXPOSURE_S)) * 1e6,
        }


class SimulatedCam(CameraBase):

    default_params = _SimDefaultParams()

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._exposure = DEFAULT_CAMERA_EXPOSURE_S
        self._next_k: Optional[int] = None
        self._background: Optional[numpy.ndarray] = None
        self._disc: Optional[numpy.ndarray] = None

    @property
    def exposure(self) -> float:
        """Exposure in seconds."""
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = float(value)

    @property
    def is_trigger_synchronized(self) -> bool:
        # Every simulated camera exposes on the shared grid, as triggered
        # secondaries expose on the primary's strobe.
        return True

    def init(self) -> None:
        width, height = self._width, self._height
        ramp = numpy.linspace(16, 200, width, dtype=numpy.float32)
        shade = numpy.linspace(0.6, 1.0, height, dtype=numpy.float32)[:, None]
        self._background = (ramp[None, :] * shade).astype(numpy.uint8)
        radius = max(1, min(width, height) // 10)
        yy, xx = numpy.mgrid[-radius:radius + 1, -radius:radius + 1]
        self._disc = (xx * xx + yy * yy) <= radius * radius

    def prepare_capture(self) -> None:
        super().prepare_capture()
        self._next_k = None
        if self._background is None:
            self.init()

    def capture(self) -> Tuple[numpy.ndarray, int]:
        # Does CameraBase.capture's bookkeeping itself: the times here are the
        # grid's, not the moment this call happened to return.
        period = 1.0 / self._fps
        epoch = sim_epoch()
        lag = self._exposure + TRANSFER_S
        poll = time.perf_counter()
        # Index of the newest exposure that has already reached the host.
        arrived_k = math.floor((poll - epoch - lag) / period)
        if self._next_k is None:
            # Exposure 0 is the first: ids stay non-negative before the epoch.
            k = max(arrived_k + 1, 0)
        elif arrived_k - self._next_k >= 1:
            # More than a period late: jump to the grid, leaving the id gap a
            # real dropped frame leaves.
            k = arrived_k
        else:
            k = self._next_k
        self._next_k = k + 1

        start = epoch + k * period
        arrival = start + lag
        delay = arrival - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

        stamp = round((start + self._exposure) * 1e9)
        if self._frame_count == 0:
            self._capture_start = stamp
        self._last_when = stamp
        self._last_frame_id = k
        self._last_frame_perf_c = arrival
        self._last_poll_perf_c = min(poll, arrival)
        self._last_arrival_perf_c = arrival
        self._last_frame_time = time.time() - (time.perf_counter() - arrival)
        self._frame_count += 1

        return self._draw(k), stamp

    def latch_clock(self) -> Optional[Tuple[float, int, float, float, float]]:
        # The camera clock is perf_counter itself, so the latch is a perf read
        # taken inside the bracket.
        perf_before = time.perf_counter()
        raw_before = monotonic_raw_time()
        perf_mid = time.perf_counter()
        raw_after = monotonic_raw_time()
        perf_after = time.perf_counter()
        return perf_before, int(perf_mid * 1e9), perf_after, raw_before, raw_after

    def set_property(self, name: str, value: str) -> bool:
        if name.lower() == "exposure":
            # Microseconds in the URL, as for Spinnaker.
            self.exposure = float(value) * 1e-6
            return True
        return super().set_property(name, value)

    def _draw(self, k: int) -> numpy.ndarray:
        """Gradient with a disc circling once every two seconds, a function of k
        alone so cameras showing the same exposure show the same image."""
        frame = self._background.copy()
        height, width = frame.shape
        radius = self._disc.shape[0] // 2
        angle = 2.0 * math.pi * k / (2.0 * self._fps)
        cx = int(round(width / 2 + 0.35 * width * math.cos(angle)))
        cy = int(round(height / 2 + 0.35 * height * math.sin(angle)))
        cx = min(max(cx, radius), width - 1 - radius)
        cy = min(max(cy, radius), height - 1 - radius)
        if cx >= radius and cy >= radius:
            window = frame[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
            window[self._disc] = 255
        return frame
