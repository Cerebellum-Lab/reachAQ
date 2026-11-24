"""Pellet presence from the brightness of a rectangle in one camera's image.

The cue gate needs to know whether the pellet is still on its post at the
Tone 2 deadline. Live pose answers that when inference runs; this answers it
from the pixels alone, so a rig without live pose, or one whose model misses
the pellet, can still gate the cue.

The detector runs in the camera's capture process on every frame, as the
stim-camera detector does, and publishes its latest answer to a small shared
array the application reads. The answer carries the frame's own host
perf_counter capture time, so the cue policy's staleness limit judges it as
it judges pose: a camera that stops delivering frames goes stale, not
"present".

The rectangle, threshold and hysteresis are rig calibration, set in the
camera's ``params`` in the system configuration (``pellet_roi_*``), like the
stim camera's ``stim_roi_*``. Nothing here guesses them.
"""

import dataclasses
import logging
import math
from typing import Mapping, Optional, Tuple

import numpy

logger = logging.getLogger(__name__)

#: Published states. Unknown until the detector has judged a frame.
_UNKNOWN, _ABSENT, _PRESENT = -1.0, 0.0, 1.0

#: Largest ROI, in pixels: its mean is taken on every frame, up to 900 Hz.
MAX_PELLET_ROI_PIXELS = 256 * 256
#: Bounds on the shared lock. A capture child killed while it held it would
#: otherwise block the cue gate, and the trial executor's lock with it, for
#: good: Stop could then not reach the laser.
_PUBLISH_LOCK_TIMEOUT_S = 0.005
_READ_LOCK_TIMEOUT_S = 0.05


@dataclasses.dataclass(frozen=True)
class PelletRoiConfiguration:
    """One rectangle of one camera's image, and how to read it.

    ``present_when`` says which side of ``threshold`` the mean pixel value is
    on when the pellet is there: ``above`` for a pellet brighter than its
    post, ``below`` for one darker. ``hysteresis`` is how far back across the
    threshold the value must go before a present pellet reads absent (or an
    absent one present), so noise at the threshold does not flicker.
    """

    x: int
    y: int
    width: int
    height: int
    threshold: float
    hysteresis: float = 0.0
    present_when: str = "above"

    def __post_init__(self):
        for name in ("x", "y", "width", "height"):
            object.__setattr__(self, name, int(getattr(self, name)))
        if min(self.x, self.y) < 0:
            raise ValueError("Pellet ROI coordinates cannot be negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Pellet ROI width and height must be positive")
        if self.width * self.height > MAX_PELLET_ROI_PIXELS:
            raise ValueError(
                f"Pellet ROI is {self.width}x{self.height}; at most "
                f"{MAX_PELLET_ROI_PIXELS} pixels are read per frame")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "hysteresis", float(self.hysteresis))
        if not math.isfinite(self.threshold) or not math.isfinite(self.hysteresis):
            raise ValueError("Pellet ROI threshold and hysteresis must be finite")
        if self.hysteresis < 0:
            raise ValueError("Pellet ROI hysteresis cannot be negative")
        present_when = str(self.present_when).strip().lower()
        if present_when not in {"above", "below"}:
            raise ValueError("Pellet ROI present_when must be 'above' or 'below'")
        object.__setattr__(self, "present_when", present_when)

    @classmethod
    def from_params(cls, params: Mapping) -> Optional["PelletRoiConfiguration"]:
        """From a camera's configuration params; None when it has no pellet ROI."""
        width = int(params.get("pellet_roi_width", 0) or 0)
        height = int(params.get("pellet_roi_height", 0) or 0)
        if width <= 0 and height <= 0:
            return None
        if "pellet_roi_threshold" not in params:
            raise ValueError("A pellet ROI needs pellet_roi_threshold")
        return cls(
            x=int(params.get("pellet_roi_x", 0) or 0),
            y=int(params.get("pellet_roi_y", 0) or 0),
            width=width,
            height=height,
            threshold=float(params["pellet_roi_threshold"]),
            hysteresis=float(params.get("pellet_roi_hysteresis", 0) or 0),
            present_when=str(params.get("pellet_roi_present_when", "above")),
        )


class PelletRoiState:
    """The detector's latest answer, shared between the capture child and the app.

    Three doubles under one lock: the state, the frame's perf_counter time
    and the ROI value. Made in the parent, before the capture child starts.
    """

    def __init__(self, mp_ctx):
        self._values = mp_ctx.Array("d", [_UNKNOWN, math.nan, math.nan])

    def reset(self) -> None:
        self.publish(None, math.nan, math.nan)

    def publish(self, present: Optional[bool], frame_perf_time: float, value: float) -> None:
        state = _UNKNOWN if present is None else (_PRESENT if present else _ABSENT)
        lock = self._values.get_lock()
        if not lock.acquire(timeout=_PUBLISH_LOCK_TIMEOUT_S):
            return  # this frame's answer is dropped; the next one is published
        try:
            self._values[0] = state
            self._values[1] = float(frame_perf_time)
            self._values[2] = float(value)
        finally:
            lock.release()

    def latest(self) -> Optional[Tuple[bool, float, float]]:
        """(present, frame perf time, value); None while unknown or unreadable."""
        lock = self._values.get_lock()
        if not lock.acquire(timeout=_READ_LOCK_TIMEOUT_S):
            logger.warning("Pellet ROI state is locked; presence is unknown")
            return None
        try:
            state, perf, value = self._values[0], self._values[1], self._values[2]
        finally:
            lock.release()
        if state == _UNKNOWN or not math.isfinite(perf):
            return None
        return state == _PRESENT, perf, value


class PelletRoiDetector:
    """Judge each frame's ROI and publish the answer; never raises into capture."""

    def __init__(self, configuration: PelletRoiConfiguration, state: PelletRoiState):
        self.configuration = configuration
        self._state = state
        self._present: Optional[bool] = None
        self._failed = False

    def value(self, frame) -> float:
        roi = self.configuration
        array = numpy.asarray(frame)
        if array.ndim == 3:
            # The cameras are monochrome; a colour frame's first channel.
            array = array[:, :, 0]
        if roi.y + roi.height > array.shape[0] or roi.x + roi.width > array.shape[1]:
            # Not clipped: a smaller rectangle is not the one calibrated.
            raise ValueError(
                f"The pellet ROI ({roi.x}, {roi.y}, {roi.width}x{roi.height}) does not "
                f"fit the {array.shape[1]}x{array.shape[0]} frame")
        array = array[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
        return float(array.mean())

    def process(self, frame, frame_perf_time: float) -> Optional[bool]:
        try:
            value = self.value(frame)
        except Exception as error:
            if not self._failed:
                self._failed = True
                logger.error("Pellet ROI cannot be read: %s", error)
            self._present = None
            self._state.publish(None, frame_perf_time, math.nan)
            return None
        self._present = self._judge(value)
        self._state.publish(self._present, frame_perf_time, value)
        return self._present

    def _judge(self, value: float) -> bool:
        roi = self.configuration
        if roi.present_when == "above":
            if self._present:
                return value > roi.threshold - roi.hysteresis
            return value > roi.threshold
        if self._present:
            return value < roi.threshold + roi.hysteresis
        return value < roi.threshold
