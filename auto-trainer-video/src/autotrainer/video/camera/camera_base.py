import dataclasses
import logging
import math
import time
from typing import Tuple, Optional, ClassVar, Dict, Any

import numpy

from autotrainer.core import get_perf_now

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CameraBaseDefaultParams:

    width: int = 300
    height: int = 200
    fps: int = 30


class CameraBase:

    default_params: ClassVar[Dict[str, Any]] = dataclasses.asdict(CameraBaseDefaultParams())
    # possible default "params" for camera class,
    # must use same keys than in config file.
    # Is used to pre-set/applied on the camera instance, once it's created,
    # but before any eventual custom params from config file, that can so override them.

    def __init__(self, name: str = "camera"):
        self._name = name
        self._width = 0
        self._height = 0
        self._fps = 0
        self._is_primary = False
        self._frame_count = 0
        self._capture_start = 0
        self._last_when = 0
        self._last_frame_id = -1
        self._last_frame_perf_c = -math.inf
        self._last_frame_time = -math.inf
        self._last_poll_perf_c = -math.inf
        self._last_arrival_perf_c = -math.inf

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: int) -> None:
        self._name = value

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        if __debug__:
            if not isinstance(value, int):
                logger.warning("Received non-int type value for width: %s: %r", type(value), value)
        self._width = value
        logger.debug(f"<{self._name}> width: {self._width}")

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        if __debug__:
            if not isinstance(value, int):
                logger.warning("Received non-int type value for height: %s: %r", type(value), value)
        self._height = value
        logger.debug(f"<{self._name}> height: {self._height}")

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, value: int) -> None:
        self._fps = value
        logger.debug(f"<{self._name}> fps: {self._fps}")

    @property
    def is_primary(self) -> bool:
        return self._is_primary

    @property
    def frame_id(self) -> int:
        """Returns the last frame id (frame counter)"""
        return self._last_frame_id

    @property
    def is_trigger_synchronized(self) -> bool:
        """Whether this camera exposes on a hardware trigger shared with the
        other reach cameras, so that its frame_id names the same exposure as
        theirs.

        A free-running source counts only its own frames: two of them started a
        few frames apart never share ids, and pairing their frames by id would
        hold every batch until the later camera caught up.
        """
        return False
    
    @property
    def frame_perf_c(self) -> float:
        """Returns the last frame system perf_counter, as precise as possible"""
        return self._last_frame_perf_c

    @property
    def frame_poll_perf_c(self) -> float:
        """perf_counter just before the poll that returned the last frame."""
        return self._last_poll_perf_c

    @property
    def frame_arrival_perf_c(self) -> float:
        """perf_counter when the last frame was handed to the host.

        Not the same as frame_perf_c, which is an estimate fitted to these
        host poll times; the latency record needs the measured times.
        """
        return self._last_arrival_perf_c

    @property
    def frame_unix_time(self) -> float:
        return self._last_frame_time

    def init(self) -> None:
        """ Actions that should be taken once after camera creation, but should not happen in the constructor.

        This method will be called after any property changes from the camera url params have been applied.
        """

    def prepare_capture(self) -> None:
        """ Should be called once before capturing frames.

        Subclasses must call this method when overriding to reset frame count.
        """
        self._last_when = 0
        self._frame_count = 0

    def end_capture(self) -> None:
        """ Should be called once after capturing frames."""

        logger.debug(f"<{self._name}> approximate fps: ~{self._calculate_fps():.1f}")

    def capture(self) -> Tuple[Optional[numpy.ndarray], int]:
        """ Called repeatedly to capture the next frame

        Subclasses must call this method when overriding or take responsibility for increasing frame count and tracking
        performance times (capture start and last frame time).

        :returns: tuple containing a numpy array with shape (height, width) and an associated timestamp in nanoseconds
        :rtype: (numpy.ndarray, int)
        """

        if self._frame_count == 0:
            self._last_when = self._capture_start = time.time_ns()

        self._last_when = time.time_ns()
        self._last_frame_perf_c = get_perf_now()
        self._last_frame_time = time.time()
        self._last_poll_perf_c = self._last_arrival_perf_c = self._last_frame_perf_c

        self._frame_count += 1
        self._last_frame_id += 1

        return None, self._last_when

    def latch_clock(self) -> Optional[Tuple[float, int, float]]:
        """ Latches the camera's timestamp clock between two host perf_counter reads.

        Returns (perf before, camera timestamp in ns, perf after), on the clock
        capture() timestamps come from, or None for a camera with no such clock.
        Must not raise. Called on the capture thread only.
        """
        return None

    def set_property(self, name: str, value: str) -> bool:
        """ Sets known property values, typically from the camera url.

        Subclasses should override to handle custom properties for specific camera types and fallback to calling this
        method for standard properties.

        """

        name = name.lower()

        if name == "width":
            self.width = int(value)
        elif name == "height":
            self.height = int(value)
        elif name == "fps":
            self.fps = int(value)
        elif name == "name":
            self.name = value
        elif name == "primary":
            self._is_primary = value.lower() in {"true", "yes", "on", "1"}
        else:
            logger.warning(f"<{self._name}> unknown property {name}")
            return False

        return True

    def _calculate_fps(self) -> float:
        if self._frame_count > 1:
            return self._frame_count * 1e9 / (self._last_when - self._capture_start)

        return 0
