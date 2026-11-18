"""The camera contract: the CameraBase surface the capture path relies on.

A backend is created by VideoManager.create_camera from a URL whose scheme
names it (spinnaker, random, playback, sim). The URL's query values reach
set_property before init(); then the capture process calls prepare_capture
once, capture() repeatedly on its capture thread, and end_capture once.

Timing semantics the app relies on:
- capture() returns (frame, stamp): frame is a uint8 (height, width) array the
  caller may keep, and stamp is in ns on the camera's own clock. For a
  trigger-synchronised camera that clock is the one latch_clock() reads.
- frame_id names the exposure. When is_trigger_synchronized is True every
  camera in the group gives the same exposure the same id, and the app pairs
  their frames by it; a gap in ids is a dropped frame.
- frame_perf_c is the frame's time on the host perf_counter clock;
  frame_poll_perf_c and frame_arrival_perf_c bracket the poll that returned it.
- latch_clock() returns (perf before, camera ns, perf after, RAW before, RAW
  after), or None for a camera without a latchable clock. It never raises and
  is called on the capture thread only.
- set_property returns False for a name the backend does not know.
"""

from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy


@runtime_checkable
class CameraBackend(Protocol):

    @property
    def name(self) -> str: ...

    @property
    def fps(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def is_primary(self) -> bool: ...

    @property
    def is_trigger_synchronized(self) -> bool: ...

    @property
    def frame_id(self) -> int: ...

    @property
    def frame_perf_c(self) -> float: ...

    @property
    def frame_poll_perf_c(self) -> float: ...

    @property
    def frame_arrival_perf_c(self) -> float: ...

    def init(self) -> None: ...

    def prepare_capture(self) -> None: ...

    def capture(self) -> Tuple[Optional[numpy.ndarray], int]: ...

    def end_capture(self) -> None: ...

    def latch_clock(self) -> Optional[Tuple[float, int, float, float, float]]: ...

    def set_property(self, name: str, value: str) -> bool: ...
