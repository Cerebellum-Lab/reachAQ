"""CLOCK_MONOTONIC_RAW, read beside a perf_counter stamp.

time.perf_counter() is CLOCK_MONOTONIC, which systemd-timesyncd slews at every
poll; the RAW clock is the same oscillator without the corrections. A RAW
value taken right beside a perf stamp lets the frame alignment and the latency
record relate NI time to a clock that only drifts, and bridge back to
perf_counter through the recorded pairs. It costs 40 ns a read on
christielab10. Where the platform has no such clock (Windows) it is NaN, so
every stamp keeps its shape.
"""

from __future__ import annotations

import math
import time

_RAW_CLOCK_ID = getattr(time, "CLOCK_MONOTONIC_RAW", None)
RAW_CLOCK_AVAILABLE = _RAW_CLOCK_ID is not None


def monotonic_raw_time() -> float:
    """Seconds on CLOCK_MONOTONIC_RAW, or NaN where there is none."""
    if _RAW_CLOCK_ID is None:
        return math.nan
    return time.clock_gettime(_RAW_CLOCK_ID)
