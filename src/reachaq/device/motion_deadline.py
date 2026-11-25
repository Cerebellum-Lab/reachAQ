"""When a pellet stepper move must have been acknowledged: its own motion profile, plus a margin.

Pellet firmware 2.3.1 acknowledges a move (STEPPER_MOVE, STEPPER_HOME,
FIXED_XYZ) when the axis has finished it, and has no motion timeout of its own.
A move that is still unacknowledged well after its profile says it should be
done has stalled or run away, and the host stops the stage (ESTOP). The ack of
a move of known length is due

    deadline = max(DEADLINE_FLOOR_S, profile_time * DEADLINE_FACTOR + DEADLINE_MARGIN_S)

profile_time is the trapezoid (or, for a move too short to reach max_velocity,
the triangle) of the axis's max_velocity and max_acceleration (mm/s, mm/s^2).

UNFINALIZED constants, from christielab10 on 2026-11-24 (app logs 000-006,
683 MOVEs and 60 HOMEs acknowledged, max_vel 120 mm/s, max_acc 600 mm/s^2,
home_vel 30 mm/s; send to "Received ack" in the app log):
- ack delay minus profile time: median 0.14 s, p99 0.21 s, max 0.24 s (MOVE);
  max 0.10 s (HOME). The excess is the board's 100 ms status tick, on which it
  acks, plus CAN and host latency;
- ack delay over profile time, profile above 0.3 s: max 1.68, median 1.55;
- the longest ack was a 35 mm home, 1.24 s (profile at 30 mm/s: 1.17 s).
DEADLINE_FACTOR 1.25 with DEADLINE_MARGIN_S 0.4 leaves at least 0.15 s over
every measured ack; DEADLINE_FLOOR_S 0.6 covers a zero-length move (acked in
0.001-0.24 s). The 15:55 runaway, MOVE_Z 1.0 -> 15.5 mm (profile 0.31 s), is
then stopped 0.79 s after it was sent, not 3 s.
"""

from __future__ import annotations

import math
from typing import Optional

DEADLINE_FACTOR = 1.25
DEADLINE_MARGIN_S = 0.4
DEADLINE_FLOOR_S = 0.6


def profile_time(distance: float, max_velocity: float, max_acceleration: float) -> Optional[float]:
    """Seconds a move of `distance` mm takes at these limits, or None when they are unusable."""
    try:
        d, v, a = abs(float(distance)), float(max_velocity), float(max_acceleration)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (d, v, a)) or v <= 0 or a <= 0:
        return None
    if d >= v * v / a:  # reaches max_velocity: ramp up, cruise, ramp down
        return d / v + v / a
    return 2.0 * math.sqrt(d / a)


def ack_deadline(profile_s: Optional[float]) -> Optional[float]:
    """Seconds after the send by which the move's ack is due; None when the profile is unknown."""
    if profile_s is None or not math.isfinite(profile_s):
        return None
    return max(DEADLINE_FLOOR_S, profile_s * DEADLINE_FACTOR + DEADLINE_MARGIN_S)


__all__ = ["DEADLINE_FACTOR", "DEADLINE_FLOOR_S", "DEADLINE_MARGIN_S", "ack_deadline", "profile_time"]
