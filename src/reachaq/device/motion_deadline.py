"""When a pellet stepper move must have been acknowledged: the firmware's own timing, plus a margin.

Pellet firmware 2.3.1 acknowledges a move (STEPPER_MOVE, STEPPER_HOME,
FIXED_XYZ) on its 100 ms status tick after the axis has finished, and has no
motion timeout of its own. A move still unacknowledged well after the firmware
can have finished it has stalled or run away, and the host stops the stage
(ESTOP). The firmware's timing (reachAQ-hardware v2.3.1):

- a move ramps with a raised-cosine acceleration, a/2 * (1 - cos wt)
  (motor_math.c:58-79), so each ramp takes 2v/a: t = d/v + 2v/a when
  d >= 2v^2/a, else t = 2 * sqrt(2d/a) (profile_time);
- a home runs at homing_velocity with no ramp until the switch closes, from
  wherever the carriage really is (motor_motion_workq.c:835-880), so it is given
  the full travel, 36.55 mm (home_deadline);
- SEND_TO_FIXED_XYZ moves X, Z then Y, its state machine advancing one state
  per 100 ms status tick (stepper.c:196-243): the sum of the axes' profiles
  plus FIXED_XYZ_TICKS_S (fixed_xyz_deadline).

Deadlines:
    move:      max(DEADLINE_FLOOR_S, DEADLINE_FACTOR * t + DEADLINE_MARGIN_S)
    home:      DEADLINE_FACTOR * 36.55 mm / home_vel + DEADLINE_MARGIN_S
    fixed XYZ: DEADLINE_FACTOR * sum(t_axis) + FIXED_XYZ_TICKS_S + DEADLINE_MARGIN_S

UNFINALIZED constants. Basis, christielab10 (max_vel 120 mm/s, max_acc 600 mm/s^2,
home_vel 30 mm/s): 683 MOVE and 60 HOME acks on 2026-11-24 (logs 000-006) and
every SEND_TO_FIXED_XYZ ack in the trial logs of 2026-09-24, 2026-10-02 and
2026-11-24. Measured against the firmware profile, a single move is acked at
most 0.09 s late (the status tick plus CAN and host latency), whatever its
length; a 35 mm home at most 1.24 s after it was sent; the trial send
(25, 0, 10) -> (0, 20, 0) at 1.80-1.86 s (14 sends). DEADLINE_MARGIN_S is the
0.1 s tick plus a 0.3 s host-stall budget; DEADLINE_FACTOR covers clock and
rounding; FIXED_XYZ_TICKS_S allows a tick to start, about two per axis
transition, and one to ack. Re-measure on a GUI trial session (several hundred
sends) before finalising: the scripted calibration run has none of the GUI's
stalls.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

DEADLINE_FACTOR = 1.10
DEADLINE_MARGIN_S = 0.40
DEADLINE_FLOOR_S = 0.60
FIXED_XYZ_TICKS_S = 0.60
#: How long past its deadline a move is stopped even when the CAN reader has delivered
#: nothing since (CanDevice: a deadline fires only once a frame read after it was processed).
READER_SILENT_BACKSTOP_S = 1.0


def profile_time(distance: float, max_velocity: float, max_acceleration: float) -> Optional[float]:
    """Seconds the firmware takes to move `distance` mm, or None when the limits are unusable."""
    try:
        d, v, a = abs(float(distance)), float(max_velocity), float(max_acceleration)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (d, v, a)) or v <= 0 or a <= 0:
        return None
    if d >= 2.0 * v * v / a:  # reaches max_velocity: two raised-cosine ramps of 2v/a, cruise
        return d / v + 2.0 * v / a
    return 2.0 * math.sqrt(2.0 * d / a)


def ack_deadline(profile_s: Optional[float]) -> Optional[float]:
    """Seconds after the send by which a single move's ack is due; None when the profile is unknown."""
    if profile_s is None or not math.isfinite(profile_s):
        return None
    return max(DEADLINE_FLOOR_S, profile_s * DEADLINE_FACTOR + DEADLINE_MARGIN_S)


def home_time(homing_velocity: float, travel: float) -> Optional[float]:
    """Seconds a home takes over the full `travel` at `homing_velocity` (no ramp)."""
    try:
        v, d = float(homing_velocity), float(travel)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(v) and math.isfinite(d)) or v <= 0:
        return None
    return d / v


def home_deadline(homing_velocity: float, travel: float) -> Optional[float]:
    """Seconds after the send by which a home's ack is due, from the full travel."""
    time_s = home_time(homing_velocity, travel)
    return None if time_s is None else time_s * DEADLINE_FACTOR + DEADLINE_MARGIN_S


def fixed_xyz_deadline(profiles_s: Iterable[Optional[float]]) -> Optional[float]:
    """Seconds after the send by which SEND_TO_FIXED_XYZ's ack is due, from its axes' profiles."""
    profiles = list(profiles_s)
    if any(p is None or not math.isfinite(p) for p in profiles):
        return None
    return DEADLINE_FACTOR * sum(profiles) + FIXED_XYZ_TICKS_S + DEADLINE_MARGIN_S


__all__ = [
    "DEADLINE_FACTOR", "DEADLINE_FLOOR_S", "DEADLINE_MARGIN_S", "FIXED_XYZ_TICKS_S",
    "READER_SILENT_BACKSTOP_S", "ack_deadline", "fixed_xyz_deadline", "home_deadline", "home_time",
    "profile_time",
]
