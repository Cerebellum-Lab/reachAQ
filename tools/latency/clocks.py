"""Clock mappings for the latency record.

Every mapping is a straight line, y = offset + drift * x, fitted once per
session. Each fit carries its own quality numbers and a validity verdict, so a
figure derived through it can say how far to trust it, and an invalid mapping
lowers the confidence of what depends on it instead of quietly producing a
number.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import numpy

MIN_FIT_POINTS = 100
NI_MAX_RESIDUAL_SECONDS = 0.0005
WALL_SEGMENT_JUMP_SECONDS = 0.001
ENVELOPE_BINS = 20
PAIRING_MARGIN_SECONDS = 0.0001
PAIRING_MIN_AGREEMENT = 0.95


@dataclasses.dataclass(frozen=True)
class ClockFit:
    offset: float
    drift: float
    residual_rms: float
    points: int
    valid: bool
    reason: str = ""
    x_min: float = math.nan
    x_max: float = math.nan

    def map(self, x):
        return self.offset + self.drift * numpy.asarray(x, dtype=numpy.float64)

    def as_record(self) -> dict:
        return dataclasses.asdict(self)


def invalid_fit(reason: str) -> ClockFit:
    return ClockFit(math.nan, math.nan, math.nan, 0, False, reason)


def fit_line(x, y, *, max_residual: float, min_points: int = MIN_FIT_POINTS) -> ClockFit:
    """Least squares, then one refit without points beyond 5 robust sigmas."""
    x = numpy.asarray(x, dtype=numpy.float64)
    y = numpy.asarray(y, dtype=numpy.float64)
    keep = numpy.isfinite(x) & numpy.isfinite(y)
    x, y = x[keep], y[keep]
    needed = max(2, int(min_points))
    if x.size < needed:
        return invalid_fit(f"{x.size} points, need {needed}")
    for attempt in range(2):
        # Centre x first: perf_counter and sample times are large numbers, and
        # an uncentred fit loses the drift term to rounding.
        x0 = float(x.mean())
        drift, centred_offset = numpy.polyfit(x - x0, y, 1)
        residual = y - (centred_offset + drift * (x - x0))
        if attempt == 1:
            break
        median_residual = numpy.median(residual)
        spread = 1.4826 * float(numpy.median(numpy.abs(residual - median_residual)))
        inliers = numpy.abs(residual - median_residual) <= max(5.0 * spread, 1e-9)
        if inliers.all() or inliers.sum() < needed:
            break
        x, y = x[inliers], y[inliers]
    rms = float(numpy.sqrt(numpy.mean(residual ** 2)))
    valid = rms <= max_residual
    return ClockFit(
        offset=float(centred_offset - drift * x0),
        drift=float(drift),
        residual_rms=rms,
        points=int(x.size),
        valid=bool(valid),
        reason="" if valid else f"residual rms {rms:.6f} s above {max_residual:.6f} s",
        x_min=float(x.min()),
        x_max=float(x.max()),
    )


def transitions(values) -> numpy.ndarray:
    """Positions where a 0/1 line changes state, excluding its initial state.

    The primary camera's exposure output toggles once per exposure, so every
    transition, rising or falling, is one frame.
    """
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.size < 2:
        return numpy.empty(0, dtype=numpy.int64)
    high = numpy.isfinite(values) & (values > 0.5)
    return (numpy.flatnonzero(high[1:] != high[:-1]) + 1).astype(numpy.int64)


@dataclasses.dataclass(frozen=True)
class EdgeResult:
    positions: numpy.ndarray
    threshold: float
    usable: bool
    reason: str = ""


def rising_edges(values, *, min_swing: float) -> EdgeResult:
    """Rising crossings of the midpoint between a signal's baseline and peak.

    Laser pulses are sparse, so the baseline is the median and the peak the
    maximum. A channel whose swing is not well clear of its own noise - a
    floating input such as christielab10's laser-1 diode - is reported
    unusable rather than producing edges from noise.
    """
    values = numpy.asarray(values, dtype=numpy.float64)
    finite = values[numpy.isfinite(values)]
    empty = numpy.empty(0, dtype=numpy.int64)
    if finite.size < 2:
        return EdgeResult(empty, math.nan, False, "no samples")
    baseline = float(numpy.median(finite))
    noise = 1.4826 * float(numpy.median(numpy.abs(finite - baseline)))
    swing = float(finite.max()) - baseline
    if swing < max(float(min_swing), 10.0 * noise):
        return EdgeResult(empty, math.nan, False,
                          f"swing {swing:.4f} against noise {noise:.4f} (min {min_swing})")
    threshold = baseline + swing / 2
    above = numpy.isfinite(values) & (values >= threshold)
    positions = numpy.flatnonzero(above[1:] & ~above[:-1]) + 1
    return EdgeResult(positions.astype(numpy.int64), threshold, True)


def ni_block_ends(sample_index, observation_perf) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """The last sample each host read delivered, and when the host saw it."""
    sample_index = numpy.asarray(sample_index, dtype=numpy.int64)
    observation_perf = numpy.asarray(observation_perf, dtype=numpy.float64)
    if observation_perf.size == 0:
        return numpy.empty(0, dtype=numpy.int64), numpy.empty(0)
    ends = numpy.flatnonzero(observation_perf[1:] != observation_perf[:-1])
    ends = numpy.append(ends, observation_perf.size - 1)
    keep = numpy.isfinite(observation_perf[ends])
    return sample_index[ends][keep], observation_perf[ends][keep]


def fit_ni_to_host(end_index, seen_perf, sample_rate: float) -> ClockFit:
    """NI sample time (index / rate) -> host perf_counter, from the lower envelope.

    A block can only be seen after its last sample was taken, so each read's
    (seen - sample time) is the delivery delay plus whatever the reader was
    doing. The smallest delays across the session trace the clock relation
    with only the minimum delivery delay left in it; that delay is unknown and
    makes every mapped time late by about that much. A sync pulse (sub-project
    2) is what would bound it.
    """
    t = numpy.asarray(end_index, dtype=numpy.float64) / float(sample_rate)
    seen = numpy.asarray(seen_perf, dtype=numpy.float64)
    if t.size < MIN_FIT_POINTS:
        return invalid_fit(f"{t.size} host reads, need {MIN_FIT_POINTS}")
    if numpy.any(numpy.diff(t) <= 0):
        return invalid_fit("NI sample index is not increasing (task restarted mid-session)")
    edges = numpy.linspace(t[0], t[-1], ENVELOPE_BINS + 1)
    which = numpy.clip(numpy.searchsorted(edges, t, side="right") - 1, 0, ENVELOPE_BINS - 1)
    delay = seen - t
    picks = []
    for bin_index in range(ENVELOPE_BINS):
        members = numpy.flatnonzero(which == bin_index)
        if members.size:
            picks.append(int(members[numpy.argmin(delay[members])]))
    picks = numpy.asarray(picks, dtype=numpy.int64)
    return fit_line(t[picks], seen[picks], max_residual=NI_MAX_RESIDUAL_SECONDS,
                    min_points=ENVELOPE_BINS // 2)


@dataclasses.dataclass(frozen=True)
class Pairing:
    shift: int
    """Transition number = frame id - shift."""
    median_lag: float
    agreement: float
    ambiguous: bool
    reason: str = ""


def choose_pairing(frame_ids, arrival_perf, transition_perf, frame_period: float) -> Pairing:
    """Which exposure transition belongs to which frame, decided by host time.

    A straight-line fit of camera time to NI time cannot tell the right pairing
    from a one-frame offset: every shift fits equally well. Host time can,
    because an exposure starts before the host receives its frame. For each
    frame the latest transition at or before its arrival is a candidate, and
    the shift most frames agree on wins. This assumes the exposure-to-arrival
    lag is under one frame period; if the winning lag sits within a margin of
    zero or of a whole period, the NI-to-host bias could have tipped the
    choice, and the pairing is marked ambiguous.
    """
    frame_ids = numpy.asarray(frame_ids, dtype=numpy.int64)
    arrival = numpy.asarray(arrival_perf, dtype=numpy.float64)
    transition_perf = numpy.asarray(transition_perf, dtype=numpy.float64)
    keep = numpy.isfinite(arrival)
    if not keep.any() or transition_perf.size == 0:
        return Pairing(0, math.nan, 0.0, True, "no frames or no transitions")
    ids, arrival = frame_ids[keep], arrival[keep]
    candidate = numpy.searchsorted(transition_perf, arrival, side="right") - 1
    found = candidate >= 0
    if not found.any():
        return Pairing(0, math.nan, 0.0, True, "every frame arrived before the first transition")
    shifts = ids[found] - candidate[found]
    values, counts = numpy.unique(shifts, return_counts=True)
    shift = int(values[numpy.argmax(counts)])
    agreement = float(counts.max() / shifts.size)
    numbers = ids - shift
    inside = (numbers >= 0) & (numbers < transition_perf.size)
    lag = arrival[inside] - transition_perf[numbers[inside]]
    median_lag = float(numpy.median(lag)) if lag.size else math.nan
    lag_ambiguous = not (PAIRING_MARGIN_SECONDS <= median_lag
                         <= frame_period - PAIRING_MARGIN_SECONDS)
    agreement_ambiguous = agreement < PAIRING_MIN_AGREEMENT
    ambiguous = lag_ambiguous or agreement_ambiguous
    if lag_ambiguous:
        reason = (
            f"median arrival lag {median_lag * 1e3:.3f} ms is within "
            f"{PAIRING_MARGIN_SECONDS * 1e3:.1f} ms of 0 or one frame period"
        )
    elif agreement_ambiguous:
        reason = f"agreement {agreement:.2f} below {PAIRING_MIN_AGREEMENT}"
    else:
        reason = ""
    return Pairing(shift, median_lag, agreement, bool(ambiguous), reason)


def fit_camera_to_ni(frame_ids, camera_ts_ns, transition_index, sample_rate: float,
                     pairing: Pairing, frame_period: float) -> ClockFit:
    """Camera timestamp (s) -> NI time (s), over the paired frames.

    Each frame's camera timestamp maps to a specific transition index via the
    pairing shift. A lost or spurious NI transition renumbers every later frame
    by one, causing a ~P/4 residual slip if undetected; check pairing.ambiguous.
    """
    frame_ids = numpy.asarray(frame_ids, dtype=numpy.int64)
    camera_ts = numpy.asarray(camera_ts_ns, dtype=numpy.float64) / 1e9
    transition_index = numpy.asarray(transition_index, dtype=numpy.int64)
    numbers = frame_ids - int(pairing.shift)
    inside = (numbers >= 0) & (numbers < transition_index.size)
    return fit_line(camera_ts[inside], transition_index[numbers[inside]] / float(sample_rate),
                    max_residual=0.5 * frame_period)


@dataclasses.dataclass(frozen=True)
class WallMap:
    fits: Tuple[ClockFit, ...]

    def map(self, wall):
        wall = numpy.asarray(wall, dtype=numpy.float64)
        out = numpy.full(wall.shape, math.nan)
        # First pass: apply padded ranges (extrapolation zone)
        for fit in self.fits:
            if not fit.valid:
                continue
            inside = (wall >= fit.x_min - 2.0) & (wall <= fit.x_max + 2.0)
            out[inside] = fit.map(wall[inside])
        # Second pass: apply exact ranges (measured zone), overwriting extrapolation
        for fit in self.fits:
            if not fit.valid:
                continue
            inside = (wall >= fit.x_min) & (wall <= fit.x_max)
            out[inside] = fit.map(wall[inside])
        # Third pass: detect overlaps from backward wall steps.
        # If any wall value falls in multiple segments' exact ranges, it's ambiguous.
        overlapping = numpy.zeros(wall.shape, dtype=bool)
        for fit in self.fits:
            if not fit.valid:
                continue
            inside = (wall >= fit.x_min) & (wall <= fit.x_max)
            overlapping |= inside
        # Mark overlaps (wall values in more than one segment) as NaN
        overlap_count = numpy.zeros(wall.shape, dtype=int)
        for fit in self.fits:
            if not fit.valid:
                continue
            inside = (wall >= fit.x_min) & (wall <= fit.x_max)
            overlap_count[inside] += 1
        out[overlap_count > 1] = math.nan
        return out


def fit_wall_to_host(wall, perf) -> WallMap:
    """Wall clock -> perf_counter, one segment per wall-clock step (NTP)."""
    wall = numpy.asarray(wall, dtype=numpy.float64)
    perf = numpy.asarray(perf, dtype=numpy.float64)
    keep = numpy.isfinite(wall) & numpy.isfinite(perf)
    wall, perf = wall[keep], perf[keep]
    if wall.size < 2:
        return WallMap(())
    # Sort by perf (monotonic recorded order), not by wall (may step backward).
    order = numpy.argsort(perf, kind="stable")
    wall, perf = wall[order], perf[order]
    offset = perf - wall
    breaks = numpy.flatnonzero(numpy.abs(numpy.diff(offset)) > WALL_SEGMENT_JUMP_SECONDS) + 1
    fits = tuple(
        fit_line(wall[segment], perf[segment],
                 max_residual=WALL_SEGMENT_JUMP_SECONDS, min_points=2)
        for segment in numpy.split(numpy.arange(wall.size), breaks)
        if segment.size >= 2
    )
    return WallMap(fits)
