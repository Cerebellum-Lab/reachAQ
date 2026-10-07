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
# A camera clock latch is one USB round trip: christielab10's BFS-U3-16S2M
# brackets it in 0.18-0.36 ms (median 0.20 ms), idle or streaming. 2 ms admits
# ten times that on a busy bus while keeping the latch's own uncertainty, half
# the bracket, under a quarter of a 150 fps frame period (1.67 ms).
LATCH_MAX_BRACKET_SECONDS = 0.002
# Latches further apart than this are fitted with a line, which takes up the
# camera oscillator's drift; closer ones give an offset only, trusted for
# frames within LATCH_WINDOW_SECONDS (20 ppm over 30 s is 0.6 ms).
LATCH_DRIFT_SPAN_SECONDS = 10.0
LATCH_WINDOW_SECONDS = 30.0
# What may separate the NI edge from the latched camera time it is paired with
# (and the NI path's exposure-to-arrival from the latch path's): a quarter
# period, never more than this. A bias of whole periods plus a little passes a
# quarter-period test at low frame rates; christielab10's NI stamps, paired
# with their own blocks, put the edges 16.6 ms early and 50 fps paired a frame
# off with a 3.39 ms residual. With the bias gone the residual is about 0.1 ms.
LATCH_MAX_RESIDUAL_SECONDS = 0.002
HOST_ARRIVAL = "host_arrival"
CAMERA_LATCH = "camera_latch"


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
    """Each read's stamp, paired with the last sample of the read before it.

    The reader stamps a block with perf_counter taken before the blocking read
    that returns it, so the stamp says nothing about the block's own samples:
    with back-to-back reads it is when the previous read returned, about one
    block before this block's last sample. It is an upper bound on when the
    previous block's last sample was taken, though, which is what the lower
    envelope in fit_ni_to_host needs; reads that are not back to back only
    loosen the bound. The first read's stamp bounds nothing and is dropped.
    """
    sample_index = numpy.asarray(sample_index, dtype=numpy.int64)
    observation_perf = numpy.asarray(observation_perf, dtype=numpy.float64)
    if observation_perf.size == 0:
        return numpy.empty(0, dtype=numpy.int64), numpy.empty(0)
    ends = numpy.flatnonzero(observation_perf[1:] != observation_perf[:-1])
    ends = numpy.append(ends, observation_perf.size - 1)
    previous_end, stamp = sample_index[ends[:-1]], observation_perf[ends[1:]]
    keep = numpy.isfinite(stamp)
    return previous_end[keep], stamp[keep]


def fit_ni_to_host(end_index, seen_perf, sample_rate: float) -> ClockFit:
    """NI sample time (index / rate) -> host perf_counter, from the lower envelope.

    Each point pairs a sample with a host time no earlier than when it was
    taken (ni_block_ends: a read's stamp, taken before the blocking read,
    bounds the previous block's last sample). So each (seen - sample time) is
    the delivery delay plus whatever the reader did before starting its next
    read. The smallest across the session trace the clock relation with only
    the minimum of that left in it; it is unknown and makes every mapped time
    late by about that much. A sync pulse (sub-project 2) is what would bound it.
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
    method: str = HOST_ARRIVAL
    median_residual: float = math.nan
    """Camera-clock method: transition minus expected host time, over the chosen shift."""


def choose_pairing(frame_ids, arrival_perf, transition_perf, frame_period: float) -> Pairing:
    """Which exposure transition belongs to which frame, decided by host time.

    A straight-line fit of camera time to NI time cannot tell the right pairing
    from a one-frame offset: every shift fits equally well. Host time can,
    because an exposure starts before the host receives its frame. For each
    frame the latest transition at or before its arrival is a candidate, and
    the shift most frames agree on wins. This assumes the exposure-to-arrival
    lag is under one frame period; if the winning lag sits within a margin of
    zero or of a whole period, the NI-to-host bias could have tipped the
    choice, and the pairing is marked ambiguous. A longer lag is not detected:
    every frame is paired with an exposure whole periods after its own.
    choose_pairing_by_camera_clock needs no such assumption and is preferred
    when the camera stream has clock latches.
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


def usable_latches(latches) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """(camera seconds, host perf, uncertainty) of each clock latch worth using.

    The latch happened somewhere inside its perf bracket, so the midpoint is its
    host time and half the width its uncertainty. A bracket that is not finite
    or wider than LATCH_MAX_BRACKET_SECONDS is left out.
    """
    if latches is None or len(latches) == 0:
        return numpy.empty(0), numpy.empty(0), numpy.empty(0)
    before = numpy.asarray(latches["perf_before"], dtype=numpy.float64)
    after = numpy.asarray(latches["perf_after"], dtype=numpy.float64)
    camera = numpy.asarray(latches["camera_ns"], dtype=numpy.float64) / 1e9
    width = after - before
    keep = numpy.isfinite(width) & (width >= 0) & (width <= LATCH_MAX_BRACKET_SECONDS)
    return camera[keep], ((before + after) / 2)[keep], (width / 2)[keep]


def fit_latches(latches) -> ClockFit:
    """Camera timestamp (s) -> host perf_counter, from the camera clock latches.

    Latches more than LATCH_DRIFT_SPAN_SECONDS apart are fitted with a line,
    which takes up the camera's drift. Closer together they cannot measure it,
    and the tightest one's offset is used alone; its uncertainty stands in for
    the residual. Either way, x_min and x_max are the latched span, and only
    timestamps latch_window() admits should be mapped.
    """
    camera, host, uncertainty = usable_latches(latches)
    if camera.size == 0:
        rows = 0 if latches is None else len(latches)
        return invalid_fit(f"no usable clock latch: {rows} rows, none with a finite bracket "
                           f"within {LATCH_MAX_BRACKET_SECONDS * 1e3:.1f} ms")
    if camera.max() - camera.min() > LATCH_DRIFT_SPAN_SECONDS:
        return fit_line(camera, host, max_residual=LATCH_MAX_BRACKET_SECONDS, min_points=2)
    tightest = int(numpy.argmin(uncertainty))
    return ClockFit(offset=float(host[tightest] - camera[tightest]), drift=1.0,
                    residual_rms=float(uncertainty[tightest]), points=1, valid=True,
                    x_min=float(camera[tightest]), x_max=float(camera[tightest]))


def latch_tolerance(frame_period: float) -> float:
    """How far the NI path may sit from the latch path: P/4, at most LATCH_MAX_RESIDUAL_SECONDS."""
    if not math.isfinite(frame_period):
        return math.nan
    return min(frame_period / 4, LATCH_MAX_RESIDUAL_SECONDS)


def latch_window(fit: ClockFit, camera_seconds) -> numpy.ndarray:
    """Camera timestamps within LATCH_WINDOW_SECONDS of the latched span."""
    camera_seconds = numpy.asarray(camera_seconds, dtype=numpy.float64)
    return ((camera_seconds >= fit.x_min - LATCH_WINDOW_SECONDS)
            & (camera_seconds <= fit.x_max + LATCH_WINDOW_SECONDS))


def choose_pairing_by_camera_clock(frame_ids, camera_ts_ns, latches, transition_perf,
                                   frame_period: float) -> Pairing:
    """Which exposure transition belongs to which frame, decided by the camera's clock.

    The latches place camera time on host time, so a frame's camera timestamp
    says when on the host clock it was exposed, however long it then waited to
    be delivered. Each frame's candidate is the transition nearest that time,
    and the shift most frames agree on wins.

    The median residual (transition minus expected time, over the chosen shift)
    is where the camera stamps a frame relative to the NI edge, plus the NI-to-
    host bias, plus the latch uncertainty. All of it must stay well under half a
    period for the nearest transition to be the right one, and a bias near a
    whole period would pair a frame off with a small residual, so a residual
    over latch_tolerance() (a quarter period, at most 2 ms), or agreement below
    PAIRING_MIN_AGREEMENT, marks the pairing ambiguous. The lag is not measured
    here: median_lag is NaN.

    The limit that sets: a Blackfly S stamps a frame at the end of its exposure
    and the NI edge marks its start, so the residual approaches minus the
    exposure time. An exposure longer than about 1.8 ms therefore reads
    ambiguous, visibly, rather than being paired on a guess.
    """
    fit = fit_latches(latches)
    if not fit.valid:
        return Pairing(0, math.nan, 0.0, True, fit.reason, method=CAMERA_LATCH)
    frame_ids = numpy.asarray(frame_ids, dtype=numpy.int64)
    camera = numpy.asarray(camera_ts_ns, dtype=numpy.float64) / 1e9
    transition_perf = numpy.asarray(transition_perf, dtype=numpy.float64)
    keep = numpy.isfinite(camera) & latch_window(fit, camera)
    if not keep.any() or transition_perf.size == 0:
        return Pairing(0, math.nan, 0.0, True, "no frames near a clock latch or no transitions",
                       method=CAMERA_LATCH)
    ids = frame_ids[keep]
    expected = fit.map(camera[keep])
    later = numpy.clip(numpy.searchsorted(transition_perf, expected), 0, transition_perf.size - 1)
    earlier = numpy.clip(later - 1, 0, transition_perf.size - 1)
    nearest = numpy.where(numpy.abs(transition_perf[later] - expected)
                          < numpy.abs(transition_perf[earlier] - expected), later, earlier)
    shifts = ids - nearest
    values, counts = numpy.unique(shifts, return_counts=True)
    shift = int(values[numpy.argmax(counts)])
    agreement = float(counts.max() / shifts.size)
    numbers = ids - shift
    inside = (numbers >= 0) & (numbers < transition_perf.size)
    residual = transition_perf[numbers[inside]] - expected[inside]
    median_residual = float(numpy.median(residual)) if residual.size else math.nan
    tolerance = latch_tolerance(frame_period)
    residual_ambiguous = not abs(median_residual) <= tolerance
    agreement_ambiguous = agreement < PAIRING_MIN_AGREEMENT
    if residual_ambiguous:
        reason = (f"median residual {median_residual * 1e3:.3f} ms is over the latch "
                  f"tolerance ({tolerance * 1e3:.3f} ms)")
    elif agreement_ambiguous:
        reason = f"agreement {agreement:.2f} below {PAIRING_MIN_AGREEMENT}"
    else:
        reason = ""
    return Pairing(shift, math.nan, agreement, bool(residual_ambiguous or agreement_ambiguous),
                   reason, method=CAMERA_LATCH, median_residual=median_residual)


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
        # Second pass: apply exact ranges (measured zone) and detect overlaps.
        # If any wall value falls in multiple segments' exact ranges, it's ambiguous.
        overlap_count = numpy.zeros(wall.shape, dtype=int)
        for fit in self.fits:
            if not fit.valid:
                continue
            inside = (wall >= fit.x_min) & (wall <= fit.x_max)
            out[inside] = fit.map(wall[inside])
            overlap_count[inside] += 1
        # Mark overlaps (wall values in more than one segment) as NaN
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
