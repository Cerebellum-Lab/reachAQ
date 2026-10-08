"""Clock mappings for the latency record and the session's frame alignment.

A mapping is a straight line, y = offset + drift * x, or for NI to host a
continuous piecewise-linear table (PiecewiseClockFit) with a knot every
KNOT_SECONDS: time.perf_counter() is CLOCK_MONOTONIC, which systemd-timesyncd
slews at every poll, and one line per session leaves millisecond errors after
an hour. Each fit carries its own quality numbers and a validity verdict, so a
figure derived through it can say how far to trust it, and an invalid mapping
lowers the confidence of what depends on it instead of quietly producing a
number.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import numpy

MIN_FIT_POINTS = 100
NI_MAX_RESIDUAL_SECONDS = 0.0005
WALL_SEGMENT_JUMP_SECONDS = 0.001
ENVELOPE_BINS = 20
PAIRING_MARGIN_SECONDS = 0.0001
PAIRING_MIN_AGREEMENT = 0.95
# NI-to-host is fitted window by window, joined at the knots, because the
# host clock is slewed by NTP: timesyncd polls every 2048 s and the kernel
# steps the rate by up to about 6.5 ppm, decaying over 512 s. Simulated on
# christielab10's settings (ntp/pll_sim.py, 2026-10-07): one line per session
# leaves p95 errors of 1.9-4.3 ms at 1 h; 120 s windows leave under 0.4 ms.
# A window with fewer envelope points than MIN_WINDOW_POINTS (a short tail, a
# gap) merges into its neighbour rather than fitting noise.
KNOT_SECONDS = 120.0
MIN_WINDOW_POINTS = 10
# A RAW-bridged fit is exact at every stamp; its stored table samples it this
# often, which keeps linear interpolation under 50 us across a 10 ppm kink.
RECORD_KNOT_SECONDS = 10.0
PIECEWISE_ENVELOPE = "piecewise_envelope"
RAW_BRIDGED = "raw_bridged"
STRAIGHT_LINE = "straight_line"
# A bridge knot's perf-minus-RAW offset is the median over the stamp pairs
# this close to it (200 pairs at the 10 ms copy cadence), so one pair whose
# reader was preempted between its two clock reads cannot move a knot. A knot
# closer than BRIDGE_TAIL_SECONDS to the end is dropped for the end knot, so
# no segment is milliseconds long with a drift made of noise.
BRIDGE_WINDOW_SECONDS = 1.0
BRIDGE_TAIL_SECONDS = 5.0
# A camera clock latch is one USB round trip: christielab10's BFS-U3-16S2M
# brackets it in 0.14-0.58 ms inside the running app (60 latches, median
# 0.27 ms) and in about 0.20 ms with the app closed. 2 ms admits several times
# that on a busy bus while keeping the latch's own uncertainty, half the
# bracket, under a quarter of a 150 fps frame period (1.67 ms).
LATCH_MAX_BRACKET_SECONDS = 0.002
# Latches further apart than this are fitted with a line, which takes up the
# camera oscillator's drift; closer ones give an offset only, trusted for
# frames within LATCH_WINDOW_SECONDS (20 ppm over 30 s is 0.6 ms).
LATCH_DRIFT_SPAN_SECONDS = 10.0
LATCH_WINDOW_SECONDS = 30.0
# What may separate the NI edge from the latched camera time it is paired with:
# a quarter period, never more than this. This residual gate is the safeguard
# against whole-period mispairing, because a bias of whole periods plus a
# little passes a quarter-period test at low frame rates: christielab10's NI
# stamps, paired with their own blocks, put the edges 16.6 ms early, and
# 50 fps paired a frame off with a 3.39 ms residual. With the bias gone the
# residual is about 0.1 ms. The finalizer's NI-versus-latch lag check uses the
# same tolerance but sees this same residual again, through the fitted
# camera-to-NI line; it is no reason to relax this gate.
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

    def as_record(self, *, include_knots: bool = True) -> dict:
        # include_knots is PiecewiseClockFit's; accepted here so a caller that
        # may hold either kind of fit does not have to ask which.
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


@dataclasses.dataclass(frozen=True)
class BlockStamps:
    """One row per recorder copy: the last sample of the copy before it and the
    copy's stamps. raw is CLOCK_MONOTONIC_RAW taken beside the perf stamp, NaN
    where the host has no such clock or the file predates it."""
    end_index: numpy.ndarray
    perf: numpy.ndarray
    raw: numpy.ndarray


def run_ends(observation_perf) -> numpy.ndarray:
    """Position of the last sample of each run of equal block stamps."""
    observation_perf = numpy.asarray(observation_perf, dtype=numpy.float64)
    if observation_perf.size == 0:
        return numpy.empty(0, dtype=numpy.int64)
    ends = numpy.flatnonzero(observation_perf[1:] != observation_perf[:-1])
    return numpy.append(ends, observation_perf.size - 1).astype(numpy.int64)


def block_stamps_from_runs(index_at_ends, perf_at_ends, raw_at_ends=None) -> BlockStamps:
    """Pair each run's stamps with the last sample of the run before it.

    In nidaq.h5 a run of equal block_observation_perf_time values is not one NI
    read. It is one copy the session recorder makes from the NI sample ring
    (every 10 ms) of whatever the ring gained since its last copy, stamped with
    the stamp of the newest read in it. That stamp is perf_counter taken before
    the blocking read, so it says nothing about the run's own samples: with
    back-to-back reads it is when the read before the newest one returned. It
    is an upper bound on when the previous run's last sample was taken, though
    (that sample came from an earlier read, which had returned), and that is
    what the lower envelope in fit_ni_to_host needs. The first run's stamp
    bounds nothing and is dropped. How loose the bound is depends on the read
    size; see fit_ni_to_host.
    """
    index_at_ends = numpy.asarray(index_at_ends, dtype=numpy.int64)
    perf_at_ends = numpy.asarray(perf_at_ends, dtype=numpy.float64)
    raw_at_ends = (numpy.full(perf_at_ends.shape, math.nan) if raw_at_ends is None
                   else numpy.asarray(raw_at_ends, dtype=numpy.float64))
    if perf_at_ends.size < 2:
        return BlockStamps(numpy.empty(0, dtype=numpy.int64), numpy.empty(0), numpy.empty(0))
    previous_end, stamp, raw = index_at_ends[:-1], perf_at_ends[1:], raw_at_ends[1:]
    keep = numpy.isfinite(stamp)
    return BlockStamps(previous_end[keep], stamp[keep], raw[keep])


def ni_block_stamps(sample_index, observation_perf, observation_raw=None) -> BlockStamps:
    """block_stamps_from_runs over per-sample arrays, as nidaq.h5 stores them."""
    sample_index = numpy.asarray(sample_index, dtype=numpy.int64)
    observation_perf = numpy.asarray(observation_perf, dtype=numpy.float64)
    ends = run_ends(observation_perf)
    raw = (None if observation_raw is None
           else numpy.asarray(observation_raw, dtype=numpy.float64)[ends])
    return block_stamps_from_runs(sample_index[ends], observation_perf[ends], raw)


def ni_block_ends(sample_index, observation_perf) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """(last sample of the previous copy, this copy's perf stamp); see block_stamps_from_runs."""
    stamps = ni_block_stamps(sample_index, observation_perf)
    return stamps.end_index, stamps.perf


def fit_ni_to_host(end_index, seen_perf, sample_rate: float) -> ClockFit:
    """NI sample time (index / rate) -> host perf_counter, from the lower envelope.

    Each point pairs a sample with a host time no earlier than when it was
    taken (ni_block_ends: a recorder copy's stamp, taken before the blocking
    read, bounds the previous copy's last sample). So each (seen - sample
    time) is at least the delivery delay. The smallest across the session
    trace the clock relation with only their minimum left in it, which makes
    every mapped time late by about that much. A sync pulse (sub-project 2) is
    what would bound it.

    That minimum depends on the display refresh rate, which sets the NI read
    size (sample rate / refresh). While reads are long against the 10 ms copy
    period, some copies end on the read just before the stamped one and the
    minimum is the delivery floor. When every copy spans two or more reads,
    the previous copy ends a whole read or more before the stamp. Simulated
    (Task 16 review): 60-165 Hz maps about 0.25 ms late, 200 Hz fails the
    fit, and 240 Hz maps 4.45 ms late. christielab10 runs at 60 Hz.
    """
    t = numpy.asarray(end_index, dtype=numpy.float64) / float(sample_rate)
    seen = numpy.asarray(seen_perf, dtype=numpy.float64)
    if t.size < MIN_FIT_POINTS:
        return invalid_fit(f"{t.size} read pairs, need {MIN_FIT_POINTS} "
                           "(each a stamp and the last sample before it)")
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


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _interp_extrapolate(x, xs, ys):
    """Piecewise-linear interpolation through (xs, ys), continued with the end
    segments outside them; xs strictly increasing with at least two points."""
    x = numpy.asarray(x, dtype=numpy.float64)
    j = numpy.clip(numpy.searchsorted(xs, x, side="right") - 1, 0, xs.size - 2)
    slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
    return ys[j] + slope * (x - xs[j])


@dataclasses.dataclass(frozen=True)
class PiecewiseClockFit:
    """NI seconds -> host seconds through a continuous piecewise-linear table.

    The same interface as ClockFit (map, as_record, valid, reason,
    residual_rms, points, offset, drift, x_min, x_max) plus inverse() and the
    per-segment drifts. offset and drift describe the straight line through
    the end knots, so a reader of the old record keys still gets a usable
    summary of the whole session.
    """
    knots: numpy.ndarray
    knot_host: numpy.ndarray
    residual_rms: float
    points: int
    valid: bool
    reason: str = ""
    method: str = PIECEWISE_ENVELOPE

    @property
    def x_min(self) -> float:
        return float(self.knots[0]) if self.knots.size else math.nan

    @property
    def x_max(self) -> float:
        return float(self.knots[-1]) if self.knots.size else math.nan

    @property
    def drift(self) -> float:
        if self.knots.size < 2:
            return math.nan
        return float((self.knot_host[-1] - self.knot_host[0]) / (self.knots[-1] - self.knots[0]))

    @property
    def offset(self) -> float:
        if self.knots.size < 2:
            return math.nan
        return float(self.knot_host[0] - self.drift * self.knots[0])

    def segment_drifts(self) -> numpy.ndarray:
        if self.knots.size < 2:
            return numpy.empty(0)
        return numpy.diff(self.knot_host) / numpy.diff(self.knots)

    def map(self, x):
        x = numpy.asarray(x, dtype=numpy.float64)
        if not self.valid or self.knots.size < 2:
            return numpy.full(x.shape, math.nan)
        return _interp_extrapolate(x, self.knots, self.knot_host)

    def inverse(self, host):
        """Host seconds -> NI seconds; the table is increasing, so it inverts segment by segment."""
        host = numpy.asarray(host, dtype=numpy.float64)
        if not self.valid or self.knots.size < 2:
            return numpy.full(host.shape, math.nan)
        return _interp_extrapolate(host, self.knot_host, self.knots)

    def as_record(self, *, include_knots: bool = True) -> dict:
        drifts = self.segment_drifts()
        record = {
            "method": self.method,
            "valid": bool(self.valid),
            "reason": self.reason,
            "residual_rms": _finite_or_none(self.residual_rms),
            "points": int(self.points),
            "offset": _finite_or_none(self.offset),
            "drift": _finite_or_none(self.drift),
            "x_min": _finite_or_none(self.x_min),
            "x_max": _finite_or_none(self.x_max),
            "knotCount": int(self.knots.size),
            "knotSeconds": (_finite_or_none(numpy.median(numpy.diff(self.knots)))
                            if self.knots.size > 1 else None),
            "segmentDriftPpmMin": (_finite_or_none((drifts.min() - 1.0) * 1e6)
                                   if drifts.size else None),
            "segmentDriftPpmMax": (_finite_or_none((drifts.max() - 1.0) * 1e6)
                                   if drifts.size else None),
        }
        if include_knots:
            record["knots"] = [_finite_or_none(value) for value in self.knots]
            record["knotHost"] = [_finite_or_none(value) for value in self.knot_host]
        return record


def invalid_piecewise_fit(reason: str) -> PiecewiseClockFit:
    return PiecewiseClockFit(numpy.empty(0), numpy.empty(0), math.nan, 0, False, reason)


def _envelope_picks(t, seen, bin_seconds: float) -> numpy.ndarray:
    """In each bin of bin_seconds, the pair with the smallest (seen - t): the lower envelope."""
    which = numpy.floor((t - t[0]) / bin_seconds).astype(numpy.int64)
    # The last pair sits on the far edge of the last bin, which floor would
    # make a bin of its own (21 points over a single window); clip it in.
    bins = max(1, int(numpy.ceil((t[-1] - t[0]) / bin_seconds)))
    which = numpy.minimum(which, bins - 1)
    order = numpy.lexsort((seen - t, which))  # by bin, then by delay
    first_of_bin = numpy.concatenate([[True], numpy.diff(which[order]) > 0])
    return order[first_of_bin]


def _hat_design(x, knots):
    """Rows of the continuous piecewise-linear basis, one hat function per knot."""
    j = numpy.clip(numpy.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
    u = (x - knots[j]) / (knots[j + 1] - knots[j])
    design = numpy.zeros((x.size, knots.size))
    design[numpy.arange(x.size), j] = 1 - u
    design[numpy.arange(x.size), j + 1] = u
    return design


def _merge_sparse_windows(knots, x, min_points: int) -> numpy.ndarray:
    """Drop the knot after a window with fewer than min_points points, so the
    window joins the one after it; the last window joins the one before."""
    knots = numpy.array(knots, dtype=numpy.float64)
    while knots.size > 2:
        counts = numpy.histogram(x, bins=knots)[0]
        sparse = numpy.flatnonzero(counts < min_points)
        if sparse.size == 0:
            break
        window = int(sparse[0])
        knots = numpy.delete(knots, window + 1 if window < knots.size - 2 else window)
    return knots


def fit_piecewise_envelope(end_index, seen_perf, sample_rate: float, *,
                           knot_seconds: float = KNOT_SECONDS) -> PiecewiseClockFit:
    """NI sample time -> host time, a continuous piecewise-linear lower envelope.

    The envelope points are chosen as fit_ni_to_host chooses them (the
    smallest delay in each bin), with a bin every knot_seconds / ENVELOPE_BINS
    so each knot window holds about 20 of them, and a session shorter than one
    window still has 20. A least-squares fit through them in the hat basis is
    continuous at every knot. The bias is the same as fit_ni_to_host's: mapped
    times are late by about the minimum delivery delay.
    """
    t = numpy.asarray(end_index, dtype=numpy.float64) / float(sample_rate)
    seen = numpy.asarray(seen_perf, dtype=numpy.float64)
    finite = numpy.isfinite(seen)
    t, seen = t[finite], seen[finite]
    if t.size < MIN_FIT_POINTS:
        return invalid_piecewise_fit(f"{t.size} read pairs, need {MIN_FIT_POINTS} "
                                     "(each a stamp and the last sample before it)")
    if numpy.any(numpy.diff(t) <= 0):
        return invalid_piecewise_fit("NI sample index is not increasing (task restarted mid-session)")
    span = float(t[-1] - t[0])
    if span <= 0:
        return invalid_piecewise_fit("the read pairs span no time")
    picks = _envelope_picks(t, seen, min(knot_seconds, span) / ENVELOPE_BINS)
    x, y = t[picks], seen[picks]
    # arange can reach t[-1] itself, or a rounding step past it, when the span is a
    # whole number of windows; only knots strictly inside make a real last window.
    inner = numpy.arange(t[0], t[-1], knot_seconds)
    knots = _merge_sparse_windows(numpy.append(inner[inner < t[-1]], t[-1]), x, MIN_WINDOW_POINTS)
    needed = ENVELOPE_BINS // 2
    for attempt in range(2):
        if x.size < max(needed, knots.size):
            return invalid_piecewise_fit(f"{x.size} envelope points, need {max(needed, knots.size)}")
        design = _hat_design(x, knots)
        # Centre y: perf_counter values are large and the knot values would
        # otherwise lose their last digits to the solve.
        y0 = float(y.mean())
        coefficients, _, rank, _ = numpy.linalg.lstsq(design, y - y0, rcond=None)
        if rank < knots.size:
            return invalid_piecewise_fit("a knot window has no envelope point")
        residual = y - y0 - design @ coefficients
        if attempt == 1:
            break
        median_residual = numpy.median(residual)
        spread = 1.4826 * float(numpy.median(numpy.abs(residual - median_residual)))
        inliers = numpy.abs(residual - median_residual) <= max(5.0 * spread, 1e-9)
        if inliers.all() or inliers.sum() < needed:
            break
        x, y = x[inliers], y[inliers]
    rms = float(numpy.sqrt(numpy.mean(residual ** 2)))
    valid = rms <= NI_MAX_RESIDUAL_SECONDS
    return PiecewiseClockFit(
        knots=knots, knot_host=coefficients + y0, residual_rms=rms, points=int(x.size),
        valid=bool(valid),
        reason="" if valid else f"residual rms {rms:.6f} s above {NI_MAX_RESIDUAL_SECONDS:.6f} s",
    )


def fit_from_record(record) -> Optional[PiecewiseClockFit]:
    """The fit a stored record describes, or None when there is none worth using.

    A record with a knot table (PiecewiseClockFit.as_record) comes back as it
    was; a straight-line record (ClockFit.as_record) becomes a two-knot table
    over its fitted span. Anything invalid, malformed or absent is None, so a
    caller falls back rather than mapping through nonsense.
    """
    if not isinstance(record, dict) or not record.get("valid"):
        return None
    try:
        knots = numpy.asarray(record.get("knots") or (), dtype=numpy.float64)
        host = numpy.asarray(record.get("knotHost") or (), dtype=numpy.float64)
        residual = float(record.get("residual_rms") if record.get("residual_rms") is not None else math.nan)
        points = int(record.get("points") or 0)
        # Both columns strictly increasing, or inverse() would not be a function.
        if (knots.size >= 2 and host.size == knots.size and numpy.all(numpy.diff(knots) > 0)
                and numpy.all(numpy.diff(host) > 0)):
            return PiecewiseClockFit(knots, host, residual, points, True, "",
                                     str(record.get("method") or PIECEWISE_ENVELOPE))
        offset, drift = float(record["offset"]), float(record["drift"])
        x_min, x_max = float(record["x_min"]), float(record["x_max"])
        if all(math.isfinite(value) for value in (offset, drift, x_min, x_max)) and x_max > x_min:
            knots = numpy.array([x_min, x_max])
            return PiecewiseClockFit(knots, offset + drift * knots, residual, points, True, "",
                                     STRAIGHT_LINE)
    except (KeyError, TypeError, ValueError):
        pass
    return None


def fit_ni_to_host_timeline(stamps: BlockStamps, sample_rate: float, *,
                            knot_seconds: float = KNOT_SECONDS,
                            record_knot_seconds: float = RECORD_KNOT_SECONDS) -> PiecewiseClockFit:
    """NI sample time -> host perf_counter: through CLOCK_MONOTONIC_RAW when the
    stamps carry it, else the piecewise envelope of the perf stamps.

    RAW is not slewed by NTP, so NI-to-RAW is the smooth relation of two
    oscillators and the envelope fits it with nothing to hide; RAW-to-perf is
    then exact at every stamp (both were read within a microsecond of each
    other) and is interpolated between them, kinks and all. The result is
    tabulated every record_knot_seconds so it can be stored and rebuilt. With
    no RAW column (an older session, or a host without the clock) the perf
    stamps are fitted directly, window by window.
    """
    finite = numpy.isfinite(stamps.raw) & numpy.isfinite(stamps.perf)
    raw_reason = ""
    if int(finite.sum()) >= MIN_FIT_POINTS:
        ni_to_raw = fit_piecewise_envelope(stamps.end_index[finite], stamps.raw[finite], sample_rate,
                                           knot_seconds=knot_seconds)
        if ni_to_raw.valid:
            raw, perf = stamps.raw[finite], stamps.perf[finite]
            order = numpy.argsort(raw, kind="stable")
            raw, offset = raw[order], (perf - raw)[order]
            knots = numpy.arange(ni_to_raw.x_min, ni_to_raw.x_max, record_knot_seconds)
            if knots.size and knots[-1] > ni_to_raw.x_max - BRIDGE_TAIL_SECONDS:
                knots = knots[:-1]
            knots = numpy.append(knots, ni_to_raw.x_max)
            if knots.size < 2:
                knots = numpy.array([ni_to_raw.x_min, ni_to_raw.x_max])
            knot_raw = ni_to_raw.map(knots)
            # perf - raw at each knot: the median of the pairs around it, not a
            # line between the two nearest, so a single late pair is ignored.
            low = numpy.searchsorted(raw, knot_raw - BRIDGE_WINDOW_SECONDS, side="left")
            high = numpy.searchsorted(raw, knot_raw + BRIDGE_WINDOW_SECONDS, side="right")
            nearest = numpy.clip(numpy.searchsorted(raw, knot_raw), 0, raw.size - 1)
            knot_offset = numpy.array([
                float(numpy.median(offset[a:b])) if b > a else float(offset[n])
                for a, b, n in zip(low, high, nearest)
            ])
            return PiecewiseClockFit(knots, knot_raw + knot_offset, ni_to_raw.residual_rms,
                                     ni_to_raw.points, True, "", RAW_BRIDGED)
        raw_reason = f"NI-to-RAW: {ni_to_raw.reason}; "
    fit = fit_piecewise_envelope(stamps.end_index, stamps.perf, sample_rate, knot_seconds=knot_seconds)
    if fit.valid or not raw_reason:
        return fit
    return dataclasses.replace(fit, reason=raw_reason + fit.reason)


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
    if not math.isfinite(tolerance):
        reason = "the frame period is unknown, so the residual cannot be judged"
    elif not math.isfinite(median_residual):
        reason = "no frame of the chosen shift has a transition"
    elif residual_ambiguous:
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
