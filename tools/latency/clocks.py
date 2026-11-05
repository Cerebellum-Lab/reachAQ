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
import statistics
import time
from fractions import Fraction
from typing import Optional, Tuple

import numpy

MIN_FIT_POINTS = 100
NI_MAX_RESIDUAL_SECONDS = 0.0005
WALL_SEGMENT_JUMP_SECONDS = 0.001
ENVELOPE_BINS = 20
# The chance, per check and per session, that a pairing check flags a session
# whose pairing is sound. It sets every statistical limit on a pairing: the
# arrival margin (coverage_factor() standard errors) and how long a run of
# frames voting another shift must be to count as a slip (find_slip).
FALSE_ALARM_PROBABILITY = 1e-6
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
# no segment is milliseconds long with a drift made of noise. RAW stamps that
# span less than BRIDGE_MIN_RAW_COVERAGE of the stamps' span are not bridged
# over the rest: the bridge would carry the last offset it saw across slews it
# never observed.
BRIDGE_WINDOW_SECONDS = 1.0
BRIDGE_TAIL_SECONDS = 5.0
BRIDGE_MIN_RAW_COVERAGE = 0.9
# A camera clock latch is one USB round trip: christielab10's BFS-U3-16S2M
# brackets it in 0.14-0.58 ms inside the running app (60 latches, median
# 0.27 ms) and in about 0.20 ms with the app closed; the widest in the
# 2026-11-03 sweep was 1.78 ms. No width is refused: fit_latches fits through
# the brackets themselves and says how far its map could be wrong, and
# latch_gate_ok judges that against the frame period.
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
    clock: str = "perf"
    """A latch map's host clock: "perf" or "raw"."""
    uncertainty: float = math.nan
    """A latch map's bound, in seconds over x_min..x_max, on how far any map of its
    kind (a line, or an offset at drift 1) through its latches' brackets is from it."""
    dropped: tuple = ()
    """Latch rows a latch map left out: no map of its kind through the others' brackets
    passes theirs."""

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


# eq=False on the two classes that hold arrays: a field-wise == raises on an
# ndarray, and the field-wise hash a frozen dataclass would add is unhashable.
@dataclasses.dataclass(frozen=True, eq=False)
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


def _sample_rate_problem(sample_rate) -> str:
    """Why sample_rate cannot turn sample indices into seconds, or ''."""
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError):
        rate = math.nan
    return "" if math.isfinite(rate) and rate > 0 else f"sample rate {sample_rate} is not a positive number"


def _read_pairs(end_index, seen_perf, sample_rate):
    """(NI seconds, host seconds, why not): the read pairs a fit can use.

    A pair with a value that is not finite is left out, as a NaN stamp always
    has been. A sample rate that is not a positive number, or arrays of
    different lengths, leave nothing to fit: the reason is returned for the
    caller to make an invalid fit of, never raised.
    """
    problem = _sample_rate_problem(sample_rate)
    if problem:
        return None, None, problem
    index = numpy.asarray(end_index, dtype=numpy.float64)
    seen = numpy.asarray(seen_perf, dtype=numpy.float64)
    if index.shape != seen.shape:
        return None, None, f"{index.size} sample indices against {seen.size} stamps"
    t = index / float(sample_rate)
    finite = numpy.isfinite(t) & numpy.isfinite(seen)
    return t[finite], seen[finite], ""


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
    t, seen, problem = _read_pairs(end_index, seen_perf, sample_rate)
    if problem:
        return invalid_fit(problem)
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


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _interp_extrapolate(x, xs, ys):
    """Piecewise-linear interpolation through (xs, ys), continued with the end
    segments outside them; xs strictly increasing with at least two points."""
    x = numpy.asarray(x, dtype=numpy.float64)
    j = numpy.clip(numpy.searchsorted(xs, x, side="right") - 1, 0, xs.size - 2)
    slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
    return ys[j] + slope * (x - xs[j])


@dataclasses.dataclass(frozen=True, eq=False)
class PiecewiseClockFit:
    """NI seconds -> host seconds through a continuous piecewise-linear table.

    The same interface as ClockFit (map, as_record, valid, reason,
    residual_rms, points, offset, drift, x_min, x_max) plus inverse() and the
    per-segment drifts. offset and drift describe the straight line through
    the end knots, so a reader of the old record keys still gets a usable
    summary of the whole session.

    A RAW-bridged fit also keeps knot_raw, the NI-to-RAW envelope at the same
    knots, which to_raw() interpolates: NI time on CLOCK_MONOTONIC_RAW, which
    no NTP slew touches. Any other fit, and a record stored before the column
    was, has none (has_raw is False).
    """
    knots: numpy.ndarray
    knot_host: numpy.ndarray
    residual_rms: float
    points: int
    valid: bool
    reason: str = ""
    method: str = PIECEWISE_ENVELOPE
    knot_raw: Optional[numpy.ndarray] = None

    @property
    def has_raw(self) -> bool:
        """Whether to_raw() can map: a valid table with a RAW value at every knot."""
        return bool(self.valid and self.knot_raw is not None and self.knots.size >= 2
                    and self.knot_raw.shape == self.knots.shape)

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

    def to_raw(self, x):
        """NI seconds -> CLOCK_MONOTONIC_RAW seconds; all NaN without a RAW view (has_raw).

        Late by the delivery floor, as map() is: both rest on the same NI-to-RAW
        envelope, and map() adds only the perf-minus-RAW offset to it.
        """
        x = numpy.asarray(x, dtype=numpy.float64)
        if not self.has_raw:
            return numpy.full(x.shape, math.nan)
        return _interp_extrapolate(x, self.knots, self.knot_raw)

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
            if self.knot_raw is not None:
                record["knotRaw"] = [_finite_or_none(value) for value in self.knot_raw]
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
    continuous at every knot. The bias is the same as fit_ni_to_host's within
    one window and with no NTP poll in it: mapped times are late by about the
    minimum delivery delay. Across polls this is not late-never-early: the
    host clock's slew curves inside a window and a straight segment through
    the envelope points cannot follow it, so a mapped time can be early. A 4 h
    session of NTP polls measured it up to 46 us early and 260 us late at most,
    inside the 0.36 ms budget. The RAW bridge (fit_ni_to_host_timeline), which
    fits the envelope on the unslewed RAW clock and adds the slew back from the
    stamp pairs, is the one that maps late and never early.
    """
    t, seen, problem = _read_pairs(end_index, seen_perf, sample_rate)
    if problem:
        return invalid_piecewise_fit(problem)
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
    knot_host = coefficients + y0
    valid = rms <= NI_MAX_RESIDUAL_SECONDS
    reason = "" if valid else f"residual rms {rms:.6f} s above {NI_MAX_RESIDUAL_SECONDS:.6f} s"
    drifts = numpy.diff(knot_host) / numpy.diff(knots)
    if valid and not numpy.all(drifts > 0):
        # Host time that runs backwards across a window is garbage stamps, and
        # inverse() would not be a function.
        valid = False
        reason = f"a knot window runs backwards (segment drift {numpy.min(drifts):.6f}, not above 0)"
    return PiecewiseClockFit(
        knots=knots, knot_host=knot_host, residual_rms=rms, points=int(x.size),
        valid=bool(valid), reason=reason,
    )


def fit_from_record(record) -> Optional[PiecewiseClockFit]:
    """The fit a stored record describes, or None when there is none worth using.

    A record with a knot table (PiecewiseClockFit.as_record) comes back as it
    was, provided the whole table is there: both columns finite and strictly
    increasing, as many knots as knotCount says. A straight-line record
    (ClockFit.as_record: no knot keys at all) and a two-knot record stored
    without its table (as_record(include_knots=False)) become a two-knot table
    over the fitted span, which is exact for both. A record that needs a table
    and has none, or a damaged one, is None and is never replaced by the line
    through its summary keys: after an hour that line is milliseconds off,
    unflagged. Anything invalid, malformed or absent is None, so a caller falls
    back rather than mapping through nonsense.

    The RAW column (knotRaw) comes back with the table when it is whole: one
    finite value per knot, strictly increasing. A record without it, which is
    every record written before it was stored, or with a damaged one, rebuilds
    its perf table all the same, without a RAW view (has_raw is False).
    """
    if not isinstance(record, dict) or not record.get("valid"):
        return None
    try:
        count = record.get("knotCount")
        count = None if count is None else int(count)
        stored_knots, stored_host = record.get("knots") or (), record.get("knotHost") or ()
        residual = float(record.get("residual_rms") if record.get("residual_rms") is not None else math.nan)
        points = int(record.get("points") or 0)
        if stored_knots or stored_host or (count is not None and count > 2):
            knots = numpy.asarray(stored_knots, dtype=numpy.float64)
            host = numpy.asarray(stored_host, dtype=numpy.float64)
            # Both columns strictly increasing, or inverse() would not be a function.
            if (knots.size >= 2 and host.size == knots.size and count in (None, knots.size)
                    and numpy.isfinite(knots).all() and numpy.isfinite(host).all()
                    and numpy.all(numpy.diff(knots) > 0) and numpy.all(numpy.diff(host) > 0)):
                return PiecewiseClockFit(knots, host, residual, points, True, "",
                                         str(record.get("method") or PIECEWISE_ENVELOPE),
                                         _stored_knot_raw(record.get("knotRaw"), knots))
            return None
        offset, drift = float(record["offset"]), float(record["drift"])
        x_min, x_max = float(record["x_min"]), float(record["x_max"])
        if (all(math.isfinite(value) for value in (offset, drift, x_min, x_max)) and x_max > x_min
                and drift > 0):
            knots = numpy.array([x_min, x_max])
            return PiecewiseClockFit(knots, offset + drift * knots, residual, points, True, "",
                                     STRAIGHT_LINE)
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _stored_knot_raw(stored, knots) -> Optional[numpy.ndarray]:
    """A record's RAW column at knots, or None unless it is whole: one finite,
    strictly increasing value per knot."""
    if stored is None:
        return None
    try:
        raw = numpy.asarray(stored, dtype=numpy.float64)
    except (TypeError, ValueError):
        return None
    if (raw.shape != knots.shape or not numpy.isfinite(raw).all()
            or not numpy.all(numpy.diff(raw) > 0)):
        return None
    return raw


def fit_ni_to_host_timeline(stamps: BlockStamps, sample_rate: float, *,
                            knot_seconds: float = KNOT_SECONDS,
                            record_knot_seconds: float = RECORD_KNOT_SECONDS) -> PiecewiseClockFit:
    """NI sample time -> host perf_counter: through CLOCK_MONOTONIC_RAW when the
    stamps carry it, else the piecewise envelope of the perf stamps.

    RAW is not slewed by NTP, so NI-to-RAW is the smooth relation of two
    oscillators and the envelope fits it with nothing to hide. What NTP did to
    the host clock is perf minus RAW, read at a knot every record_knot_seconds
    (10 s) as the median of the stamp pairs within BRIDGE_WINDOW_SECONDS (1 s)
    of it, and interpolated between knots. The median is what keeps a pair
    whose reader was preempted between its two clock reads from moving a knot;
    its price is that a kink in the slew is smoothed over the window, to about
    50 us at 10 s knots. The table is stored and rebuilt as it is, with the
    NI-to-RAW values at its knots beside it (to_raw).

    When RAW cannot be used (no RAW column, an older session or a host without
    the clock; too few stamps; stamps over too little of the session; a NI-to-
    RAW fit that fails; a bridge that is not increasing) the perf stamps are
    fitted directly, window by window, and a RAW that was present and failed
    is named at the front of the fit's reason.
    """
    problem = _sample_rate_problem(sample_rate)
    if problem:
        return invalid_piecewise_fit(problem)
    finite = numpy.isfinite(stamps.raw) & numpy.isfinite(stamps.perf)
    usable = int(finite.sum())
    raw_reason = ""
    if usable >= MIN_FIT_POINTS:
        bridged, raw_reason = _bridge_through_raw(stamps, finite, sample_rate, knot_seconds,
                                                  record_knot_seconds)
        if bridged is not None:
            return bridged
    elif usable:
        raw_reason = f"RAW on {usable} stamps, need {MIN_FIT_POINTS}; "
    fit = fit_piecewise_envelope(stamps.end_index, stamps.perf, sample_rate, knot_seconds=knot_seconds)
    if not raw_reason:
        return fit
    return dataclasses.replace(fit, reason=(raw_reason + fit.reason).rstrip("; "))


def _bridge_through_raw(stamps: BlockStamps, finite, sample_rate: float, knot_seconds: float,
                        record_knot_seconds: float):
    """(the RAW-bridged fit, "") or (None, why the bridge cannot be used)."""
    end_index = numpy.asarray(stamps.end_index, dtype=numpy.float64)
    stamped = end_index[numpy.isfinite(stamps.perf)] / float(sample_rate)
    covered = end_index[finite] / float(sample_rate)
    span = float(stamped.max() - stamped.min())
    if span > 0 and covered.max() - covered.min() < BRIDGE_MIN_RAW_COVERAGE * span:
        return None, (f"RAW covers {(covered.max() - covered.min()) / span:.0%} of the {span:.0f} s "
                      f"of stamps, need {BRIDGE_MIN_RAW_COVERAGE:.0%}; ")
    ni_to_raw = fit_piecewise_envelope(stamps.end_index[finite], stamps.raw[finite], sample_rate,
                                       knot_seconds=knot_seconds)
    if not ni_to_raw.valid:
        return None, f"NI-to-RAW: {ni_to_raw.reason}; "
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
    # perf - raw at each knot: the median of the pairs around it, not a line
    # between the two nearest, so a single late pair is ignored. A knot with no
    # pair that close takes the closer of the pairs on either side of it.
    low = numpy.searchsorted(raw, knot_raw - BRIDGE_WINDOW_SECONDS, side="left")
    high = numpy.searchsorted(raw, knot_raw + BRIDGE_WINDOW_SECONDS, side="right")
    after = numpy.clip(numpy.searchsorted(raw, knot_raw), 0, raw.size - 1)
    before = numpy.clip(after - 1, 0, raw.size - 1)
    nearest = numpy.where(numpy.abs(raw[before] - knot_raw) <= numpy.abs(raw[after] - knot_raw),
                          before, after)
    knot_offset = numpy.array([
        float(numpy.median(offset[a:b])) if b > a else float(offset[n])
        for a, b, n in zip(low, high, nearest)
    ])
    knot_host = knot_raw + knot_offset
    if not numpy.all(numpy.diff(knot_host) > 0):
        # Corrupt perf stamps, not a slew: NTP moves the clock by at most a few
        # ms a second, and inverse() needs a table that only goes forward.
        return None, "RAW bridge is not increasing; "
    return PiecewiseClockFit(knots, knot_host, ni_to_raw.residual_rms, ni_to_raw.points, True, "",
                             RAW_BRIDGED, knot_raw), ""


@dataclasses.dataclass(frozen=True)
class Pairing:
    shift: int
    """Transition number = frame id - shift."""
    median_lag: float
    agreement: float
    """Share of frames voting the chosen shift: a diagnostic, not a gate."""
    ambiguous: bool
    reason: str = ""
    method: str = HOST_ARRIVAL
    median_residual: float = math.nan
    """Camera-clock method: transition minus expected host time, over the chosen shift."""
    slip: Optional[dict] = None
    """Where the votes change shift, as find_slip gives it; None when they do not."""
    limit: float = math.nan
    """What the verdict was judged against, in seconds: the arrival margin or the latch tolerance."""
    clock: str = "perf"
    """The host clock the frames and transitions were compared on: "perf" or "raw"."""


def coverage_factor() -> float:
    """How many standard errors a two-sided test must allow to false-alarm at
    FALSE_ALARM_PROBABILITY: 4.89 at 1e-6. A Gaussian quantile on a robust
    spread is an approximation, not a guarantee."""
    return statistics.NormalDist().inv_cdf(1 - FALSE_ALARM_PROBABILITY / 2)


def robust_spread(values) -> float:
    """1.4826 times the median absolute deviation of the finite values: the
    standard deviation of a Gaussian, which a few stray values cannot move.
    0.0 with fewer than two values."""
    values = numpy.asarray(values, dtype=numpy.float64)
    values = values[numpy.isfinite(values)]
    if values.size < 2:
        return 0.0
    return 1.4826 * float(numpy.median(numpy.abs(values - numpy.median(values))))


def find_slip(frame_ids, votes, majority: int) -> Optional[dict]:
    """Where the pairing changes shift, or None.

    votes[i] is the shift frame_ids[i]'s own transition implies, in frame-id
    order, and majority is the shift most of them name. A lost or spurious NI
    transition renumbers every later frame by one, so every frame from there on
    votes the same other shift: a long run. A stray vote is a run of one.
    The stray rate is (non-majority votes + 1) / (n + 2), and the longest run L
    of one non-majority shift is a slip when n * rate ** L, the number of runs
    that long to expect among n votes at that rate, is under
    FALSE_ALARM_PROBABILITY.

    The record names the frame where the votes change: the first of the run,
    or, when the run opens the session, the frame after it, which is the first
    to vote the majority. frames is the run's length. Frame-id gaps are not
    runs: a dropped frame had its exposure edge and still votes the same shift.
    """
    votes = numpy.asarray(votes, dtype=numpy.int64)
    frame_ids = numpy.asarray(frame_ids, dtype=numpy.int64)
    n = int(votes.size)
    others = int(numpy.count_nonzero(votes != majority))
    if not others:
        return None
    starts = numpy.concatenate(([0], numpy.flatnonzero(numpy.diff(votes)) + 1))
    lengths = numpy.diff(numpy.append(starts, n))
    longest = int(numpy.argmax(numpy.where(votes[starts] != majority, lengths, 0)))
    start, length = int(starts[longest]), int(lengths[longest])
    stray_rate = (others + 1) / (n + 2)
    if not n * stray_rate ** length < FALSE_ALARM_PROBABILITY:
        return None
    shift = int(votes[start])
    opens_session = start == 0 and length < n
    return {
        "frameId": int(frame_ids[start + length] if opens_session else frame_ids[start]),
        "frames": length,
        "fromShift": shift if opens_session else int(majority),
        "toShift": int(majority) if opens_session else shift,
    }


def _slip_reason(slip: dict, majority: int) -> str:
    stray = slip["toShift"] if slip["fromShift"] == majority else slip["fromShift"]
    return (f"slip at frame {slip['frameId']}: {slip['frames']} frames vote shift {stray}, "
            f"not {majority}")


def frames_within_transitions(arrival_perf, transition_perf, frame_period) -> numpy.ndarray:
    """Mask of the frames that arrived while the NI spool held transitions.

    The camera runs on before the NI task starts and after it stops; those
    frames have no transition of their own, and each would pair with an edge it
    does not belong to, a stray vote in either pairing. A frame arrives after
    its exposure edge, so one inside the spool arrives no later than a period
    past the last transition (with no period known, no later than the last).

    The mask is built on arrival_perf for both pairings, although the
    camera-clock pairing matches by camera timestamp. That field is always
    present in CAMERA_FRAME_DTYPE and camera_base falls it back to the
    frame's own host stamp, so only a frame with no arrival yet (-inf)
    is left out of the latch pairing for want of one.
    """
    arrival = numpy.asarray(arrival_perf, dtype=numpy.float64)
    last = float(transition_perf[-1]) + (0.0 if frame_period is None else float(frame_period))
    return numpy.isfinite(arrival) & (arrival >= transition_perf[0]) & (arrival <= last)


def choose_pairing(frame_ids, arrival_perf, transition_perf, frame_period: float, *,
                   ni_residual_rms: float) -> Pairing:
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

    The margin is measured: coverage_factor() standard errors of the median
    lag, which are the NI-to-host fit's residual (ni_residual_rms) and the
    median's own error from the spread of the lags (1.2533, the square root
    of pi/2, is a median's error over a mean's). A shift that changes partway
    through (find_slip) is ambiguous too, and is reported where it changes.
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
    order = numpy.argsort(ids[found], kind="stable")
    voters, votes = ids[found][order], (ids[found] - candidate[found])[order]
    values, counts = numpy.unique(votes, return_counts=True)
    shift = int(values[numpy.argmax(counts)])
    agreement = float(counts.max() / votes.size)
    numbers = ids - shift
    inside = (numbers >= 0) & (numbers < transition_perf.size)
    lag = arrival[inside] - transition_perf[numbers[inside]]
    median_lag = float(numpy.median(lag)) if lag.size else math.nan
    margin = math.nan if not lag.size else coverage_factor() * math.hypot(
        ni_residual_rms, 1.2533 * robust_spread(lag) / math.sqrt(lag.size))
    lag_ambiguous = not (margin <= median_lag <= frame_period - margin)
    slip = find_slip(voters, votes, shift)
    problems = []
    if slip is not None:
        problems.append(_slip_reason(slip, shift))
    if lag_ambiguous:
        problems.append(
            f"median arrival lag {median_lag * 1e3:.3f} ms is within the measured margin "
            f"{margin * 1e3:.3f} ms of 0 or one frame period"
        )
    return Pairing(shift, median_lag, agreement, bool(problems), "; ".join(problems),
                   slip=slip, limit=margin)


_LATCH_BRACKETS = {"perf": ("perf_before", "perf_after"), "raw": ("raw_before", "raw_after")}


def _latch_brackets(latches, clock: str):
    """(row numbers, camera ns, bracket start, bracket end) of the latches whose
    bracket on clock is finite and does not end before it starts. A latch table
    without that clock's pair of fields has none."""
    fields = _LATCH_BRACKETS.get(clock)
    names = getattr(getattr(latches, "dtype", None), "names", None) or ()
    if latches is None or fields is None or not set(fields) <= set(names) or len(latches) == 0:
        return (numpy.empty(0, dtype=numpy.int64), numpy.empty(0, dtype=numpy.int64),
                numpy.empty(0), numpy.empty(0))
    before = numpy.asarray(latches[fields[0]], dtype=numpy.float64)
    after = numpy.asarray(latches[fields[1]], dtype=numpy.float64)
    width = after - before
    rows = numpy.flatnonzero(numpy.isfinite(width) & (width >= 0))
    camera_ns = numpy.asarray(latches["camera_ns"], dtype=numpy.int64)
    return rows, camera_ns[rows], before[rows], after[rows]


def usable_latches(latches, *, clock: str) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """(camera seconds, host time, half the bracket) of each clock latch on clock.

    clock is "perf" (perf_before, perf_after) or "raw" (raw_before, raw_after).
    The latch happened somewhere inside its bracket, so the midpoint is its
    host time and half the width its uncertainty. Only the physical checks
    apply: a bracket that is not finite, or ends before it starts, is left out.
    Rows with the RAW pair NaN, or a table without it (older streams), have no
    RAW latch.
    """
    _, camera_ns, before, after = _latch_brackets(latches, clock)
    return camera_ns / 1e9, (before + after) / 2, (after - before) / 2


def fit_latches(latches, *, clock: str) -> ClockFit:
    """Camera timestamp (s) -> host time on clock ("perf" or "raw"), from the camera clock latches.

    A latch says the camera clock read camera_ns somewhere inside the latch's
    host bracket, which is widened by the host clock's resolution
    (time.get_clock_info, measured) so that a zero-width bracket still holds
    it. A straight map must pass through every bracket, and no other test is
    applied. When no line passes through them all, the largest set of latches
    one line passes through is kept (ties: the narrower brackets), and each
    latch left out is in dropped and named in the reason. The set is found
    exactly, by trying every line through two bracket ends
    (_largest_consistent_set).

    Kept latches more than LATCH_DRIFT_SPAN_SECONDS apart give a line, which
    takes up the camera's drift. The lines through all their brackets range,
    at each camera time, between a highest and a lowest; the fit runs through
    the middle of that range at the first and the last latch, and uncertainty
    is half the range where it is widest. The range's width is convex in time,
    so it is widest at one end, and every straight map through the brackets
    lies within uncertainty of the fit between x_min and x_max.

    When that set spans no more than LATCH_DRIFT_SPAN_SECONDS it cannot
    measure drift, and the map is an offset at drift 1. A free slope says
    little within one mark (any two latches there have a line through both),
    so the latches are chosen again under the map actually used
    (_largest_common_offset): each bracket minus its camera time is the range
    of offsets it allows, and the largest set of those with a common offset is
    kept (ties: the narrower brackets), the rest dropped and named. The offset
    is the middle of their overlap and uncertainty half its width.

    residual_rms is the rms of the kept latches' midpoints about the map, a
    diagnostic that caps nothing. Either way x_min and x_max are the kept
    latches' span, and only timestamps latch_window() admits should be mapped.
    valid needs one usable latch.
    """
    if clock not in _LATCH_BRACKETS:
        return dataclasses.replace(invalid_fit(f"unknown clock {clock!r}, not perf or raw"),
                                   clock=str(clock))
    rows, camera_ns, before, after = _latch_brackets(latches, clock)
    if not rows.size:
        return dataclasses.replace(invalid_fit(_no_latch_reason(latches, clock)), clock=clock)
    resolution = Fraction(time.get_clock_info("perf_counter").resolution)
    x = [Fraction(int(value), 1_000_000_000) for value in camera_ns]
    low = [Fraction(float(value)) - resolution for value in before]
    high = [Fraction(float(value)) + resolution for value in after]
    kept, lines = _largest_consistent_set(x, low, high)
    inside = [k for k in range(len(x)) if kept[k]]
    first, last = min(x[k] for k in inside), max(x[k] for k in inside)
    if last - first > LATCH_DRIFT_SPAN_SECONDS:
        model = "line"
        at_first = [value + slope * (first - start) for start, value, slope in lines]
        at_last = [value + slope * (last - start) for start, value, slope in lines]
        middle_first = (max(at_first) + min(at_first)) / 2
        middle_last = (max(at_last) + min(at_last)) / 2
        drift = (middle_last - middle_first) / (last - first)
        offset = middle_first - drift * first
        uncertainty = max(max(at_first) - min(at_first), max(at_last) - min(at_last)) / 2
    else:
        model = "offset at drift 1"
        kept, offset, uncertainty = _largest_common_offset(x, low, high)
        inside = [k for k in range(len(x)) if kept[k]]
        first, last = min(x[k] for k in inside), max(x[k] for k in inside)
        drift = Fraction(1)
    dropped = tuple(int(rows[k]) for k in range(len(x)) if not kept[k])
    residual = [float((Fraction(float(before[k])) + Fraction(float(after[k]))) / 2
                      - offset - drift * x[k]) for k in inside]
    return ClockFit(
        offset=float(offset),
        drift=float(drift),
        residual_rms=math.sqrt(sum(value * value for value in residual) / len(residual)),
        points=len(inside),
        valid=True,
        reason=_dropped_reason(dropped, len(inside), model),
        x_min=float(first),
        x_max=float(last),
        clock=clock,
        uncertainty=float(uncertainty),
        dropped=dropped,
    )


def _largest_consistent_set(x, low, high):
    """(which intervals one line passes through together, the lines found through exactly those).

    x, low and high are Fractions: interval k is [low[k], high[k]] at x[k].
    Each line is (x0, value at x0, slope). Exact, in O(n^3): the lines (an
    offset and a slope) through every interval of a set form a convex polygon,
    bounded once two of the intervals are at different x, and each corner of
    it lies on the boundaries of two intervals at different x, which is a line
    through an end of each. So of the lines through two interval ends at
    different x, the one through the most intervals finds the largest set any
    line passes through, and those through exactly that set include every
    corner of its polygon, where a line's value at any x is highest and
    lowest. Ties go to the set whose intervals are narrower in sum. With every
    interval at one x there is no such line, and an interval end inside the
    most intervals stands in for one, of slope 0.
    """
    count = len(x)
    width = [high[k] - low[k] for k in range(count)]
    lines = []
    for i in range(count):
        offsets = [x[k] - x[i] for k in range(count)]
        for j in range(i + 1, count):
            if not offsets[j]:
                continue
            for start in (low[i], high[i]):
                for end in (low[j], high[j]):
                    slope = (end - start) / offsets[j]
                    passes = tuple(low[k] <= start + slope * offsets[k] <= high[k]
                                   for k in range(count))
                    lines.append(((x[i], start, slope), passes))
    if not lines:
        for value in low + high:
            lines.append(((x[0], value, Fraction(0)),
                          tuple(low[k] <= value <= high[k] for k in range(count))))
    _, best = max(lines, key=lambda line: (sum(line[1]),
                                           -sum(w for w, on in zip(width, line[1]) if on)))
    return best, [line for line, passes in lines if passes == best]


def _largest_common_offset(x, low, high):
    """(which intervals share an offset at drift 1, the middle of their overlap, half its width).

    x, low and high are as _largest_consistent_set takes them. At drift 1,
    interval k allows the offsets [low[k] - x[k], high[k] - x[k]]. A set of these with a common
    point has the largest of their lower ends among its common points, so of
    the lower ends, the one inside the most intervals finds the largest such
    set, exactly, in O(n^2). Ties go to the set whose intervals are narrower
    in sum.
    """
    count = len(x)
    start = [low[k] - x[k] for k in range(count)]
    end = [high[k] - x[k] for k in range(count)]
    width = [high[k] - low[k] for k in range(count)]
    sets = [tuple(start[k] <= value <= end[k] for k in range(count)) for value in start]
    best = max(sets, key=lambda inside: (sum(inside),
                                         -sum(w for w, on in zip(width, inside) if on)))
    common_start = max(start[k] for k in range(count) if best[k])
    common_end = min(end[k] for k in range(count) if best[k])
    return best, (common_start + common_end) / 2, (common_end - common_start) / 2


def _no_latch_reason(latches, clock: str) -> str:
    rows = 0 if latches is None else len(latches)
    names = getattr(getattr(latches, "dtype", None), "names", None) or ()
    before, after = _LATCH_BRACKETS[clock]
    if rows and not {before, after} <= set(names):
        return f"no usable clock latch: {rows} rows, none with {before} and {after} fields"
    return f"no usable clock latch: {rows} rows, none with a finite, non-negative {clock} bracket"


def _dropped_reason(dropped: tuple, kept: int, model: str) -> str:
    """model names the map: "line", or "offset at drift 1"."""
    if not dropped:
        return ""
    if len(dropped) == 1:
        return (f"latch {dropped[0]} left out: no {model} through the other {kept} latches' "
                f"brackets passes through its own")
    rows = ", ".join(str(row) for row in dropped)
    return (f"latches {rows} left out: no {model} through the {kept} kept latches' brackets "
            f"passes through any of theirs")


def latch_gate_ok(fit: ClockFit, ni_residual_rms: float, frame_period: float) -> bool:
    """Whether a latch map is trusted to choose the transitions: its uncertainty
    plus coverage_factor() NI-to-host residuals (ni_residual_rms) is under a
    quarter frame period, the bound latch_tolerance() also puts on the residual
    (below its cap). At 150 fps the quarter is 1.67 ms; the worst session of
    the 2026-11-03 sweep needs 0.89 + 4.89 x 0.039 = 1.08 ms. False when the
    fit's uncertainty, the residual or the period is not finite, which covers
    an invalid fit.
    """
    try:
        values = (float(fit.uncertainty), float(ni_residual_rms), float(frame_period))
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    uncertainty, residual, period = values
    return uncertainty + coverage_factor() * residual < period / 4


def latch_gate_reason(fit: ClockFit, ni_residual_rms: float, frame_period: float) -> str:
    """Why latch_gate_ok() does not trust fit, or "" when it does."""
    if latch_gate_ok(fit, ni_residual_rms, frame_period):
        return ""
    if not getattr(fit, "valid", False):
        return getattr(fit, "reason", "") or "no latch map"
    residual, period = _float_or_nan(ni_residual_rms), _float_or_nan(frame_period)
    if not math.isfinite(period):
        return "the frame period is unknown, so the latch map cannot be judged"
    if not math.isfinite(residual):
        return "the NI fit's residual is unknown, so the latch map cannot be judged"
    spread = coverage_factor() * residual
    return (f"latch mapping uncertainty {fit.uncertainty * 1e3:.3f} ms plus "
            f"{coverage_factor():.2f} NI residuals ({spread * 1e3:.3f} ms) "
            f"exceeds a quarter frame period ({period / 4 * 1e3:.3f} ms)")


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


def latch_clock(latches, ni_fit) -> str:
    """The host clock a camera-clock pairing runs on, "raw" or "perf": one rule for every caller.

    "raw" needs CLOCK_MONOTONIC_RAW on both sides: the NI fit has a RAW view
    (has_raw), every latch usable on perf is usable on RAW too, and the RAW
    latches make a valid fit. RAW on only some of the latches, one burst say,
    is "perf": a RAW map through those alone would give up latches perf keeps,
    and with a whole burst gone, the drift. Anything else is "perf", and this
    never raises: latches that cannot be read here say so when the pairing
    reads them.
    """
    try:
        if not getattr(ni_fit, "has_raw", False):
            return "perf"
        on_raw = usable_latches(latches, clock="raw")[0].size
        if on_raw != usable_latches(latches, clock="perf")[0].size:
            return "perf"
        return "raw" if fit_latches(latches, clock="raw").valid else "perf"
    except Exception:
        return "perf"


def choose_pairing_by_camera_clock(frame_ids, camera_ts_ns, latches, transition_times,
                                   frame_period: float, *, ni_residual_rms: float,
                                   clock: str) -> Pairing:
    """Which exposure transition belongs to which frame, decided by the camera's clock.

    The latches place camera time on the host clock named by clock ("perf" or
    "raw", as fit_latches reads them), and transition_times are the NI
    transitions on that same clock. So a frame's camera timestamp says when on
    that clock it was exposed, however long it then waited to be delivered.
    Each frame's candidate is the transition nearest that time, and the shift
    most frames agree on wins.

    The median residual (transition minus expected time, over the chosen shift)
    is where the camera stamps a frame relative to the NI edge, plus the NI-to-
    host bias, plus the latch map's error. All of it must stay well under half a
    period for the nearest transition to be the right one, and a bias near a
    whole period would pair a frame off with a small residual, so a residual
    over latch_tolerance() (a quarter period, at most 2 ms), or a shift that
    changes partway through (find_slip), marks the pairing ambiguous. So does a
    latch map too uncertain to choose a transition by (latch_gate_ok: the map's
    uncertainty plus coverage_factor() times ni_residual_rms, the NI-to-host
    fit's residual, against a quarter period); the shift it found is still
    reported. The lag is not measured here: median_lag is NaN. The residual
    tolerance is in Pairing.limit.

    The limit that sets: a Blackfly S stamps a frame at the end of its exposure
    and the NI edge marks its start, so the residual approaches minus the
    exposure time. An exposure longer than about 1.8 ms therefore reads
    ambiguous, visibly, rather than being paired on a guess.
    """
    fit = fit_latches(latches, clock=clock)
    if not fit.valid:
        return Pairing(0, math.nan, 0.0, True, fit.reason, method=CAMERA_LATCH, clock=clock)
    frame_ids = numpy.asarray(frame_ids, dtype=numpy.int64)
    camera = numpy.asarray(camera_ts_ns, dtype=numpy.float64) / 1e9
    transition_times = numpy.asarray(transition_times, dtype=numpy.float64)
    keep = numpy.isfinite(camera) & latch_window(fit, camera)
    if not keep.any() or transition_times.size == 0:
        return Pairing(0, math.nan, 0.0, True, "no frames near a clock latch or no transitions",
                       method=CAMERA_LATCH, clock=clock)
    ids = frame_ids[keep]
    expected = fit.map(camera[keep])
    later = numpy.clip(numpy.searchsorted(transition_times, expected), 0,
                       transition_times.size - 1)
    earlier = numpy.clip(later - 1, 0, transition_times.size - 1)
    nearest = numpy.where(numpy.abs(transition_times[later] - expected)
                          < numpy.abs(transition_times[earlier] - expected), later, earlier)
    order = numpy.argsort(ids, kind="stable")
    voters, votes = ids[order], (ids - nearest)[order]
    values, counts = numpy.unique(votes, return_counts=True)
    shift = int(values[numpy.argmax(counts)])
    agreement = float(counts.max() / votes.size)
    numbers = ids - shift
    inside = (numbers >= 0) & (numbers < transition_times.size)
    residual = transition_times[numbers[inside]] - expected[inside]
    median_residual = float(numpy.median(residual)) if residual.size else math.nan
    tolerance = latch_tolerance(frame_period)
    residual_ambiguous = not abs(median_residual) <= tolerance
    gate_ok = latch_gate_ok(fit, ni_residual_rms, frame_period)
    slip = find_slip(voters, votes, shift)
    problems = []
    if slip is not None:
        problems.append(_slip_reason(slip, shift))
    if not math.isfinite(tolerance):
        problems.append("the frame period is unknown, so the residual cannot be judged")
    elif not math.isfinite(median_residual):
        problems.append("no frame of the chosen shift has a transition")
    elif residual_ambiguous:
        problems.append(f"median residual {median_residual * 1e3:.3f} ms is over the latch "
                        f"tolerance ({tolerance * 1e3:.3f} ms)")
    # An unknown frame period fails the gate too, and is already the reason.
    if not gate_ok and math.isfinite(tolerance):
        problems.append(latch_gate_reason(fit, ni_residual_rms, frame_period))
    return Pairing(shift, math.nan, agreement,
                   bool(residual_ambiguous or slip is not None or not gate_ok),
                   "; ".join(problems), method=CAMERA_LATCH, median_residual=median_residual,
                   slip=slip, limit=tolerance, clock=clock)


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
