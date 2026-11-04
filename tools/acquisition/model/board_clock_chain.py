"""Pellet-board time to NI-DAQ time, through the IO module's clap and barcode.

The IO module (the timekeeper) claps on CAN once a second and plays the same
tick as a barcode on the NI-DAQ `barcode` line. Every board stamps the end of
the clap frame on its own clock and reports SYNC_CLAP_OBSERVED; the timekeeper
reports SYNC_BARCODE with the board time of the barcode's start edge. Pairing a
board's clap stamps with the timekeeper's gives board -> timekeeper time;
pairing the timekeeper's barcode times with the edges the NI-DAQ sampled gives
timekeeper -> NI time. Composing them puts pellet events on the NI sample
timeline the cameras and tones are already on.

Post-session and read-only over the recorded evidence: it never rewrites a
device row's event time.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Dict, Mapping, Optional, Tuple

import numpy

from tools.latency import clocks

BARCODE_CHANNEL = "barcode"
BARCODE_BIT_SECONDS = 0.001
BARCODE_BITS = 32
BARCODE_FRAME_SECONDS = (1 + BARCODE_BITS) * BARCODE_BIT_SECONDS
#: A start edge follows at least this much low line. A frame holds at most 32 ms
#: of low (tick 0), so no edge inside a frame qualifies.
BARCODE_QUIET_SECONDS = 0.040
#: How far a sampled edge may sit from where the decoded tick puts it.
BARCODE_EDGE_TOLERANCE_SECONDS = 0.0003
#: Five samples per bit: below this a bit's middle is not reliably inside it.
MIN_BARCODE_RATE_HZ = 5000.0


@dataclasses.dataclass(frozen=True)
class BarcodeFrame:
    tick: int
    start_position: int
    #: (sample position, seconds after the start edge) of every edge in the frame.
    edges: Tuple[Tuple[int, float], ...]


def barcode_edges(tick: int) -> Tuple[Tuple[float, bool], ...]:
    """Seconds after the start edge, and whether it rises, of every edge of barcode `tick`."""
    levels = [1] + [(int(tick) >> bit) & 1 for bit in range(BARCODE_BITS)] + [0]
    edges = [(0.0, True)]
    for segment in range(1, len(levels)):
        if levels[segment] != levels[segment - 1]:
            edges.append((segment * BARCODE_BIT_SECONDS, levels[segment] == 1))
    return tuple(edges)


def _is_high(samples) -> numpy.ndarray:
    """Samples above the 0.5 threshold, read in the line's own dtype (no float64 copy). NaN and +inf read low."""
    high = samples > 0.5
    if samples.dtype.kind == "f":
        numpy.logical_and(high, samples < numpy.inf, out=high)
    return high


def decode_barcode_frames(values, sample_rate_hz) -> Tuple[Tuple[BarcodeFrame, ...], int]:
    """Barcode frames on one sampled line, and how many candidate starts were rejected.

    Bits are read at the middle of their millisecond, by position from the start
    edge: bit 0 set merges with the start pulse, so edges alone cannot delimit
    bits. A frame is kept only if every edge its tick implies is where it should
    be, which rejects glitches and frames the recording cut.
    """
    if not sample_rate_hz or float(sample_rate_hz) < MIN_BARCODE_RATE_HZ:
        return (), 0
    rate = float(sample_rate_hz)
    # The line in its own dtype: a float64 copy is 8 bytes a sample, 290 MB an hour at 10 kHz.
    high = _is_high(numpy.asarray(values))
    if high.size < 2:
        return (), 0
    changes = numpy.flatnonzero(high[1:] != high[:-1]) + 1
    bit = BARCODE_BIT_SECONDS * rate
    quiet = BARCODE_QUIET_SECONDS * rate
    tolerance = max(1.0, BARCODE_EDGE_TOLERANCE_SECONDS * rate)
    frames, rejected = [], 0
    next_start = 0
    # Only a rising edge can start a frame. Edges inside a frame already read are
    # skipped by a jump, so the loop runs about once a frame, not once an edge.
    rising = numpy.flatnonzero(high[changes])
    rising_positions = changes[rising]
    candidate = 0
    while candidate < rising.size:
        index, position = int(rising[candidate]), int(rising_positions[candidate])
        if position < next_start:
            candidate = int(numpy.searchsorted(rising_positions, next_start, side="left"))
            continue
        candidate += 1
        low_since = int(changes[index - 1]) if index else 0
        if position - low_since < quiet:
            continue
        trailing = position + int(round((1 + BARCODE_BITS + 0.5) * bit))
        if trailing >= high.size:
            break
        next_start = position + int(round(BARCODE_FRAME_SECONDS * rate))
        if not high[position + int(round(0.5 * bit))] or high[trailing]:
            rejected += 1
            continue
        tick = 0
        for bit_index in range(BARCODE_BITS):
            if high[position + int(round((1.5 + bit_index) * bit))]:
                tick |= 1 << bit_index
        expected_edges = barcode_edges(tick)
        # Bit-centre sampling misses a glitch on a bit boundary; the frame must
        # also hold exactly the edges its tick implies, no more.
        frame_end = position + BARCODE_FRAME_SECONDS * rate + tolerance
        within = int(numpy.searchsorted(changes, frame_end, side="right")) - index
        if within != len(expected_edges):
            rejected += 1
            continue
        edges = []
        for offset, _rises in expected_edges:
            expected = position + offset * rate
            nearest = int(numpy.searchsorted(changes, expected - tolerance, side="left"))
            if nearest >= changes.size or changes[nearest] > expected + tolerance:
                break
            edges.append((int(changes[nearest]), offset))
        else:
            frames.append(BarcodeFrame(tick=tick, start_position=int(position), edges=tuple(edges)))
            continue
        rejected += 1
    return tuple(frames), rejected


NIDAQ_CHANNELS = (BARCODE_CHANNEL, "tone1", "tone2")
TONE_CHANNEL_BY_FREQUENCY_HZ = {5000: "tone1", 6000: "tone2"}
METHOD = "board_clap_and_nidaq_barcode"
COMPOSED_METHOD = "piecewise_line_composed"
PAIRING_HOST_TIME, PAIRING_HOST_ENDPOINTS = "host_time", "host_endpoints"
#: The line through the NI recording's first and last host times is a reference only if its
#: slope is within this of 1: the two clocks differ by tens of ppm, so anything else is a bad stamp.
ENDPOINT_SLOPE_TOLERANCE = 0.01
NI_TIME_NOTE = "NI sample index / sample rate (s); cameraNidaqAlignment.niToHost maps it to host perf time"
#: Every edge of a kept frame is a fit point, and a frame has 2 to 34 of them.
BARCODE_FIT_MIN_FRAMES = 10
BARCODE_FIT_MIN_POINTS = 20
#: Edge quantization alone is about 29 us rms at 10 kHz.
BARCODE_FIT_MAX_RESIDUAL_SECONDS = 0.0001
CLAP_FIT_MIN_POINTS = 10
#: Both clap stamps are taken in a CAN RX interrupt: a few us of jitter.
CLAP_FIT_MAX_RESIDUAL_SECONDS = 0.0001
#: How much of the clock a link's points must span. A rate read off fewer
#: seconds is noise multiplied by the whole recording; the host's TIME_SYNC fit
#: asks the same (board_clock.MIN_SLOPE_SPAN_SECONDS).
LINK_FIT_MIN_SPAN_SECONDS = 10.0
#: The timekeeper reports a barcode this long after the frame's start edge (the
#: firmware's BARCODE_CALLBACK_SLOT), and the host receives it a few ms later.
BARCODE_REPORT_DELAY_SECONDS = 0.5
#: A report pairs with a frame only if the host received it within this of that
#: delay after the frame's start edge, the NI time of the edge mapped to host
#: time: the same tick a boot reports hours earlier or later is not the frame.
BARCODE_REPORT_WINDOW_SECONDS = 0.5
#: Frames paired to one boot's reports agree on (NI seconds - board seconds) to
#: within the clocks' drift, which is milliseconds an hour. A pair that does not
#: (a mistimed stamp, a repeated tick that slipped through) sits seconds or more
#: away. Pairs chain into one cluster while neighbours are this close.
PAIRING_CLUSTER_SECONDS = 0.5
#: An observer's clap and the timekeeper's own clap of the same tick are one
#: event only if the host received them this close together (row column 9).
CLAP_HOST_WINDOW_SECONDS = 0.5
#: The device-row ring holds 100 000 rows and is not cleared between sessions,
#: so older sessions' rows are in it. A row counts only if the host received it
#: within the NI recording's host span widened by this.
NI_SPAN_MARGIN_SECONDS = 5.0
TONE_MATCH_WINDOW_SECONDS = 0.005
#: The spec's end-to-end acceptance: |error| 95th percentile under 0.1 ms over at least 100 tones.
TONE_TARGET_P95_SECONDS = 0.0001
TONE_ACCEPTANCE_COUNT = 100
#: Two knots this close are one.
_SAME_KNOT_SECONDS = 1e-6

_KIND_CLAP, _KIND_BARCODE, _KIND_TONE = "SYNC_CLAP_OBSERVATION", "SYNC_BARCODE_REPORT", "TONE_STATUS"
_STAMP_START_EDGE, _STAMP_OBSERVED = "physical_start", "observed"
#: Column 9 is host_receive_perf_time. Column 0 (perf_time) is the event time, which for
#: a time-synced board is its aligned board time and can be tens of ms from the receive time.
_ROW_KIND, _ROW_PAYLOAD, _ROW_HOST_RECEIVE, _ROW_BOOT, _ROW_BOARD_US, _ROW_STAMP_KIND = 3, 8, 9, 12, 14, 15


@dataclasses.dataclass(frozen=True)
class _Envelope:
    address: Optional[int]
    boot_id: int
    board_us: int
    host_perf: Optional[float]
    stamp_kind: str
    payload: Mapping


@dataclasses.dataclass(frozen=True)
class _Candidate:
    """One timekeeper boot's frames, paired to its reports."""
    key: tuple
    ticks: Dict[int, _Envelope]
    paired: int
    kept: list
    tied: bool


def _int_or_none(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _record(status, reason, *, timekeeper=None, ignored=(), rejected=None, boards=(), pairing=None, outside=0) -> dict:
    """Every record has these keys, whatever the status, so a reader never has to ask which are there."""
    rejected = rejected or {}
    return {
        "status": status,
        "reason": reason,
        "method": METHOD,
        "niTime": None if timekeeper is None else NI_TIME_NOTE,
        "pairing": pairing,
        "timekeeper": timekeeper,
        "ignoredTimekeeperBoots": list(ignored),
        "rowsRejectedForStampKind": {
            _KIND_BARCODE: int(rejected.get(_KIND_BARCODE, 0)), _KIND_CLAP: int(rejected.get(_KIND_CLAP, 0)),
        },
        "rowsOutsideNiSpan": int(outside),
        "boards": list(boards),
    }


def unavailable(reason: str, *, rejected=None, ignored=(), pairing=None, outside=0) -> dict:
    return _record("unavailable", reason, rejected=rejected, ignored=ignored, pairing=pairing, outside=outside)


def _envelopes(device_rows, kind_name):
    found = []
    for row in device_rows:
        if len(row) <= _ROW_STAMP_KIND or row[_ROW_KIND] != kind_name:
            continue
        boot, board_us = _int_or_none(row[_ROW_BOOT]), _int_or_none(row[_ROW_BOARD_US])
        if boot is None or board_us is None:
            continue
        try:
            payload = json.loads(row[_ROW_PAYLOAD] or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            found.append(_Envelope(
                _int_or_none(payload.get("board_address")), boot, board_us,
                _float_or_none(row[_ROW_HOST_RECEIVE]), str(row[_ROW_STAMP_KIND] or ""), payload,
            ))
    return tuple(found)


def _stamped(envelopes, stamp_kind):
    """The envelopes stamped as `stamp_kind` says they must be, and how many were not."""
    usable = tuple(envelope for envelope in envelopes if envelope.stamp_kind == stamp_kind)
    return usable, len(envelopes) - len(usable)


def _within(envelopes, span):
    """The envelopes the host received inside `span`, and how many were not."""
    inside = tuple(
        envelope for envelope in envelopes if envelope.host_perf is not None and span[0] <= envelope.host_perf <= span[1]
    )
    return inside, len(envelopes) - len(inside)


def _host_span(ni_to_host, indices, rate):
    """(first, last) host perf seconds of the NI recording, widened by the margin; None without a valid fit."""
    if ni_to_host is None or not getattr(ni_to_host, "valid", False) or not indices.size:
        return None
    ends = numpy.asarray(
        ni_to_host.map(numpy.array([float(indices[0]), float(indices[-1])]) / rate), dtype=numpy.float64,
    )
    if ends.shape != (2,) or not numpy.isfinite(ends).all():
        return None
    return float(ends[0]) - NI_SPAN_MARGIN_SECONDS, float(ends[1]) + NI_SPAN_MARGIN_SECONDS


def _endpoint_line(endpoints):
    """A line from NI seconds to host seconds through two (NI seconds, host seconds) points, or None.

    None unless both points are finite, the second is later in NI time, and the
    slope is within ENDPOINT_SLOPE_TOLERANCE of 1. A failed gate is no reference.
    """
    try:
        (ni0, host0), (ni1, host1) = ((float(ni), float(host)) for ni, host in endpoints)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (ni0, host0, ni1, host1)) or not ni1 > ni0:
        return None
    drift = (host1 - host0) / (ni1 - ni0)
    if not abs(drift - 1.0) <= ENDPOINT_SLOPE_TOLERANCE:
        return None
    return clocks.ClockFit(
        offset=host0 - drift * ni0, drift=drift, residual_rms=math.nan, points=2, valid=True, x_min=ni0, x_max=ni1,
    )


def _host_reference(ni_to_host, ni_host_endpoints, indices, rate):
    """(NI-to-host map, which reference it is, the recording's host span widened by the margin), or None.

    The session's NI-to-host fit when it is valid, else the line through the
    recording's first and last host times. Either must put both ends of the
    recording at finite host times.
    """
    for fit, label in ((ni_to_host, PAIRING_HOST_TIME), (_endpoint_line(ni_host_endpoints), PAIRING_HOST_ENDPOINTS)):
        span = _host_span(fit, indices, rate)
        if span is not None:
            return fit, label, span
    return None


def has_barcode_reports(device_rows) -> bool:
    """Whether the rows hold a timekeeper's SYNC_BARCODE report, from the rows alone: no NI-DAQ read needed."""
    return bool(_stamped(_envelopes(device_rows, _KIND_BARCODE), _STAMP_START_EDGE)[0])


def _largest_cluster(values, gap):
    """(positions in `values` of the largest run with no gap above `gap` between sorted neighbours, whether another run is as large)."""
    order = numpy.argsort(values, kind="stable")
    runs = numpy.split(order, numpy.flatnonzero(numpy.diff(values[order]) > gap) + 1)
    sizes = [len(run) for run in runs]
    largest = max(sizes)
    return runs[sizes.index(largest)], sizes.count(largest) > 1


def _consistent_frames(pairs, indices, rate):
    """The (frame, report) pairs that agree with each other on NI seconds minus board seconds, and whether two runs tie."""
    if not pairs:
        return [], False
    starts = numpy.fromiter((frame.start_position for frame, _ in pairs), dtype=numpy.int64, count=len(pairs))
    boards = numpy.fromiter((report.board_us for _, report in pairs), dtype=numpy.float64, count=len(pairs))
    offsets = indices[starts].astype(numpy.float64) / rate - boards / 1e6
    positions, tied = _largest_cluster(offsets, PAIRING_CLUSTER_SECONDS)
    return [pairs[position] for position in sorted(positions.tolist())], tied


def _pair_frames(frames, ticks, frame_host):
    """One boot's (frame, report) pairs: the same tick, and a report the host received
    BARCODE_REPORT_DELAY_SECONDS (plus or minus the window) after the frame's start edge.

    A tick never pairs alone: a boot that repeats it hours earlier or later reported it at another host time.
    """
    pairs = []
    for position, frame in enumerate(frames):
        report = ticks.get(frame.tick)
        if report is None or report.host_perf is None:
            continue
        if abs(report.host_perf - frame_host[position] - BARCODE_REPORT_DELAY_SECONDS) <= BARCODE_REPORT_WINDOW_SECONDS:
            pairs.append((frame, report))
    return pairs


def compose(inner, outer):
    """`outer` after `inner` as one exact table: x -> outer(inner(x)).

    The knots are inner's and the x at which inner, carried on past its ends by
    its end segments, reaches each of outer's knots, so every bend of either
    link is a knot and the table between them is straight. The residual adds in
    quadrature, outer's rate carrying inner's. Invalid unless both links are valid.
    """
    if not inner.valid or not outer.valid:
        return clocks.invalid_piecewise_line_fit("a link of the chain is not valid")
    knots = numpy.sort(numpy.concatenate([inner.knots, inner.inverse(outer.knots)]))
    knots = knots[numpy.concatenate([[True], numpy.diff(knots) > _SAME_KNOT_SECONDS])]
    return clocks.PiecewiseClockFit(
        knots, outer.map(inner.map(knots)),
        float(math.hypot(outer.drift * inner.residual_rms, outer.residual_rms)),
        min(inner.points, outer.points), True, "", COMPOSED_METHOD,
    )


def _percentile(ascending, percent) -> float:
    """numpy's linear percentile of an ascending array that may end in infinities (which numpy turns to nan)."""
    position = (ascending.size - 1) * percent / 100.0
    lower = int(math.floor(position))
    low = float(ascending[lower])
    fraction = position - lower
    if fraction == 0 or math.isinf(low) or lower + 1 >= ascending.size:
        return low
    high = float(ascending[lower + 1])
    return high if math.isinf(high) else low + (high - low) * fraction


def _rising_edge_seconds(samples, indices, rate) -> numpy.ndarray:
    high = _is_high(numpy.asarray(samples))
    rising = numpy.flatnonzero(high[1:] & ~high[:-1]) + 1
    return numpy.sort(indices[rising].astype(numpy.float64) / rate)


def _tone_cross_check(tones, link, indices, rate, channels) -> dict:
    """Pellet tone stamps mapped to NI seconds, against the tone edges the NI-DAQ recorded.

    This is the accuracy evidence and feeds no fit. A tone whose NI second is
    inside the recording and has no edge within the match window is a failure of
    the chain, not a tone to leave out: its error counts as infinite in the
    95th percentile. A tone outside the recording, or on a line that was not
    recorded, cannot be judged and is counted apart.
    """
    first, last = float(indices[0]) / rate, float(indices[-1]) / rate
    by_channel: Dict[str, list] = {}
    for tone in tones:
        channel = TONE_CHANNEL_BY_FREQUENCY_HZ.get(_int_or_none(tone.payload.get("frequency_hz")))
        if tone.stamp_kind == _STAMP_START_EDGE and channel is not None:
            by_channel.setdefault(channel, []).append(tone.board_us / 1e6)
    errors: list = []
    unmatched = outside = without_channel = 0
    missing = []
    for channel, board_seconds in sorted(by_channel.items()):
        samples = channels.get(channel)
        if samples is None or numpy.shape(samples) != indices.shape:
            without_channel += len(board_seconds)
            missing.append(channel)
            continue
        predicted = link.map(numpy.asarray(board_seconds, dtype=numpy.float64))
        inside = (predicted >= first + TONE_MATCH_WINDOW_SECONDS) & (predicted <= last - TONE_MATCH_WINDOW_SECONDS)
        outside += int((~inside).sum())
        predicted = predicted[inside]
        if not predicted.size:
            continue
        edges = _rising_edge_seconds(samples, indices, rate)
        if not edges.size:
            unmatched += int(predicted.size)
            continue
        after = numpy.searchsorted(edges, predicted)
        right, left = edges[numpy.minimum(after, edges.size - 1)], edges[numpy.maximum(after - 1, 0)]
        nearest = numpy.where(numpy.abs(right - predicted) < numpy.abs(left - predicted), right, left)
        error = predicted - nearest
        landed = numpy.abs(error) <= TONE_MATCH_WINDOW_SECONDS
        errors.extend(error[landed].tolist())
        unmatched += int((~landed).sum())
    judged = len(errors) + unmatched
    summary = {
        "tonesConsidered": judged, "matched": len(errors), "unmatched": unmatched,
        "outsideRecording": outside, "tonesWithoutChannel": without_channel, "channelsMissing": missing,
        "matchWindowSeconds": TONE_MATCH_WINDOW_SECONDS, "targetP95Seconds": TONE_TARGET_P95_SECONDS,
        "acceptanceTones": TONE_ACCEPTANCE_COUNT, "enoughTonesForAcceptance": bool(judged >= TONE_ACCEPTANCE_COUNT),
        "p95AbsResidualSeconds": None, "meetsTarget": None,
    }
    if judged:
        everything = numpy.sort(numpy.concatenate([numpy.abs(errors), numpy.full(unmatched, numpy.inf)]))
        p95 = _percentile(everything, 95.0)
        summary["p95AbsResidualSeconds"] = p95 if math.isfinite(p95) else None
        summary["meetsTarget"] = bool(p95 < TONE_TARGET_P95_SECONDS)
    if errors:
        magnitude = numpy.abs(numpy.asarray(errors, dtype=numpy.float64))
        summary.update({
            "meanResidualSeconds": float(numpy.mean(errors)),
            "p50AbsResidualSeconds": float(numpy.percentile(magnitude, 50)),
            "maxAbsResidualSeconds": float(magnitude.max()),
        })
    return summary


def _timekeeper_link(frames, chosen, indices, rate):
    """Timekeeper boot seconds -> NI seconds from every edge of the kept frames, or why not."""
    if not frames:
        return clocks.invalid_piecewise_line_fit("no barcode frames decoded (is the barcode line wired?)")
    if not chosen.paired:
        return clocks.invalid_piecewise_line_fit(
            f"{len(frames)} barcode frames decoded, none matches a SYNC_BARCODE report of the same tick and host time",
        )
    if len(chosen.kept) < BARCODE_FIT_MIN_FRAMES:
        return clocks.invalid_piecewise_line_fit(
            f"only {len(chosen.kept)} barcode frames matched SYNC_BARCODE reports, need {BARCODE_FIT_MIN_FRAMES}",
        )
    positions = numpy.fromiter(
        (position for frame, _ in chosen.kept for position, _ in frame.edges), dtype=numpy.int64,
    )
    board_seconds = numpy.fromiter(
        (report.board_us / 1e6 + offset for frame, report in chosen.kept for _, offset in frame.edges),
        dtype=numpy.float64,
    )
    return clocks.fit_piecewise_line(
        board_seconds, indices[positions].astype(numpy.float64) / rate,
        max_residual=BARCODE_FIT_MAX_RESIDUAL_SECONDS, min_points=BARCODE_FIT_MIN_POINTS,
        min_span=LINK_FIT_MIN_SPAN_SECONDS,
    )


def board_clock_chain(
    device_rows, *, sample_indices, sample_rate_hz, channels, ni_to_host=None, ni_host_endpoints=None,
) -> dict:
    """Fit every clapping board's time to NI seconds through the timekeeper.

    NI seconds are NI sample index / sample rate; cameraNidaqAlignment.niToHost
    maps them to host perf time. Each link is a piecewise-linear table, one
    segment per KNOT_SECONDS, so a drifting crystal does not fail a link.

    Ticks pair with their boot, never alone (spec section 9): a timekeeper reset
    restarts the tick numbering, and an NI frame carries only a tick. What ties a
    frame to its boot is the host time at which it was played, so the chain needs
    a host reference for the NI recording: `ni_to_host`, the session's NI-to-host
    fit (anything with `valid` and `map`), or failing that `ni_host_endpoints`,
    ((NI seconds, host seconds), (NI seconds, host seconds)) of the recording's
    first and last samples, taken as a line when its slope is within 1 % of 1. A
    report pairs with a frame by tick and by being received about half a second
    after the frame's start edge, and only rows received within the NI
    recording's host span count. With neither reference the chain is
    `unavailable`; so is a tie between boots (or inside one), never a pick.
    """
    barcode_rows = _envelopes(device_rows, _KIND_BARCODE)
    reports, wrong_stamp = _stamped(barcode_rows, _STAMP_START_EDGE)
    rejected = {_KIND_BARCODE: wrong_stamp}
    if not reports:
        if wrong_stamp:
            return unavailable(
                f"{wrong_stamp} SYNC_BARCODE reports, none stamped {_STAMP_START_EDGE}: no usable timekeeper",
                rejected=rejected,
            )
        return unavailable("no SYNC_BARCODE reports: no timekeeper was on the bus")
    if BARCODE_CHANNEL not in channels:
        return unavailable("the NI-DAQ recorded no barcode channel", rejected=rejected)
    rate = _float_or_none(sample_rate_hz)
    if rate is None or rate <= 0:
        return unavailable("the NI-DAQ sample rate is unknown", rejected=rejected)
    if rate < MIN_BARCODE_RATE_HZ:
        return unavailable(
            f"NI-DAQ rate {rate:g} Hz is below the {MIN_BARCODE_RATE_HZ:g} Hz a barcode needs", rejected=rejected,
        )
    line = numpy.asarray(channels[BARCODE_CHANNEL])
    indices = numpy.asarray(sample_indices)
    if line.ndim != 1 or indices.shape != line.shape:
        return unavailable(
            f"NI-DAQ sample index ({indices.size} samples) and barcode line ({line.size}) differ in length",
            rejected=rejected,
        )

    reference = _host_reference(ni_to_host, ni_host_endpoints, indices, rate)
    if reference is None:
        return unavailable(
            "ticks cannot be tied to a timekeeper boot without host time (spec section 9): the session has no "
            "valid NI-to-host fit and no usable first and last NI sample host times", rejected=rejected,
        )
    host_map, pairing, span = reference
    # Rows the host received outside the recording (earlier sessions', left in the ring) are not evidence.
    barcode_rows, outside = _within(barcode_rows, span)
    claps, outside_claps = _within(_envelopes(device_rows, _KIND_CLAP), span)
    tones, outside_tones = _within(_envelopes(device_rows, _KIND_TONE), span)
    outside += outside_claps + outside_tones
    reports, rejected[_KIND_BARCODE] = _stamped(barcode_rows, _STAMP_START_EDGE)
    claps, rejected[_KIND_CLAP] = _stamped(claps, _STAMP_OBSERVED)
    if not reports:
        return unavailable(
            f"no SYNC_BARCODE report was received within the NI recording's host span ({span[0]:.1f} to "
            f"{span[1]:.1f} s, which includes {NI_SPAN_MARGIN_SECONDS:g} s either side)",
            rejected=rejected, pairing=pairing, outside=outside,
        )
    frames, rejected_frames = decode_barcode_frames(line, rate)

    ticks_by_boot: Dict[tuple, Dict[int, _Envelope]] = {}
    for report in reports:
        tick = _int_or_none(report.payload.get("tick"))
        if tick is not None:
            ticks_by_boot.setdefault((report.address, report.boot_id), {})[tick] = report
    if not ticks_by_boot:
        return unavailable("SYNC_BARCODE reports carry no tick", rejected=rejected, pairing=pairing, outside=outside)

    # Per timekeeper boot: the frames whose tick it reported when the frame was played, and of those the
    # largest set that agree on NI seconds minus board seconds.
    starts = numpy.fromiter((frame.start_position for frame in frames), dtype=numpy.int64, count=len(frames))
    frame_host = numpy.asarray(host_map.map(indices[starts].astype(numpy.float64) / rate), dtype=numpy.float64)
    candidates = []
    for key, ticks in ticks_by_boot.items():
        pairs = _pair_frames(frames, ticks, frame_host)
        kept, tied = _consistent_frames(pairs, indices, rate)
        candidates.append(_Candidate(key, ticks, len(pairs), kept, tied))

    def listed(candidate):
        return {
            "address": candidate.key[0], "bootId": int(candidate.key[1]),
            "reports": len(candidate.ticks), "framesPaired": len(candidate.kept),
        }

    best = max(len(candidate.kept) for candidate in candidates)
    leaders = [candidate for candidate in candidates if len(candidate.kept) == best]
    if best and (len(leaders) > 1 or leaders[0].tied):
        how = "by tick and host time"
        if len(leaders) > 1:
            names = " and ".join(f"{leader.key[1]:#x}" for leader in leaders)
            why = (f"tie between timekeeper boots {names}: each pairs {best} barcode frames with its reports {how}, "
                   "so which one drove the NI-DAQ line cannot be told")
        else:
            why = (f"tie inside timekeeper boot {leaders[0].key[1]:#x}: two runs of {best} barcode frames paired "
                   f"{how} disagree on NI time minus board time, so which belongs to it cannot be told")
        return unavailable(
            f"timekeeper to NI-DAQ: {why}", rejected=rejected, ignored=[listed(c) for c in candidates],
            pairing=pairing, outside=outside,
        )
    # With nothing paired there is no link to make; show the boot that reported most.
    chosen = leaders[0] if best else max(candidates, key=lambda candidate: len(candidate.ticks))
    ignored = [listed(candidate) for candidate in candidates if candidate is not chosen]
    (address, boot_id) = chosen.key

    to_ni = _timekeeper_link(frames, chosen, indices, rate)
    timekeeper = {
        "address": address, "bootId": int(boot_id), "reports": len(chosen.ticks),
        "barcodeFramesDecoded": len(frames), "barcodeFramesRejected": int(rejected_frames),
        "barcodeFramesMatched": len(chosen.kept), "barcodeFramesInconsistent": chosen.paired - len(chosen.kept),
        "toNi": to_ni.as_record(),
    }
    if not to_ni.valid:
        return _record(
            "unavailable", f"timekeeper to NI-DAQ: {to_ni.reason}", timekeeper=timekeeper, ignored=ignored,
            rejected=rejected, pairing=pairing, outside=outside,
        )

    own: Dict[int, _Envelope] = {}
    heard: Dict[tuple, list] = {}
    for clap in claps:
        tick = _int_or_none(clap.payload.get("tick"))
        if tick is None or clap.host_perf is None or _int_or_none(clap.payload.get("clapper_node")) != address:
            continue
        if clap.address == address:
            # The timekeeper's own claps are what the boards are paired to, and are never a board.
            if clap.boot_id == boot_id:
                own[tick] = clap
        else:
            heard.setdefault((clap.address, clap.boot_id), []).append((tick, clap))
    boards = []
    for (board_address, board_boot), entries in sorted(heard.items(), key=lambda item: (item[0][0] or 0, item[0][1])):
        pairs = [
            (clap.board_us / 1e6, own[tick].board_us / 1e6) for tick, clap in entries
            if tick in own and abs(clap.host_perf - own[tick].host_perf) <= CLAP_HOST_WINDOW_SECONDS
        ]
        to_timekeeper = clocks.fit_piecewise_line(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            max_residual=CLAP_FIT_MAX_RESIDUAL_SECONDS, min_points=CLAP_FIT_MIN_POINTS,
            min_span=LINK_FIT_MIN_SPAN_SECONDS,
        )
        board = {
            "address": board_address, "bootId": int(board_boot),
            "clapsHeard": len(entries), "clapsMatched": len(pairs),
            "toTimekeeper": to_timekeeper.as_record(),
            "toNi": clocks.invalid_piecewise_line_fit(f"board to timekeeper: {to_timekeeper.reason}").as_record(),
            "toneCrossCheck": None,
        }
        if to_timekeeper.valid:
            to_ni_time = compose(to_timekeeper, to_ni)
            board["toNi"] = to_ni_time.as_record()
            board["toneCrossCheck"] = _tone_cross_check(
                [tone for tone in tones if tone.boot_id == board_boot], to_ni_time, indices, rate, channels,
            )
        boards.append(board)
    return _record(
        "fitted", "", timekeeper=timekeeper, ignored=ignored, rejected=rejected, boards=boards,
        pairing=pairing, outside=outside,
    )
