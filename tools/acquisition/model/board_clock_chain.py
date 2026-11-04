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
#: Frames paired to one boot's reports by tick agree on (NI seconds - board
#: seconds) to within the clocks' drift, which is milliseconds an hour. A reboot
#: restarts the ticks, so frames of another boot that repeat a tick sit seconds
#: or more away. Pairs chain into one cluster while neighbours are this close.
PAIRING_CLUSTER_SECONDS = 0.5
#: An observer's clap and the timekeeper's own clap of the same tick are one
#: event only if the host heard them this close together.
CLAP_HOST_WINDOW_SECONDS = 0.5
TONE_MATCH_WINDOW_SECONDS = 0.005
#: The spec's end-to-end acceptance: |error| 95th percentile under 0.1 ms over at least 100 tones.
TONE_TARGET_P95_SECONDS = 0.0001
TONE_ACCEPTANCE_COUNT = 100
#: Two knots this close are one.
_SAME_KNOT_SECONDS = 1e-6

_KIND_CLAP, _KIND_BARCODE, _KIND_TONE = "SYNC_CLAP_OBSERVATION", "SYNC_BARCODE_REPORT", "TONE_STATUS"
_STAMP_START_EDGE, _STAMP_OBSERVED = "physical_start", "observed"
_ROW_PERF, _ROW_KIND, _ROW_PAYLOAD, _ROW_BOOT, _ROW_BOARD_US, _ROW_STAMP_KIND = 0, 3, 8, 12, 14, 15


@dataclasses.dataclass(frozen=True)
class _Envelope:
    address: Optional[int]
    boot_id: int
    board_us: int
    perf: Optional[float]
    stamp_kind: str
    payload: Mapping


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


def _record(status, reason, *, timekeeper=None, ignored=(), rejected=None, boards=()) -> dict:
    """Every record has these keys, whatever the status, so a reader never has to ask which are there."""
    rejected = rejected or {}
    return {
        "status": status,
        "reason": reason,
        "method": METHOD,
        "niTime": None if timekeeper is None else NI_TIME_NOTE,
        "timekeeper": timekeeper,
        "ignoredTimekeeperBoots": list(ignored),
        "rowsRejectedForStampKind": {
            _KIND_BARCODE: int(rejected.get(_KIND_BARCODE, 0)), _KIND_CLAP: int(rejected.get(_KIND_CLAP, 0)),
        },
        "boards": list(boards),
    }


def unavailable(reason: str, *, rejected=None) -> dict:
    return _record("unavailable", reason, rejected=rejected)


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
                _float_or_none(row[_ROW_PERF]), str(row[_ROW_STAMP_KIND] or ""), payload,
            ))
    return tuple(found)


def _stamped(envelopes, stamp_kind):
    """The envelopes stamped as `stamp_kind` says they must be, and how many were not."""
    usable = tuple(envelope for envelope in envelopes if envelope.stamp_kind == stamp_kind)
    return usable, len(envelopes) - len(usable)


def has_barcode_reports(device_rows) -> bool:
    """Whether the rows hold a timekeeper's SYNC_BARCODE report, from the rows alone: no NI-DAQ read needed."""
    return bool(_stamped(_envelopes(device_rows, _KIND_BARCODE), _STAMP_START_EDGE)[0])


def _largest_cluster(values, gap) -> numpy.ndarray:
    """Positions in `values` of the largest run that never has a gap above `gap` between sorted neighbours."""
    order = numpy.argsort(values, kind="stable")
    breaks = numpy.flatnonzero(numpy.diff(values[order]) > gap) + 1
    runs = numpy.split(order, breaks)
    return max(runs, key=len)


def _consistent_frames(pairs, indices, rate):
    """The (frame, board_us) pairs that agree with each other on NI seconds minus board seconds."""
    if not pairs:
        return []
    starts = numpy.fromiter((frame.start_position for frame, _ in pairs), dtype=numpy.int64, count=len(pairs))
    boards = numpy.fromiter((board_us for _, board_us in pairs), dtype=numpy.float64, count=len(pairs))
    offsets = indices[starts].astype(numpy.float64) / rate - boards / 1e6
    return [pairs[position] for position in sorted(_largest_cluster(offsets, PAIRING_CLUSTER_SECONDS).tolist())]


def compose(inner, outer):
    """`outer` after `inner` as one exact table: x -> outer(inner(x)), over inner's span.

    The knots are inner's and the board seconds at which inner reaches each of
    outer's knots, so every bend of either link is a knot and the table between
    them is straight. The residual adds in quadrature, outer's rate carrying
    inner's. Invalid unless both links are valid.
    """
    if not inner.valid or not outer.valid:
        return clocks.invalid_piecewise_fit("a link of the chain is not valid")
    bends = inner.inverse(outer.knots)
    bends = bends[(bends > inner.knots[0]) & (bends < inner.knots[-1])]
    if bends.size:
        bends = bends[numpy.abs(inner.knots[:, None] - bends[None, :]).min(axis=0) > _SAME_KNOT_SECONDS]
    knots = numpy.sort(numpy.concatenate([inner.knots, bends]))
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


def _timekeeper_link(frames, matched_pairs, tick_matched, indices, rate):
    """Timekeeper boot seconds -> NI seconds from every edge of the kept frames, or why not."""
    if not frames:
        return clocks.invalid_piecewise_fit("no barcode frames decoded (is the barcode line wired?)")
    if not tick_matched:
        return clocks.invalid_piecewise_fit(
            f"{len(frames)} barcode frames decoded, none matches a SYNC_BARCODE report of the same tick",
        )
    if len(matched_pairs) < BARCODE_FIT_MIN_FRAMES:
        return clocks.invalid_piecewise_fit(
            f"only {len(matched_pairs)} barcode frames matched SYNC_BARCODE reports, need {BARCODE_FIT_MIN_FRAMES}",
        )
    positions = numpy.fromiter(
        (position for frame, _ in matched_pairs for position, _ in frame.edges), dtype=numpy.int64,
    )
    board_seconds = numpy.fromiter(
        (board_us / 1e6 + offset for frame, board_us in matched_pairs for _, offset in frame.edges),
        dtype=numpy.float64,
    )
    return clocks.fit_piecewise_line(
        board_seconds, indices[positions].astype(numpy.float64) / rate,
        max_residual=BARCODE_FIT_MAX_RESIDUAL_SECONDS, min_points=BARCODE_FIT_MIN_POINTS,
        min_span=LINK_FIT_MIN_SPAN_SECONDS,
    )


def board_clock_chain(device_rows, *, sample_indices, sample_rate_hz, channels) -> dict:
    """Fit every clapping board's time to NI seconds through the timekeeper.

    NI seconds are NI sample index / sample rate; cameraNidaqAlignment.niToHost
    maps them to host perf time. Each link is a piecewise-linear table, one
    segment per KNOT_SECONDS, so a drifting crystal does not fail a link.
    Ticks pair with their boot, never alone (spec section 9): a timekeeper
    reset restarts the tick numbering.
    """
    reports, wrong_stamp = _stamped(_envelopes(device_rows, _KIND_BARCODE), _STAMP_START_EDGE)
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
    frames, rejected_frames = decode_barcode_frames(line, rate)

    by_boot: Dict[tuple, Dict[int, int]] = {}
    for report in reports:
        tick = _int_or_none(report.payload.get("tick"))
        if tick is not None:
            by_boot.setdefault((report.address, report.boot_id), {})[tick] = report.board_us
    if not by_boot:
        return unavailable("SYNC_BARCODE reports carry no tick", rejected=rejected)

    # Per timekeeper boot: the frames whose tick it reported, and of those the
    # largest set that agree on NI seconds minus board seconds. Another boot's
    # frames that repeat a tick disagree, and are dropped.
    candidates = []
    for key, ticks in by_boot.items():
        pairs = [(frame, ticks[frame.tick]) for frame in frames if frame.tick in ticks]
        candidates.append((key, ticks, len(pairs), _consistent_frames(pairs, indices, rate)))
    (address, boot_id), ticks, tick_matched, kept = max(candidates, key=lambda c: (len(c[3]), len(c[1])))
    ignored = [
        {"address": key[0], "bootId": int(key[1]), "reports": len(other_ticks), "framesPaired": len(other_kept)}
        for key, other_ticks, _, other_kept in candidates if key != (address, boot_id)
    ]

    to_ni = _timekeeper_link(frames, kept, tick_matched, indices, rate)
    timekeeper = {
        "address": address, "bootId": int(boot_id), "reports": len(ticks),
        "barcodeFramesDecoded": len(frames), "barcodeFramesRejected": int(rejected_frames),
        "barcodeFramesMatched": len(kept), "barcodeFramesInconsistent": tick_matched - len(kept),
        "toNi": to_ni.as_record(),
    }
    if not to_ni.valid:
        return _record(
            "unavailable", f"timekeeper to NI-DAQ: {to_ni.reason}",
            timekeeper=timekeeper, ignored=ignored, rejected=rejected,
        )

    claps, wrong_stamp = _stamped(_envelopes(device_rows, _KIND_CLAP), _STAMP_OBSERVED)
    rejected[_KIND_CLAP] = wrong_stamp
    own: Dict[int, tuple] = {}
    heard: Dict[tuple, list] = {}
    for clap in claps:
        tick = _int_or_none(clap.payload.get("tick"))
        if tick is None or clap.perf is None or _int_or_none(clap.payload.get("clapper_node")) != address:
            continue
        if clap.address == address:
            # The timekeeper's own claps are what the boards are paired to, and are never a board.
            if clap.boot_id == boot_id:
                own[tick] = (clap.board_us, clap.perf)
        else:
            heard.setdefault((clap.address, clap.boot_id), []).append((tick, clap.board_us, clap.perf))
    tones = _envelopes(device_rows, _KIND_TONE)
    boards = []
    for (board_address, board_boot), entries in sorted(heard.items(), key=lambda item: (item[0][0] or 0, item[0][1])):
        pairs = [
            (board_us / 1e6, own[tick][0] / 1e6) for tick, board_us, perf in entries
            if tick in own and abs(perf - own[tick][1]) <= CLAP_HOST_WINDOW_SECONDS
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
            "toNi": clocks.invalid_piecewise_fit(f"board to timekeeper: {to_timekeeper.reason}").as_record(),
            "toneCrossCheck": None,
        }
        if to_timekeeper.valid:
            to_ni_time = compose(to_timekeeper, to_ni)
            board["toNi"] = to_ni_time.as_record()
            board["toneCrossCheck"] = _tone_cross_check(
                [tone for tone in tones if tone.boot_id == board_boot], to_ni_time, indices, rate, channels,
            )
        boards.append(board)
    return _record("fitted", "", timekeeper=timekeeper, ignored=ignored, rejected=rejected, boards=boards)
