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
    samples = numpy.asarray(values, dtype=numpy.float64)
    high = numpy.isfinite(samples) & (samples > 0.5)
    if high.size < 2:
        return (), 0
    changes = numpy.flatnonzero(high[1:] != high[:-1]) + 1
    bit = BARCODE_BIT_SECONDS * rate
    quiet = BARCODE_QUIET_SECONDS * rate
    tolerance = max(1.0, BARCODE_EDGE_TOLERANCE_SECONDS * rate)
    frames, rejected = [], 0
    next_start = 0
    for index, position in enumerate(changes.tolist()):
        if not high[position] or position < next_start:
            continue
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
#: 33 edges a frame at most; 20 already spans more than one frame.
BARCODE_FIT_MIN_POINTS = 20
#: Edge quantization alone is about 29 us rms at 10 kHz.
BARCODE_FIT_MAX_RESIDUAL_SECONDS = 0.0001
CLAP_FIT_MIN_POINTS = 5
#: Both clap stamps are taken in a CAN RX interrupt: a few us of jitter.
CLAP_FIT_MAX_RESIDUAL_SECONDS = 0.0001
TONE_MATCH_WINDOW_SECONDS = 0.005
#: The spec's end-to-end acceptance: |error| 95th percentile under 0.1 ms.
TONE_TARGET_P95_SECONDS = 0.0001

_KIND_CLAP, _KIND_BARCODE, _KIND_TONE = "SYNC_CLAP_OBSERVATION", "SYNC_BARCODE_REPORT", "TONE_STATUS"
_ROW_KIND, _ROW_PAYLOAD, _ROW_BOOT, _ROW_BOARD_US, _ROW_STAMP_KIND = 3, 8, 12, 14, 15


@dataclasses.dataclass(frozen=True)
class _Envelope:
    address: Optional[int]
    boot_id: int
    board_us: int
    stamp_kind: str
    payload: Mapping


def _int_or_none(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _fit_record(fit) -> dict:
    return {
        "offset": _finite(fit.offset), "drift": _finite(fit.drift),
        "residualRms": _finite(fit.residual_rms), "points": int(fit.points),
        "valid": bool(fit.valid), "reason": str(fit.reason),
        "xMin": _finite(fit.x_min), "xMax": _finite(fit.x_max),
    }


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason, "method": METHOD, "boards": []}


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
                str(row[_ROW_STAMP_KIND] or ""), payload,
            ))
    return tuple(found)


def _tone_cross_check(tones, offset, drift, indices, rate, channels) -> dict:
    residuals, unmatched, considered = [], 0, 0
    edges_by_channel: Dict[str, numpy.ndarray] = {}
    for tone in tones:
        channel = TONE_CHANNEL_BY_FREQUENCY_HZ.get(_int_or_none(tone.payload.get("frequency_hz")))
        if tone.stamp_kind != "physical_start" or channel not in channels:
            continue
        considered += 1
        if channel not in edges_by_channel:
            samples = numpy.asarray(channels[channel], dtype=numpy.float64)
            high = numpy.isfinite(samples) & (samples > 0.5)
            rising = numpy.flatnonzero(high[1:] & ~high[:-1]) + 1
            edges_by_channel[channel] = indices[rising] / rate
        edges = edges_by_channel[channel]
        predicted = offset + drift * (tone.board_us / 1e6)
        if edges.size == 0:
            unmatched += 1
            continue
        nearest = float(edges[int(numpy.argmin(numpy.abs(edges - predicted)))])
        if abs(nearest - predicted) > TONE_MATCH_WINDOW_SECONDS:
            unmatched += 1
            continue
        residuals.append(predicted - nearest)
    summary = {
        "tonesConsidered": considered, "matched": len(residuals), "unmatched": unmatched,
        "targetP95Seconds": TONE_TARGET_P95_SECONDS, "meetsTarget": None,
    }
    if residuals:
        magnitude = numpy.abs(numpy.asarray(residuals, dtype=numpy.float64))
        p95 = float(numpy.percentile(magnitude, 95))
        summary.update({
            "meanResidualSeconds": float(numpy.mean(residuals)),
            "p50AbsResidualSeconds": float(numpy.percentile(magnitude, 50)),
            "p95AbsResidualSeconds": p95,
            "maxAbsResidualSeconds": float(magnitude.max()),
            "meetsTarget": bool(p95 < TONE_TARGET_P95_SECONDS),
        })
    return summary


def board_clock_chain(device_rows, *, sample_indices, sample_rate_hz, channels) -> dict:
    """Fit every clapping board's time to NI seconds through the timekeeper.

    NI seconds are NI sample index / sample rate; cameraNidaqAlignment.niToHost
    maps them to host perf time.
    """
    reports = _envelopes(device_rows, _KIND_BARCODE)
    if not reports:
        return _unavailable("no SYNC_BARCODE reports: no timekeeper was on the bus")
    if BARCODE_CHANNEL not in channels or not sample_rate_hz:
        return _unavailable("the NI-DAQ recorded no barcode channel")
    rate = float(sample_rate_hz)
    if rate < MIN_BARCODE_RATE_HZ:
        return _unavailable(f"NI-DAQ rate {rate:g} Hz is below the {MIN_BARCODE_RATE_HZ:g} Hz a barcode needs")
    indices = numpy.asarray(sample_indices, dtype=numpy.float64)
    frames, rejected = decode_barcode_frames(channels[BARCODE_CHANNEL], rate)

    by_timekeeper: Dict[tuple, Dict[int, int]] = {}
    for report in reports:
        tick = _int_or_none(report.payload.get("tick"))
        if tick is not None:
            by_timekeeper.setdefault((report.address, report.boot_id), {})[tick] = report.board_us
    timekeeper, ticks = max(
        by_timekeeper.items(), key=lambda item: sum(frame.tick in item[1] for frame in frames),
    )
    x, y, matched = [], [], 0
    for frame in frames:
        start_us = ticks.get(frame.tick)
        if start_us is None:
            continue
        matched += 1
        for position, offset in frame.edges:
            x.append(start_us / 1e6 + offset)
            y.append(indices[position] / rate)
    to_ni = clocks.fit_line(
        x, y, max_residual=BARCODE_FIT_MAX_RESIDUAL_SECONDS, min_points=BARCODE_FIT_MIN_POINTS,
    )
    record = {
        "status": "fitted" if to_ni.valid else "unavailable",
        "reason": "" if to_ni.valid else f"timekeeper to NI-DAQ: {to_ni.reason}",
        "method": METHOD,
        "niTime": "NI sample index / sample rate (s); cameraNidaqAlignment.niToHost maps it to host perf time",
        "timekeeper": {
            "address": timekeeper[0], "bootId": int(timekeeper[1]),
            "barcodeFramesDecoded": len(frames), "barcodeFramesRejected": int(rejected),
            "barcodeFramesMatched": matched, "toNi": _fit_record(to_ni),
        },
        "boards": [],
    }
    if not to_ni.valid:
        return record

    claps = _envelopes(device_rows, _KIND_CLAP)
    own: Dict[int, int] = {}
    heard: Dict[tuple, Dict[int, int]] = {}
    for clap in claps:
        tick = _int_or_none(clap.payload.get("tick"))
        if tick is None or _int_or_none(clap.payload.get("clapper_node")) != timekeeper[0]:
            continue
        if (clap.address, clap.boot_id) == timekeeper:
            own[tick] = clap.board_us
        else:
            heard.setdefault((clap.address, clap.boot_id), {})[tick] = clap.board_us
    tones = _envelopes(device_rows, _KIND_TONE)
    for (address, boot_id), board_ticks in sorted(heard.items(), key=lambda item: (item[0][0] or 0, item[0][1])):
        common = sorted(set(board_ticks) & set(own))
        to_timekeeper = clocks.fit_line(
            [board_ticks[tick] / 1e6 for tick in common], [own[tick] / 1e6 for tick in common],
            max_residual=CLAP_FIT_MAX_RESIDUAL_SECONDS, min_points=CLAP_FIT_MIN_POINTS,
        )
        board = {
            "address": address, "bootId": int(boot_id), "clapsMatched": len(common),
            "toTimekeeper": _fit_record(to_timekeeper),
            "toNi": {"offset": None, "drift": None, "valid": False}, "toneCrossCheck": None,
        }
        if to_timekeeper.valid:
            offset = float(to_ni.offset + to_ni.drift * to_timekeeper.offset)
            drift = float(to_ni.drift * to_timekeeper.drift)
            board["toNi"] = {"offset": offset, "drift": drift, "valid": True}
            board["toneCrossCheck"] = _tone_cross_check(
                [tone for tone in tones if tone.boot_id == boot_id], offset, drift, indices, rate, channels,
            )
        record["boards"].append(board)
    return record
