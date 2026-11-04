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
from typing import Tuple

import numpy

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
