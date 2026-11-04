"""Synthetic NI-DAQ lines and device rows for the board clock chain tests."""

import json
import math

import numpy as np

from tools.latency import clocks

RATE = 10_000.0
#: Host perf seconds of NI second 0: the rigs below put the host clock at NI seconds plus this.
NI_HOST_OFFSET = 1000.0
#: The timekeeper reports a barcode this long after its start edge (the firmware's BARCODE_CALLBACK_SLOT).
BARCODE_REPORT_DELAY = 0.5


def barcode_line(rate, total_seconds, frames):
    """frames: (start_seconds, tick). Each edge at the first sample at or after its time."""
    line = np.zeros(int(math.ceil(total_seconds * rate)), dtype=np.float32)
    for start, tick in frames:
        levels = [1] + [(tick >> bit) & 1 for bit in range(32)]
        for segment, level in enumerate(levels):
            if level:
                first = int(math.ceil((start + segment * 0.001) * rate))
                last = int(math.ceil((start + (segment + 1) * 0.001) * rate))
                line[first:last] = 1.0
    return line


def device_row(perf, kind, payload, boot, sequence, board_us, stamp_kind, event_perf=None):
    """A normalized 23-field device row. `perf` is the host receive time (column 9). The row's own
    perf_time (column 0) is the same unless `event_perf` gives a board-timed row's aligned event time."""
    event = perf if event_perf is None else event_perf
    return (event, perf, "inbound", kind, "", None, None, None, json.dumps(payload),
            perf, perf, 0, boot, sequence, board_us, stamp_kind, None, None, None, "{}", event,
            "host_receive_perf_time", "host_timestamp")


def rig_ni_endpoints(indices, rate=RATE):
    """((NI seconds, host seconds) of the first sample, and of the last): what the recorder takes of the
    per-sample host stamps of a rig whose host clock is NI seconds plus NI_HOST_OFFSET."""
    first, last = float(indices[0]) / rate, float(indices[-1]) / rate
    return (first, NI_HOST_OFFSET + first), (last, NI_HOST_OFFSET + last)


def rig_ni_to_host():
    """The NI-to-host fit of a rig whose host clock is NI seconds plus NI_HOST_OFFSET."""
    return clocks.PiecewiseClockFit(
        np.array([0.0, 1e6]), np.array([NI_HOST_OFFSET, NI_HOST_OFFSET + 1e6]), 0.0, 1000, True, "",
    )


IO_ADDRESS, PELLET_ADDRESS, IO_BOOT, PELLET_BOOT = 8, 0, 0xB007, 0x7E11


def io_to_ni(io_s):
    return 2.0 + 1.00002 * io_s


def pellet_to_io(pellet_s):
    return 0.5 + 0.99999 * pellet_s


def synthetic_rig(*, seconds=30, tone_pellet_times=(15.25, 21.75)):
    """Claps and barcodes once a second, two pellet tones, sampled at RATE."""
    rows, frames, sequence = [], [], 1
    for tick in range(1, seconds):
        io_barcode_s = 10.0 + tick
        frames.append((io_to_ni(io_barcode_s), tick))
        rows.append(device_row(NI_HOST_OFFSET + io_to_ni(io_barcode_s) + BARCODE_REPORT_DELAY, "SYNC_BARCODE_REPORT",
                               {"board_address": IO_ADDRESS, "tick": tick},
                               IO_BOOT, sequence, int(round(io_barcode_s * 1e6)), "physical_start"))
        io_clap_s = io_barcode_s + 0.2
        pellet_clap_s = (io_clap_s - 0.5) / 0.99999
        rows.append(device_row(NI_HOST_OFFSET + io_to_ni(io_clap_s), "SYNC_CLAP_OBSERVATION",
                               {"board_address": IO_ADDRESS, "tick": tick, "clapper_node": IO_ADDRESS},
                               IO_BOOT, sequence + 1, int(round(io_clap_s * 1e6)), "observed"))
        rows.append(device_row(NI_HOST_OFFSET + io_to_ni(io_clap_s) + 0.002, "SYNC_CLAP_OBSERVATION",
                               {"board_address": PELLET_ADDRESS, "tick": tick, "clapper_node": IO_ADDRESS},
                               PELLET_BOOT, sequence, int(round(pellet_clap_s * 1e6)), "observed"))
        sequence += 2
    total = io_to_ni(10.0 + seconds) + 0.5
    tone1 = np.zeros(int(math.ceil(total * RATE)), dtype=np.float32)
    for pellet_s in tone_pellet_times:
        onset = int(math.ceil(io_to_ni(pellet_to_io(pellet_s)) * RATE))
        tone1[onset:onset + 2000] = 1.0
        rows.append(device_row(NI_HOST_OFFSET + io_to_ni(pellet_to_io(pellet_s)), "TONE_STATUS",
                               {"frequency_hz": 5000, "time_remaining_ms": 200},
                               PELLET_BOOT, 9000, int(round(pellet_s * 1e6)), "physical_start"))
    channels = {"barcode": barcode_line(RATE, total, frames)[: tone1.size], "tone1": tone1}
    return tuple(rows), channels, np.arange(tone1.size, dtype=np.int64)


def linear_clock(offset, drift):
    """A board clock: board seconds as a function of true seconds (true seconds are NI seconds)."""
    return lambda true_s: offset + drift * true_s


def wandering_clock(offset, drift, wander, seconds):
    """A clock whose drift moves linearly by `wander` over `seconds`, as a crystal does with temperature."""
    return lambda true_s: offset + drift * true_s + 0.5 * (wander / seconds) * true_s ** 2


def _true_time(clock, board_s):
    """The true second at which `clock` reads `board_s` (clocks run at about 1, so this converges fast)."""
    true_s = board_s
    for _ in range(8):
        true_s -= clock(true_s) - board_s
    return true_s


def clock_rig(*, io_eras, pellet_eras, ni_seconds, tones=(), rate=RATE, index_dtype=np.int64,
              pellet_event_skew=0.0):
    """Claps, barcodes and tones stamped by clocks that are functions of true time.

    True seconds are NI seconds: sample i is at i / rate. `io_eras` are the
    timekeeper's boots, (boot_id, clock, first_tick, count, first_true_s): its
    clock reads a whole number of seconds more at each tick, so tick n plays at
    clock(first_true_s) + n - first_tick on its own clock. `pellet_eras` are the
    pellet's boots, (boot_id, clock, from_true_s, to_true_s): the era a clap or
    tone falls in stamps it. `tones` are (true_s, frequency_hz). Rows outside
    0..ni_seconds are pre-roll and post-roll: the device rows exist, the NI-DAQ
    did not record them. An io era may end with False to say its barcodes never
    reached the NI-DAQ line. Host receive times (column 9) are NI_HOST_OFFSET plus
    true seconds, a barcode report BARCODE_REPORT_DELAY later than its start edge;
    the pellet's rows carry perf_time (column 0) `pellet_event_skew` away from
    that, as an aligned board time can be.
    """
    def pellet_era(true_s):
        return next((era for era in pellet_eras if era[2] <= true_s < era[3]), None)

    rows, frames, sequence = [], [], 1
    for boot, clock, first_tick, count, first_true_s, *on_ni in io_eras:
        first_board_s = clock(first_true_s)
        for tick in range(first_tick, first_tick + count):
            board_s = first_board_s + tick - first_tick
            start = _true_time(clock, board_s)
            rows.append(device_row(NI_HOST_OFFSET + start + BARCODE_REPORT_DELAY, "SYNC_BARCODE_REPORT",
                                   {"board_address": IO_ADDRESS, "tick": tick},
                                   boot, sequence, int(round(board_s * 1e6)), "physical_start"))
            clap = _true_time(clock, board_s + 0.2)
            rows.append(device_row(NI_HOST_OFFSET + clap, "SYNC_CLAP_OBSERVATION",
                                   {"board_address": IO_ADDRESS, "tick": tick, "clapper_node": IO_ADDRESS},
                                   boot, sequence + 1, int(round((board_s + 0.2) * 1e6)), "observed"))
            era = pellet_era(clap)
            if era is not None:
                rows.append(device_row(NI_HOST_OFFSET + clap + 0.002, "SYNC_CLAP_OBSERVATION",
                                       {"board_address": PELLET_ADDRESS, "tick": tick, "clapper_node": IO_ADDRESS},
                                       era[0], sequence, int(round(era[1](clap) * 1e6)), "observed",
                                       event_perf=NI_HOST_OFFSET + clap + 0.002 + pellet_event_skew))
            sequence += 2
            if (not on_ni or on_ni[0]) and start >= 0.05 and start + 0.04 <= ni_seconds:
                frames.append((start, tick))
    samples = int(math.ceil(ni_seconds * rate))
    channels = {"barcode": barcode_line(rate, ni_seconds, frames)[:samples],
                "tone1": np.zeros(samples, dtype=np.float32)}
    if any(frequency_hz == 6000 for _, frequency_hz in tones):
        channels["tone2"] = np.zeros(samples, dtype=np.float32)
    for true_s, frequency_hz in tones:
        era = pellet_era(true_s)
        rows.append(device_row(NI_HOST_OFFSET + true_s, "TONE_STATUS",
                               {"frequency_hz": frequency_hz, "time_remaining_ms": 200},
                               era[0], 9000, int(round(era[1](true_s) * 1e6)), "physical_start"))
        onset = int(math.ceil(true_s * rate))
        if 0 <= onset and onset + 2000 <= samples:
            channels["tone1" if frequency_hz == 5000 else "tone2"][onset:onset + 2000] = 1.0
    return tuple(rows), channels, np.arange(samples, dtype=index_dtype)
