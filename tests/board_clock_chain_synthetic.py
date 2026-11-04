"""Synthetic NI-DAQ lines and device rows for the board clock chain tests."""

import json
import math

import numpy as np

RATE = 10_000.0


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


def device_row(perf, kind, payload, boot, sequence, board_us, stamp_kind):
    return (perf, perf, "inbound", kind, "", None, None, None, json.dumps(payload),
            perf, perf, 0, boot, sequence, board_us, stamp_kind, None, None, None, "{}", perf,
            "host_receive_perf_time", "host_timestamp")


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
        rows.append(device_row(1000 + tick, "SYNC_BARCODE_REPORT", {"board_address": IO_ADDRESS, "tick": tick},
                               IO_BOOT, sequence, int(round(io_barcode_s * 1e6)), "physical_start"))
        io_clap_s = io_barcode_s + 0.2
        pellet_clap_s = (io_clap_s - 0.5) / 0.99999
        rows.append(device_row(1000 + tick, "SYNC_CLAP_OBSERVATION",
                               {"board_address": IO_ADDRESS, "tick": tick, "clapper_node": IO_ADDRESS},
                               IO_BOOT, sequence + 1, int(round(io_clap_s * 1e6)), "observed"))
        rows.append(device_row(1000 + tick, "SYNC_CLAP_OBSERVATION",
                               {"board_address": PELLET_ADDRESS, "tick": tick, "clapper_node": IO_ADDRESS},
                               PELLET_BOOT, sequence, int(round(pellet_clap_s * 1e6)), "observed"))
        sequence += 2
    total = io_to_ni(10.0 + seconds) + 0.5
    tone1 = np.zeros(int(math.ceil(total * RATE)), dtype=np.float32)
    for pellet_s in tone_pellet_times:
        onset = int(math.ceil(io_to_ni(pellet_to_io(pellet_s)) * RATE))
        tone1[onset:onset + 2000] = 1.0
        rows.append(device_row(1000 + pellet_s, "TONE_STATUS", {"frequency_hz": 5000, "time_remaining_ms": 200},
                               PELLET_BOOT, 9000, int(round(pellet_s * 1e6)), "physical_start"))
    channels = {"barcode": barcode_line(RATE, total, frames)[: tone1.size], "tone1": tone1}
    return tuple(rows), channels, np.arange(tone1.size, dtype=np.int64)
