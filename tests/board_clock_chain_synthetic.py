"""Synthetic NI-DAQ lines and device rows for the board clock chain tests."""

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
