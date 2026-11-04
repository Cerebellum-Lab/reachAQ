import json

import numpy as np
import pytest

from board_clock_chain_synthetic import RATE, barcode_line
from tools.acquisition.model import board_clock_chain as chain


def test_barcode_edges_follow_the_start_pulse_and_bits():
    assert chain.barcode_edges(0) == ((0.0, True), (0.001, False))
    # Bit 0 set merges with the start pulse; bit 31 set ends at 33 ms.
    edges = chain.barcode_edges(0x80000001)
    assert edges[0] == (0.0, True)
    assert edges[1] == pytest.approx((0.002, False))
    assert edges[-2] == pytest.approx((0.032, True))
    assert edges[-1] == pytest.approx((0.033, False))


@pytest.mark.parametrize("tick", [0, 1, 0x80000001, 0xFFFFFFFF, 0x12345678])
def test_frames_decode_by_position(tick):
    line = barcode_line(RATE, 2.0, [(0.5, tick), (1.5, tick + 1 & 0xFFFFFFFF)])

    frames, rejected = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [tick, (tick + 1) & 0xFFFFFFFF]
    assert rejected == 0
    assert frames[0].start_position == 5000
    assert frames[0].edges[0] == (5000, 0.0)
    assert len(frames[0].edges) == len(chain.barcode_edges(tick))


def test_a_frame_cut_by_the_recording_start_is_not_decoded():
    full = barcode_line(RATE, 2.0, [(0.010, 0x00F0F0F0), (1.0, 7)])
    line = full[int(0.025 * RATE):]  # starts 15 ms into the first frame

    frames, _ = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [7]


def test_a_glitch_inside_a_frame_rejects_it():
    line = barcode_line(RATE, 2.0, [(0.5, 0), (1.5, 9)])
    line[int(0.512 * RATE)] = 1.0  # a one-sample spike in a low bit of the first frame

    frames, rejected = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [9]
    assert rejected == 1


def test_a_truncated_last_frame_is_ignored():
    line = barcode_line(RATE, 1.52, [(0.5, 3), (1.5, 4)])  # second frame cut at 20 ms

    frames, _ = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [3]


def test_rates_below_the_minimum_decode_nothing():
    line = barcode_line(1000.0, 2.0, [(0.5, 3)])
    assert chain.decode_barcode_frames(line, 1000.0) == ((), 0)


def test_five_kilohertz_still_decodes():
    line = barcode_line(5000.0, 2.0, [(0.5, 0xA5A5A5A5)])
    frames, _ = chain.decode_barcode_frames(line, 5000.0)
    assert [frame.tick for frame in frames] == [0xA5A5A5A5]
