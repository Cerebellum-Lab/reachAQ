"""Buffered digital is read per port; these pin where each line ends up.

The rewrite that introduced this read the port correctly and then unpacked it
wrongly, putting tone1's signal on cam_frames - because the bit was taken from
the line's position in the configured channel list rather than from its line
number. On this rig the configuration lists lines 2, 3, 0, 1 in that order, so
the two disagree, and the stream recorded four channels' worth of plausible
nonsense until a driven line proved it.
"""

from types import SimpleNamespace

import numpy
import pytest

from autotrainer.device.nidaq_signal_stream import (
    NidaqSignalStreamController,
    _line_number,
)


def _channel(physical_channel):
    return SimpleNamespace(name=physical_channel.rsplit("/", 1)[-1],
                           physical_channel=physical_channel)


def _controller(layouts=None, ports=None):
    stream = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    stream._digital_layouts = dict(layouts or {})
    stream._digital_port_counts = dict(ports or {})
    return stream


def test_a_lines_bit_is_its_line_number():
    assert _line_number("PXI1Slot5/port0/line0") == 0
    assert _line_number("PXI1Slot5/port0/line3") == 3
    assert _line_number("PXI1Slot5/port2/line15") == 15


def test_a_channel_that_names_no_line_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        _line_number("PXI1Slot5/port0")


def test_each_line_is_unpacked_from_its_own_bit():
    """The configured order is 2, 3, 0, 1 here, which is the whole point."""
    stream = _controller(layouts={"Dev1": ((0, 2), (0, 3), (0, 0), (0, 1))})
    # One port word per sample: only line 0 high, then only line 3 high.
    ports = numpy.array([[0b0001, 0b1000]], dtype=numpy.uint32)

    rows = stream._unpack_digital("Dev1", ports, 4)

    assert [list(row) for row in rows] == [
        [0, 0],   # line2
        [0, 1],   # line3
        [1, 0],   # line0
        [0, 0],   # line1
    ]


def test_lines_on_different_ports_come_from_different_words():
    stream = _controller(layouts={"Dev1": ((0, 1), (1, 1))})
    ports = numpy.array([[0b0010], [0b0000]], dtype=numpy.uint32)

    rows = stream._unpack_digital("Dev1", ports, 2)

    assert [list(row) for row in rows] == [[1], [0]]


def test_a_high_line_number_survives_a_wide_port():
    stream = _controller(layouts={"Dev1": ((0, 31),)})
    ports = numpy.array([[1 << 31]], dtype=numpy.uint32)

    assert list(stream._unpack_digital("Dev1", ports, 1)[0]) == [1]


def test_without_a_layout_the_words_are_passed_through_unchanged():
    """The fallback path, for a task built before a layout was recorded."""
    stream = _controller()
    ports = numpy.array([[1, 0, 1]], dtype=numpy.uint32)

    rows = stream._unpack_digital("Dev1", ports, 1)

    assert [list(row) for row in rows] == [[1, 0, 1]]
