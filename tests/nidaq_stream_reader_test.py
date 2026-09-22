"""The preallocated stream readers, and what happens when one is unavailable.

There is no line-based many-sample digital read in nidaqmx - the analog
readers have read_many_sample, the digital readers have only the per-PORT
variants, and read_many_sample_multi_line is absent from every reader class.
An earlier version of this file drew the wrong conclusion from that true
observation: it treated the absence as proof that buffered digital could not
be preallocated for at all, and settled for task.read(), which allocates a
fresh list of lists every chunk.

The port variants are the point rather than a consolation. NI's buffered
digital path is port-based: a channel spanning several lines is read by
read_many_sample_port_uint32 into a buffer the stream already owns, and each
line is recovered from its bit. These tests pin that, and pin the fallback for
a nidaqmx that somehow lacks it.
"""

from types import SimpleNamespace

import numpy
import pytest

from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController


class _PortDigitalReader:
    """What nidaqmx ships: port reads, and no line-based many-sample read."""

    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_many_sample_port_uint32(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called by these tests")


class _ReaderWithoutPortRead:
    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_one_sample_multi_line(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called by these tests")


class _AnalogReader:
    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_many_sample(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called by these tests")


def _controller(digital_reader_class, port_count=1):
    readers = SimpleNamespace(
        AnalogMultiChannelReader=_AnalogReader,
        DigitalMultiChannelReader=digital_reader_class,
    )
    return SimpleNamespace(
        _nidaqmx=SimpleNamespace(stream_readers=readers, __version__="1.0.2"),
        _configuration=SimpleNamespace(read_chunk_size=500),
        _analog_readers={},
        _digital_readers={},
        _analog_buffers={},
        _digital_buffers={},
        _digital_port_counts={"PXI1Slot5": port_count},
    )


def _make(controller, *, analog, channels=("a", "b")):
    NidaqSignalStreamController._make_stream_reader(
        controller, "PXI1Slot5", SimpleNamespace(in_stream=object()), channels,
        analog=analog,
    )


def test_the_port_reader_is_registered_with_a_buffer_shaped_by_port():
    """One row per port, not per line: the lines come out of the bits."""
    controller = _controller(_PortDigitalReader, port_count=2)

    _make(controller, analog=False, channels=("a", "b", "c", "d"))

    assert isinstance(controller._digital_readers["PXI1Slot5"],
                      _PortDigitalReader)
    buffer = controller._digital_buffers["PXI1Slot5"]
    assert buffer.shape == (2, 500)
    assert buffer.dtype == numpy.uint32


def test_a_reader_without_the_port_read_registers_nothing():
    """Unregistered selects task.read(), which allocates but still acquires."""
    controller = _controller(_ReaderWithoutPortRead)

    _make(controller, analog=False)

    assert controller._digital_readers == {}
    assert controller._digital_buffers == {}


def test_the_analog_reader_is_preallocated_per_channel():
    """Only the digital side is per port; read_many_sample is per channel."""
    controller = _controller(_PortDigitalReader)

    _make(controller, analog=True)

    assert isinstance(controller._analog_readers["PXI1Slot5"], _AnalogReader)
    assert controller._analog_buffers["PXI1Slot5"].shape == (2, 500)
    assert controller._analog_buffers["PXI1Slot5"].dtype == numpy.float64


@pytest.mark.parametrize("missing", ["in_stream", "stream_readers"])
def test_nothing_is_registered_when_the_runtime_lacks_the_pieces(missing):
    controller = _controller(_PortDigitalReader)
    task = SimpleNamespace(in_stream=None if missing == "in_stream" else object())
    if missing == "stream_readers":
        controller._nidaqmx = SimpleNamespace(stream_readers=None)

    NidaqSignalStreamController._make_stream_reader(
        controller, "PXI1Slot5", task, ("a",), analog=False)

    assert controller._digital_readers == {}
