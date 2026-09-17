"""The preallocated stream readers, and what happens when one is unavailable.

There is no line-based many-sample digital read in nidaqmx to preallocate for.
The analog readers have read_many_sample; the digital readers have only the
per-PORT variants, and read_many_sample_multi_line is absent from every reader
class at 1.6.0, the current release. So the call never resolved on any
version: the rig raised AttributeError on the first digital chunk, failed the
signal stream, and blocked Record entirely - with no metadata written for any
session.
"""

from types import SimpleNamespace

import numpy
import pytest

from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController


class _HypotheticalDigitalReader:
    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_many_sample_multi_line(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called by these tests")


class _RealDigitalReader:
    """What nidaqmx actually ships: no line-based many-sample read."""

    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_one_sample_multi_line(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called by these tests")


class _AnalogReader:
    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_many_sample(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called by these tests")


def _controller(digital_reader_class):
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
    )


def _make(controller, *, analog, channels=("a", "b")):
    NidaqSignalStreamController._make_stream_reader(
        controller, "PXI1Slot5", SimpleNamespace(in_stream=object()), channels,
        analog=analog,
    )


def test_a_reader_that_offers_the_read_is_registered_for_it():
    controller = _controller(_HypotheticalDigitalReader)
    _make(controller, analog=False)
    assert isinstance(controller._digital_readers["PXI1Slot5"], _HypotheticalDigitalReader)
    assert controller._digital_buffers["PXI1Slot5"].shape == (2, 500)
    assert controller._digital_buffers["PXI1Slot5"].dtype == numpy.bool_


def test_the_shipped_nidaqmx_reader_registers_nothing():
    """Unregistered selects the task.read() path, which allocates per chunk.

    Slower than a preallocated read would be, and unboundedly better than a
    stream that fails on its first chunk and takes recording down with it.
    This is the path every real deployment takes today.
    """
    controller = _controller(_RealDigitalReader)
    _make(controller, analog=False)
    assert controller._digital_readers == {}
    assert controller._digital_buffers == {}


def test_the_analog_reader_is_unaffected():
    """Only the digital side lacks it; read_many_sample is a real analog API."""
    controller = _controller(_RealDigitalReader)
    _make(controller, analog=True)
    assert isinstance(controller._analog_readers["PXI1Slot5"], _AnalogReader)
    assert controller._analog_buffers["PXI1Slot5"].shape == (2, 500)


@pytest.mark.parametrize("missing", ["in_stream", "stream_readers"])
def test_nothing_is_registered_when_the_runtime_lacks_the_pieces(missing):
    controller = _controller(_HypotheticalDigitalReader)
    task = SimpleNamespace(in_stream=None if missing == "in_stream" else object())
    if missing == "stream_readers":
        controller._nidaqmx = SimpleNamespace(stream_readers=None)
    NidaqSignalStreamController._make_stream_reader(
        controller, "PXI1Slot5", task, ("a",), analog=False)
    assert controller._digital_readers == {}
