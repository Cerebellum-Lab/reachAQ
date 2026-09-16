"""The preallocated stream readers, and what happens when one is unavailable.

read_many_sample_multi_line arrived in nidaqmx 1.1, which needs Python 3.9. A
3.8 deployment can install no newer than 1.0.2, so the bench rig raised
AttributeError on the first digital chunk, failed the signal stream, and
blocked Record entirely - with no metadata written for any session.
"""

from types import SimpleNamespace

import numpy
import pytest

from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController


class _ModernDigitalReader:
    def __init__(self, in_stream):
        self.in_stream = in_stream

    def read_many_sample_multi_line(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("not called by these tests")


class _LegacyDigitalReader:
    """nidaqmx 1.0.2: one-sample reads only, no preallocated many-sample read."""

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


def test_modern_nidaqmx_registers_the_preallocated_digital_reader():
    controller = _controller(_ModernDigitalReader)
    _make(controller, analog=False)
    assert isinstance(controller._digital_readers["PXI1Slot5"], _ModernDigitalReader)
    assert controller._digital_buffers["PXI1Slot5"].shape == (2, 500)
    assert controller._digital_buffers["PXI1Slot5"].dtype == numpy.bool_


def test_legacy_nidaqmx_registers_no_digital_reader():
    """Unregistered selects the task.read() path, which allocates per chunk.

    Slower than the preallocated read, and unboundedly better than a stream
    that fails on its first chunk and takes recording down with it.
    """
    controller = _controller(_LegacyDigitalReader)
    _make(controller, analog=False)
    assert controller._digital_readers == {}
    assert controller._digital_buffers == {}


def test_the_analog_reader_is_unaffected():
    """Only the digital reader lost a method; read_many_sample is in 1.0.2."""
    controller = _controller(_LegacyDigitalReader)
    _make(controller, analog=True)
    assert isinstance(controller._analog_readers["PXI1Slot5"], _AnalogReader)
    assert controller._analog_buffers["PXI1Slot5"].shape == (2, 500)


@pytest.mark.parametrize("missing", ["in_stream", "stream_readers"])
def test_nothing_is_registered_when_the_runtime_lacks_the_pieces(missing):
    controller = _controller(_ModernDigitalReader)
    task = SimpleNamespace(in_stream=None if missing == "in_stream" else object())
    if missing == "stream_readers":
        controller._nidaqmx = SimpleNamespace(stream_readers=None)
    NidaqSignalStreamController._make_stream_reader(
        controller, "PXI1Slot5", task, ("a",), analog=False)
    assert controller._digital_readers == {}
