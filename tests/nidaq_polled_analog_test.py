"""Analog input is read straight from the FIFO, and waited on differently.

This chassis's IOMMU blocks the NI board's DMA, so the analog task is put on
the polled transfer mechanism, where the board's FIFO is the only buffer there
is. Two things follow, and both are easy to lose in a later edit: such a task
cannot be asked how many samples are available, and an overflow means the
reader was late rather than that the host buffer was small.
"""

import types

import pytest

from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController


class _Channels:
    """The `all` collection nidaqmx exposes for setting channel properties."""

    def __init__(self, accepts=True):
        object.__setattr__(self, "accepts", accepts)
        object.__setattr__(self, "ai_data_xfer_mech", None)

    def __setattr__(self, name, value):
        if name == "ai_data_xfer_mech" and not self.accepts:
            raise RuntimeError("Specified property is not supported")
        object.__setattr__(self, name, value)


class _Task:
    def __init__(self, accepts=True):
        self.ai_channels = types.SimpleNamespace(all=_Channels(accepts))


class _InStream:
    """A polled task refuses the availability property outright."""

    @property
    def avail_samp_per_chan(self):
        raise RuntimeError("the task is not a buffered input task")


class _PolledTask:
    in_stream = _InStream()
    ai_channels = types.SimpleNamespace(all=[object()])


def _stream(polled=()):
    stream = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    stream._nidaqmx = types.SimpleNamespace(
        constants=types.SimpleNamespace(
            DataTransferActiveTransferMode=types.SimpleNamespace(POLLED="polled")
        )
    )
    stream._polled_analog_devices = set(polled)
    return stream


def test_the_analog_task_is_taken_off_dma():
    stream = _stream()
    task = _Task()

    stream._configure_analog_transfer(task, "PXI1Slot5")

    assert task.ai_channels.all.ai_data_xfer_mech == "polled"
    assert stream._polled_analog_devices == {"PXI1Slot5"}


def test_a_board_that_refuses_the_mechanism_is_not_recorded_as_polled():
    """It keeps the driver default, so it must still be waited on normally."""
    stream = _stream()

    stream._configure_analog_transfer(_Task(accepts=False), "PXI1Slot5")

    assert stream._polled_analog_devices == set()


def test_the_barrier_does_not_ask_a_polled_task_how_much_is_available():
    """hasattr does not catch a DaqError, so the check has to come first."""
    stream = _stream(polled={"PXI1Slot5"})
    stream._owned_task_records = lambda: (("PXI1Slot5.ai", _PolledTask()),)
    stream._read_telemetry = {"late_barriers": 0}

    # No task is left to wait on, so this returns rather than raising.
    stream._wait_all_available(167, 1.0)


def test_a_polled_overflow_says_what_actually_overran():
    stream = _stream(polled={"PXI1Slot5"})
    stream._configuration = types.SimpleNamespace(
        analog_channels=(), sample_rate_hz=10_000.0
    )
    stream._channels_by_device = lambda _channels: {
        "PXI1Slot5": tuple(range(10))
    }

    explained = stream._explain_analog_read_failure(
        "PXI1Slot5", RuntimeError("Onboard device memory overflow.")
    )

    message = str(explained)
    assert "10 channels at 10000 Hz" in message
    assert "intel_iommu=off" in message


def test_an_unrelated_failure_is_passed_through_untouched():
    """Only the overflow has this explanation; nothing else is reinterpreted."""
    stream = _stream(polled={"PXI1Slot5"})
    original = RuntimeError("Some or all of the samples requested...")

    assert stream._explain_analog_read_failure("PXI1Slot5", original) is original
