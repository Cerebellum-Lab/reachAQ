"""A reference clock is set only on boards that actually have one.

The timing plan carries a single terminal name for the whole chassis, but the
boards differ: a PXI-6221 exposes /PXI1Slot5/PXI_Clk10 and the PXI-6713 beside
it exposes no Clk10 at all. Setting a reference clock on the board without one
fails the task with "property is not supported by the device", which took the
whole signal stream down as soon as hardware-timed laser output was enabled.
"""

import types

import pytest

from autotrainer.device.nidaq_signal_stream import (
    NidaqSignalStreamController,
)


class _Device:
    def __init__(self, terminals):
        self.terminals = terminals


def _stream(devices):
    """A stream stub carrying only what the reference-clock lookup reads."""
    stream = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    stream._nidaqmx = types.SimpleNamespace(
        system=types.SimpleNamespace(
            Device=lambda name: _Device(devices[name])))
    return stream


SLOT5 = ["/PXI1Slot5/ai/StartTrigger", "/PXI1Slot5/PXI_Clk10"]
SLOT4 = ["/PXI1Slot4/ao/StartTrigger"]          # a 6713: no Clk10


def _plan(source, rate=10_000_000.0):
    return types.SimpleNamespace(reference_clock_source=source,
                                 reference_clock_rate_hz=rate)


def test_a_board_without_the_terminal_gets_no_reference_clock():
    stream = _stream({"PXI1Slot4": SLOT4})
    stream._timing_plan = _plan("PXI_CLK10")

    assert stream._reference_clock_for("PXI1Slot4") is None


def test_a_board_with_the_terminal_gets_its_own_qualified_name():
    stream = _stream({"PXI1Slot5": SLOT5})
    stream._timing_plan = _plan("PXI_CLK10")

    assert stream._reference_clock_for("PXI1Slot5") == "/PXI1Slot5/PXI_Clk10"


def test_the_default_spelling_is_corrected_to_the_device_s_own():
    """The plan writes PXI_CLK10; NI calls the terminal PXI_Clk10."""
    stream = _stream({"PXI1Slot5": SLOT5})
    stream._timing_plan = _plan("PXI_CLK10")

    assert stream._reference_clock_for("PXI1Slot5").endswith("PXI_Clk10")


def test_an_explicitly_configured_terminal_is_left_alone():
    """Naming a terminal is deliberate and must not be second-guessed."""
    stream = _stream({"PXI1Slot5": SLOT5})
    stream._timing_plan = _plan("/PXI1Slot9/PXI_Clk10")

    assert stream._reference_clock_for("PXI1Slot5") == "/PXI1Slot9/PXI_Clk10"


def test_no_configured_source_means_no_reference_clock():
    stream = _stream({"PXI1Slot5": SLOT5})
    stream._timing_plan = _plan(None)

    assert stream._reference_clock_for("PXI1Slot5") is None


def test_an_unreadable_device_is_left_unset_rather_than_guessed():
    def explode(_name):
        raise RuntimeError("device offline")

    stream = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    stream._nidaqmx = types.SimpleNamespace(
        system=types.SimpleNamespace(Device=explode))
    stream._timing_plan = _plan("PXI_CLK10")

    assert stream._reference_clock_for("PXI1Slot5") is None


def test_the_task_is_only_touched_when_a_clock_was_resolved():
    stream = _stream({"PXI1Slot4": SLOT4})
    stream._timing_plan = _plan("PXI_CLK10")
    timing = types.SimpleNamespace(ref_clk_src="untouched", ref_clk_rate=0.0)
    task = types.SimpleNamespace(timing=timing)

    stream._configure_reference_clock(task, "PXI1Slot4")

    assert timing.ref_clk_src == "untouched"


def test_the_rate_is_applied_alongside_a_resolved_clock():
    stream = _stream({"PXI1Slot5": SLOT5})
    stream._timing_plan = _plan("PXI_CLK10")
    timing = types.SimpleNamespace(ref_clk_src=None, ref_clk_rate=None)
    task = types.SimpleNamespace(timing=timing)

    stream._configure_reference_clock(task, "PXI1Slot5")

    assert timing.ref_clk_src == "/PXI1Slot5/PXI_Clk10"
    assert timing.ref_clk_rate == pytest.approx(10_000_000.0)


class _RejectingTiming:
    """A task whose type does not accept a reference clock.

    A PXI-6221 exposes PXI_Clk10 but its digital-input task rejects
    DAQmx_RefClk_Src with -200452, "not applicable to the task".
    """

    def __init__(self):
        self.ref_clk_rate = None

    @property
    def ref_clk_src(self):
        return None

    @ref_clk_src.setter
    def ref_clk_src(self, _value):
        raise RuntimeError(
            "Specified property is not supported by the device or is not "
            "applicable to the task. Property: DAQmx_RefClk_Src")


def test_a_task_that_rejects_the_clock_does_not_fail_the_whole_stream():
    stream = _stream({"PXI1Slot5": SLOT5})
    stream._timing_plan = _plan("PXI_CLK10")
    task = types.SimpleNamespace(timing=_RejectingTiming())

    # Must not raise: the task runs on its own timebase instead.
    stream._configure_reference_clock(task, "PXI1Slot5")


def test_a_rejecting_task_leaves_the_rate_unset():
    stream = _stream({"PXI1Slot5": SLOT5})
    stream._timing_plan = _plan("PXI_CLK10")
    timing = _RejectingTiming()
    task = types.SimpleNamespace(timing=timing)

    stream._configure_reference_clock(task, "PXI1Slot5")

    assert timing.ref_clk_rate is None


class _Stream:
    def __init__(self, available):
        self.avail_samp_per_chan = available


class _UnbufferedStream:
    """A polled task holds no host buffer, so the query itself fails."""

    @property
    def avail_samp_per_chan(self):
        raise RuntimeError("not a buffered input task")


def _described(records):
    stream = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    stream._owned_task_records = lambda: records
    return stream._describe_task_state()


def _task(available, *, acquires=True, done=False, fault=None):
    def is_task_done():
        if fault is not None:
            raise fault
        return done

    channels = [object()] if acquires else []
    return types.SimpleNamespace(
        in_stream=_Stream(available) if available is not None
        else _UnbufferedStream(),
        ai_channels=channels, di_channels=[], is_task_done=is_task_done)


def test_each_task_is_named_with_what_it_is_holding():
    """"available=(513, 0)" said a task was starved without saying which."""
    described = _described([("stream_PXI1Slot5_ai", _task(513)),
                            ("stream_PXI1Slot5_di", _task(0))])

    assert "stream_PXI1Slot5_ai=513[running]" in described
    assert "stream_PXI1Slot5_di=0[running]" in described


def test_a_task_with_no_input_channels_is_left_out():
    """A pure output task's avail_samp_per_chan is zero forever.

    Reporting it beside a genuinely starved input would read as two stalled
    tasks, which is how the PXI-6713 made the barrier wait for a number that
    could never arrive.
    """
    described = _described([("stream_PXI1Slot5_ai", _task(0)),
                            ("stream_PXI1Slot4_ao", _task(0, acquires=False))])

    assert "stream_PXI1Slot4_ao" not in described
    assert described == "stream_PXI1Slot5_ai=0[running]"


def test_a_faulted_task_is_told_apart_from_one_that_never_started():
    """DAQmx answers both through the same query; only the text separates them."""
    described = _described([("stream_PXI1Slot5_ai", _task(
        0, fault=RuntimeError("Onboard device memory overflow")))])

    assert "not-running(RuntimeError: Onboard device memory overflow)" in described


def test_a_task_with_no_host_buffer_says_so_rather_than_zero():
    described = _described([("stream_PXI1Slot5_ai", _task(None))])

    assert "stream_PXI1Slot5_ai=unbuffered[running]" == described
