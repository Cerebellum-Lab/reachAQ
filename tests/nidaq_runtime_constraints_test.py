"""Three constraints the NI-DAQmx runtime imposes, each learned by a failure.

None of these is guessable from the API, and each produced a message that
pointed somewhere other than the cause, so they are pinned here rather than
left to be rediscovered by whoever meets them next.
"""

import multiprocessing
from types import SimpleNamespace

import pytest

from autotrainer.core.multiproc import get_mp_ctx, get_nidaq_mp_ctx
from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController


# ------------------------------------------------- the worker start method


def test_the_nidaq_context_is_not_the_spawn_everything_else_uses():
    """Spawn fails once NI-DAQmx is resident; the fault is not in our code.

    Measured on christielab10, one process moments apart: a spawned child
    starts before the laser loads and fails after it, with the exec dying at
    "[Errno 14] Bad address" naming the Python executable. Through
    multiprocessing that surfaces only as a child that exits 255 with no
    result, which is how the exact-task preflight reported it and why the
    signal stream refused to start at all.
    """
    nidaq = get_nidaq_mp_ctx()

    if "forkserver" in multiprocessing.get_all_start_methods():
        assert nidaq.get_start_method() == "forkserver"
        assert nidaq.get_start_method() != get_mp_ctx().get_start_method()
    else:
        # Windows has no forkserver, and no NI-DAQmx Linux runtime either.
        assert nidaq.get_start_method() == get_mp_ctx().get_start_method()


def test_the_nidaq_context_is_the_same_object_every_time():
    """One forkserver per process, and it must be the warmed one.

    A context asked for late looks identical and is not: starting the server
    is itself an exec, so a forkserver first started after the laser has
    loaded dies with a broken pipe. The one cached here was started while the
    process was still clean.
    """
    assert get_nidaq_mp_ctx() is get_nidaq_mp_ctx()


def test_everything_else_still_spawns():
    """The divergence is the NI-DAQ path's alone.

    Cameras and inference are untouched: nothing has shown they need a
    different start method, and that is not a thing to change across an
    application on one subsystem's evidence.
    """
    assert get_mp_ctx().get_start_method() == "spawn"


# --------------------------------------------------------- the declared rate


def _configuration(rate=10_000.0):
    return SimpleNamespace(sample_rate_hz=rate, is_enabled=True, channels=(1,))


def test_a_stream_clocked_at_another_rate_than_it_declares_is_refused():
    """C7. Nothing fails when these disagree, which is the problem.

    The rate handed to cfg_samp_clk_timing beside an external source does not
    set the rate - the clock does - it sizes the buffer, and every sample
    count downstream is computed from it. A tenfold disagreement once
    stretched a two-second pulse train to twenty seconds with no error
    anywhere.
    """
    plan = SimpleNamespace(sample_clock_rate_hz=1_000.0)

    with pytest.raises(RuntimeError) as refused:
        NidaqSignalStreamController._require_matching_clock_rate(
            _configuration(10_000.0), plan)

    message = str(refused.value)
    assert "10000 Hz" in message and "1000 Hz" in message
    assert "would really take 10 seconds" in message


def test_the_same_rate_through_yaml_and_a_plan_is_not_a_disagreement():
    """These are floats that have been through a file; exactness is wrong."""
    NidaqSignalStreamController._require_matching_clock_rate(
        _configuration(10_000.0),
        SimpleNamespace(sample_clock_rate_hz=10_000.000000001))


@pytest.mark.parametrize("plan", [
    None,
    SimpleNamespace(sample_clock_rate_hz=None),
    SimpleNamespace(sample_clock_rate_hz=0.0),
])
def test_a_plan_that_declares_no_clock_rate_is_not_second_guessed(plan):
    NidaqSignalStreamController._require_matching_clock_rate(
        _configuration(10_000.0), plan)


# ----------------------------------------------------- the transfer mechanism


class _Modes:
    INTERRUPT = "interrupt"
    DMA = "dma"
    POLLED = "polled"


def _controller(tasks=()):
    controller = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    controller._nidaqmx = SimpleNamespace(
        constants=SimpleNamespace(DataTransferActiveTransferMode=_Modes))
    controller._timing_plan = SimpleNamespace(
        task_graph=SimpleNamespace(tasks=tuple(tasks)))
    return controller


def _task(device="PXI1Slot5", subsystem="di", mechanism=None):
    return SimpleNamespace(device=device, subsystem=subsystem,
                           transfer_mechanism=mechanism)


def test_a_configured_transfer_mechanism_is_applied():
    """C4. It reached the plan and was read by nothing at all."""
    controller = _controller([_task(mechanism="dma")])

    assert controller._transfer_mechanism("PXI1Slot5", "di") == _Modes.DMA


def test_no_override_leaves_the_default_to_the_caller():
    controller = _controller([_task(mechanism=None), _task(subsystem="ai")])

    assert controller._transfer_mechanism("PXI1Slot5", "di") is None
    assert controller._transfer_mechanism("PXI1Slot5", "ai") is None
    assert controller._transfer_mechanism("PXI1Slot9", "di") is None


def test_a_mechanism_daqmx_does_not_have_is_refused_not_ignored():
    """The whole point of the setting is the case where the default is wrong.

    Silently dropping an unrecognised name would leave somebody believing an
    override took effect while the default they were trying to escape stayed
    in force.
    """
    controller = _controller([_task(mechanism="turbo")])

    with pytest.raises(RuntimeError) as refused:
        controller._transfer_mechanism("PXI1Slot5", "di")

    assert "turbo" in str(refused.value)
    assert "DMA" in str(refused.value) and "INTERRUPT" in str(refused.value)


def test_the_mechanism_is_matched_per_device_and_subsystem():
    controller = _controller([
        _task(device="PXI1Slot5", subsystem="di", mechanism="interrupt"),
        _task(device="PXI1Slot5", subsystem="ai", mechanism="polled"),
        _task(device="PXI1Slot4", subsystem="di", mechanism="dma"),
    ])

    assert controller._transfer_mechanism("PXI1Slot5", "di") == _Modes.INTERRUPT
    assert controller._transfer_mechanism("PXI1Slot5", "ai") == _Modes.POLLED
    assert controller._transfer_mechanism("PXI1Slot4", "di") == _Modes.DMA


# ------------------------------------------------- per-channel referencing


def test_referencing_is_asked_of_each_channel_rather_than_forced():
    """A board's AO readback channel reports DIFF and only DIFF.

    Naming RSE on it fails the whole task with "requested value is not a
    supported value", which takes down the sixteen real inputs that do accept
    RSE. Measured: 16 inputs at RSE plus two readbacks at their own stream
    together in one task.
    """
    controller = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    controller._terminal_config_support = {}
    supported = {"PXI1Slot5/ai0": ("RSE", "NRSE", "DIFF"),
                 "PXI1Slot5/_ao0_vs_aognd": ("DIFF",)}
    controller._nidaqmx = SimpleNamespace(system=SimpleNamespace(
        PhysicalChannel=lambda name: SimpleNamespace(
            ai_term_cfgs=supported[name])))

    assert controller._channel_accepts("PXI1Slot5/ai0", "RSE")
    assert not controller._channel_accepts("PXI1Slot5/_ao0_vs_aognd", "RSE")
    assert controller._channel_accepts("PXI1Slot5/_ao0_vs_aognd", "DIFF")


def test_a_channel_that_cannot_be_asked_keeps_what_was_configured():
    """This can only ever remove a setting that would have failed."""
    controller = NidaqSignalStreamController.__new__(NidaqSignalStreamController)
    controller._terminal_config_support = {}
    controller._nidaqmx = SimpleNamespace(system=SimpleNamespace(
        PhysicalChannel=lambda name: (_ for _ in ()).throw(RuntimeError("no"))))

    assert controller._channel_accepts("Dev1/ai0", "RSE")
    # And the answer is remembered, so a task is not re-probed per channel.
    assert "Dev1/ai0" in controller._terminal_config_support
