"""A pulse train's clocked digital outputs, against a stand-in for NI-DAQmx.

The PMT shutter and each channel's trigger output and timing trigger output
are clocked digital output tasks beside the laser's analog output task. Each
named its sample clock and its start trigger on the analog output's board,
/PXI1Slot4/ao/SampleClock and /PXI1Slot4/PXI_Trig0 on christielab10. A clocked
digital line has to sit on the PXI-6221 there, so both names crossed boards:
the implicit route DAQmx refuses on this unidentified chassis with -89125.
christielab10 configures none of these lines, so nothing failed yet.

Nothing here touches a driver or a board (nidaq_daqmx_fake).
"""

import dataclasses

import pytest

from autotrainer.core import NidaqTimingPlan
from autotrainer.device import (
    LaserChannelId,
    LaserOperationState,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser

from nidaq_daqmx_fake import FakeDaqmx, rig_lasers


#: The pellet board's STIM line, arriving on the 6221's PFI0 and driven onto
#: PXI_Trig0 for the 6713, which arms on its own view of that line.
BOARD_STIM = "/PXI1Slot4/PXI_Trig0"
TRIGGER_ROUTE = ("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")
#: The 6713's own clock, and the backplane route the ramp puts it on.
AO_CLOCK_ROUTE = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")
#: The input stream's clock, which a synchronized pulse train runs on.
SHARED_CLOCK = "/PXI1Slot5/ai/SampleClock"
SHARED_CLOCK_ROUTE = (SHARED_CLOCK, "/PXI1Slot5/PXI_Trig1")

#: Each output: the laser configuration it needs, the pulse asking for it,
#: its task's name and its line.
OUTPUTS = {
    "pmt_shutter": (
        dict(), dict(enable_pmt_shutter=True),
        "laser_pmt_shutter_do", "PXI1Slot5/port0/line6"),
    "trigger_output": (
        dict(trigger_output="PXI1Slot5/port0/line7"),
        dict(emit_trigger_output=True),
        "laser_1_trigger_do", "PXI1Slot5/port0/line7"),
    "timing_trigger_output": (
        dict(timing_trigger_output="PXI1Slot5/port0/line7"),
        dict(emit_timing_trigger_output=True),
        "laser_1_timing_trigger_do", "PXI1Slot5/port0/line7"),
}


@pytest.fixture
def daq(monkeypatch):
    fake = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: fake)
    return fake


def _synchronized_plan(clock_device="PXI1Slot5", output_device="PXI1Slot4"):
    """christielab10's plan: the 6221 clocks the stream, the 6713 is its output."""
    return NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device=clock_device,
        slave_devices=(output_device,),
        sample_clock_source=f"/{clock_device}/ai/SampleClock",
        sample_clock_rate_hz=10_000.0,
        start_trigger_source=f"/{clock_device}/ai/StartTrigger",
        hardware_output_devices=(output_device,),
        hardware_output_timing_status="declared_not_armed",
    )


def _controller(plan=None, **channel):
    """The rig's split, with its board STIM route, as the controller opens it."""
    values = dict(trigger_source=BOARD_STIM, trigger_route_source="/PXI1Slot5/PFI0")
    values.update(channel)
    return NidaqLaserController(rig_lasers(**values), timing_plan=plan)


def _pulse(*, trigger_source=BOARD_STIM, wait=True, defer_start=False, **fields):
    return LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=1.0,
            **fields,
        ),),
        trigger_source=trigger_source,
        wait=wait,
        defer_start=defer_start,
        timeout_seconds=5.0,
    )


def _output(name):
    configuration, fields, task, line = OUTPUTS[name]
    return configuration, fields, task, line


# ------------------------------------------------ unsynchronized: the AO clock


@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_a_cross_board_output_takes_the_ao_clock_over_the_backplane(daq, output):
    # With no shared timing the AO runs on its own clock. That clock ticks
    # only once the AO has been triggered, and the digital tasks start
    # before the AO: slaved to it, they need no start trigger of their own,
    # and naming the AO board's was the cross-board route DAQmx refuses.
    configuration, fields, task_name, line = _output(output)
    controller = _controller(**configuration)

    controller.run_synchronized_pulse_train(_pulse(**fields))

    digital = daq.task(task_name)
    analog = daq.task("laser_sync_pulse_ao")
    assert digital.channels == [line]
    assert digital.timing_kwargs["source"] == "/PXI1Slot5/PXI_Trig1"
    assert digital.start_trigger is None
    # The AO arms as it did, on its own clock.
    assert analog.start_trigger == (BOARD_STIM, "rising")
    assert "source" not in analog.timing_kwargs
    assert daq.starts.index(digital.name) < daq.starts.index(analog.name)
    # Held for the pulse alone; the controller's trigger route stays.
    assert daq.connected == [TRIGGER_ROUTE, AO_CLOCK_ROUTE]
    assert daq.disconnected == [AO_CLOCK_ROUTE]
    assert controller._trigger_routes == [TRIGGER_ROUTE]
    controller.close()


def test_the_ao_clock_route_is_released_when_the_pulse_fails(monkeypatch):
    daq = FakeDaqmx(failing_task="laser_sync_pulse_ao")
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = _controller()

    with pytest.raises(RuntimeError, match="refused to start"):
        controller.run_synchronized_pulse_train(_pulse(enable_pmt_shutter=True))

    assert daq.connected == [TRIGGER_ROUTE, AO_CLOCK_ROUTE]
    assert daq.disconnected == [AO_CLOCK_ROUTE]
    assert controller._trigger_routes == [TRIGGER_ROUTE]
    assert daq.reserved == {}


def test_the_ao_clock_route_is_released_when_the_pulse_is_cancelled(daq):
    # A deferred software start, as a protocol's direct NI route arms one.
    controller = _controller()
    operation = controller.run_synchronized_pulse_train(_pulse(
        trigger_source=None, wait=False, defer_start=True,
        enable_pmt_shutter=True))
    assert daq.connected == [TRIGGER_ROUTE, AO_CLOCK_ROUTE]

    assert operation.cancel()
    assert operation.wait(5.0) is LaserOperationState.CANCELLED

    assert daq.disconnected == [AO_CLOCK_ROUTE]
    assert controller._trigger_routes == [TRIGGER_ROUTE]
    assert daq.starts == []


# -------------------------------------- synchronized: the shared clock, locally


@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_a_synchronized_cross_board_output_takes_the_shared_clock_on_its_own_board(
    daq, output,
):
    # In a synchronized pulse train the 6221's clock already runs to the
    # 6713 on PXI_Trig1, for as long as the controller is open. Putting the
    # 6713's AO clock on that line as well would be two drivers on one line,
    # which DAQmx cannot see across these boards. The digital task takes the
    # shared clock as its own board sees it. That clock runs whether or not
    # the AO has triggered, so the task keeps its start trigger, named on
    # its own board: PXI_Trig lines are bussed, and both boards see PXI_Trig0.
    configuration, fields, task_name, line = _output(output)
    controller = _controller(_synchronized_plan(), **configuration)

    controller.run_synchronized_pulse_train(_pulse(**fields))

    digital = daq.task(task_name)
    analog = daq.task("laser_sync_pulse_ao")
    assert digital.channels == [line]
    assert digital.timing_kwargs["source"] == SHARED_CLOCK
    assert digital.timing_kwargs["rate"] == 10_000.0
    assert digital.start_trigger == ("/PXI1Slot5/PXI_Trig0", "rising")
    assert analog.timing_kwargs["source"] == "/PXI1Slot4/PXI_Trig1"
    assert analog.start_trigger == (BOARD_STIM, "rising")
    assert daq.starts.index(digital.name) < daq.starts.index(analog.name)
    # Nothing puts the 6713's clock anywhere; the shared clock's route is the
    # one the pulse train always made, and stays with the controller.
    assert daq.connected == [TRIGGER_ROUTE, SHARED_CLOCK_ROUTE]
    assert daq.disconnected == []
    assert controller._trigger_routes == [TRIGGER_ROUTE, SHARED_CLOCK_ROUTE]
    controller.close()


def test_a_synchronized_output_refuses_a_trigger_its_board_cannot_see(daq):
    # A PFI on the AO board reaches the AO and nothing else: no bussed line
    # carries it to the 6221.
    controller = NidaqLaserController(rig_lasers(), timing_plan=_synchronized_plan())

    with pytest.raises(RuntimeError) as refused:
        controller.run_synchronized_pulse_train(_pulse(
            trigger_source="/PXI1Slot4/PFI0", enable_pmt_shutter=True))

    message = str(refused.value)
    assert "PXI1Slot5/port0/line6" in message and "/PXI1Slot4/PFI0" in message
    assert "PXI_Trig" in message
    assert daq.starts == []
    assert AO_CLOCK_ROUTE not in daq.connected
    assert daq.reserved == {}


# --------------------------------------------------- the backplane clock line


def test_an_unsynchronized_output_refuses_a_clock_line_the_shared_clock_holds(daq):
    # A synchronized pulse leaves the 6221's clock on PXI_Trig1 until the
    # controller closes. An unsynchronized one after it, the AO on its own
    # clock, would drive the 6713's clock onto the same line.
    controller = _controller(_synchronized_plan())
    controller.run_synchronized_pulse_train(_pulse())
    assert SHARED_CLOCK_ROUTE in daq.connected
    started = len(daq.starts)

    with pytest.raises(RuntimeError) as refused:
        controller.run_synchronized_pulse_train(_pulse(
            trigger_source=None, enable_pmt_shutter=True))

    message = str(refused.value)
    assert "PXI_Trig1" in message and SHARED_CLOCK in message
    assert AO_CLOCK_ROUTE not in daq.connected
    assert daq.starts[started:] == []
    assert daq.reserved == {}
    controller.close()


def test_the_clock_line_is_not_driven_over_a_trigger_route(daq):
    # A backplane clock line on the line the stimulus takes. A configuration
    # refuses that as it is built (LaserSystemConfiguration); the controller
    # refuses it too, for one that reaches it another way.
    configuration = rig_lasers(trigger_source=BOARD_STIM,
                               trigger_route_source="/PXI1Slot5/PFI0")
    object.__setattr__(configuration, "backplane_clock_line", "PXI_Trig0")
    controller = NidaqLaserController(configuration)

    with pytest.raises(RuntimeError) as refused:
        controller.run_synchronized_pulse_train(_pulse(enable_pmt_shutter=True))

    message = str(refused.value)
    assert "PXI_Trig0" in message and "/PXI1Slot5/PFI0" in message
    assert daq.connected == [TRIGGER_ROUTE]
    assert daq.starts == []


def test_the_clock_line_is_not_driven_over_the_streams_own_export(daq):
    # The stream exports its sample clock to a configured terminal, from its
    # own process; that line is driven too.
    plan = dataclasses.replace(
        _synchronized_plan(), hardware_output_devices=(),
        sample_clock_export_terminal="/PXI1Slot5/PXI_Trig1")
    controller = _controller(plan)

    with pytest.raises(RuntimeError, match="PXI_Trig1"):
        controller.run_synchronized_pulse_train(_pulse(enable_pmt_shutter=True))

    assert AO_CLOCK_ROUTE not in daq.connected
    assert daq.starts == []


# ---------------------------------------------------------------- unchanged


@pytest.mark.parametrize("plan", [None, _synchronized_plan("PXI1Slot5", "PXI1Slot5")],
                         ids=["unsynchronized", "synchronized"])
@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_an_output_on_the_ao_board_keeps_its_clock_and_trigger(daq, output, plan):
    configuration, fields, task_name, line = _output(output)
    controller = _controller(
        plan, analog_output="PXI1Slot5/ao0", trigger_source="/PXI1Slot5/PFI0",
        trigger_route_source=None, **configuration)

    controller.run_synchronized_pulse_train(_pulse(trigger_source="/PXI1Slot5/PFI0", **fields))

    digital = daq.task(task_name)
    assert digital.timing_kwargs["source"] == "/PXI1Slot5/ao/SampleClock"
    assert digital.start_trigger == ("/PXI1Slot5/PFI0", "rising")
    assert daq.connected == []
    assert daq.disconnected == []


@pytest.mark.parametrize("plan", [None, _synchronized_plan()],
                         ids=["unsynchronized", "synchronized"])
def test_a_pulse_with_no_digital_outputs_makes_the_same_calls(daq, plan):
    controller = _controller(plan)
    created = len(daq.tasks)

    controller.run_synchronized_pulse_train(_pulse())

    calls = [(task.name, tuple(task.channels), task.timing_kwargs, task.start_trigger)
             for task in daq.tasks[created:]]
    synchronized = plan is not None
    samples = 10 if synchronized else 100
    analog_timing = dict(rate=10_000.0 if synchronized else 100_000.0,
                         sample_mode="finite", samps_per_chan=samples)
    if synchronized:
        analog_timing["source"] = "/PXI1Slot4/PXI_Trig1"
    assert calls == [
        ("laser_sync_pulse_ao", ("PXI1Slot4/ao0",), analog_timing,
         (BOARD_STIM, "rising")),
        ("laser_1_manual_ao", ("PXI1Slot4/ao0",), None, None),
    ]
    assert daq.starts == ["laser_sync_pulse_ao"]
    assert daq.connected == (
        [TRIGGER_ROUTE, SHARED_CLOCK_ROUTE] if synchronized else [TRIGGER_ROUTE])
    assert daq.disconnected == []
