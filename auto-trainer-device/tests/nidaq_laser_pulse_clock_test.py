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
#: The input stream's clock, which a synchronized pulse train runs on.
SHARED_CLOCK = "/PXI1Slot5/ai/SampleClock"
SHARED_CLOCK_ROUTE = (SHARED_CLOCK, "/PXI1Slot5/PXI_Trig1")
#: The 6713's own clock on pulseClockLine, for any pulse's DO on the 6221.
PULSE_CLOCK_ROUTE = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig3")
#: The pulse's two modes: the AO on its own clock, or on the input stream's.
MODES = ("unsynchronized", "synchronized")

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


# ------------------------------- a DO line on another board: pulseClockLine


def _plan_for(mode):
    return _synchronized_plan() if mode == "synchronized" else None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_a_cross_board_output_runs_on_the_ao_clock_over_the_pulse_clock_line(
    daq, output, mode,
):
    # A clocked line on the 6221 can neither name the 6713's clock nor take
    # a start trigger: the board's clocked DO takes none (do_trig_usage is
    # empty, and TASK_VERIFY refuses one with -200452; christielab10,
    # 2026-09-25). So it runs on the AO's own clock, which ticks only once
    # the AO has triggered, carried on pulseClockLine (PXI_Trig3) for this
    # pulse alone. backplaneClockLine is the shared clock's, and the ramp's.
    configuration, fields, task_name, line = _output(output)
    controller = _controller(_plan_for(mode), **configuration)

    controller.run_synchronized_pulse_train(_pulse(**fields))

    digital = daq.task(task_name)
    analog = daq.task("laser_sync_pulse_ao")
    assert digital.channels == [line]
    assert digital.timing_kwargs["source"] == "/PXI1Slot5/PXI_Trig3"
    assert digital.start_trigger is None
    # The AO arms as it always did.
    assert analog.start_trigger == (BOARD_STIM, "rising")
    if mode == "synchronized":
        assert analog.timing_kwargs["source"] == "/PXI1Slot4/PXI_Trig1"
        assert digital.timing_kwargs["rate"] == 10_000.0
    else:
        assert "source" not in analog.timing_kwargs
    # Routed, the AO committed, the line started before the AO, and the
    # route released after.
    log = daq.log
    assert (log.index(("connect", PULSE_CLOCK_ROUTE))
            < log.index(("commit", analog.name))
            < log.index(("start", digital.name))
            < log.index(("start", analog.name))
            < log.index(("disconnect", PULSE_CLOCK_ROUTE)))
    shared = [SHARED_CLOCK_ROUTE] if mode == "synchronized" else []
    assert daq.connected == [TRIGGER_ROUTE, *shared, PULSE_CLOCK_ROUTE]
    assert daq.disconnected == [PULSE_CLOCK_ROUTE]
    # The shared clock's route stays with the controller, as it always did.
    assert controller._trigger_routes == [TRIGGER_ROUTE, *shared]
    controller.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_the_pulse_clock_route_is_released_when_the_pulse_fails(monkeypatch, output, mode):
    daq = FakeDaqmx(failing_task="laser_sync_pulse_ao")
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    configuration, fields, _task_name, _line = _output(output)
    controller = _controller(_plan_for(mode), **configuration)

    with pytest.raises(RuntimeError, match="refused to start"):
        controller.run_synchronized_pulse_train(_pulse(**fields))

    assert daq.disconnected == [PULSE_CLOCK_ROUTE]
    assert PULSE_CLOCK_ROUTE not in controller._trigger_routes
    assert daq.reserved == {}


@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_the_pulse_clock_route_is_released_when_a_software_start_is_cancelled(daq, output):
    # A deferred software start, as a protocol's direct NI route arms one.
    configuration, fields, _task_name, _line = _output(output)
    controller = _controller(**configuration)
    operation = controller.run_synchronized_pulse_train(_pulse(
        trigger_source=None, wait=False, defer_start=True, **fields))
    assert daq.connected == [TRIGGER_ROUTE, PULSE_CLOCK_ROUTE]

    assert operation.cancel()
    assert operation.wait(5.0) is LaserOperationState.CANCELLED

    assert daq.disconnected == [PULSE_CLOCK_ROUTE]
    assert controller._trigger_routes == [TRIGGER_ROUTE]
    assert daq.starts == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_the_pulse_clock_route_is_released_when_an_armed_pulse_is_cancelled(
    monkeypatch, output, mode,
):
    # Armed on the board STIM, waiting for it, as a trial's pulse is.
    daq = FakeDaqmx(block_wait=True, hold_waits=True, stop_unblocks=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    configuration, fields, _task_name, _line = _output(output)
    controller = _controller(_plan_for(mode), **configuration)
    try:
        operation = controller.run_synchronized_pulse_train(_pulse(wait=False, **fields))
        assert PULSE_CLOCK_ROUTE in daq.connected

        assert operation.cancel()
        assert operation.wait(5.0) is LaserOperationState.CANCELLED

        assert daq.disconnected == [PULSE_CLOCK_ROUTE]
        shared = [SHARED_CLOCK_ROUTE] if mode == "synchronized" else []
        assert controller._trigger_routes == [TRIGGER_ROUTE, *shared]
    finally:
        daq.waits_released.set()


@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_a_commit_that_fails_leaves_nothing_behind(monkeypatch, output):
    # The AO is committed before its lines start. A refused commit is the
    # pulse's failure like any other: its tasks closed, its clock route
    # released, the command put back, and the operation ended FAILED.
    daq = FakeDaqmx()
    daq.failing_commit = "laser_sync_pulse_ao"
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    configuration, fields, task_name, _line = _output(output)
    controller = _controller(**configuration)
    ended = []
    release = controller._release_operation
    controller._release_operation = lambda operation: (
        ended.append(operation.state), release(operation))

    with pytest.raises(RuntimeError, match="refused to commit"):
        controller.run_synchronized_pulse_train(_pulse(**fields))

    assert ended == [LaserOperationState.FAILED]
    assert daq.task("laser_sync_pulse_ao").closed
    assert daq.task(task_name).closed
    assert daq.disconnected == [PULSE_CLOCK_ROUTE]
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao")[-1] == 0.0
    assert daq.starts == []
    assert daq.reserved == {}
    assert controller._live_operations == {}


def test_a_deferred_pulse_is_committed_before_it_is_armed(daq):
    # A protocol's direct NI route arms the pulse, then starts it by
    # software. The AO is committed while the operation is still prepared,
    # so that nothing is programmed on the board once it reads armed.
    configuration, fields, task_name, _line = _output("pmt_shutter")
    controller = _controller(**configuration)
    states = []

    def at_commit(task):
        operation, = controller._live_operations.values()
        states.append((task.name, operation.state))

    daq.before_control = at_commit
    operation = controller.run_synchronized_pulse_train(_pulse(
        trigger_source=None, wait=False, defer_start=True, **fields))

    assert operation.state is LaserOperationState.ARMED
    assert states == [("laser_sync_pulse_ao", LaserOperationState.PREPARED)]
    assert daq.starts == []
    operation.trigger()
    assert operation.wait(5.0) is LaserOperationState.COMPLETED
    assert daq.log.index(("commit", "laser_sync_pulse_ao")) < daq.log.index(
        ("start", task_name))


def _two_lasers_on_one_board():
    lasers = rig_lasers(trigger_source=BOARD_STIM, trigger_route_source="/PXI1Slot5/PFI0")
    laser_2 = dataclasses.replace(
        lasers.channels[0], channel_id=LaserChannelId.LASER_2,
        analog_output="PXI1Slot4/ao1", diode_input="PXI1Slot5/ai9",
        shutter_output="PXI1Slot5/port0/line5", command_copy_input=None,
        trigger_source="/PXI1Slot4/PXI_Trig2", trigger_route_source="/PXI1Slot5/PFI1")
    return dataclasses.replace(lasers, channels=(lasers.channels[0], laser_2))


def test_a_second_pulse_on_the_same_board_is_refused_while_the_first_holds_it(
    monkeypatch,
):
    # The resource check was per channel, so a pulse on ao1 went ahead
    # while one on ao0 of the same 6713 was armed. It borrowed the first
    # pulse's route onto pulseClockLine without owning it, and lost its
    # lines' clock when the first released it. A board has one timed
    # analog output: DAQmx would refuse the second task (-50103) after
    # the route and the tasks were made. Refused first now, by name.
    daq = FakeDaqmx(block_wait=True, hold_waits=True, stop_unblocks=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(_two_lasers_on_one_board())
    try:
        first = controller.run_synchronized_pulse_train(_pulse(
            wait=False, enable_pmt_shutter=True))
        created, connected = len(daq.tasks), list(daq.connected)
        second = LaserSynchronizedPulseTrain(
            pulse_trains=(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_2, amplitude_volts=1.0,
                duration_ms=1.0, enable_pmt_shutter=True),),
            timeout_seconds=5.0)

        with pytest.raises(RuntimeError) as refused:
            controller.run_synchronized_pulse_train(second)

        assert str(refused.value) == (
            "The analog output of PXI1Slot4 is in use by laser operation "
            f"{first.operation_id} (on PXI1Slot4/ao0), armed or running: a "
            "board runs one timed analog output at a time, so a pulse on "
            "PXI1Slot4/ao1 is refused until that operation ends or is cancelled")
        assert (len(daq.tasks), daq.connected) == (created, connected)
        assert PULSE_CLOCK_ROUTE in controller._trigger_routes

        daq.waits_released.set()
        assert first.wait(5.0) is LaserOperationState.COMPLETED
        controller.run_synchronized_pulse_train(second)
        assert daq.disconnected == [PULSE_CLOCK_ROUTE, PULSE_CLOCK_ROUTE]
    finally:
        daq.waits_released.set()


def test_one_train_on_both_lasers_of_a_board_is_one_task_and_runs(daq):
    controller = NidaqLaserController(_two_lasers_on_one_board())
    both = LaserSynchronizedPulseTrain(
        pulse_trains=tuple(
            LaserPulseTrain(channel_id=channel_id, amplitude_volts=1.0, duration_ms=1.0)
            for channel_id in (LaserChannelId.LASER_1, LaserChannelId.LASER_2)),
        timeout_seconds=5.0)

    controller.run_synchronized_pulse_train(both)

    assert daq.task("laser_sync_pulse_ao").channels == ["PXI1Slot4/ao0", "PXI1Slot4/ao1"]
    assert daq.starts == ["laser_sync_pulse_ao"]


def test_an_unsynchronized_pulse_after_a_synchronized_one_runs(daq):
    # A synchronized pulse leaves the shared clock on PXI_Trig1 until the
    # controller closes. The AO clock of a pulse on its own clock went on
    # that line too, and was refused after one: which pulses ran depended on
    # their order. It has a line of its own now.
    controller = _controller(_synchronized_plan())
    controller.run_synchronized_pulse_train(_pulse())
    assert SHARED_CLOCK_ROUTE in daq.connected

    controller.run_synchronized_pulse_train(_pulse(
        trigger_source=None, enable_pmt_shutter=True))

    # On its own clock: no trigger to arm on, so not on the stream's.
    assert "source" not in daq.task("laser_sync_pulse_ao").timing_kwargs
    assert daq.task("laser_pmt_shutter_do").timing_kwargs["source"] == "/PXI1Slot5/PXI_Trig3"
    assert daq.disconnected == [PULSE_CLOCK_ROUTE]
    controller.close()


def test_an_output_needs_no_trigger_its_board_can_see(daq):
    # Its clock is the AO's own, so an AO armed on a PFI of its own board
    # starts the line too, although nothing carries that PFI to the 6221.
    controller = NidaqLaserController(rig_lasers(), timing_plan=_synchronized_plan())

    controller.run_synchronized_pulse_train(_pulse(
        trigger_source="/PXI1Slot4/PFI0", enable_pmt_shutter=True))

    digital = daq.task("laser_pmt_shutter_do")
    assert digital.timing_kwargs["source"] == "/PXI1Slot5/PXI_Trig3"
    assert digital.start_trigger is None


def test_the_fake_refuses_a_start_trigger_on_a_clocked_output_as_the_6221_does(daq):
    # A guard on the stand-in, not on the controller: it models the board's
    # refusal (-200452 at verify), so a design relying on a DO start trigger
    # fails here as it does on christielab10.
    controller = NidaqLaserController(rig_lasers())

    with pytest.raises(RuntimeError, match="-200452"):
        controller._create_finite_digital_output_task(
            "PXI1Slot5/port0/line6", "laser_pmt_shutter_do", [True] * 10,
            10_000.0, 10, SHARED_CLOCK, "/PXI1Slot5/PXI_Trig0", "rising")


# --------------------------------------------------------- the clock lines


@pytest.mark.parametrize("mode", MODES)
def test_the_pulse_clock_line_is_not_driven_over_the_streams_own_export(daq, mode):
    # The stream exports its sample clock to a configured terminal, from its
    # own process; that line is driven too.
    plan = dataclasses.replace(
        _synchronized_plan(), sample_clock_export_terminal="/PXI1Slot5/PXI_Trig3")
    if mode == "unsynchronized":
        # A plan that does not declare the output leaves it on its own clock.
        plan = dataclasses.replace(plan, hardware_output_devices=())
    controller = _controller(plan)

    with pytest.raises(RuntimeError, match="PXI_Trig3"):
        controller.run_synchronized_pulse_train(_pulse(enable_pmt_shutter=True))

    assert PULSE_CLOCK_ROUTE not in daq.connected
    assert daq.starts == []


def test_the_pulse_clock_line_is_not_driven_over_a_trigger_route(daq):
    # A pulse clock line on the line the stimulus takes. A configuration
    # refuses that as it is built (LaserSystemConfiguration); the controller
    # refuses it too, for one that reaches it another way.
    configuration = rig_lasers(trigger_source=BOARD_STIM,
                               trigger_route_source="/PXI1Slot5/PFI0")
    object.__setattr__(configuration, "pulse_clock_line", "PXI_Trig0")
    controller = NidaqLaserController(configuration)

    with pytest.raises(RuntimeError) as refused:
        controller.run_synchronized_pulse_train(_pulse(enable_pmt_shutter=True))

    message = str(refused.value)
    assert "PXI_Trig0" in message and "/PXI1Slot5/PFI0" in message
    assert daq.connected == [TRIGGER_ROUTE]
    assert daq.starts == []


# ---------------------------------------------------------------- the AO board


@pytest.mark.parametrize("plan", [None, _synchronized_plan("DevX", "DevX")],
                         ids=["unsynchronized", "synchronized"])
@pytest.mark.parametrize("output", sorted(OUTPUTS))
def test_an_output_on_the_ao_board_runs_on_its_clock_with_no_trigger(daq, output, plan):
    # On the AO's own board the line takes that board's AO clock by name,
    # and no start trigger: the clock ticks only once the AO has triggered,
    # and the line starts first. An M Series board refuses a clocked DO start
    # trigger (-200452), as christielab10's 6221 does. DevX is hypothetical.
    configuration, fields, task_name, _line = _output(output)
    line = "DevX/port0/line6" if output == "pmt_shutter" else "DevX/port0/line7"
    lasers = rig_lasers(
        analog_output="DevX/ao0", diode_input="DevX/ai0",
        shutter_output="DevX/port0/line4", command_copy_input=None,
        trigger_source="/DevX/PFI0", **{name: line for name in configuration})
    lasers = dataclasses.replace(lasers, pmt_shutter_output="DevX/port0/line6")
    controller = NidaqLaserController(lasers, timing_plan=plan)

    controller.run_synchronized_pulse_train(_pulse(trigger_source="/DevX/PFI0", **fields))

    digital = daq.task(task_name)
    assert digital.channels == [line]
    assert digital.timing_kwargs["source"] == "/DevX/ao/SampleClock"
    assert digital.start_trigger is None
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
    # Nothing committed ahead of its start: that is for a pulse with lines.
    assert daq.controlled == []
    assert daq.connected == (
        [TRIGGER_ROUTE, SHARED_CLOCK_ROUTE] if synchronized else [TRIGGER_ROUTE])
    assert daq.disconnected == []


def test_a_route_the_driver_would_not_release_is_still_counted(daq):
    # It was taken off the held list before its disconnect, and a failed one
    # left it unlisted though it may still be driven: the collision check no
    # longer saw it, and close() did not try it again.
    daq.failing_disconnects.add(PULSE_CLOCK_ROUTE)
    controller = _controller()

    with pytest.raises(RuntimeError, match="trigger route"):
        controller.run_synchronized_pulse_train(_pulse(enable_pmt_shutter=True))

    assert PULSE_CLOCK_ROUTE in controller._trigger_routes
    daq.failing_disconnects.clear()
    controller.close()
    assert daq.disconnected == [TRIGGER_ROUTE, PULSE_CLOCK_ROUTE]
    assert controller._trigger_routes == []
