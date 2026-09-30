"""Closing the laser controller while a synchronous pulse train runs.

Run Pulse waits for its train (wait=True), and that path ran on the caller's
thread without an operation: close() could neither cancel it nor wait for it,
and its reset met the train's task on the output at -50103, leaving the
train to run out with the shutter state unmanaged.

Nothing here touches a driver or a board (nidaq_daqmx_fake).
"""

import threading
import time

import pytest

from autotrainer.device import (
    LaserChannelId,
    LaserOperationState,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser

from nidaq_daqmx_fake import FakeDaqmx, rig_lasers


PULSE = LaserPulseTrain(
    channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0, duration_ms=1.0)


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met")
        time.sleep(0.01)


def _in_thread(function, *args):
    outcome = []

    def run():
        try:
            outcome.append(function(*args))
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _pulse_is_running(daq):
    return any(task.name == "laser_sync_pulse_ao" and task.started for task in daq.tasks)


@pytest.fixture
def held(monkeypatch):
    """A fake whose waits hold until released, as a train still running does.

    As the hardware does (H5a, H5b): a stop from another thread waits behind
    the wait, and an abort wakes it with -88709.
    """
    daq = FakeDaqmx(block_wait=True, hold_waits=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    try:
        yield daq
    finally:
        daq.waits_released.set()


def test_closing_mid_pulse_cancels_a_synchronous_run_and_resets_the_laser(held):
    daq = held
    controller = NidaqLaserController(rig_lasers())
    pulse_thread, pulse_outcome = _in_thread(controller.run_pulse_train, PULSE)
    _wait_for(lambda: _pulse_is_running(daq))
    shutter = daq.task("laser_1_shutter")
    assert shutter.writes[-1] is True

    started = time.monotonic()
    close_thread, close_outcome = _in_thread(controller.close)
    close_thread.join(10.0)

    # The cancel stopped the train's task from close()'s thread, and on the
    # hardware that stop waits behind the train's wait (H5a): close() waited
    # for the whole train. The abort wakes the wait at once (H5b).
    assert not close_thread.is_alive()
    assert time.monotonic() - started < 1.0
    assert close_outcome == [None]
    pulse_thread.join(5.0)
    assert not pulse_thread.is_alive()
    error, = pulse_outcome
    assert isinstance(error, RuntimeError)
    assert "cancelled" in str(error) and "-50103" not in str(error)
    # close()'s own reset, made on the output the train had held.
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao",
                         thread=close_thread) == [0.0]
    assert shutter.writes[-1] is False
    assert controller._live_operations == {}


def test_a_synchronous_run_is_an_operation_while_it_runs(held):
    daq = held
    controller = NidaqLaserController(rig_lasers())
    pulse_thread, pulse_outcome = _in_thread(controller.run_pulse_train, PULSE)
    _wait_for(lambda: _pulse_is_running(daq))

    operation, = controller._live_operations.values()
    assert operation.resources == ("PXI1Slot4/ao0",)

    daq.waits_released.set()
    pulse_thread.join(5.0)
    assert pulse_outcome == [None]
    assert controller._live_operations == {}


def _armed_pulse(controller):
    """A pulse on laser 1 that runs on its own thread, as a trial's does."""
    return controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(PULSE,), wait=False))


def test_a_waited_for_pulse_is_refused_while_an_armed_one_owns_the_output(held):
    # DAQmx would refuse the second task on ao0 anyway (-50103); refused
    # before any task is made, it says which operation holds the output.
    daq = held
    controller = NidaqLaserController(rig_lasers())
    operation = _armed_pulse(controller)
    created = len(daq.tasks)

    with pytest.raises(RuntimeError) as refused:
        controller.run_pulse_train(PULSE)

    message = str(refused.value)
    assert message == (
        "Laser 1: refused while the pulse on laser 1 holds the analog output "
        "of PXI1Slot4; wait for it to end, or cancel it. A board runs one "
        "timed analog output at a time (laser operation "
        f"{operation.operation_id} on PXI1Slot4/ao0)")
    assert len(daq.tasks) == created
    daq.waits_released.set()
    assert operation.wait(5.0) is LaserOperationState.COMPLETED
    controller.run_pulse_train(PULSE)


def test_a_pulse_ended_by_a_base_exception_still_lets_go_of_its_output(monkeypatch):
    # Only an Exception brought the operation to an end: a KeyboardInterrupt
    # left it live, owning ao0, and every later pulse on it was refused.
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    execute = controller._execute_synchronized_pulse_train
    operations = []

    def interrupted(pulse_train, *, operation=None):
        operations.append(operation)
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "_execute_synchronized_pulse_train", interrupted)
    with pytest.raises(KeyboardInterrupt):
        controller.run_pulse_train(PULSE)

    operation, = operations
    assert operation.state is LaserOperationState.FAILED
    assert operation._done.is_set()
    assert controller._live_operations == {}
    monkeypatch.setattr(controller, "_execute_synchronized_pulse_train", execute)
    controller.run_pulse_train(PULSE)


def test_a_close_beside_a_pulse_a_base_exception_ends_still_resets(monkeypatch):
    # The pulse's KeyboardInterrupt was stored as its operation's error, and
    # close(), waiting for the operation, had it raised at it: close() catches
    # Exception only, so the reset, the task close and the route release were
    # skipped. It is stored as a RuntimeError now; the pulse's own thread
    # still raises the KeyboardInterrupt.
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))
    go, operations = threading.Event(), []

    def interrupted(pulse_train, *, operation=None):
        operations.append(operation)
        go.wait(5.0)
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "_execute_synchronized_pulse_train", interrupted)
    pulse_outcome = []

    def pulse():
        try:
            controller.run_pulse_train(PULSE)
        except BaseException as error:
            pulse_outcome.append(error)

    pulse_thread = threading.Thread(target=pulse, daemon=True)
    pulse_thread.start()
    _wait_for(lambda: operations)
    operation, = operations
    close_outcome = []

    def close():
        try:
            close_outcome.append(controller.close())
        except BaseException as error:
            close_outcome.append(error)

    close_thread = threading.Thread(target=close, daemon=True)
    set_shutter_open = controller.set_shutter_open

    def shutter_then_interrupt(channel_id, is_open):
        # After close() has taken the operation to wait for, and before it
        # waits: the pulse ends by its KeyboardInterrupt in between.
        if threading.current_thread() is close_thread and not go.is_set():
            go.set()
            assert operation._done.wait(5.0)
        return set_shutter_open(channel_id, is_open)

    monkeypatch.setattr(controller, "set_shutter_open", shutter_then_interrupt)
    close_thread.start()
    close_thread.join(10.0)
    pulse_thread.join(5.0)

    assert not close_thread.is_alive()
    error, = pulse_outcome
    assert isinstance(error, KeyboardInterrupt)
    assert operation.state is LaserOperationState.FAILED
    assert isinstance(operation.error, RuntimeError)
    assert "KeyboardInterrupt" in str(operation.error)
    # Whatever close() reports of the pulse, it reset the laser and let go.
    assert not any(isinstance(outcome, KeyboardInterrupt) for outcome in close_outcome)
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao",
                         thread=close_thread) == [0.0]
    assert daq.disconnected == [("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")]
    assert controller._live_operations == {}


def test_an_aborted_pulse_puts_the_command_back_straight_after_its_task(held):
    # DAQmx leaves an aborted output on the last sample it wrote, a pulse's
    # high level, until something writes the minimum. Nothing but the output
    # task's own stop and close comes between the wait's return and that.
    daq = held
    controller = NidaqLaserController(rig_lasers())
    pulse = LaserPulseTrain(channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
                            duration_ms=1.0, enable_pmt_shutter=True)
    pulse_thread, pulse_outcome = _in_thread(controller.run_pulse_train, pulse)
    _wait_for(lambda: _pulse_is_running(daq))

    close_thread, _close_outcome = _in_thread(controller.close)
    close_thread.join(5.0)
    pulse_thread.join(5.0)

    assert not close_thread.is_alive() and not pulse_thread.is_alive()
    assert "cancelled" in str(pulse_outcome[0])
    abort = next(entry for entry in daq.timeline
                 if entry.event == "abort" and entry.task == "laser_sync_pulse_ao")
    own = [entry for entry in daq.timeline if entry.thread is pulse_thread]
    returned = next(index for index, entry in enumerate(own)
                    if entry.event == "wait returned" and entry.task == "laser_sync_pulse_ao")
    assert [(entry.event, entry.task) for entry in own[returned + 1:returned + 4]] == [
        ("stop", "laser_sync_pulse_ao"),
        ("close", "laser_sync_pulse_ao"),
        ("write", "laser_1_manual_ao"),
    ]
    reset = own[returned + 3]
    assert reset.data == 0.0
    assert reset.time - abort.time < 0.5


def _armed(controller, **fields):
    """A pulse armed on its own thread, as a trial's is."""
    return controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(PULSE,), wait=False, timeout_seconds=30.0, **fields))


@pytest.mark.parametrize("armed", ["board_stim", "deferred"])
def test_closing_over_an_armed_pulse_ends_it_at_once(held, armed):
    # Waiting for its trigger, the pulse's wait held until its own timeout,
    # and the cancel's stop waited behind it for all of that.
    daq = held
    controller = NidaqLaserController(rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))
    fields = (dict(trigger_source="/PXI1Slot4/PXI_Trig0") if armed == "board_stim"
              else dict(defer_start=True))
    operation = _armed(controller, **fields)
    assert operation.state is LaserOperationState.ARMED

    started = time.monotonic()
    close_thread, close_outcome = _in_thread(controller.close)
    close_thread.join(5.0)

    assert not close_thread.is_alive()
    assert time.monotonic() - started < 1.0
    assert close_outcome == [None]
    assert operation.wait(1.0) is LaserOperationState.CANCELLED
    assert daq.task("laser_1_shutter").writes[-1] is False
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao")[-1] == 0.0


def test_a_pulse_that_fails_by_itself_still_ends_failed(monkeypatch):
    # Not every DAQmx error in the wait is a cancel: one with none asked for
    # is the pulse's failure, as it was.
    daq = FakeDaqmx(block_wait=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(PULSE,), wait=False, timeout_seconds=0.2))

    with pytest.raises(RuntimeError, match="-200560"):
        operation.wait(5.0)
    assert operation.state is LaserOperationState.FAILED
    assert controller._live_operations == {}


def test_an_interrupt_as_a_pulse_thread_starts_leaves_it_tracked(monkeypatch):
    # A KeyboardInterrupt can land in Thread.start() after the thread has
    # started, while start() waits for it: the pulse runs, and was marked
    # FAILED and let go of, running untracked.
    daq = FakeDaqmx(block_wait=True, hold_waits=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    start = threading.Thread.start

    def start_then_interrupt(thread):
        start(thread)
        if thread.name.startswith("NidaqLaser-"):
            raise KeyboardInterrupt

    monkeypatch.setattr(threading.Thread, "start", start_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
                pulse_trains=(PULSE,), wait=False, timeout_seconds=30.0))
        monkeypatch.setattr(threading.Thread, "start", start)

        operation, = controller._live_operations.values()
        assert operation._thread.is_alive()
        assert operation.state is not LaserOperationState.FAILED
    finally:
        daq.waits_released.set()
    _wait_for(lambda: controller._live_operations == {})


def test_close_closes_the_shutters_before_it_waits_for_a_pulse(monkeypatch):
    # The laser model closed the shutters with a driver call before the
    # controller's close could mark it closed; the close then reset them
    # only after the wait for each pulse it cancels.
    monkeypatch.setattr(nidaq_laser, "_OPERATION_CANCEL_TIMEOUT_S", 3.0)
    # A sick driver, whose abort does not wake the wait: the pulse's wait
    # holds, and close() waits for it.
    daq = FakeDaqmx(block_wait=True, hold_waits=True, abort_unblocks=False)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    try:
        _armed_pulse(controller)
        shutter = daq.task("laser_1_shutter")
        controller.set_shutter_open(LaserChannelId.LASER_1, True)

        close_thread, _close_outcome = _in_thread(controller.close)
        _wait_for(lambda: shutter.writes[-1] is False, timeout=1.0)

        assert close_thread.is_alive(), "close() had already stopped waiting"
        daq.waits_released.set()
        close_thread.join(5.0)
        assert not close_thread.is_alive()
    finally:
        daq.waits_released.set()


def test_close_reports_a_pulse_it_gave_up_on_until_the_pulse_ends(monkeypatch):
    # close() waits a bounded time for each pulse it cancels, then goes on.
    # One that has not ended can still act later, releasing its clock route
    # or writing the PMT line after a new controller has opened on the same
    # lines: close() says so, and for as long as it lasts.
    monkeypatch.setattr(nidaq_laser, "_OPERATION_CANCEL_TIMEOUT_S", 0.2)
    # A sick driver, whose abort does not wake the wait.
    daq = FakeDaqmx(block_wait=True, hold_waits=True, abort_unblocks=False)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    try:
        operation = _armed_pulse(controller)
        assert controller.work_left_running() == ()

        with pytest.raises(RuntimeError, match=f"laser operation {operation.operation_id} cancel"):
            controller.close()

        assert controller.work_left_running() == (
            f"laser operation {operation.operation_id}",)
        assert controller.wait_for_work_left_running(0.1) is False
        daq.waits_released.set()
        assert controller.wait_for_work_left_running(5.0) is True
        assert controller.work_left_running() == ()
    finally:
        daq.waits_released.set()


def test_a_synchronous_run_without_a_close_returns_and_fails_as_before(monkeypatch):
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())

    assert controller.run_pulse_train(PULSE) is None
    assert daq.starts == ["laser_sync_pulse_ao"]

    daq.failing_task = "laser_sync_pulse_ao"
    with pytest.raises(RuntimeError) as refused:
        controller.run_pulse_train(PULSE)
    assert str(refused.value) == "DAQmx refused to start laser_sync_pulse_ao"
    assert controller._live_operations == {}
    assert daq.reserved == {}
