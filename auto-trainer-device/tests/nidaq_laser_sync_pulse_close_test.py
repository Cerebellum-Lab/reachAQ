"""Closing the laser controller while a synchronous pulse train runs.

Run Pulse waits for its train (wait=True), and that path ran on the caller's
thread without an operation: close() could neither cancel it nor wait for it,
and its reset met the train's task on the output at -50103, leaving the
train to run out with the shutter state unmanaged.

Nothing here touches a driver or a board (nidaq_daqmx_fake).
"""

import dataclasses
import logging
import threading
import time
import warnings

import pytest

from autotrainer.device import (
    LaserChannelId,
    LaserOperationState,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser

from nidaq_daqmx_fake import (
    ABORT_WARNING_TEXT,
    FakeDaqError,
    FakeDaqWarning,
    FakeDaqmx,
    rig_lasers,
)


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
    return any(task.label == "laser_sync_pulse_ao" and task.started for task in daq.tasks)


@pytest.fixture(autouse=True)
def _every_controller_closed():
    """Close each controller a test opened, then check no fake task is open.

    Most tests here never closed theirs, which kept its channel tasks, and
    any trigger route, open on the fake. This is torn down after the test's
    own fixtures, so its patches are undone by then. A stand-in the test set
    on the controller itself is dropped, and any wait or call it left held
    is let go, so the close meets the fake as a controller's close would.
    """
    opened = []
    init = NidaqLaserController.__init__

    def opening(self, *args, **kwargs):
        init(self, *args, **kwargs)
        opened.append(self)

    NidaqLaserController.__init__ = opening
    try:
        yield
    finally:
        NidaqLaserController.__init__ = init
    fakes = {id(controller._nidaqmx): controller._nidaqmx for controller in opened}
    for daq in fakes.values():
        daq.waits_released.set()
        daq.hang_released.set()
    for controller in opened:
        for name in [name for name in vars(controller)
                     if callable(getattr(NidaqLaserController, name, None))]:
            delattr(controller, name)
        if not controller._closed:
            controller.close()
        assert controller.wait_for_work_left_running(5.0)
    for daq in fakes.values():
        assert [task.label for task in daq.tasks if not task.closed] == []


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
    # A running task's abort returned only once its woken owner had closed
    # it, in every cancel here: the fake never had to give up on one.
    assert not _gave_up_on_an_owner(daq)


def _gave_up_on_an_owner(daq):
    return [entry.task for entry in daq.timeline if entry.event == "abort gave up on owner"]


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


def test_a_pulse_that_fails_by_itself_as_close_begins_is_no_close_failure(monkeypatch, caplog):
    # close() takes the live pulses, then cancels each. One that failed by
    # itself in between, its wait timing out as Stop was pressed, was not
    # cancelled, and close() raised its own error as a failure to close:
    # Stop then logged "make the laser safe by hand" for a laser the pulse's
    # own cleanup had reset.
    daq = FakeDaqmx(block_wait=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(PULSE,), wait=False, timeout_seconds=0.2))
    mark_closed = controller._mark_closed

    def the_pulse_fails_meanwhile():
        taken = mark_closed()
        assert operation.wait_until_finished(5.0)
        return taken

    monkeypatch.setattr(controller, "_mark_closed", the_pulse_fails_meanwhile)
    with caplog.at_level(logging.ERROR):
        assert controller.close() is None

    assert operation.state is LaserOperationState.FAILED
    assert [record.getMessage() for record in caplog.records
            if record.levelno >= logging.ERROR] == []
    assert controller.work_left_running() == ()


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


# ------------------------------------------------ round 5: the cancel's edges


def _routed():
    return rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0")


def _events(daq, *wanted):
    """The timeline's (event, task) pairs of these kinds, in order."""
    return [(entry.event, entry.task, entry.data) for entry in daq.timeline
            if (entry.event, entry.task) in wanted]


@pytest.mark.parametrize("armed", ["board_stim", "deferred_then_triggered"])
def test_a_cancel_just_before_the_start_is_not_lost(held, armed):
    # The pulse looked for a cancel, then started its tasks with no look
    # after: a cancel landing between them aborted tasks not yet started,
    # which does nothing (H5c), and the train ran.
    daq = held
    controller = NidaqLaserController(_routed())
    cancelled = []
    after_the_cancel = []

    def cancel_as_it_starts(task):
        if task.label == "laser_sync_pulse_ao":
            daq.before_start = None
            operation, = controller._live_operations.values()
            cancelled.append(operation)
            operation.cancel()
            # The cancel's own shutter close is on this thread too.
            after_the_cancel.append(len(daq.writes))

    if armed == "board_stim":
        # Cancelled as it arms: the arming itself reports it.
        daq.before_start = cancel_as_it_starts
        with pytest.raises(RuntimeError, match="cancelled"):
            _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")
        operation, = cancelled
    else:
        operation = _armed(controller, defer_start=True)
        daq.before_start = cancel_as_it_starts
        operation.trigger()

    assert operation.wait_until_finished(1.0) is True
    assert operation.state is LaserOperationState.CANCELLED
    # The pulse's own cleanup: its thread's reset and shutter close.
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao",
                         thread=operation._thread) == [0.0]
    assert daq.writes_to("PXI1Slot5/port0/line4", thread=operation._thread,
                         since=after_the_cancel[0]) == [False]


def test_cancelling_an_armed_deferred_pulse_logs_no_error(held, caplog):
    # H8b: the cancel woke the deferred wait before it aborted, and the
    # pulse's cleanup stopped the task while the abort was under way:
    # -88710, and two ERROR lines, on a cancel that went as it should.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, defer_start=True)

    with caplog.at_level("DEBUG"):
        assert operation.cancel()
        assert operation.wait(1.0) is LaserOperationState.CANCELLED

    assert not [record.getMessage() for record in caplog.records
                if record.levelno >= logging.ERROR]
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao",
                         thread=operation._thread) == [0.0]
    # #5: woken only once the abort has returned, the cleanup's stop comes
    # after it, and meets no abort under way: no -88710 at all, refused or
    # logged. Woken before the abort, its first stop was refused during it.
    own = [(entry.event, entry.task) for entry in daq.timeline
           if entry.task == "laser_sync_pulse_ao"
           and entry.event in ("abort returned", "stop", "stop refused")]
    assert own == [("abort returned", "laser_sync_pulse_ao"), ("stop", "laser_sync_pulse_ao")]
    assert not [record for record in caplog.records if "-88710" in record.getMessage()]


def test_cancelling_a_running_pulse_logs_no_error(held, caplog):
    # Its owner is woken during the abort, and stops and clears the task
    # before the abort returns: clean on the 6713 (H8a, H8c).
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")

    with caplog.at_level("DEBUG"):
        assert operation.cancel()
        assert operation.wait(1.0) is LaserOperationState.CANCELLED

    assert not [record.getMessage() for record in caplog.records
                if record.levelno >= logging.ERROR]
    assert ("stop", "laser_sync_pulse_ao", None) in _events(
        daq, ("stop", "laser_sync_pulse_ao"))


def test_a_stop_that_fails_otherwise_is_still_an_error(monkeypatch, caplog):
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    task = daq.Task("some_task")

    def refused():
        raise FakeDaqError(-50103, "the resource is reserved")

    task.stop = refused
    errors = []
    with caplog.at_level("DEBUG"):
        controller._stop_and_close_task("some task", task, errors)

    assert [location for location, _error in errors] == ["some task stop"]
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_a_cancel_closes_the_pulses_shutter_before_it_aborts(held):
    # close() closes the shutters first; a trial's cancel went straight to
    # the abort, and the light stayed on for the abort and the reset, 24-40
    # ms after the cancel on the 6713 (H8a, H8c on c12189cd).
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")
    assert daq.task("laser_1_shutter").writes[-1] is True
    marked = len(daq.timeline)

    assert operation.cancel()
    assert operation.wait(1.0) is LaserOperationState.CANCELLED

    after = [(entry.event, entry.task, entry.data) for entry in daq.timeline[marked:]]
    shut = after.index(("write", "laser_1_shutter", False))
    abort = after.index(("abort", "laser_sync_pulse_ao", None))
    reset = after.index(("write", "laser_1_manual_ao", 0.0))
    assert shut < abort < reset


def test_a_cancel_after_the_pulse_has_finished_changes_nothing(monkeypatch):
    # A cancel between the waits returning and the end marked the delivered
    # pulse CANCELLED, skipped its baseline, and aborted tasks its own thread
    # was closing.
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    stop_and_close = controller._stop_and_close_task
    results, operations = [], []

    def cancel_then_close(name, task, errors):
        if not results:
            operation, = controller._live_operations.values()
            operations.append(operation)
            canceller = threading.Thread(target=lambda: results.append(operation.cancel()))
            canceller.start()
            canceller.join(5.0)
        return stop_and_close(name, task, errors)

    monkeypatch.setattr(controller, "_stop_and_close_task", cancel_then_close)

    controller.run_pulse_train(PULSE)

    assert results == [False]
    operation, = operations
    assert operation.state is LaserOperationState.COMPLETED
    assert not [entry for entry in daq.timeline if entry.event == "abort"]


def test_a_pulse_close_gave_up_on_puts_the_command_back_when_it_ends(monkeypatch):
    # Past its wait, close() reset the output against the pulse's still-open
    # task (-50103) and let go of the channel tasks; the pulse's own late
    # reset then went through those tasks, and failed too (KeyError): the
    # output held its last sample until reachAQ restarted.
    monkeypatch.setattr(nidaq_laser, "_OPERATION_CANCEL_TIMEOUT_S", 0.2)
    # A sick driver: its abort neither wakes the wait nor lets go of ao0.
    daq = FakeDaqmx(block_wait=True, hold_waits=True, abort_unblocks=False,
                    abort_releases=False)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    try:
        operation = _armed(controller)
        with pytest.raises(RuntimeError, match="channel 1 reset"):
            controller.close()
        assert operation._thread.is_alive()

        daq.waits_released.set()
        operation._thread.join(5.0)

        assert not operation._thread.is_alive()
        assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao",
                             thread=operation._thread)[-1] == 0.0
    finally:
        daq.waits_released.set()


def test_the_aborts_warning_is_kept_to_the_log(held, caplog):
    # Aborting a running task warns (DaqWarning 200010, H5b, H8a), through
    # Python's warnings, onto stderr. It is the abort doing what was asked.
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")

    with warnings.catch_warnings(record=True) as escaped:
        with caplog.at_level("DEBUG"):
            assert operation.cancel()
            assert operation.wait(1.0) is LaserOperationState.CANCELLED

    assert [str(warning.message) for warning in escaped] == []
    assert any("200010" in record.getMessage() and record.levelno == logging.DEBUG
               for record in caplog.records)


def test_a_waited_for_pulse_cancelled_by_itself_says_cancelled_not_closed(held):
    daq = held
    controller = NidaqLaserController(rig_lasers())
    pulse_thread, pulse_outcome = _in_thread(controller.run_pulse_train, PULSE)
    _wait_for(lambda: _pulse_is_running(daq))
    operation, = controller._live_operations.values()

    assert operation.cancel()
    pulse_thread.join(5.0)

    error, = pulse_outcome
    assert str(error) == f"Laser operation {operation.operation_id} was cancelled"
    assert operation.wait_until_finished(1.0) is True


# ------------------------------------------------ round 6: the abort's logging


def test_cancelling_a_running_pulse_asks_the_driver_nothing_after_its_abort(held, caplog):
    # After the abort, the name of each task was read to log its warning: a
    # driver query (nidaqmx's Task.name). The pulse's own thread, woken
    # during the abort, had cleared the task by then: -200088, logged as a
    # failed abort, and the 200010 note lost (christielab10, H8c).
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")

    with caplog.at_level("DEBUG"):
        assert operation.cancel()
        assert operation.wait(1.0) is LaserOperationState.CANCELLED

    messages = [record.getMessage() for record in caplog.records]
    assert not [message for message in messages if "abort during cancellation failed" in message]
    assert not [record for record in caplog.records if record.exc_info]
    assert "laser 1: aborting its running task laser_sync_pulse_ao (DAQmx may warn 200010)" in messages


def test_the_abort_leaves_the_warnings_state_alone(held, monkeypatch):
    # Recording the abort's warnings swapped the process's warnings state
    # for the abort's duration: on Python 3.10 a thread doing the same
    # meanwhile (numpy, torch, pyqtgraph) could leave it corrupted, and a
    # hung abort held the swap.
    swaps = []
    catch_warnings = warnings.catch_warnings

    def recorded(*args, **kwargs):
        swaps.append(threading.current_thread().name)
        return catch_warnings(*args, **kwargs)

    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")
    monkeypatch.setattr(nidaq_laser.warnings, "catch_warnings", recorded)

    assert operation.cancel()
    assert operation.wait(1.0) is LaserOperationState.CANCELLED

    monkeypatch.setattr(nidaq_laser.warnings, "catch_warnings", catch_warnings)
    assert swaps == []


def test_only_the_aborts_own_warning_is_filtered(held):
    # One filter, set as the driver loads: 200010 by its text. Any other
    # DAQmx warning still reaches the warnings machinery.
    NidaqLaserController(_routed())

    with warnings.catch_warnings(record=True) as seen:
        warnings.warn(FakeDaqWarning(ABORT_WARNING_TEXT))
        warnings.warn(FakeDaqWarning(
            "\nWarning 200015 occurred.\n\nWhile writing, a test warning "
            f"{time.monotonic()!r} arrived."))

    assert [str(warning.message).split("occurred")[0].strip() for warning in seen] == [
        "Warning 200015"]


def _no_time(monkeypatch):
    """A clock that moves only as the code under test sleeps."""
    from types import SimpleNamespace

    now = [0.0]
    monkeypatch.setattr(nidaq_laser, "time", SimpleNamespace(
        monotonic=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        perf_counter=time.perf_counter))


def _stopping(errors_to_raise):
    calls = []

    def stop():
        calls.append(None)
        raise errors_to_raise()

    return stop, calls


def test_an_abort_that_never_finishes_ends_the_retry_in_an_error(held, monkeypatch, caplog):
    controller = NidaqLaserController(rig_lasers())
    task = held.Task("some_task")
    task.stop, calls = _stopping(lambda: FakeDaqError(-88710, "an abort is in progress"))
    _no_time(monkeypatch)
    errors = []

    with caplog.at_level("DEBUG"):
        controller._stop_and_close_task("some task", task, errors)

    assert [location for location, _error in errors] == ["some task stop"]
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    # Tried for the whole bound, 200 ms every 5 ms, then given up.
    assert 30 < len(calls) < 60


@pytest.mark.parametrize("error", [
    lambda: FakeDaqError(-50103, "the resource is reserved"),
    lambda: RuntimeError("DAQmx -88710: its text alone says an abort is in progress"),
], ids=["another_code", "the_code_in_text_only"])
def test_only_the_abort_in_progress_code_is_tried_again(held, monkeypatch, error):
    controller = NidaqLaserController(rig_lasers())
    task = held.Task("some_task")
    task.stop, calls = _stopping(error)
    _no_time(monkeypatch)
    errors = []

    controller._stop_and_close_task("some task", task, errors)

    assert len(calls) == 1
    assert [location for location, _error in errors] == ["some task stop"]


# ------------------------------------------------ round 6: the channel tasks going


def _let_go_of_the_channel_tasks(controller):
    """As close() lets go of the channel tasks: each closed, then all cleared."""
    for tasks in controller._tasks.values():
        tasks.close()
    controller._tasks.clear()


def test_a_pulse_cleanup_whose_channel_tasks_go_meanwhile_still_resets(monkeypatch):
    # The cleanup looked for the channel's tasks, then used them: close()
    # clearing them in between raised KeyError, and nothing was reset.
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    set_command_voltage = controller.set_command_voltage

    def tasks_go_first(channel_id, volts):
        _let_go_of_the_channel_tasks(controller)
        return set_command_voltage(channel_id, volts)

    monkeypatch.setattr(controller, "set_command_voltage", tasks_go_first)
    before = len(daq.writes)

    controller.run_pulse_train(PULSE)

    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao", since=before) == [0.0]


def test_a_cancels_shutter_close_whose_channel_tasks_go_meanwhile_still_closes(held, monkeypatch, caplog):
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")
    set_shutter_open = controller.set_shutter_open
    canceller = threading.current_thread()

    def tasks_go_first(channel_id, is_open):
        if threading.current_thread() is canceller:
            _let_go_of_the_channel_tasks(controller)
        return set_shutter_open(channel_id, is_open)

    monkeypatch.setattr(controller, "set_shutter_open", tasks_go_first)
    before = len(daq.writes)

    with caplog.at_level("ERROR"):
        assert operation.cancel()
        assert operation.wait(1.0) is LaserOperationState.CANCELLED

    assert daq.writes_to("PXI1Slot5/port0/line4", thread=canceller, since=before)[:1] == [False]
    assert not [record for record in caplog.records if "before a cancel" in record.getMessage()]


# ------------------------------------------------ round 6: the shutter and a cancel


def test_a_cancel_closes_the_shutter_of_a_pulse_meant_to_leave_it_open(held):
    # A cancel makes the laser safe: the pulse's shutter is closed whether
    # or not the pulse would have left it open, and whoever opened it.
    daq = held
    controller = NidaqLaserController(_routed())
    controller.set_shutter_open(LaserChannelId.LASER_1, True)
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
            duration_ms=1.0, open_shutter=False, close_shutter=False),),
        trigger_source="/PXI1Slot4/PXI_Trig0", wait=False, timeout_seconds=30.0))

    assert operation.cancel()
    assert operation.wait(1.0) is LaserOperationState.CANCELLED

    assert daq.task("laser_1_shutter").writes[-1] is False


def test_a_cancel_just_before_the_shutter_opens_leaves_it_closed(held):
    # The cancel closed the shutter, and the pulse then opened it: with a
    # pulse that leaves its shutter open, it stayed open. The pulse now
    # looks whether it is cancelled before it opens it, and again after. The
    # cancel here ends as the pulse writes its output's buffer, its last
    # driver call before the shutter: the first look finds it, and the
    # shutter is never opened after it.
    daq = held
    controller = NidaqLaserController(_routed())
    create = controller._create_synchronized_analog_output_task
    shutter = daq.task("laser_1_shutter")
    cancelled = []
    after_the_cancel = []

    def cancel_as_the_buffer_is_written(channels, name):
        task = create(channels, name)
        write = task.write

        def written(*args, **kwargs):
            result = write(*args, **kwargs)
            operation, = controller._live_operations.values()
            canceller = threading.Thread(target=lambda: cancelled.append(operation.cancel()))
            canceller.start()
            canceller.join(5.0)
            after_the_cancel.append(len(shutter.writes))
            return result

        task.write = written
        return task

    controller._create_synchronized_analog_output_task = cancel_as_the_buffer_is_written
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
            duration_ms=1.0, close_shutter=False),),
        trigger_source="/PXI1Slot4/PXI_Trig0", wait=False, timeout_seconds=30.0)

    with pytest.raises(RuntimeError, match="cancelled"):
        controller.run_synchronized_pulse_train(pulse)
    _wait_for(lambda: cancelled == [True])
    _wait_for(lambda: controller._live_operations == {})

    assert shutter.writes[-1] is False
    # Not opened and closed again: the second look would leave it closed too.
    assert True not in shutter.writes[after_the_cancel[0]:]


def test_a_cancel_as_the_shutter_opens_has_the_pulse_close_it_again(held):
    # The other order: the pulse has looked and is about to open. The cancel
    # is marked and closes the shutter at once; the pulse opens it, looks
    # again, finds itself cancelled, and closes it itself: a pulse that
    # leaves its shutter open would not have.
    daq = held
    controller = NidaqLaserController(_routed())
    set_shutter_open = controller.set_shutter_open
    seen = []

    def opening_meets_a_cancel(channel_id, is_open):
        if is_open and not seen:
            operation, = controller._live_operations.values()
            canceller = threading.Thread(target=operation.cancel)
            canceller.start()
            canceller.join(0.2)
            seen.append((canceller.is_alive(), operation.state))
        return set_shutter_open(channel_id, is_open)

    controller.set_shutter_open = opening_meets_a_cancel
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
            duration_ms=1.0, close_shutter=False),),
        trigger_source="/PXI1Slot4/PXI_Trig0", wait=False, timeout_seconds=30.0)

    with pytest.raises(RuntimeError, match="cancelled"):
        controller.run_synchronized_pulse_train(pulse)
    _wait_for(lambda: controller._live_operations == {})

    assert seen == [(False, LaserOperationState.CANCELLED)]
    # The cancel's close, the pulse's opening, and its own close after it.
    assert daq.task("laser_1_shutter").writes[-3:] == [False, True, False]


def test_a_cancel_during_a_hung_shutter_open_marks_cancelled_at_once(held):
    # A cancel waited, on a lock, for a shutter opening in progress: a
    # driver hung in that write held the cancel's mark, and close() behind
    # it. The mark is made at once now, and the pulse, once its write
    # returns, closes the shutter. The cancel's own shutter close, and
    # close()'s first loop, still write that shutter's task: where DAQmx
    # makes calls on one task wait for each other, those wait behind a hung
    # opening. The fake does not, so this pins the mark alone.
    daq = held
    controller = NidaqLaserController(_routed())
    shutter = daq.task("laser_1_shutter")
    write = shutter.write
    opening, release = threading.Event(), threading.Event()

    def hung_as_it_opens(data, auto_start=False):
        if data is True and not release.is_set():
            opening.set()
            release.wait(5.0)
        return write(data, auto_start=auto_start)

    shutter.write = hung_as_it_opens
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
            duration_ms=1.0, close_shutter=False),),
        trigger_source="/PXI1Slot4/PXI_Trig0", wait=False, timeout_seconds=30.0)
    arming, arming_outcome = _in_thread(controller.run_synchronized_pulse_train, pulse)
    try:
        assert opening.wait(5.0)
        operation, = controller._live_operations.values()
        canceller, cancelled = _in_thread(operation.cancel)
        canceller.join(0.2)

        assert not canceller.is_alive()
        assert cancelled == [True]
        assert operation.state is LaserOperationState.CANCELLED
    finally:
        release.set()
    arming.join(5.0)

    assert "cancelled" in str(arming_outcome[0])
    _wait_for(lambda: controller._live_operations == {})
    assert shutter.writes[-3:] == [False, True, False]


# ------------------------------------------------ round 7: the cancel's order


@pytest.mark.parametrize("armed", ["deferred", "board_stim"])
def test_a_cancel_closes_the_shutter_then_aborts_the_output_then_its_lines(held, armed):
    # The output is what holds the laser on: DAQmx keeps its last sample
    # until the cleanup's reset, and the lines run on its clock. The lines
    # were aborted first, delaying the output's abort and the reset. A
    # deferred pulse's line is aborted after the output. A running pulse's
    # owner, woken by the output's abort, has let go of its tasks by the
    # time that returns, and stops the line itself: the cancel skips it.
    daq = held
    controller = NidaqLaserController(_routed())
    fields = (dict(trigger_source="/PXI1Slot4/PXI_Trig0") if armed == "board_stim"
              else dict(defer_start=True))
    operation = _armed(controller, enable_pmt_shutter=True, **fields)
    before = len(daq.timeline)

    assert operation.cancel()
    assert operation.wait(1.0) is LaserOperationState.CANCELLED

    wanted = {("write", "laser_1_shutter"), ("abort", "laser_sync_pulse_ao"),
              ("abort", "laser_pmt_shutter_do")}
    order = [(entry.event, entry.task, entry.data) for entry in daq.timeline[before:]
             if (entry.event, entry.task) in wanted]
    expected = [("write", "laser_1_shutter", False), ("abort", "laser_sync_pulse_ao", None)]
    if armed == "deferred":
        expected.append(("abort", "laser_pmt_shutter_do", None))
    assert order[:len(expected)] == expected
    if armed == "board_stim":
        # Not even tried: an abort that met the cleanup's close would have
        # raised, and left nothing on the timeline.
        assert ("laser_pmt_shutter_do", "abort") not in daq.controlled
        assert ("abort", "laser_pmt_shutter_do", None) not in order
        assert ("stop", "laser_pmt_shutter_do", None) in _events(
            daq, ("stop", "laser_pmt_shutter_do"))


# ------------------------------------------------ round 7: the "running" label


def test_a_line_started_before_a_cancel_is_logged_as_running(held, caplog):
    # The lines start first, then the output. A cancel landing between the
    # starts found the pulse not yet marked started, and logged a running
    # line as "not started". Each task is marked as it starts now.
    daq = held
    controller = NidaqLaserController(_routed())

    def cancel_as_the_output_starts(task):
        if task.label == "laser_sync_pulse_ao":
            daq.before_start = None
            operation, = controller._live_operations.values()
            operation.cancel()

    daq.before_start = cancel_as_the_output_starts
    with caplog.at_level("DEBUG"):
        with pytest.raises(RuntimeError, match="cancelled"):
            _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0", enable_pmt_shutter=True)
        _wait_for(lambda: controller._live_operations == {})

    messages = [record.getMessage() for record in caplog.records]
    assert "laser 1: aborting its task laser_sync_pulse_ao, not started" in messages
    assert ("laser 1: aborting its running task laser_pmt_shutter_do "
            "(DAQmx may warn 200010)") in messages


# ------------------------------------------------ round 7: the filter


def test_the_aborts_filter_is_not_shadowed_by_a_later_catch_all(held):
    # The filter was added only when absent: a catch-all set after it, in
    # front of it, then showed 200010 again, whatever controller came next.
    NidaqLaserController(_routed())

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        NidaqLaserController(_routed())
        warnings.warn(FakeDaqWarning(ABORT_WARNING_TEXT))

    assert [str(warning.message) for warning in seen if "200010" in str(warning.message)] == []


# ------------------------------------------------ round 9: several shutters


def _two_lasers():
    """christielab10's two lasers: both outputs on the 6713, both shutters on the 6221."""
    lasers = rig_lasers()
    one, = lasers.channels
    two = dataclasses.replace(
        one, channel_id=LaserChannelId.LASER_2, analog_output="PXI1Slot4/ao1",
        diode_input="PXI1Slot5/ai4", command_copy_input="PXI1Slot5/ai5",
        shutter_output="PXI1Slot5/port0/line5")
    return dataclasses.replace(lasers, channels=(one, two))


def _both_left_open():
    """Both lasers, each pulse leaving its shutter open."""
    return LaserSynchronizedPulseTrain(
        pulse_trains=tuple(
            LaserPulseTrain(channel_id=channel_id, amplitude_volts=1.0,
                            duration_ms=1.0, close_shutter=False)
            for channel_id in (LaserChannelId.LASER_1, LaserChannelId.LASER_2)),
        wait=False, timeout_seconds=30.0)


def test_a_cancel_as_the_second_shutter_fails_to_open_still_closes_the_first(held):
    # The cancel lands after the pulse's first look, and closes both
    # shutters; the pulse then opens laser 1's, and laser 2's opening
    # raises. With no second look after a raise, laser 1's stayed open under
    # a cancel, the pulse leaving it so.
    daq = held
    controller = NidaqLaserController(_two_lasers())
    set_shutter_open = controller.set_shutter_open

    def cancelled_as_they_open(channel_id, is_open):
        if is_open and int(channel_id) == 1:
            operation, = controller._live_operations.values()
            operation.cancel()
        if is_open and int(channel_id) == 2:
            raise RuntimeError("DAQmx refused laser 2's shutter")
        return set_shutter_open(channel_id, is_open)

    controller.set_shutter_open = cancelled_as_they_open

    with pytest.raises(RuntimeError, match="cancelled"):
        controller.run_synchronized_pulse_train(_both_left_open())
    _wait_for(lambda: controller._live_operations == {})

    assert daq.task("laser_1_shutter").writes[-2:] == [True, False]


def test_a_shutter_that_fails_to_close_under_a_cancel_leaves_the_others_closed(held, caplog):
    # The cancel lands as laser 2's shutter opens, after laser 1's. The
    # second look closes both; laser 1's close raising skipped laser 2's,
    # which stayed open.
    daq = held
    controller = NidaqLaserController(_two_lasers())
    set_shutter_open = controller.set_shutter_open
    opened = []

    def cancelled_as_the_second_opens(channel_id, is_open):
        if is_open and int(channel_id) == 2:
            operation, = controller._live_operations.values()
            operation.cancel()
            opened.append(2)
        elif not is_open and int(channel_id) == 1 and opened:
            raise RuntimeError("DAQmx refused to close laser 1's shutter")
        return set_shutter_open(channel_id, is_open)

    controller.set_shutter_open = cancelled_as_the_second_opens

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="cancelled"):
            controller.run_synchronized_pulse_train(_both_left_open())
        _wait_for(lambda: controller._live_operations == {})

    assert daq.task("laser_2_shutter").writes[-2:] == [True, False]
    assert any("laser 1 shutter" in record.getMessage() for record in caplog.records)


# ------------------------------------------------ round 9: a pulse never woken


def test_a_never_woken_pulses_cancel_aborts_the_output_then_its_line(monkeypatch):
    # Where the abort does not wake the pulse's wait, its own cleanup never
    # takes its tasks, and the cancel aborts them all: the output first,
    # then its line.
    daq = FakeDaqmx(block_wait=True, hold_waits=True, abort_unblocks=False)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0",
                       enable_pmt_shutter=True)
    before = len(daq.timeline)
    try:
        assert operation.cancel()

        wanted = {("write", "laser_1_shutter"), ("abort", "laser_sync_pulse_ao"),
                  ("abort", "laser_pmt_shutter_do")}
        order = [(entry.event, entry.task, entry.data) for entry in daq.timeline[before:]
                 if (entry.event, entry.task) in wanted]
        assert order == [("write", "laser_1_shutter", False),
                         ("abort", "laser_sync_pulse_ao", None),
                         ("abort", "laser_pmt_shutter_do", None)]
    finally:
        daq.waits_released.set()
    assert operation.wait(5.0) is LaserOperationState.CANCELLED



# ------------------------------------------------ the final fix round


def test_a_command_reset_refused_after_a_cancel_is_critical(held, caplog):
    # After a cancel the output holds the pulse's level until the cleanup
    # puts the command back. That reset refused left it driven, and only an
    # ERROR said so, the cancel's own error being the run's.
    daq = held
    controller = NidaqLaserController(_routed())
    try:
        operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")
        daq.failing_write = "laser_1_manual_ao"
        try:
            with caplog.at_level(logging.DEBUG):
                assert operation.cancel()
                assert operation.wait(1.0) is LaserOperationState.CANCELLED
        finally:
            daq.failing_write = None

        critical = [record.getMessage() for record in caplog.records
                    if record.levelno == logging.CRITICAL]
        assert len(critical) == 1
        assert "Laser 1" in critical[0] and "PXI1Slot4/ao0" in critical[0]
        assert "make the laser safe by hand" in critical[0]
    finally:
        # Left open, the controller kept its tasks and its trigger route.
        controller.close()
    assert controller._trigger_routes == []


# ------------------------------------------------ workstream D: the follow-ups


@pytest.mark.parametrize("wait", [True, False])
def test_a_pulse_outside_its_lasers_range_is_refused_before_its_operation(monkeypatch, wait):
    # The range check ran in the pulse's own thread, once its operation had
    # been made: the refusal came back as that operation's failure, a
    # ValueError, and the laser model kept the amplitude as what the output
    # may hold. Refused first, it drives nothing and makes nothing.
    from autotrainer.device.laser import LaserPulseRefused

    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    made = []

    class Counted(nidaq_laser.NidaqLaserOperation):
        def __init__(self, **kwargs):
            made.append(self)
            super().__init__(**kwargs)

    monkeypatch.setattr(nidaq_laser, "NidaqLaserOperation", Counted)
    tasks_before = len(daq.tasks)
    try:
        with pytest.raises(LaserPulseRefused, match="6.0 V is outside the configured range 0.0..5.0 V"):
            controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
                pulse_trains=(dataclasses.replace(PULSE, amplitude_volts=6.0),),
                wait=wait, timeout_seconds=5.0))

        assert made == []
        assert daq.tasks[tasks_before:] == []
        assert controller._live_operations == {}
    finally:
        controller.close()


def _critical(caplog):
    return [record.getMessage() for record in caplog.records
            if record.levelno == logging.CRITICAL]


def _reset_after_all(caplog):
    return [record.getMessage() for record in caplog.records
            if record.levelno == logging.WARNING and "after all" in record.getMessage()]


def _cancelled_with_its_reset_refused(controller, daq, amplitude_volts):
    """A board STIM pulse at `amplitude_volts`, cancelled; its reset refused."""
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(dataclasses.replace(PULSE, amplitude_volts=amplitude_volts),),
        wait=False, timeout_seconds=30.0, trigger_source="/PXI1Slot4/PXI_Trig0"))
    daq.failing_write = "laser_1_manual_ao"
    try:
        assert operation.cancel()
        assert operation.wait(1.0) is LaserOperationState.CANCELLED
    finally:
        daq.failing_write = None


def test_a_refused_reset_after_a_cancel_names_the_pulses_amplitude(held, caplog):
    # It named the level it could not reach, the minimum, and not the one
    # the output may still hold.
    daq = held
    controller = NidaqLaserController(_routed())
    try:
        with caplog.at_level(logging.WARNING):
            _cancelled_with_its_reset_refused(controller, daq, 2.5)

        critical, = _critical(caplog)
        assert "The output may still hold the pulse's amplitude, 2.5 V" in critical
    finally:
        controller.close()


def test_a_close_that_resets_a_laser_a_critical_left_driven_says_so(held, caplog):
    # The CRITICAL said to make the laser safe by hand; close()'s own reset
    # of it then succeeded, and nothing said the laser was reset after all.
    daq = held
    controller = NidaqLaserController(_routed())
    with caplog.at_level(logging.WARNING):
        _cancelled_with_its_reset_refused(controller, daq, 2.5)
        assert _reset_after_all(caplog) == []

        controller.close()

    warning, = _reset_after_all(caplog)
    assert "Laser 1" in warning and "PXI1Slot4/ao0" in warning and "0 V" in warning
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao")[-1] == 0.0


def test_a_close_after_no_critical_says_nothing_of_a_reset(held, caplog):
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _armed(controller, trigger_source="/PXI1Slot4/PXI_Trig0")
    with caplog.at_level(logging.WARNING):
        assert operation.cancel()
        assert operation.wait(1.0) is LaserOperationState.CANCELLED
        controller.close()

    assert _critical(caplog) == [] and _reset_after_all(caplog) == []
