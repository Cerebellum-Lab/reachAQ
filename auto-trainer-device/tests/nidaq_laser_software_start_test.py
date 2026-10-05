"""trigger() starts an armed, deferred pulse on the thread that calls it.

It woke the pulse's own thread through an Event, which then started the
output: 8.2 ms p50 under a GIL hog on christielab10, against 0.6 ms for the
start() itself on the deciding thread (latency-measurements.md, M3 C and D).
The start is now the caller's. The pulse's own thread still owns everything
else: its waits, and its cleanup, which never meets a start still running.

Nothing here touches a driver or a board (nidaq_daqmx_fake).
"""

import dataclasses
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
AO = "laser_sync_pulse_ao"
SHUTTER = "laser_1_shutter"


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met")
        time.sleep(0.005)


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


@pytest.fixture(autouse=True)
def _every_controller_closed(monkeypatch):
    """Close each controller a test opened; then no fake task is left open.

    As nidaq_laser_sync_pulse_close_test's: anything the test held is let go
    first, and a stand-in assigned on the controller itself is dropped.
    """
    opened = []
    init = NidaqLaserController.__init__

    def opening(self, *args, **kwargs):
        init(self, *args, **kwargs)
        opened.append(self)

    with pytest.MonkeyPatch.context() as tracking:
        tracking.setattr(NidaqLaserController, "__init__", opening)
        yield
    monkeypatch.undo()
    fakes = {id(controller._nidaqmx): controller._nidaqmx for controller in opened}
    for daq in fakes.values():
        daq.waits_released.set()
        daq.hang_released.set()
        daq.before_start = None
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
    """A fake whose waits hold until released, as a train still running does."""
    daq = FakeDaqmx(block_wait=True, hold_waits=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    try:
        yield daq
    finally:
        daq.waits_released.set()


class _StartWaitOnCue:
    """An operation's start-wait event, whose wait runs out when the test says.

    set() and is_set() as an Event's. wait() returns True once it is set, or
    False once run_out() is called, whatever its timeout: the start wait's
    end, made at a point the test chooses rather than by a clock. Bounded,
    so a fault cannot hang the test.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._set = False
        self._ran_out = False
        self._error = None

    def set(self):
        with self._condition:
            self._set = True
            self._condition.notify_all()

    def is_set(self):
        with self._condition:
            return self._set

    def run_out(self):
        with self._condition:
            self._ran_out = True
            self._condition.notify_all()

    def raise_in_wait(self, error):
        """The wait raises `error`, as Event.wait(inf) raised OverflowError."""
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def wait(self, timeout=None):
        with self._condition:
            self._condition.wait_for(
                lambda: self._set or self._ran_out or self._error is not None, 10.0)
            if self._error is not None and not self._set:
                raise self._error
            return self._set


class _NotedEvent(threading.Event):
    """An Event that notes when anyone waits on it."""

    def __init__(self):
        super().__init__()
        self.waited_on = threading.Event()

    def wait(self, timeout=None):
        self.waited_on.set()
        return super().wait(timeout)


@pytest.fixture
def on_cue(monkeypatch):
    """Each operation made with a start wait that runs out on cue.

    Its `_start_done`, the event its thread waits on for a start in flight,
    notes that wait.
    """
    made = []

    class OnCue(nidaq_laser.NidaqLaserOperation):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._start_requested = _StartWaitOnCue()
            self._start_done = _NotedEvent()
            made.append(self)

    monkeypatch.setattr(nidaq_laser, "NidaqLaserOperation", OnCue)
    return made


def _routed():
    return rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0")


def _arm(controller, pulse=PULSE, **fields):
    """A deferred pulse, armed on its own thread, as Test stim and a protocol arm one."""
    return controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(pulse,), wait=False, defer_start=True, timeout_seconds=30.0,
        **fields))


def _held_in_the_driver(daq, task_name=AO):
    """A start of `task_name` held, as a driver slow to start it; (starting, release)."""
    starting, release = threading.Event(), threading.Event()

    def hold(task):
        if task.label == task_name:
            starting.set()
            assert release.wait(5.0)

    daq.before_start = hold
    return starting, release


def _output_events(daq, since=0, events=("start", "stop", "close", "abort")):
    return [(entry.event, entry.thread) for entry in daq.timeline[since:]
            if entry.task == AO and entry.event in events]


def _left_safe(daq, controller, operation, since):
    """The laser as every cancel, failure and timeout leaves it.

    The command reset to its minimum by the pulse's own thread, the shutter
    closed, by whichever task wrote it last, nothing reserved and the
    operation let go of.
    """
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao",
                         thread=operation._thread, since=since) == [0.0]
    assert daq.writes_to("PXI1Slot5/port0/line4")[-1] is False
    assert daq.reserved == {}
    assert controller._live_operations == {}


# ------------------------------------------------ the plan's D2.3 tests


def test_trigger_starts_the_armed_output_on_the_calling_thread(held):
    # With PMT margins, so that a clocked line starts before the output.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller, dataclasses.replace(
        PULSE, enable_pmt_shutter=True,
        pmt_shutter_open_delay_ms=1.0, pmt_shutter_close_delay_ms=1.0))
    starts, states = [], []
    daq.before_start = lambda task: starts.append((task.label, threading.current_thread()))

    def trigger():
        operation.trigger()
        states.append(operation.state)

    caller, outcome = _in_thread(trigger)
    caller.join(5.0)

    assert outcome == [None]
    assert starts == [("laser_pmt_shutter_do", caller), (AO, caller)]
    assert states == [LaserOperationState.TRIGGERED]
    daq.waits_released.set()
    assert operation.wait(5.0) is LaserOperationState.COMPLETED


def test_a_second_trigger_starts_nothing(held):
    # One while the first's start is still in the driver, and one after.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    starting, release = _held_in_the_driver(daq)
    first, first_outcome = _in_thread(operation.trigger)
    try:
        assert starting.wait(5.0)
        with pytest.raises(RuntimeError, match="must be armed"):
            operation.trigger()
    finally:
        release.set()
    first.join(5.0)

    assert isinstance(first_outcome[0], float)
    with pytest.raises(RuntimeError, match="must be armed"):
        operation.trigger()
    daq.waits_released.set()
    assert operation.wait(5.0) is LaserOperationState.COMPLETED
    assert daq.starts.count(AO) == 1


def test_a_cancel_during_the_callers_start_aborts_what_it_started(held):
    # The cancel's abort met an output not yet started, which does nothing
    # (H5c): the start that follows is aborted by trigger() itself.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    shutter = daq.task(SHUTTER)
    cancelled_at = []

    def cancel_as_it_starts(task):
        if task.label == AO:
            daq.before_start = None
            operation.cancel()
            cancelled_at.append(len(shutter.writes))

    daq.before_start = cancel_as_it_starts
    before = len(daq.timeline)
    operation.trigger()

    assert operation.wait_until_finished(5.0)
    assert operation.state is LaserOperationState.CANCELLED
    events = [event for event, _thread in _output_events(daq, before, ("start", "abort"))]
    assert events == ["abort", "start", "abort"]
    after_the_cancel = shutter.writes[cancelled_at[0] - 1:]
    assert after_the_cancel[0] is False and True not in after_the_cancel
    _left_safe(daq, controller, operation, 0)


def test_a_refused_start_fails_the_operation(held):
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    daq.failing_task = AO
    before = len(daq.writes)

    assert isinstance(operation.trigger(), float)

    with pytest.raises(RuntimeError, match="refused to start"):
        operation.wait(1.0)
    assert operation.state is LaserOperationState.FAILED
    assert daq.starts == []
    _left_safe(daq, controller, operation, before)


def test_the_cleanup_waits_for_a_start_in_flight(held, on_cue):
    # A cancel wakes the pulse's thread while trigger()'s start is still in
    # the driver: it waits for that start before it stops or closes a task.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    starting, release = _held_in_the_driver(daq)
    caller, _outcome = _in_thread(operation.trigger)
    try:
        assert starting.wait(5.0)
        canceller, cancelled = _in_thread(operation.cancel)
        canceller.join(5.0)
        assert cancelled == [True]
        _wait_for(lambda: operation._start_done.waited_on.is_set()
                  or operation._done.is_set())

        assert [event for event, _ in _output_events(daq, events=("stop", "close"))] == []
        assert not operation._done.is_set()
    finally:
        release.set()
    caller.join(5.0)

    assert operation.wait_until_finished(5.0)
    assert operation.state is LaserOperationState.CANCELLED
    events = [event for event, _ in _output_events(daq, events=("start", "stop", "close"))]
    assert events == ["start", "stop", "close"]
    _left_safe(daq, controller, operation, 0)


def test_close_is_not_held_by_a_start_held_in_the_driver(held, monkeypatch):
    # close() closes the shutters first and gives up on the pulse at its
    # bound, as with any pulse it cannot end; shortened here from 5 s.
    monkeypatch.setattr(nidaq_laser, "_OPERATION_CANCEL_TIMEOUT_S", 1.0)
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    starting, release = _held_in_the_driver(daq)
    caller, _outcome = _in_thread(operation.trigger)
    try:
        assert starting.wait(5.0)
        before = len(daq.timeline)
        started = time.monotonic()
        closer, close_outcome = _in_thread(controller.close)
        closer.join(10.0)

        assert not closer.is_alive()
        assert time.monotonic() - started < 1.0 + 2.0
        first = next(entry for entry in daq.timeline[before:] if entry.thread is closer)
        assert (first.event, first.task, first.data) == ("write", SHUTTER, False)
        error, = close_outcome
        assert f"laser operation {operation.operation_id} cancel" in str(error)
        assert controller.work_left_running() == (f"laser operation {operation.operation_id}",)
    finally:
        release.set()
    caller.join(5.0)

    assert controller.wait_for_work_left_running(5.0)
    assert operation.state is LaserOperationState.CANCELLED
    assert daq.writes_to("PXI1Slot5/port0/line4")[-1] is False
    assert daq.writes_to("PXI1Slot4/ao0")[-1] == 0.0
    assert daq.reserved == {}


# ------------------------------------------------ the two races (controller ruling)


def test_a_start_wait_that_runs_out_during_a_triggers_start_waits_for_it(held, on_cue):
    # Race A, the trigger first: it claims the start, and the start wait runs
    # out while that start is in the driver. The pulse's thread ended the
    # operation then, its cleanup stopping and closing the output under the
    # start. One of them wins: here the trigger, and its start is the pulse's.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    starting, release = _held_in_the_driver(daq)
    caller, outcome = _in_thread(operation.trigger)
    try:
        assert starting.wait(5.0)
        operation._start_requested.run_out()
        # The pulse's thread decides: it waits for the start, or it ends.
        _wait_for(lambda: operation._start_done.waited_on.is_set()
                  or operation._done.is_set())
    finally:
        release.set()
    caller.join(5.0)
    daq.waits_released.set()

    assert operation.wait_until_finished(5.0)
    events = _output_events(daq, events=("start", "stop", "close"))
    assert [event for event, _thread in events] == ["start", "stop", "close"]
    assert events[0][1] is caller
    assert {thread for _event, thread in events[1:]} == {operation._thread}
    assert daq.starts == [AO]
    assert isinstance(outcome[0], float)
    assert operation.state is LaserOperationState.COMPLETED


def test_a_trigger_after_the_start_wait_ran_out_starts_nothing(held, on_cue):
    # Race A, the start wait first: it runs out, and a trigger lands while the
    # cleanup stops the output. The trigger found the operation still armed
    # and started it under the cleanup. The window is closed before the
    # cleanup begins, and the trigger is refused.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    stop_and_close = controller._stop_and_close_task
    triggered = []

    def a_trigger_lands_as_the_output_stops(name, task, errors):
        if task.label == AO and not triggered:
            caller, outcome = _in_thread(operation.trigger)
            caller.join(5.0)
            triggered.append(outcome)
        return stop_and_close(name, task, errors)

    controller._stop_and_close_task = a_trigger_lands_as_the_output_stops
    before = len(daq.writes)
    operation._start_requested.run_out()

    assert operation.wait_until_finished(5.0)
    (refused,), = triggered
    assert isinstance(refused, RuntimeError) and "must be armed" in str(refused)
    assert daq.starts == []
    assert operation.state is LaserOperationState.FAILED
    assert isinstance(operation.error, TimeoutError)
    with pytest.raises(RuntimeError, match="must be armed"):
        operation.trigger()
    _left_safe(daq, controller, operation, before)


def test_a_start_wait_that_raises_leaves_no_window_to_claim(held, on_cue):
    # The start wait itself raised: Event.wait(inf) raised OverflowError, an
    # infinite timeout_seconds passing validation then. The window was never
    # closed, the operation stayed armed through its cleanup, and a trigger
    # landing then started the output under the cleanup's stop and close
    # (the review of tasks 7-8, M1). It is closed before the error goes on.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    stop_and_close = controller._stop_and_close_task
    triggered = []

    def a_trigger_lands_as_the_output_stops(name, task, errors):
        if task.label == AO and not triggered:
            caller, outcome = _in_thread(operation.trigger)
            caller.join(5.0)
            triggered.append(outcome)
        return stop_and_close(name, task, errors)

    controller._stop_and_close_task = a_trigger_lands_as_the_output_stops
    before = len(daq.writes)
    operation._start_requested.raise_in_wait(OverflowError("timeout value is too large"))

    assert operation.wait_until_finished(5.0)
    (refused,), = triggered
    assert isinstance(refused, RuntimeError) and "must be armed" in str(refused)
    assert daq.starts == []
    assert operation.state is LaserOperationState.FAILED
    assert isinstance(operation.error, OverflowError)
    _left_safe(daq, controller, operation, before)


def test_a_cancel_after_the_triggers_start_closes_the_shutter_before_its_abort(held):
    # Race B: a cancel from another thread lands between trigger()'s start
    # and its look, and is held before its own shutter close. trigger() found
    # it and aborted the output at once: on the 6713 the output then holds
    # its last sample, the amplitude just after a start, with the shutter
    # still open until the cancel's close lands. Shutter, then abort.
    daq = held
    controller = NidaqLaserController(_routed())
    operation = _arm(controller)
    output = daq.task(AO)
    start = output.start
    close_pulse_shutters = controller._close_pulse_shutters
    cancellers = []
    cancel_marked, let_the_cancel_close = threading.Event(), threading.Event()

    def held_before_its_shutter_close(pulse_train):
        if threading.current_thread() in cancellers:
            # cancel() marks the operation cancelled before this.
            cancel_marked.set()
            assert let_the_cancel_close.wait(5.0)
        return close_pulse_shutters(pulse_train)

    def started_then_a_cancel_lands():
        start()
        canceller = threading.Thread(target=operation.cancel, daemon=True)
        cancellers.append(canceller)
        canceller.start()
        assert cancel_marked.wait(5.0)

    controller._close_pulse_shutters = held_before_its_shutter_close
    output.start = started_then_a_cancel_lands
    before = len(daq.timeline)
    try:
        operation.trigger()
    finally:
        let_the_cancel_close.set()
    cancellers[0].join(5.0)

    assert operation.wait_until_finished(5.0)
    after = [(entry.event, entry.task, entry.data) for entry in daq.timeline[before:]]
    shut = after.index(("write", SHUTTER, False))
    abort = after.index(("abort", AO, None))
    assert after.index(("start", AO, None)) < shut < abort
    assert operation.state is LaserOperationState.CANCELLED
    _left_safe(daq, controller, operation, 0)
