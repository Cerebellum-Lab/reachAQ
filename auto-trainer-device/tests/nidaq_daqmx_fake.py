"""A stand-in for the parts of nidaqmx the laser controller uses.

Nothing here touches a driver or a board. The fake records what is done to
each task and every write that happens, with the thread that made it. It
refuses a channel another started task reserves, as DAQmx does at -50103, and
it records the terminals routed.

It is shared by the controller's own tests and by the application's close-path
tests. Those load it by path, because tests/ and auto-trainer-device/tests are
separate import roots.
"""

import threading
import time
import warnings
from types import SimpleNamespace

from autotrainer.device import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
)


class FakeDaqError(RuntimeError):
    """A DAQmx error as nidaqmx raises one: its status code on error_code."""

    def __init__(self, error_code, message):
        super().__init__(f"DAQmx {error_code}: {message}")
        self.error_code = error_code


class FakeDaqWarning(UserWarning):
    """A DAQmx warning, which nidaqmx issues through warnings.warn."""


#: nidaqmx's text for the warning an abort of a running task gives (H8a).
ABORT_WARNING_TEXT = (
    "\nWarning 200010 occurred.\n\nFinite acquisition or generation has been "
    "stopped before the requested number of samples were acquired or generated.")


class FakeTask:
    def __init__(self, daq, name):
        self.daq = daq
        #: The name the task was made with, as the fake and the tests read
        #: it; `name` is nidaqmx's, a driver query.
        self.label = name
        self.channels = []
        self.timing_kwargs = None
        self.writes = []
        self.started = False
        self.closed = False
        self.aborted = threading.Event()
        #: The thread in wait_until_done, and set whenever none is: a stop()
        #: from another thread waits for it (FakeDaqmx, stop from another
        #: thread).
        self._waiter = None
        self._wait_over = threading.Event()
        self._wait_over.set()
        #: Set while an abort of a task never started is under way.
        self._aborting_unstarted = False
        #: Set once the task is closed, for an abort to wait on.
        self._closed_event = threading.Event()
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=self._add)
        self.ai_channels = SimpleNamespace(add_ai_voltage_chan=self._add)
        self.do_channels = SimpleNamespace(add_do_chan=self._add_do)
        self.do_channels_added = False
        self.timing = SimpleNamespace(cfg_samp_clk_timing=self._timing)
        #: The digital edge start trigger, as (source, edge), or None.
        self.start_trigger = None
        self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(
            cfg_dig_edge_start_trig=self._start_trigger))

    @property
    def name(self):
        # nidaqmx's Task.name asks the driver, and a task already cleared
        # answers -200088 (christielab10, H8c: read after an abort the
        # pulse's own thread had closed the task within).
        if self.closed:
            raise FakeDaqError(-200088, "Task specified is invalid or does not exist")
        return self.label

    def _add(self, channel, **_kwargs):
        self.channels.append(channel)

    def _note(self, event, data=None):
        """On the fake's timeline: when, on which thread, what, to which task."""
        self.daq.timeline.append(SimpleNamespace(
            time=time.monotonic(), thread=threading.current_thread(),
            event=event, task=self.label, data=data))

    def _add_do(self, channel, **_kwargs):
        self.do_channels_added = True
        self.channels.append(channel)

    def _timing(self, **kwargs):
        self.timing_kwargs = kwargs

    def _start_trigger(self, source, trigger_edge=None):
        if self.do_channels_added and not self.daq.do_takes_start_trigger:
            # As christielab10's PXI-6221 and PXI-6713 answer: do_trig_usage
            # is empty, and TASK_VERIFY refuses a clocked DO start trigger.
            raise RuntimeError(
                "DAQmx -200452: Specified property is not supported by the "
                f"device or is not applicable to the task ({self.label})")
        self.start_trigger = (source, trigger_edge)

    def _reserve(self):
        for channel in self.channels:
            holder = self.daq.reserved.get(channel)
            if holder is not None and holder is not self:
                raise RuntimeError(
                    f"DAQmx -50103: {channel} is reserved by {holder.label}")
        for channel in self.channels:
            self.daq.reserved[channel] = self

    def _release(self):
        for channel in self.channels:
            if self.daq.reserved.get(channel) is self:
                del self.daq.reserved[channel]

    def write(self, data, auto_start=False):
        if auto_start:
            # An on-demand write reserves its lines only for the write, and a
            # refused one (-50103) writes nothing.
            self._reserve()
            self._release()
        self.writes.append(data)
        self._note("write", data)
        # Every write that happened, in order and whoever made it. Asking for
        # a task by name finds only the last one of that name, and each
        # command reset makes a task of the same name, so the last write
        # alone cannot tell whose it was.
        self.daq.writes.append(SimpleNamespace(
            task=self.label,
            channels=tuple(self.channels),
            data=data,
            thread=threading.current_thread(),
        ))

    def start(self):
        if self.daq.before_start is not None:
            self.daq.before_start(self)
        self.daq.fail_start(self)
        # An abort before a start is a no-op: the task starts and runs as if
        # none had been made (christielab10, H5c).
        self.aborted.clear()
        self._reserve()
        self.started = True
        self.daq.starts.append(self.label)
        self.daq.log.append(("start", self.label))
        self._note("start")

    def wait_until_done(self, timeout):
        self._waiter = threading.current_thread()
        self._wait_over.clear()
        try:
            if self.daq.block_wait:
                # Without hold_waits, the wait times out at `timeout`, as the
                # hardware's does. With it, the wait holds past its timeout,
                # until the test releases it: a hung driver's, or a stand-in
                # for a train long enough to outlast the test.
                limit = self.daq.held_wait_limit if self.daq.hold_waits else timeout
                deadline = time.monotonic() + limit
                while not self._unblocked():
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "DAQmx -200560: Wait Until Done did not indicate done")
                    time.sleep(0.005)
            if self.aborted.is_set():
                raise FakeDaqError(
                    -88709, "The specified operation cannot be performed "
                    "because a task has been aborted")
        finally:
            self._note("wait returned")
            self._waiter = None
            self._wait_over.set()

    def _unblocked(self):
        return (
            self.daq.waits_released.is_set()
            or (self.daq.abort_unblocks and self.aborted.is_set()))

    def read(self, number_of_samples_per_channel, timeout):
        if self.daq.read_samples is not None:
            return self.daq.read_samples(self, number_of_samples_per_channel)
        values = [1.0] * number_of_samples_per_channel
        if len(self.channels) == 1:
            return values
        return [list(values) for _ in self.channels]

    def control(self, mode):
        if self.daq.before_control is not None:
            self.daq.before_control(self)
        self.daq.controlled.append((self.label, mode))
        if self.closed:
            # The fake's model of a call on a task already cleared.
            raise RuntimeError(f"{self.label} was closed before it was aborted")
        if mode == "commit":
            if self.daq.failing_commit and self.label.endswith(self.daq.failing_commit):
                raise RuntimeError(f"DAQmx refused to commit {self.label}")
            # Programmed on the board, and started later; nothing else here.
            self.daq.log.append(("commit", self.label))
            return
        if self.daq.failing_abort and self.label.endswith(self.daq.failing_abort):
            raise RuntimeError(f"DAQmx refused to abort {self.label}")
        # As measured on christielab10:
        # - running (H8a, H8c on the 6713's AO): the abort takes about 13 ms;
        #   the owner's wait is woken near its end, 11.8 ms (H8a) and 12.1 ms
        #   (H8c) from the abort's entry, and the owner stops and clears the
        #   task before the abort returns, cleanly. The fake returns once the
        #   owner has closed the task, bounded, and notes "abort gave up on
        #   owner" on the timeline when the bound runs out, which the cancel
        #   tests assert never happens; with abort_seconds 0, at once, the
        #   rest ordered by the test;
        # - never started, its buffer written (H8b): the abort takes about
        #   1 ms, and a stop made meanwhile raised -88710.
        was_started = self.started
        waited_on = self._waiter is not None
        self._note("abort")
        self._aborting_unstarted = not was_started
        try:
            self.aborted.set()
            if self.daq.abort_releases:
                self.started = False
                self._release()
            if (was_started and waited_on and self.daq.abort_unblocks
                    and self.daq.abort_seconds):
                if not self._closed_event.wait(self.daq.abort_seconds + 0.5):
                    self._note("abort gave up on owner")
            elif self.daq.abort_seconds:
                time.sleep(self.daq.abort_seconds)
        finally:
            self._aborting_unstarted = False
        self._note("abort returned")
        if was_started:
            # Aborting a running task warns, through warnings.warn, with
            # nidaqmx's own text (H5b, H8a).
            warnings.warn(FakeDaqWarning(ABORT_WARNING_TEXT))

    def _refuse_while_aborting(self, what):
        # Only while a never-started task is being aborted (H8b). A stop or
        # close of a running task's owner, woken by its abort, is clean even
        # before the abort returns (H8a, H8c). A refusal is on the timeline,
        # as "stop refused" or "close refused".
        if self._aborting_unstarted:
            self._note(f"{what} refused")
            raise FakeDaqError(
                -88710, "The specified operation cannot be performed because a "
                "task is in the process of being aborted. Wait until the abort "
                "operation is complete and attempt to perform the operation again")

    def stop(self):
        # A stop from another thread does not wake a wait on the task; it
        # waits behind it, until the task ends by itself or the wait times
        # out (christielab10, H5a). Made while a never-started task is being
        # aborted, it raises -88710 (H8b); after an abort, and by a woken
        # owner during one, it is clean (H8a, H8b again, H8c).
        waiter = self._waiter
        if waiter is not None and waiter is not threading.current_thread():
            self._wait_over.wait(self.daq.held_wait_limit)
        self._refuse_while_aborting("stop")
        self._note("stop")
        self.started = False
        self._release()

    def close(self):
        self.daq.sick("task_close")
        # By a woken owner during a running task's abort, clean (H8a, H8c);
        # after an abort, clean (H5b, H8b again). During a never-started
        # task's abort it is not measured: the fake takes it to fail as the
        # stop does.
        self._refuse_while_aborting("close")
        self._note("close")
        self.closed = True
        self._closed_event.set()
        self.started = False
        self._release()


class FakeDaqmx:
    """The parts of nidaqmx the laser controller uses."""

    def __init__(
        self,
        *,
        block_wait=False,
        failing_task=None,
        abort_releases=True,
        abort_unblocks=True,
        hold_waits=False,
        hang=(),
        do_takes_start_trigger=False,
    ):
        self.block_wait = block_wait
        self.failing_task = failing_task
        # Whether an aborted task gives up its lines, and whether the abort
        # returns a wait blocked on it. Measured on christielab10 (H5b,
        # 2026-09-29, a 6221 AI task): TASK_ABORT from another thread wakes a
        # blocked wait in about 36 ms with -88709. abort_unblocks=False is a
        # sick driver's, whose abort does not.
        self.abort_releases = abort_releases
        self.abort_unblocks = abort_unblocks
        self.hold_waits = hold_waits
        # A stop() from another thread never wakes a wait: it waits behind it
        # (FakeTask.stop, H5a). There is no option for the opposite, which
        # the hardware does not do.
        #: How long an abort of a task no one waits on takes; a running
        #: task's returns once its woken owner has closed it (FakeTask.control).
        #: 0 makes every abort return at once, whatever the owner does: a
        #: test that sets it orders what follows itself.
        self.abort_seconds = 0.01
        #: Sick-driver behaviour, not DAQmx's: each call named here -
        #: "connect_terms", "disconnect_terms" or "task_close" - blocks until
        #: hang_released is set, as a call does inside a driver that has hung.
        self.hang = set(hang)
        #: Called with the task at the start of each control(), before it
        #: acts: where a test holds an abort to interleave it.
        self.before_control = None
        #: Called with the task at the start of each start(), before it
        #: acts: where a test holds a start, as a driver slow to start one.
        self.before_start = None
        #: A task, by name suffix, whose abort fails while it is still open.
        self.failing_abort = None
        #: A task, by name suffix, whose commit (TASK_COMMIT) fails.
        self.failing_commit = None
        #: Routes, as (source, destination), the driver will not disconnect.
        self.failing_disconnects = set()
        #: Called as read_samples(task, count) for what a read returns, in
        #: place of a constant 1.0, such as a diode's step response.
        self.read_samples = None
        self.hang_released = threading.Event()
        #: Set as a call starts to hang; hung names each such call, in order.
        self.hanging = threading.Event()
        self.hung = []
        self.held_wait_limit = 20.0
        #: Set to let every blocked wait return.
        self.waits_released = threading.Event()
        self.tasks = []
        self.writes = []
        self.reserved = {}
        self.controlled = []
        self.connected = []
        self.disconnected = []
        #: Names of the tasks started, in the order they were.
        self.starts = []
        #: Routes connected and released and tasks started, in one order.
        self.log = []
        #: Starts, aborts, stops, closes, returned waits and writes of every
        #: task, with the time and the thread (FakeTask._note).
        self.timeline = []
        #: Whether a clocked digital output task takes a start trigger. The
        #: rig's M Series boards say no (do_trig_usage empty, -200452 at
        #: verify, 2026-09-25); an X Series board would say yes.
        self.do_takes_start_trigger = do_takes_start_trigger
        #: As nidaqmx.errors: the category DAQmx warnings are issued with.
        self.errors = SimpleNamespace(DaqWarning=FakeDaqWarning)
        self.constants = SimpleNamespace(
            AcquisitionType=SimpleNamespace(FINITE="finite"),
            TaskMode=SimpleNamespace(TASK_ABORT="abort", TASK_COMMIT="commit"),
            Edge=SimpleNamespace(RISING="rising", FALLING="falling"),
        )
        daq = self

        class _System:
            @staticmethod
            def local():
                return daq

        self.system = SimpleNamespace(System=_System)

    def Task(self, name):  # noqa: N802 - nidaqmx's name
        task = FakeTask(self, name)
        self.tasks.append(task)
        return task

    def fail_start(self, task):
        if self.failing_task and task.label.endswith(self.failing_task):
            raise RuntimeError(f"DAQmx refused to start {task.label}")

    def sick(self, call):
        """Block `call` until released, when it is one of the hung calls."""
        if call in self.hang and not self.hang_released.is_set():
            self.hung.append(call)
            self.hanging.set()
            self.hang_released.wait(self.held_wait_limit)

    def connect_terms(self, source, destination):
        self.sick("connect_terms")
        self.connected.append((source, destination))
        self.log.append(("connect", (source, destination)))

    def disconnect_terms(self, source, destination):
        self.sick("disconnect_terms")
        if (source, destination) in self.failing_disconnects:
            raise RuntimeError(f"DAQmx refused to disconnect {source} -> {destination}")
        self.disconnected.append((source, destination))
        self.log.append(("disconnect", (source, destination)))

    def task(self, suffix):
        return next(task for task in reversed(self.tasks) if task.label.endswith(suffix))

    def writes_to(self, channel, *, task_suffix=None, thread=None, since=0):
        """What was written to `channel`, in order.

        Narrowed to tasks whose name ends in `task_suffix`, to writes made by
        `thread`, and to writes after the first `since`, when given.
        """
        return [
            write.data
            for write in self.writes[since:]
            if write.channels == (channel,)
            and (task_suffix is None or write.task.endswith(task_suffix))
            and (thread is None or write.thread is thread)
        ]


def rig_lasers(**overrides):
    """christielab10's split: the command on the PXI-6713, inputs on the 6221."""
    values = dict(
        channel_id=LaserChannelId.LASER_1,
        analog_output="PXI1Slot4/ao0",
        diode_input="PXI1Slot5/ai8",
        shutter_output="PXI1Slot5/port0/line4",
        command_copy_input="PXI1Slot5/ai3",
    )
    values.update(overrides)
    return LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(**values),),
        backend="nidaq",
        hardware_timed=True,
        sample_rate_hz=100_000.0,
        pmt_shutter_output="PXI1Slot5/port0/line6",
    )
