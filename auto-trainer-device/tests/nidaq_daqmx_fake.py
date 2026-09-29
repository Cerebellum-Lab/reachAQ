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
from types import SimpleNamespace

from autotrainer.device import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
)


class FakeTask:
    def __init__(self, daq, name):
        self.daq = daq
        self.name = name
        self.channels = []
        self.timing_kwargs = None
        self.writes = []
        self.started = False
        self.closed = False
        self.aborted = threading.Event()
        #: Set by a stop() from another thread while the task ran, with
        #: stop_unblocks; its wait then returns with an error.
        self.stopped_while_waiting = threading.Event()
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=self._add)
        self.ai_channels = SimpleNamespace(add_ai_voltage_chan=self._add)
        self.do_channels = SimpleNamespace(add_do_chan=self._add_do)
        self.do_channels_added = False
        self.timing = SimpleNamespace(cfg_samp_clk_timing=self._timing)
        #: The digital edge start trigger, as (source, edge), or None.
        self.start_trigger = None
        self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(
            cfg_dig_edge_start_trig=self._start_trigger))

    def _add(self, channel, **_kwargs):
        self.channels.append(channel)

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
                f"device or is not applicable to the task ({self.name})")
        self.start_trigger = (source, trigger_edge)

    def _reserve(self):
        for channel in self.channels:
            holder = self.daq.reserved.get(channel)
            if holder is not None and holder is not self:
                raise RuntimeError(
                    f"DAQmx -50103: {channel} is reserved by {holder.name}")
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
        # Every write that happened, in order and whoever made it. Asking for
        # a task by name finds only the last one of that name, and each
        # command reset makes a task of the same name, so the last write
        # alone cannot tell whose it was.
        self.daq.writes.append(SimpleNamespace(
            task=self.name,
            channels=tuple(self.channels),
            data=data,
            thread=threading.current_thread(),
        ))

    def start(self):
        self.daq.fail_start(self)
        self._reserve()
        self.started = True
        self.daq.starts.append(self.name)
        self.daq.log.append(("start", self.name))

    def wait_until_done(self, timeout):
        if self.daq.block_wait:
            # With hold_waits set, the wait models a hung driver: it keeps
            # waiting past its own timeout, until the test releases it.
            limit = self.daq.held_wait_limit if self.daq.hold_waits else timeout
            deadline = time.monotonic() + limit
            while not self._unblocked():
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "DAQmx -200560: Wait Until Done did not indicate done")
                time.sleep(0.005)
        if self.aborted.is_set():
            raise RuntimeError("DAQmx -88709: the task was aborted")
        if self.stopped_while_waiting.is_set():
            raise RuntimeError("the task was stopped while Wait Until Done waited on it")

    def _unblocked(self):
        return (
            self.daq.waits_released.is_set()
            or (self.daq.abort_unblocks and self.aborted.is_set())
            or self.stopped_while_waiting.is_set())

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
        self.daq.controlled.append((self.name, mode))
        if self.closed:
            # The fake's model of a call on a task already cleared.
            raise RuntimeError(f"{self.name} was closed before it was aborted")
        if mode == "commit":
            # Programmed on the board, and started later; nothing else here.
            self.daq.log.append(("commit", self.name))
            return
        if self.daq.failing_abort and self.name.endswith(self.daq.failing_abort):
            raise RuntimeError(f"DAQmx refused to abort {self.name}")
        self.aborted.set()
        if self.daq.abort_releases:
            self.started = False
            self._release()

    def stop(self):
        if self.started and self.daq.stop_unblocks:
            self.stopped_while_waiting.set()
        self.started = False
        self._release()

    def close(self):
        self.daq.sick("task_close")
        self.closed = True
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
        stop_unblocks=False,
        hang=(),
        do_takes_start_trigger=False,
    ):
        self.block_wait = block_wait
        self.failing_task = failing_task
        # Whether an aborted task gives up its lines, and whether the abort
        # returns a wait blocked on it. DAQmx says an abort does both; the
        # tests do not rely on it.
        self.abort_releases = abort_releases
        self.abort_unblocks = abort_unblocks
        self.hold_waits = hold_waits
        #: Whether stopping a running task from another thread returns a
        #: wait blocked on it, with an error. A laser operation's cancel
        #: stops its tasks and relies on that; like the abort, it is the
        #: fake's model of DAQmx, not a measurement.
        self.stop_unblocks = stop_unblocks
        #: Sick-driver behaviour, not DAQmx's: each call named here -
        #: "connect_terms", "disconnect_terms" or "task_close" - blocks until
        #: hang_released is set, as a call does inside a driver that has hung.
        self.hang = set(hang)
        #: Called with the task at the start of each control(), before it
        #: acts: where a test holds an abort to interleave it.
        self.before_control = None
        #: A task, by name suffix, whose abort fails while it is still open.
        self.failing_abort = None
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
        #: Whether a clocked digital output task takes a start trigger. The
        #: rig's M Series boards say no (do_trig_usage empty, -200452 at
        #: verify, 2026-09-25); an X Series board would say yes.
        self.do_takes_start_trigger = do_takes_start_trigger
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
        if self.failing_task and task.name.endswith(self.failing_task):
            raise RuntimeError(f"DAQmx refused to start {task.name}")

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
        return next(task for task in reversed(self.tasks) if task.name.endswith(suffix))

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
