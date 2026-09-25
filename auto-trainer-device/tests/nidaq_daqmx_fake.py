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
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=self._add)
        self.ai_channels = SimpleNamespace(add_ai_voltage_chan=self._add)
        self.do_channels = SimpleNamespace(add_do_chan=self._add)
        self.timing = SimpleNamespace(cfg_samp_clk_timing=self._timing)

    def _add(self, channel, **_kwargs):
        self.channels.append(channel)

    def _timing(self, **kwargs):
        self.timing_kwargs = kwargs

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

    def _unblocked(self):
        return self.daq.waits_released.is_set() or (
            self.daq.abort_unblocks and self.aborted.is_set())

    def read(self, number_of_samples_per_channel, timeout):
        values = [1.0] * number_of_samples_per_channel
        if len(self.channels) == 1:
            return values
        return [list(values) for _ in self.channels]

    def control(self, mode):
        self.daq.controlled.append((self.name, mode))
        self.aborted.set()
        if self.daq.abort_releases:
            self.started = False
            self._release()

    def stop(self):
        self.started = False
        self._release()

    def close(self):
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
    ):
        self.block_wait = block_wait
        self.failing_task = failing_task
        # Whether an aborted task gives up its lines, and whether the abort
        # returns a wait blocked on it. DAQmx says an abort does both; the
        # tests do not rely on it.
        self.abort_releases = abort_releases
        self.abort_unblocks = abort_unblocks
        self.hold_waits = hold_waits
        self.held_wait_limit = 20.0
        #: Set to let every blocked wait return.
        self.waits_released = threading.Event()
        self.tasks = []
        self.writes = []
        self.reserved = {}
        self.controlled = []
        self.connected = []
        self.disconnected = []
        self.constants = SimpleNamespace(
            AcquisitionType=SimpleNamespace(FINITE="finite"),
            TaskMode=SimpleNamespace(TASK_ABORT="abort"),
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

    def connect_terms(self, source, destination):
        self.connected.append((source, destination))

    def disconnect_terms(self, source, destination):
        self.disconnected.append((source, destination))

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
