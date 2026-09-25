"""The NI-DAQ laser calibration ramp, against a stand-in for NI-DAQmx.

Nothing here touches a driver or a board: nidaqmx is replaced by a fake that
records what is done to each task, refuses a channel another started task
reserves (as DAQmx does, at -50103), and records the terminals routed.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from autotrainer.device import (
    LaserCalibrationRamp,
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser


class _FakeTask:
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
        self.writes.append(data)
        if auto_start:
            # An on-demand write reserves its lines only for the write.
            self._reserve()
            self._release()

    def start(self):
        self.daq.fail_start(self)
        self._reserve()
        self.started = True

    def wait_until_done(self, timeout):
        if self.daq.block_wait and not self.aborted.wait(timeout):
            raise RuntimeError("DAQmx -200560: Wait Until Done did not indicate done")
        if self.aborted.is_set():
            raise RuntimeError("DAQmx -88709: the task was aborted")

    def read(self, number_of_samples_per_channel, timeout):
        values = [1.0] * number_of_samples_per_channel
        if len(self.channels) == 1:
            return values
        return [list(values) for _ in self.channels]

    def control(self, mode):
        self.daq.controlled.append((self.name, mode))
        self.aborted.set()
        self.started = False
        self._release()

    def stop(self):
        self.started = False
        self._release()

    def close(self):
        self.closed = True
        self.started = False
        self._release()


class _FakeDaqmx:
    """The parts of nidaqmx the laser controller uses."""

    def __init__(self, *, block_wait=False, failing_task=None):
        self.block_wait = block_wait
        self.failing_task = failing_task
        self.tasks = []
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
        task = _FakeTask(self, name)
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


def _rig_lasers(**overrides):
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


RAMP = LaserCalibrationRamp(
    channel_id=LaserChannelId.LASER_1,
    start_volts=0.0,
    stop_volts=5.0,
    steps=3,
    samples_per_step=10,
    timeout_seconds=5.0,
)


@pytest.fixture
def daq(monkeypatch):
    fake = _FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: fake)
    return fake


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met")
        time.sleep(0.01)


def test_closing_the_controller_mid_ramp_stops_the_ramp_and_resets_the_laser(daq):
    # reachAQ closing mid-ramp closes the ramp's controller from its own
    # thread while the ramp thread waits inside DAQmx. The ramp's tasks hold
    # the analog output, so the command reset in close() failed at -50103,
    # and the 6713 kept its last sample with the shutter open.
    daq.block_wait = True
    controller = NidaqLaserController(_rig_lasers())
    outcome = []

    def ramp():
        try:
            outcome.append(controller.run_calibration_ramp(RAMP))
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=ramp)
    thread.start()
    _wait_for(lambda: any(
        task.name.endswith("calibration_ao") and task.started for task in daq.tasks))
    shutter = daq.task("laser_1_shutter")
    assert shutter.writes[-1] is True

    controller.close()
    thread.join(5.0)

    assert not thread.is_alive()
    assert ("laser_1_calibration_ao", "abort") in daq.controlled
    assert ("laser_1_calibration_ai", "abort") in daq.controlled
    # The command went back to its minimum and the shutter closed.
    assert daq.task("laser_1_manual_ao").writes == [0.0]
    assert shutter.writes[-1] is False
    error, = outcome
    assert isinstance(error, RuntimeError) and "aborted" in str(error)


def test_a_ramp_on_a_closed_controller_starts_nothing(daq):
    # A close that came first has reset the outputs; the ramp must not then
    # drive the laser after it.
    controller = NidaqLaserController(_rig_lasers())
    controller.close()

    with pytest.raises(RuntimeError, match="closed"):
        controller.run_calibration_ramp(RAMP)

    assert not any(task.started for task in daq.tasks)


# ------------------------------------------------------ the cross-board clock


def test_the_ramp_clocks_its_inputs_over_a_backplane_line(daq):
    # The AI task on the 6221 was clocked from /PXI1Slot4/ao/SampleClock by
    # name, a route DAQmx makes across the backplane only by reserving a
    # line - which it refuses on this unidentified chassis, -89125. The clock
    # is put on PXI_Trig1 on the 6713 and read as the 6221's PXI_Trig1, as
    # a pulse train's shared clock is.
    import dataclasses

    controller = NidaqLaserController(_rig_lasers())

    points = controller.run_calibration_ramp(
        dataclasses.replace(RAMP, enable_pmt_shutter=True))

    assert len(points) == 3
    route = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")
    assert daq.connected == [route]
    assert daq.task("laser_1_calibration_ai").timing_kwargs["source"] == "/PXI1Slot5/PXI_Trig1"
    assert daq.task("laser_pmt_shutter_calibration_do").timing_kwargs["source"] == (
        "/PXI1Slot5/PXI_Trig1")
    # Held for the ramp alone.
    assert daq.disconnected == [route]
    assert controller._trigger_routes == []


def test_the_ramp_releases_its_clock_route_when_it_fails(monkeypatch):
    daq = _FakeDaqmx(failing_task="calibration_ai")
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(_rig_lasers())

    with pytest.raises(RuntimeError, match="refused to start"):
        controller.run_calibration_ramp(RAMP)

    route = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")
    assert daq.connected == [route]
    assert daq.disconnected == [route]
    assert controller._trigger_routes == []


def test_the_ramp_keeps_the_controllers_own_trigger_route(daq):
    controller = NidaqLaserController(_rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))
    trigger_route = ("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")
    assert daq.connected == [trigger_route]

    controller.run_calibration_ramp(RAMP)

    assert daq.disconnected == [("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")]
    assert controller._trigger_routes == [trigger_route]


def test_a_ramp_on_one_board_needs_no_route(daq):
    controller = NidaqLaserController(_rig_lasers(analog_output="PXI1Slot5/ao0"))

    controller.run_calibration_ramp(RAMP)

    assert daq.connected == []
    assert daq.task("laser_1_calibration_ai").timing_kwargs["source"] == "/PXI1Slot5/ao/SampleClock"
