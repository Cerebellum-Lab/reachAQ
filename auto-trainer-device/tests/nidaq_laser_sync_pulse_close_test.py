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
    LaserPulseTrain,
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
    """A fake whose waits hold until released, and whose stop lets them go."""
    daq = FakeDaqmx(block_wait=True, hold_waits=True, stop_unblocks=True)
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

    assert not close_thread.is_alive()
    assert time.monotonic() - started < 5.0
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
