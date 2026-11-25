"""LaserModel over a laser process: Run Pulse, a waited train, and a fault reaching the model.

The laser process runs NidaqLaserController over the fake NI-DAQmx
(tests/device/laser_process_fake.py, loaded by path: tests/ and
tests/device are separate import roots).
"""

import importlib.util
import os
import signal
import sys
import threading
from pathlib import Path

import pytest

from reachaq.device import LaserChannelId
from reachaq.device.laser import LaserPulseTrain
from reachaq.device.laser_process import LaserProcessController

from reachaq.app.model.laser_model import LaserModel

_FAKE_PATH = (Path(__file__).resolve().parents[1]
              / "tests" / "device" / "laser_process_fake.py")
_spec = importlib.util.spec_from_file_location("laser_process_fake", _FAKE_PATH)
fake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake)


@pytest.fixture
def model(tmp_path):
    model = LaserModel()
    controller = LaserProcessController(
        fake.configuration(), factory=fake.FACTORY,
        factory_kwargs={"log_path": str(tmp_path / "laser.jsonl"), "block_wait": False},
        on_fault=model._on_controller_fault)
    model.set_controller(controller)
    yield model
    model.close()


def _pulse(**overrides):
    values = dict(channel_id=LaserChannelId.LASER_1, amplitude_volts=2.0, duration_ms=2.0)
    values.update(overrides)
    return LaserPulseTrain(**values)


def test_run_pulse_arms_fires_and_completes(model):
    armed = model.arm_manual_pulse(
        _pulse(), manual_context={"profile_id": "test"}, start_wait_seconds=5.0)
    assert armed.fire()
    assert armed.finish() is True
    assert model.last_command_volts[1] == 0.0


def test_a_waited_pulse_train_returns_at_its_baseline(model):
    model.run_pulse_train(_pulse())
    assert model.last_command_volts[1] == 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="kills the laser process by POSIX signal")
def test_a_laser_process_fault_reaches_the_model(model):
    faults = []
    told = threading.Event()

    def changed(name, value, _old):
        if name == LaserModel.CONTROLLER_FAULT:
            faults.append(value)
            told.set()

    model.property_changed += changed
    os.kill(model._controller.process_pid, signal.SIGKILL)
    assert told.wait(30)
    assert "killed by SIGKILL" in faults[0] and "reset to safe" in faults[0]
    with pytest.raises(RuntimeError, match="Laser process failed"):
        model.set_command_voltage(1, 1.0)
