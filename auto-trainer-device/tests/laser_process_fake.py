"""Helpers for the laser process tests: a fake NI controller for the child, and a stand-in application.

open_fake_controller runs in the laser process (LaserProcessController's
`factory`): NidaqLaserController over nidaq_daqmx_fake's FakeDaqmx, with every
write and abort appended to a JSON-lines file that outlives both processes, so
a test can read what the laser process drove after the processes are gone.

application_main runs in a stand-in for the application, which the tests kill
or stop: it opens a LaserProcessController, arms and starts a pulse that runs
until it is aborted, and reports to the test over a pipe.
"""

import importlib.util
import json
import os
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
FACTORY = f"{_HERE / 'laser_process_fake.py'}:open_fake_controller"

#: A command that makes the fake controller raise in a thread of its own,
#: which nothing catches: the laser process's internal-error path.
RAISE_IN_THREAD_VOLTS = 4.321


def _fake_daqmx_module():
    spec = importlib.util.spec_from_file_location(
        "nidaq_daqmx_fake", _HERE / "nidaq_daqmx_fake.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration():
    """christielab10's laser 1, as nidaq_daqmx_fake lays it out."""
    return _fake_daqmx_module().rig_lasers()


def _log(path, **record):
    record.update(pid=os.getpid(), t=time.monotonic())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def read_log(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def open_fake_controller(configuration, *, log_path, block_wait=True):
    """NidaqLaserController over a FakeDaqmx whose writes and aborts go to `log_path`."""
    fake = _fake_daqmx_module()
    daq = fake.FakeDaqmx(block_wait=block_wait)
    write, control = fake.FakeTask.write, fake.FakeTask.control

    def logged_write(task, data, auto_start=False):
        write(task, data, auto_start)
        values = data.tolist() if hasattr(data, "tolist") else data
        values = values if isinstance(values, list) else [values]
        flat = [value for item in values
                for value in (item if isinstance(item, list) else [item])]
        _log(log_path, event="write", task=task.label, channels=list(task.channels),
             timed=task.timing_kwargs is not None, n=len(flat),
             last=flat[-1] if flat else None,
             high=max(flat) if flat and not isinstance(flat[0], bool) else None)

    def logged_control(task, mode):
        _log(log_path, event=f"control:{mode}", task=task.label)
        control(task, mode)

    fake.FakeTask.write = logged_write
    fake.FakeTask.control = logged_control
    from autotrainer.device import nidaq_laser
    nidaq_laser._load_nidaqmx = lambda: daq
    controller = nidaq_laser.NidaqLaserController(configuration)
    set_command_voltage = controller.set_command_voltage

    def set_command_voltage_or_raise(channel_id, volts):
        if abs(volts - RAISE_IN_THREAD_VOLTS) < 1e-9:
            def fail():
                raise RuntimeError("injected failure in a laser process thread")
            threading.Thread(target=fail, name="InjectedFailure").start()
            return volts
        return set_command_voltage(channel_id, volts)

    controller.set_command_voltage = set_command_voltage_or_raise
    _log(log_path, event="opened")
    return controller


def armed_pulse(duration_ms=5000.0, amplitude_volts=2.5):
    """A deferred-start pulse on laser 1: armed, then started by trigger()."""
    from autotrainer.device import LaserChannelId, LaserPulseTrain, LaserSynchronizedPulseTrain
    return LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=amplitude_volts,
            duration_ms=duration_ms),),
        wait=False, defer_start=True, start_wait_seconds=30.0, timeout_seconds=60.0,
    )


def application_main(conn, log_path, simulated=False, then=None):
    """A stand-in application: a laser process with a pulse running, then `then`.

    Sends ("running", laser process pid) once the pulse has started, and
    ("fault", reason) if the laser process fails. `then` is "hold" (wait to be
    killed), "raise" (make the laser process raise in a thread) or "close".
    """
    from autotrainer.device.laser_process import LaserProcessController
    faults = []

    def on_fault(reason):
        faults.append(reason)
        conn.send(("fault", reason))

    if simulated:
        controller = LaserProcessController(_simulated_configuration(), on_fault=on_fault)
    else:
        controller = LaserProcessController(
            configuration(), factory=FACTORY,
            factory_kwargs={"log_path": str(log_path)}, on_fault=on_fault)
    operation = controller.run_synchronized_pulse_train(armed_pulse())
    operation.trigger()
    conn.send(("running", controller.process_pid))
    if then == "raise":
        controller.set_command_voltage(1, RAISE_IN_THREAD_VOLTS)
    elif then == "close":
        controller.close()
        conn.send(("closed", controller.fault, operation.state.value))
    while True:
        time.sleep(1.0)


def _simulated_configuration():
    from autotrainer.device import LaserChannelConfiguration, LaserChannelId, LaserSystemConfiguration
    return LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(
            channel_id=LaserChannelId.LASER_1, analog_output="Dev1/ao0",
            diode_input="Dev1/ai0", shutter_output="Dev1/port0/line0",
            command_copy_input="Dev1/ai1"),),
        backend="nidaq", hardware_timed=True, sample_rate_hz=100_000.0)
