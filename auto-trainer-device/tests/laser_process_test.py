"""The laser process drives the laser safe when its application dies, hangs, or it fails itself.

Each safety test runs three processes: this test; a stand-in application
(laser_process_fake.application_main), which the test kills or stops; and the
laser process it opened, over a fake NI-DAQmx whose writes and aborts land in
a file (laser_process_fake.open_fake_controller). "Safe" is read off that
file: the running pulse's output aborted, the shutter written closed and the
command written to its minimum (0 V) after the event. The simulation test reads
the simulation bus instead, where the simulated laser publishes 0 V for a
pulse cut short.
"""

import json
import multiprocessing
import os
import signal
import sys
import time

import pytest

import laser_process_fake as fake
from autotrainer.core.simulation import ENV_SIM_BUS, ENV_SIMULATION
from autotrainer.device.laser import LaserPulseRefused
from autotrainer.device.laser_process import (
    EXIT_CLOSED,
    HEARTBEAT_PERIOD_S,
    HEARTBEAT_TIMEOUT_S,
    LaserProcessController,
    LaserProcessFailed,
)
from autotrainer.device.nidaq_laser import LaserOperationState

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="kills and stops processes by POSIX signal")

AO = "PXI1Slot4/ao0"
SHUTTER = "PXI1Slot5/port0/line4"


def _gone(pid):
    try:
        with open(f"/proc/{pid}/status") as status:
            return any(line.startswith("State:") and "Z" in line for line in status)
    except FileNotFoundError:
        return True


def _wait_until(condition, timeout, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(step)
    return condition()


def _start_application(tmp_path, *, then="hold", simulated=False):
    ctx = multiprocessing.get_context("spawn")
    ours, theirs = ctx.Pipe()
    log_path = tmp_path / "laser.jsonl"
    application = ctx.Process(
        # Not daemonic: a daemonic process may not start the laser process.
        target=fake.application_main, args=(theirs, str(log_path), simulated, then))
    application.start()
    theirs.close()
    assert ours.poll(90), "the stand-in application never reported its pulse running"
    kind, laser_pid = ours.recv()
    assert kind == "running"
    return application, ours, log_path, laser_pid


def _safe_after(log_path, since):
    """What was written after `since`: the pulse aborted, the shutter closed, the command at 0 V."""
    after = [record for record in fake.read_log(log_path) if record["t"] >= since]
    aborted = any(record["event"] == "control:abort" and record["task"] == "laser_sync_pulse_ao"
                  for record in after)
    closed = any(record["event"] == "write" and record["channels"] == [SHUTTER]
                 and record["last"] is False for record in after)
    zero = [record for record in after
            if record["event"] == "write" and record["channels"] == [AO]
            and not record["timed"] and record["last"] == 0.0]
    return aborted, closed, zero


def _running_pulse_was_driven(log_path):
    return any(record["event"] == "write" and record["task"] == "laser_sync_pulse_ao"
               and record["high"] == 2.5 for record in fake.read_log(log_path))


@posix_only
def test_application_sigkill_drives_the_laser_safe(tmp_path):
    application, conn, log_path, laser_pid = _start_application(tmp_path)
    assert _running_pulse_was_driven(log_path)
    killed = time.monotonic()
    os.kill(application.pid, signal.SIGKILL)
    assert _wait_until(lambda: _gone(laser_pid), 10), "the laser process outlived its application"
    aborted, closed, zero = _safe_after(log_path, killed)
    assert aborted and closed and zero
    # The pipe's EOF, not the heartbeat deadline.
    assert zero[0]["t"] - killed < 0.9


@posix_only
def test_lost_heartbeat_drives_the_laser_safe(tmp_path):
    application, conn, log_path, laser_pid = _start_application(tmp_path)
    stopped = time.monotonic()
    # Alive but hung: its pipe stays open, and its heartbeats stop.
    os.kill(application.pid, signal.SIGSTOP)
    try:
        assert _wait_until(lambda: _gone(laser_pid), 10)
        aborted, closed, zero = _safe_after(log_path, stopped)
        assert aborted and closed and zero
        # The deadline runs from the last heartbeat, up to one period before
        # the stop.
        assert HEARTBEAT_TIMEOUT_S - HEARTBEAT_PERIOD_S <= zero[0]["t"] - stopped < 2.0
    finally:
        os.kill(application.pid, signal.SIGKILL)


@posix_only
def test_an_exception_in_the_laser_process_drives_the_laser_safe(tmp_path):
    started = time.monotonic()
    application, conn, log_path, laser_pid = _start_application(tmp_path, then="raise")
    try:
        assert conn.poll(10), "the application was not told of the laser process's failure"
        kind, reason = conn.recv()
        assert kind == "fault"
        assert "an internal error" in reason and "drove the laser safe first" in reason
        assert _gone(laser_pid)
        aborted, closed, zero = _safe_after(log_path, started)
        assert aborted and closed and zero
        # Safe by itself, so no reset process opened the laser after it.
        assert {record["pid"] for record in fake.read_log(log_path)} == {laser_pid}
    finally:
        application.kill()


@posix_only
def test_a_laser_process_killed_hard_is_reset_from_a_new_process(tmp_path):
    application, conn, log_path, laser_pid = _start_application(tmp_path)
    try:
        killed = time.monotonic()
        os.kill(laser_pid, signal.SIGKILL)
        assert conn.poll(30)
        kind, reason = conn.recv()
        assert kind == "fault"
        assert "killed by SIGKILL" in reason and "reset to safe" in reason
        after = [record for record in fake.read_log(log_path) if record["t"] >= killed]
        reset_pids = {record["pid"] for record in after} - {laser_pid}
        assert len(reset_pids) == 1
        _, closed, zero = _safe_after(log_path, killed)
        assert closed and zero and zero[0]["pid"] in reset_pids
    finally:
        application.kill()


@posix_only
def test_an_unresponsive_laser_process_is_killed_and_reset(tmp_path):
    application, conn, log_path, laser_pid = _start_application(tmp_path)
    try:
        stopped = time.monotonic()
        os.kill(laser_pid, signal.SIGSTOP)
        assert conn.poll(30)
        kind, reason = conn.recv()
        assert kind == "fault"
        assert "stopped answering" in reason and "reset to safe" in reason
        assert _gone(laser_pid)
        _, closed, zero = _safe_after(log_path, stopped)
        assert closed and zero and zero[0]["pid"] != laser_pid
    finally:
        application.kill()


@posix_only
def test_a_clean_close_is_no_fault(tmp_path):
    started = time.monotonic()
    application, conn, log_path, laser_pid = _start_application(tmp_path, then="close")
    try:
        assert conn.poll(30)
        kind, fault, state = conn.recv()
        assert (kind, fault, state) == ("closed", None, "cancelled")
        assert _wait_until(lambda: _gone(laser_pid), 10)
        aborted, closed, zero = _safe_after(log_path, started)
        assert aborted and closed and zero
        assert not conn.poll(1.5), "a clean close was reported as a fault"
    finally:
        application.kill()


@posix_only
def test_simulated_laser_goes_safe_when_its_application_dies(tmp_path, monkeypatch):
    bus = tmp_path / "bus.jsonl"
    bus.touch()
    monkeypatch.setenv(ENV_SIMULATION, "1")
    monkeypatch.setenv(ENV_SIM_BUS, str(bus))
    application, conn, _, laser_pid = _start_application(tmp_path, simulated=True)
    killed = time.perf_counter()
    os.kill(application.pid, signal.SIGKILL)
    assert _wait_until(lambda: _gone(laser_pid), 10)
    events = [json.loads(line) for line in bus.read_text().splitlines() if line.strip()]
    driven = [event for event in events if event["value"] == 2.5]
    cleared = [event for event in events if event["value"] == 0.0]
    assert driven, "the simulated pulse was never published"
    # The rest of the 5 s pulse is published at 0 V from the moment it was
    # cut short, within a second of the kill.
    assert cleared and all(killed <= event["start"] < killed + 0.9 for event in cleared)
    assert cleared[0]["end"] >= driven[0]["end"] - 1e-6


# The controller itself, in this process --------------------------------------

@pytest.fixture
def controller(tmp_path):
    log_path = tmp_path / "laser.jsonl"
    opened = LaserProcessController(
        fake.configuration(), factory=fake.FACTORY,
        factory_kwargs={"log_path": str(log_path), "block_wait": False})
    yield opened
    opened.close()


def test_commands_and_operations_cross_the_process(controller):
    assert controller.set_command_voltage(1, 1.25) == 1.25
    assert controller._call("command_volts", 1) == 1.25
    operation = controller.run_synchronized_pulse_train(fake.armed_pulse(duration_ms=2.0))
    assert operation.state is LaserOperationState.ARMED
    assert operation.to_record()["timing_status"]["status"] == "software_start"
    ended = []
    operation.add_terminal_callback(ended.append)
    entry = operation.trigger()
    assert isinstance(entry, float)
    assert operation.wait(5.0) is LaserOperationState.COMPLETED
    assert ended == [operation]
    states = [state for state, _, _ in operation.observations]
    assert states[-3:] == ["armed", "triggered", "completed"]
    with pytest.raises(RuntimeError, match="must be armed"):
        operation.trigger()


def test_errors_keep_their_class(controller):
    refused = fake.armed_pulse(amplitude_volts=9.0)
    with pytest.raises(LaserPulseRefused):
        controller.run_synchronized_pulse_train(refused)


def test_calls_after_close_are_refused(tmp_path):
    controller = LaserProcessController(
        fake.configuration(), factory=fake.FACTORY,
        factory_kwargs={"log_path": str(tmp_path / "laser.jsonl")})
    controller.close()
    assert controller._closed
    assert controller._process.exitcode == EXIT_CLOSED
    assert controller.fault is None
    with pytest.raises(LaserProcessFailed):
        controller.set_command_voltage(1, 1.0)
