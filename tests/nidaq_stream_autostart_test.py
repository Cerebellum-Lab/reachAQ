"""The rule that keeps the shared NI-DAQ input stream running by itself.

Against a stand-in monitor, so each case says only what the rule decides:
when a start is wanted, that nothing retries a failure behind the operator's
back, and that a holder such as the DAQ Monitor keeps the stream stopped
until it lets go.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.model.nidaq_stream_autostart import NidaqStreamAutoStart


class _Monitor:
    """Enough of NidaqSignalMonitorModel to see what the rule asks of it."""

    def __init__(self, *, hardware_enabled=True, configured=True, starts=True,
                 start_gate=None):
        self.hardware_enabled = hardware_enabled
        self.configuration = SimpleNamespace(is_enabled=configured)
        self.is_running = False
        self.is_starting = False
        self.starts = starts
        self.start_gate = start_gate
        self.calls = []
        self.paused = []

    def start(self):
        self.calls.append("start")
        if self.start_gate is not None:
            self.start_gate.wait(5.0)
        self.is_running = self.starts
        return self.starts

    def stop(self):
        self.calls.append("stop")
        self.is_running = False

    def show_paused(self, reason):
        self.paused.append(reason)


def _rule(monitor, *, may_start=lambda: True, background=False):
    return NidaqStreamAutoStart(monitor, may_start=may_start, background=background)


def test_a_request_starts_a_configured_stream():
    monitor = _Monitor()

    _rule(monitor).request()

    assert monitor.calls == ["start"]
    assert monitor.is_running


@pytest.mark.parametrize("monitor, may_start", [
    (_Monitor(hardware_enabled=False), True),
    (_Monitor(configured=False), True),
    (_Monitor(), False),
], ids=["hardware disabled", "nothing configured", "application busy"])
def test_nothing_starts_without_hardware_a_plan_or_an_idle_application(
    monitor, may_start,
):
    _rule(monitor, may_start=lambda: may_start).request()

    assert monitor.calls == []


def test_a_running_stream_is_left_running():
    monitor = _Monitor()
    monitor.is_running = True

    _rule(monitor).request()

    assert monitor.calls == []


def test_a_failed_start_is_not_retried_until_something_asks_again():
    # A start that fails must stay failed and visible. Retrying it here would
    # relaunch a broken worker forever, each attempt clearing the error the
    # operator needs to read.
    monitor = _Monitor(starts=False)
    rule = _rule(monitor, background=True)

    rule.request()
    assert rule.wait(5.0)
    time.sleep(0.2)
    assert monitor.calls == ["start"]

    rule.request()  # as a hardware refresh does
    assert rule.wait(5.0)
    assert monitor.calls == ["start", "start"]


def test_a_request_returns_before_a_slow_start_finishes():
    # Discovery and the exact-task preflight run inside start() and can take
    # seconds; the caller is often the Qt thread.
    gate = threading.Event()
    monitor = _Monitor(start_gate=gate)
    rule = _rule(monitor, background=True)

    began = time.monotonic()
    rule.request()
    returned = time.monotonic() - began
    gate.set()

    assert returned < 0.5
    assert rule.wait(5.0)
    assert monitor.calls == ["start"]
    assert monitor.is_running


def test_a_pause_stops_the_stream_and_holds_it_until_every_holder_resumes():
    monitor = _Monitor()
    rule = _rule(monitor)
    rule.request()
    monitor_holder, load_holder = object(), object()

    rule.pause(monitor_holder, "the DAQ Monitor is open")
    rule.pause(load_holder, "the configuration is loading")
    rule.request()

    assert not monitor.is_running
    assert monitor.calls == ["start", "stop", "stop"]
    assert rule.pause_reasons == (
        "the DAQ Monitor is open", "the configuration is loading")
    assert monitor.paused[-1] == "the configuration is loading"

    rule.resume(load_holder)
    assert not monitor.is_running

    rule.resume(monitor_holder)
    assert monitor.is_running
    assert monitor.calls[-1] == "start"
    assert rule.pause_reasons == ()
    assert monitor.paused[-1] == ""


def test_resuming_twice_or_resuming_a_stranger_changes_nothing():
    monitor = _Monitor()
    rule = _rule(monitor)
    holder = object()
    rule.pause(holder, "the DAQ Monitor is open")
    rule.resume(holder)
    starts = monitor.calls.count("start")

    rule.resume(holder)
    rule.resume(object())

    assert monitor.calls.count("start") == starts


def test_a_pause_waits_for_a_start_in_progress_and_then_stops_it():
    # The pause must win. A start that slipped in after the pause, while the
    # DAQ Monitor opens its own tasks on the same lines, is the conflict the
    # pause exists to prevent.
    gate = threading.Event()
    monitor = _Monitor(start_gate=gate)
    rule = _rule(monitor, background=True)
    rule.request()
    deadline = time.monotonic() + 5.0
    while monitor.calls != ["start"] and time.monotonic() < deadline:
        time.sleep(0.01)

    threading.Timer(0.2, gate.set).start()
    rule.pause(object(), "the DAQ Monitor is open")

    assert monitor.calls == ["start", "stop"]
    assert not monitor.is_running


def test_nothing_starts_once_closed():
    monitor = _Monitor()
    rule = _rule(monitor, background=True)

    rule.close()
    rule.request()

    assert rule.wait(1.0)
    assert monitor.calls == []


def _configuration(*names):
    return NidaqSignalStreamConfiguration(
        channels=tuple(
            NidaqSignalChannelConfiguration(
                name=name,
                physical_channel=f"Dev1/port0/line{index}",
                kind="digital",
            )
            for index, name in enumerate(names)
        ),
        is_enabled=bool(names),
    )


@pytest.mark.parametrize("setup, state", [
    (lambda m: None, "disabled"),
    (lambda m: setattr(m, "_hardware_enabled", True), "stopped"),
    (lambda m: (setattr(m, "_hardware_enabled", True),
                setattr(m, "_is_starting", True)), "starting"),
    (lambda m: (setattr(m, "_hardware_enabled", True),
                setattr(m, "_is_running", True)), "running"),
    (lambda m: (setattr(m, "_hardware_enabled", True),
                setattr(m, "_error_message", "task refused")), "error"),
], ids=["disabled", "stopped", "starting", "running", "error"])
def test_the_monitor_names_its_stream_state(setup, state):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _configuration("cam_frames")
    setup(monitor)

    assert monitor.stream_state == state


def test_a_paused_monitor_says_why_and_forgets_it_on_resume():
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _configuration("cam_frames")
    monitor._hardware_enabled = True

    monitor.show_paused("the DAQ Monitor is open")
    assert monitor.status_message == (
        "NI-DAQ signal stream paused: the DAQ Monitor is open")

    monitor.show_paused("")
    assert monitor.status_message == "NI-DAQ signal stream stopped"


def test_a_display_change_is_not_a_change_to_the_running_stream():
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _configuration("cam_frames", "tone1")
    monitor._started_signature = monitor._acquisition_signature()
    monitor._is_running = True

    monitor.set_display_channels(("tone1",))
    assert monitor.running_matches_configuration

    monitor._configuration = _configuration("cam_frames", "tone1", "tone2")
    assert not monitor.running_matches_configuration


def test_the_sample_ring_is_available_while_a_start_holds_the_lock():
    # start() holds the monitor lock through discovery and the preflight.
    # The Qt timers read sample_ring sixty times a second, and waiting on
    # that lock froze every graph for as long as the start took.
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _configuration("cam_frames")
    ring = monitor.sample_ring
    held = threading.Event()
    release = threading.Event()

    def hold_the_lock():
        with monitor._lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold_the_lock, daemon=True)
    holder.start()
    held.wait(5.0)
    try:
        result = []
        reader = threading.Thread(
            target=lambda: result.append(monitor.sample_ring), daemon=True)
        reader.start()
        reader.join(0.5)

        assert result == [ring]
    finally:
        release.set()
        holder.join(5.0)
