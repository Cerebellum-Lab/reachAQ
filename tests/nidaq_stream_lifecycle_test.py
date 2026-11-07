"""The shared NI-DAQ input stream runs by itself while reachAQ is idle.

It used to start only from a button, or with System Mode. These drive the
application model with the real stream model and a stand-in worker process,
so each start is a real one: a worker launched, a ready message received, a
process id to compare.
"""

import dataclasses
import logging
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from autotrainer.core import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NidaqPortConfiguration,
    NidaqTimingConfiguration,
)
from tools.acquisition.model import nidaq_monitor_session, nidaq_signal_monitor_model
from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts
from tools.acquisition.model.nidaq_monitor_session import NidaqMonitorSession
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState

import nidaq_stream_fakes
from nidaq_monitor_session_test import _Stream


def _with_nidaq(system_config, trainer_config_dir):
    system_config.hardware.nidaq_enabled = True
    system_config.nidaq_ports = NidaqPortConfiguration(cam_frames="Dev1/port0/line0")
    system_config.save_default(trainer_config_dir)


def _with_fake_worker(app_model, worker=nidaq_stream_fakes.idle_worker):
    monitor = app_model.nidaq_signal_monitor
    monitor._worker_target = worker
    monitor._device_discovery = nidaq_stream_fakes.discover_dev1
    monitor._exact_preflight = None
    return monitor


def _wait_until_running_is_announced(monitor, timeout=10.0):
    """Wait for the monitor to finish telling everyone its worker is ready."""
    deadline = time.monotonic() + timeout
    while (monitor.status_message != "NI-DAQ signal stream running"
           and time.monotonic() < deadline):
        time.sleep(0.01)
    return monitor.status_message == "NI-DAQ signal stream running"


def _settle(app_model, *, running=True, timeout=10.0):
    """Wait for any automatic start, then for the stream to reach `running`.

    Running includes the announcement: the monitor settles its flags before
    it tells the application, so READY is written a moment after them.
    """
    # vars(), not getattr: AppModel answers a missing attribute with an
    # EventsException rather than an AttributeError.
    rule = vars(app_model).get("_nidaq_stream_autostart")
    if rule is not None:
        assert rule.wait(timeout)
    monitor = app_model.nidaq_signal_monitor
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not monitor.is_starting and monitor.is_running == running:
            if running:
                _wait_until_running_is_announced(
                    monitor, max(0.0, deadline - time.monotonic()))
            return monitor
        time.sleep(0.02)
    return monitor


def _worker_pid(monitor):
    process = monitor._process
    return None if process is None else process.pid


def _nidaq_state(app_model):
    return app_model.subsystem_statuses[SubsystemId.NIDAQ_STREAM.value]


@pytest.fixture
def nidaq_app(app_model, system_config, trainer_config_dir):
    _with_nidaq(system_config, trainer_config_dir)
    _with_fake_worker(app_model)
    try:
        yield app_model
    finally:
        rule = vars(app_model).get("_nidaq_stream_autostart")
        if rule is not None:
            rule.close()
        app_model.nidaq_signal_monitor.close()


def test_the_stream_starts_by_itself_once_the_configuration_loads(nidaq_app):
    assert nidaq_app.load_configuration() is True

    monitor = _settle(nidaq_app)

    assert monitor.is_running
    assert not nidaq_app.acquisition_started
    assert _nidaq_state(nidaq_app).state is SubsystemState.READY


def test_a_disabled_nidaq_is_never_started(app_model, monkeypatch):
    starts = []
    monkeypatch.setattr(app_model.nidaq_signal_monitor, "start",
                        lambda: starts.append(True) or True)

    assert app_model.load_configuration() is True
    _settle(app_model, running=False, timeout=1.0)

    assert starts == []


def test_every_run_restarts_the_stream_from_a_fresh_timing_anchor(nidaq_app, monkeypatch):
    # NI sample times are the task-start anchor plus index / rate, anchored
    # once per worker, and are compared with host perf_counter times; the NI
    # and host clocks drift apart. A stream started in Idle and carried into
    # System Mode would bring hours of drift into a recording, so each Run
    # starts a new worker, as before the stream ran in Idle.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    idle_pid = _worker_pid(monitor)
    try:
        assert nidaq_app.capture_start() is True

        assert monitor.is_running
        assert _worker_pid(monitor) not in (None, idle_pid)
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY
    finally:
        nidaq_app.capture_stop()


def test_a_run_during_an_automatic_start_still_gets_a_fresh_worker(nidaq_app, monkeypatch):
    # An automatic start decides to start, then holds the monitor through
    # discovery and preflight, and says it is starting only once its worker
    # is launched. Run assigns the session's project to the monitor, under
    # the same lock, before it reaches the NI-DAQ domain, so a start already
    # inside the monitor holds Run there until the worker is launched. One
    # that decided just before Run began and reached the monitor just after
    # that assignment did not: Run found the stream neither running nor
    # starting, skipped the stop, and its start() waited on the monitor and
    # returned True for the worker the automatic start had launched in Idle.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    monitor = nidaq_app.nidaq_signal_monitor
    rule = nidaq_app._nidaq_stream_autostart
    decided = threading.Event()
    enter_monitor = threading.Event()
    in_discovery = threading.Event()
    release = threading.Event()
    may_start = rule._may_start

    def decide_then_stall():
        allowed = may_start()
        if allowed and not decided.is_set():
            decided.set()
            enter_monitor.wait(10.0)
        return allowed

    def gated_discovery():
        in_discovery.set()
        release.wait(10.0)
        return nidaq_stream_fakes.discover_dev1()

    start_can_domain = nidaq_app._start_can_domain

    def let_the_automatic_start_in(*args, **kwargs):
        # Past the project assignment, before the NI-DAQ domain.
        enter_monitor.set()
        in_discovery.wait(10.0)
        return start_can_domain(*args, **kwargs)

    rule._may_start = decide_then_stall
    monitor._device_discovery = gated_discovery
    monkeypatch.setattr(nidaq_app, "_start_can_domain", let_the_automatic_start_in)
    automatic_pids = []
    original_start = monitor.start

    def start():
        started = original_start()
        if threading.current_thread().name == "nidaq-stream-autostart":
            automatic_pids.append(_worker_pid(monitor))
        return started

    monitor.start = start
    started = []
    run = threading.Thread(
        target=lambda: started.append(nidaq_app.capture_start()),
        name="StartAcquisition", daemon=True)
    try:
        assert nidaq_app.load_configuration() is True
        assert decided.wait(5.0), "the stream did not start by itself"

        run.start()
        # Let the automatic start finish only once Run is at the NI-DAQ
        # domain, past the point where it decides whether to restart.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            status = nidaq_app.subsystem_statuses.get(SubsystemId.NIDAQ_STREAM.value)
            if status is not None and status.reason == "starting synchronized NI-DAQ tasks":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Run never reached the NI-DAQ domain")
        assert in_discovery.is_set()
        assert not (monitor.is_running or monitor.is_starting)
        time.sleep(0.3)
        release.set()
        run.join(30.0)

        assert not run.is_alive()
        assert started == [True]
        assert len(automatic_pids) == 1 and automatic_pids[0] is not None
        assert monitor.is_running
        assert _worker_pid(monitor) not in (None, automatic_pids[0])
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY
    finally:
        enter_monitor.set()
        release.set()
        if run.ident is not None:
            run.join(30.0)
            nidaq_app.capture_stop()


def test_a_run_waits_for_an_automatic_start_only_within_its_bound(
    nidaq_app, monkeypatch, caplog,
):
    # A start stuck in discovery or the preflight must not hang Run; past the
    # bound it carries on as before and says so.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    monitor = nidaq_app.nidaq_signal_monitor
    in_discovery = threading.Event()
    release = threading.Event()

    def gated_discovery():
        in_discovery.set()
        release.wait(10.0)
        return nidaq_stream_fakes.discover_dev1()

    monitor._device_discovery = gated_discovery
    releaser = threading.Timer(2.0, release.set)
    try:
        assert nidaq_app.load_configuration() is True
        assert in_discovery.wait(5.0), "the stream did not start by itself"
        releaser.start()

        assert nidaq_app._start_nidaq_domain(restart=True, timeout=1.0) is True

        assert monitor.is_running
        assert any(
            "still running after 1 seconds" in record.getMessage()
            for record in caplog.records
        )
    finally:
        release.set()
        releaser.cancel()


def test_the_daq_monitor_cannot_take_the_stream_from_a_running_acquisition(
    nidaq_app, monkeypatch,
):
    # Pausing for the monitor stopped the acquisition's stream, marked it
    # STOPPED rather than FAILED, so a recording carried on without its NI
    # data, and nothing restarted it when the monitor closed.
    monkeypatch.setattr(nidaq_monitor_session, "NidaqSignalMonitorModel", _Stream)
    monkeypatch.setattr(nidaq_monitor_session, "discover_nidaq_devices",
                        lambda: ((), None))
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    try:
        assert nidaq_app.capture_start() is True
        assert monitor.is_running
        pid = _worker_pid(monitor)

        with pytest.raises(RuntimeError, match="belongs to System Mode while it runs"):
            NidaqMonitorSession(nidaq_app)

        assert monitor.is_running
        assert _worker_pid(monitor) == pid
        assert nidaq_app._nidaq_stream_autostart.pause_reasons == ()
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY
    finally:
        nidaq_app.capture_stop()


def test_the_daq_monitor_waits_for_the_last_recording_session_to_finish(
    nidaq_app, monkeypatch,
):
    # In Idle, a recording session still aborting was refused as "belongs to
    # System Mode while it runs; set System Mode to Idle first", advice an
    # operator already in Idle cannot follow.
    monkeypatch.setattr(nidaq_monitor_session, "NidaqSignalMonitorModel", _Stream)
    monkeypatch.setattr(nidaq_monitor_session, "discover_nidaq_devices",
                        lambda: ((), None))
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    pid = _worker_pid(monitor)
    monkeypatch.setattr(nidaq_app._recording_session, "status",
                        SessionRecordingStatus.ABORTING)

    with pytest.raises(RuntimeError) as refused:
        NidaqMonitorSession(nidaq_app)

    message = str(refused.value)
    assert "recording session" in message and "aborting" in message
    assert "System Mode" not in message
    assert monitor.is_running
    assert _worker_pid(monitor) == pid
    assert nidaq_app._nidaq_stream_autostart.pause_reasons == ()


def test_saving_daq_ports_restarts_the_stream_with_the_new_plan(nidaq_app):
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    pid = _worker_pid(monitor)

    nidaq_app.update_daq_port_configuration(
        dataclasses.replace(nidaq_app.nidaq_ports, tone1="Dev1/port0/line3"),
        nidaq_app.laser.configuration,
    )
    _settle(nidaq_app)

    assert monitor.is_running
    assert _worker_pid(monitor) != pid
    assert "tone1" in monitor.sample_ring.channel_names


def test_a_daq_ports_save_clashing_with_the_trigger_readback_changes_nothing(nidaq_app):
    # Edit DAQ Ports refuses the pair itself now, but the application does
    # not depend on it: the plan refuses it too, and the laser configuration
    # was applied before the plan was built, so a refused save still changed
    # the lasers.
    assert nidaq_app.load_configuration() is True
    _settle(nidaq_app)
    laser_before = nidaq_app.laser.configuration
    ports_before = nidaq_app.nidaq_ports
    plan_before = nidaq_app.nidaq_signal_monitor.configuration
    clashing = LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="Dev1/ao0",
                diode_input="Dev1/ai0",
                shutter_output="Dev1/port0/line2",
                trigger_monitor_input="Dev1/ai0",
            ),
        ),
        backend="null",
    )

    with pytest.raises(ValueError) as refused:
        nidaq_app.update_daq_port_configuration(
            dataclasses.replace(ports_before, tone1="Dev1/port0/line3"), clashing)

    assert "laser1_diode" in str(refused.value)
    assert "laser1_trigger" in str(refused.value)
    assert nidaq_app.laser.configuration == laser_before
    assert nidaq_app.loaded_configuration.laser == laser_before
    assert nidaq_app.nidaq_ports == ports_before
    assert nidaq_app.loaded_configuration.nidaq_ports == ports_before
    assert nidaq_app.nidaq_signal_monitor.configuration == plan_before
    assert _settle(nidaq_app).is_running


def test_a_failed_start_is_shown_and_not_retried(app_model, system_config,
                                                 trainer_config_dir):
    _with_nidaq(system_config, trainer_config_dir)
    monitor = _with_fake_worker(app_model, nidaq_stream_fakes.failing_worker)
    starts = []
    original_start = monitor.start

    def counting_start():
        starts.append(True)
        return original_start()

    monitor.start = counting_start
    try:
        assert app_model.load_configuration() is True
        _settle(app_model, running=False)
        deadline = time.monotonic() + 5.0
        while not monitor.error_message and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.5)

        assert starts == [True]
        assert "the task was refused" in monitor.error_message
        assert monitor.stream_state == "error"
        failed = _nidaq_state(app_model)
        assert failed.state is SubsystemState.FAILED
        assert "the task was refused" in failed.error
    finally:
        rule = vars(app_model).get("_nidaq_stream_autostart")
        if rule is not None:
            rule.close()
        monitor.close()


class _HeldExitContext:
    """The monitor's process context, holding its first worker's thread.

    The monitor's thread reads the worker's exit code only in the finally of
    its _run, after its loop has ended. Holding that read keeps the thread
    alive past stop()'s two-second join, as a slow driver teardown can.
    """

    def __init__(self, context):
        self._context = context
        self.release = threading.Event()
        self.held = threading.Event()
        self._made = 0

    def __getattr__(self, name):
        return getattr(self._context, name)

    def __reduce_ex__(self, _protocol):
        # The sample ring keeps the context it was made with and goes to the
        # worker with it; the worker gets the stream's real one.
        from autotrainer.core.multiproc import get_nidaq_mp_ctx
        return get_nidaq_mp_ctx, ()

    def Process(self, *args, **kwargs):  # noqa: N802 - multiprocessing's name
        process = self._context.Process(*args, **kwargs)
        self._made += 1
        return _HeldExitProcess(process, self) if self._made == 1 else process


class _HeldExitProcess:
    def __init__(self, process, context):
        self._process = process
        self._context = context

    def __getattr__(self, name):
        return getattr(self._process, name)

    @property
    def exitcode(self):
        self._context.held.set()
        self._context.release.wait(30.0)
        return self._process.exitcode


def test_an_old_workers_late_error_cannot_fail_the_stream_that_replaced_it(
    nidaq_app, caplog,
):
    # stop() waits two seconds for the monitor's thread, then lets it go. A
    # thread that outlived that posted its worker's error from its finally
    # with no check that its process was still the monitor's, over a new and
    # healthy worker: the monitor showed that error, and NI-DAQ read FAILED.
    monitor = nidaq_app.nidaq_signal_monitor
    monitor._worker_target = nidaq_stream_fakes.failing_as_stopped_worker
    context = monitor._mp_ctx = _HeldExitContext(monitor._mp_ctx)
    try:
        assert nidaq_app.load_configuration() is True
        assert _settle(nidaq_app).is_running, "worker A did not start"
        old_pid = _worker_pid(monitor)
        old_thread = monitor._thread

        monitor.stop()

        assert context.held.wait(5.0) and old_thread.is_alive()
        monitor._worker_target = nidaq_stream_fakes.idle_worker
        assert monitor.start()
        assert _settle(nidaq_app).is_running, "worker B did not start"
        new_pid = _worker_pid(monitor)
        assert new_pid not in (None, old_pid)
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY

        # Worker A's error arrived while it was still the monitor's, and was
        # logged then; what follows is its thread posting it after B began.
        caplog.clear()
        with caplog.at_level("WARNING"):
            context.release.set()
            old_thread.join(5.0)

        assert not old_thread.is_alive()
        assert monitor.is_running and _worker_pid(monitor) == new_pid
        assert monitor.error_message == ""
        assert monitor.status_message == "NI-DAQ signal stream running"
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY
        late = [record for record in caplog.records
                if "the old task failed as it was stopped" in record.getMessage()]
        warning, = late
        assert warning.levelname == "WARNING"
        assert f"pid={old_pid}" in warning.getMessage()
    finally:
        context.release.set()


class _EndedProcess:
    """A worker process, for driving the monitor's thread directly."""

    def __init__(self, pid, *, alive=False):
        self.pid = pid
        self.exitcode = 0
        self._alive = alive

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self._alive = False


def _monitor_thread_log(caplog, process, messages, *, current, **settings):
    monitor = nidaq_signal_monitor_model.NidaqSignalMonitorModel()
    for name, value in settings.items():
        setattr(monitor, name, value)
    # A newer worker, or this one.
    monitor._process = process if current else object()
    with caplog.at_level("WARNING"):
        monitor._run(process, messages, threading.Event(), time.perf_counter())
    return monitor


def test_a_replaced_workers_own_error_is_a_warning_naming_it(caplog):
    # Its thread logged the error it read at ERROR, with no pid: the status
    # bar showed "Error: NI-DAQ signal stream worker error" over a healthy
    # newer worker.
    messages = queue.Queue()
    messages.put(("error", "the old task failed"))
    messages.put(("stopped", None))

    monitor = _monitor_thread_log(caplog, _EndedProcess(4242), messages, current=False)

    said = [record for record in caplog.records
            if "the old task failed" in record.getMessage()]
    assert said
    assert all(record.levelname == "WARNING" and "pid=4242" in record.getMessage()
               for record in said)
    assert monitor.error_message == ""


def test_a_replaced_workers_startup_timeout_is_a_warning_naming_it(caplog):
    monitor = _monitor_thread_log(
        caplog, _EndedProcess(4343, alive=True), queue.Queue(), current=False,
        _startup_timeout_seconds=0.0)

    said = [record for record in caplog.records
            if "did not become ready" in record.getMessage()]
    assert said
    assert all(record.levelname == "WARNING" and "pid=4343" in record.getMessage()
               for record in said)
    assert monitor.error_message == ""


@pytest.mark.parametrize(("current", "level"), [(False, "WARNING"), (True, "ERROR")])
def test_a_relayed_worker_log_is_capped_at_warning_once_the_worker_is_replaced(
    caplog, current, level,
):
    # The worker's own logger.exception, as when it fails or fails to close
    # its tasks, was relayed at ERROR whichever worker it came from: on the
    # status bar over a healthy newer one.
    messages = queue.Queue()
    messages.put(("log", (
        logging.ERROR, "tools.acquisition.model.nidaq_signal_monitor_model",
        "Failed to close the NI-DAQ signal stream", False)))
    messages.put(("stopped", None))

    _monitor_thread_log(caplog, _EndedProcess(4545), messages, current=current)

    relayed, = [record for record in caplog.records
                if "Failed to close the NI-DAQ signal stream" in record.getMessage()]
    assert relayed.levelname == level
    assert "pid=4545" in relayed.getMessage()


def test_the_current_workers_error_is_an_error_naming_it(caplog):
    messages = queue.Queue()
    messages.put(("error", "the task failed"))
    messages.put(("stopped", None))

    monitor = _monitor_thread_log(caplog, _EndedProcess(4444), messages, current=True)

    read = [record for record in caplog.records
            if "worker error: the task failed" in record.getMessage()]
    assert [record.levelname for record in read] == ["ERROR"]
    assert "pid=4444" in read[0].getMessage()
    assert monitor.error_message == "the task failed"


def test_a_hardware_refresh_starts_a_stopped_stream(nidaq_app, monkeypatch):
    from hardware_status_content_test import _patch_hardware_scans

    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    monitor.stop()
    _patch_hardware_scans(nidaq_app, monkeypatch)

    nidaq_app.refresh_hardware_bindings()

    assert _settle(nidaq_app).is_running


def _discover_dev1_lines(digital_input_max_rate):
    """Dev1 listing its port0 lines, with the digital-input rate discovery reads.

    discover_dev1 lists no lines at all, so no rule about lines applies to it.
    A PXI-6713 lists its port0 lines and reports no digital-input rate; a
    PXI-6221 reports 1 MHz (christielab10, 2026-09-25).
    """
    def discover():
        return (NidaqDevicePorts(
            name="Dev1",
            digital_inputs=tuple(f"Dev1/port0/line{line}" for line in range(8)),
            counter_outputs=("Dev1/ctr0",),
            digital_input_max_rate=digital_input_max_rate,
        ),), None
    return discover


def _counting_preflight(calls, *, valid):
    """The exact-task preflight; refused as a 6713 port0 line is refused there."""
    def preflight(_configuration, _timing_plan):
        calls.append(True)
        if valid:
            return SimpleNamespace(is_valid=True)
        return SimpleNamespace(
            is_valid=False, stage="DI_DataXferMech",
            error="DaqError -200452: Specified property is not supported by "
                  "the device or is not applicable to the task",
            corrective_action="")
    return preflight


def _status_bar_lines(caplog, text):
    """ERROR lines the status bar would show, as its handler cuts them."""
    shown = []
    for record in caplog.records:
        if record.levelno < logging.ERROR:
            continue
        lines = [line.strip() for line in record.getMessage().splitlines() if line.strip()]
        if lines and text in lines[0][:320]:
            shown.append(lines[0])
    return shown


def test_an_idle_start_refuses_a_line_its_board_cannot_clock_as_run_does(
    nidaq_app, monkeypatch, caplog,
):
    # Run and the DAQ Monitor asked whether a digital line's board could
    # clock it; the stream's own start in Idle asked only whether the line
    # existed, reached the preflight, and failed there at -200452, naming
    # neither the line nor the board.
    from hardware_status_content_test import _patch_hardware_scans
    from tools.acquisition.model import app_model as app_model_module

    discover = _discover_dev1_lines(None)
    monitor = nidaq_app.nidaq_signal_monitor
    monitor._device_discovery = discover
    preflights = []
    monitor._exact_preflight = _counting_preflight(preflights, valid=False)
    starts = _count_starts(monkeypatch, monitor)
    _patch_hardware_scans(nidaq_app, monkeypatch)
    monkeypatch.setattr(app_model_module, "discover_nidaq_devices", discover)

    with caplog.at_level("ERROR"):
        assert nidaq_app.load_configuration() is True
        _settle(nidaq_app, running=False)
    idle = _nidaq_state(nidaq_app)

    assert starts == [True]
    assert preflights == [] and monitor._process is None
    assert not monitor.is_running
    assert idle.state is SubsystemState.FAILED
    assert "'Dev1/port0/line0'" in idle.error
    assert "Dev1, which cannot clock digital input" in idle.error
    assert len(_status_bar_lines(caplog, "cannot clock digital input")) == 1

    def refresh_is_refused_again(attempts):
        # A hardware refresh is how a failed start is retried; it is refused
        # again, by name, before the preflight, and says so once.
        caplog.clear()
        with caplog.at_level("ERROR"):
            nidaq_app.refresh_hardware_bindings()
            _settle(nidaq_app, running=False)
        assert len(starts) == attempts
        assert preflights == [] and monitor._process is None
        retried = _nidaq_state(nidaq_app)
        assert retried.state is SubsystemState.FAILED
        assert idle.error in retried.error
        assert len(_status_bar_lines(caplog, "cannot clock digital input")) == 1

    refresh_is_refused_again(2)

    # Run gives the same refusal the same state, from the same check.
    assert nidaq_app._start_nidaq_domain(restart=True) is False
    run = _nidaq_state(nidaq_app)
    assert run.state is idle.state
    assert idle.error in run.error
    refresh_is_refused_again(3)


def _refused_idle_start(nidaq_app, monkeypatch):
    """Load a plan whose Dev1 line cannot be clocked; the Idle start refuses it."""
    from hardware_status_content_test import _patch_hardware_scans
    from tools.acquisition.model import app_model as app_model_module

    discover = _discover_dev1_lines(None)
    monitor = nidaq_app.nidaq_signal_monitor
    monitor._device_discovery = discover
    preflights = []
    monitor._exact_preflight = _counting_preflight(preflights, valid=False)
    _patch_hardware_scans(nidaq_app, monkeypatch)
    monkeypatch.setattr(app_model_module, "discover_nidaq_devices", discover)
    assert nidaq_app.load_configuration() is True
    _settle(nidaq_app, running=False)
    refused = _nidaq_state(nidaq_app)
    assert refused.state is SubsystemState.FAILED
    return refused.error


def _refused_again(nidaq_app, reason):
    status = _nidaq_state(nidaq_app)
    assert status.state is SubsystemState.FAILED, status
    assert reason in status.error


def test_a_repeated_idle_refusal_is_said_each_time(nidaq_app, monkeypatch):
    # The refusal's text is the same each time, and the monitor announced
    # its error only when it changed: after a Run and a Stop, a Hardware
    # refresh, or a hardware settings save, NI-DAQ read "stopped" or
    # "configured; not started" while every start was refused.
    reason = _refused_idle_start(nidaq_app, monkeypatch)

    assert nidaq_app.capture_start() is True
    nidaq_app.capture_stop()
    _settle(nidaq_app, running=False)
    _refused_again(nidaq_app, reason)

    # Something else first, so what follows is the refresh's own writing.
    nidaq_app._set_subsystem_status(
        SubsystemId.NIDAQ_STREAM, SubsystemState.STOPPED, reason="before the refresh")
    nidaq_app.refresh_hardware_bindings()
    _settle(nidaq_app, running=False)
    _refused_again(nidaq_app, reason)

    nidaq_app._set_subsystem_status(
        SubsystemId.NIDAQ_STREAM, SubsystemState.STOPPED, reason="before the save")
    hardware = nidaq_app.loaded_configuration.hardware
    nidaq_app.update_hardware_configuration(
        can_enabled=hardware.can_enabled,
        pellet_controller_enabled=hardware.pellet_controller_enabled,
        nidaq_enabled=hardware.nidaq_enabled,
        rfid_reader_enabled=hardware.rfid_reader_enabled,
        rfid_device=hardware.rfid_device,
    )
    _settle(nidaq_app, running=False)
    _refused_again(nidaq_app, reason)


def test_run_still_lists_a_laser_route_problem_after_an_idle_refusal(
    app_model, system_config, trainer_config_dir, monkeypatch,
):
    # The Idle refusal left an invalid plan with no clock on the monitor, and
    # Run's check read the clock's board off it: the laser's missing route
    # went unlisted beside the refused line. The board that clock is on is
    # the plan's own master: here cam_frames' board, Dev2, which the plan
    # prefers to Dev1, the first analog input's. Guessed "first analog", the
    # laser on Dev1 looked clocked from its own board, and went unlisted.
    from autotrainer.core import NidaqSignalChannelConfiguration, NidaqSignalStreamConfiguration
    from tools.acquisition.model import app_model as app_model_module
    from tools.acquisition.model.nidaq_routing import UNIDENTIFIED

    def board(name, **values):
        return NidaqDevicePorts(
            name=name, bus_type="PXI", pxi_chassis_number=UNIDENTIFIED,
            pxi_slot_number=UNIDENTIFIED, counter_outputs=(f"{name}/ctr0",),
            digital_inputs=(f"{name}/port0/line0",), **values)

    devices = (
        board("Dev1", analog_inputs=("Dev1/ai0",), analog_outputs=("Dev1/ao0",),
              digital_input_max_rate=1_000_000.0),
        board("Dev2", analog_outputs=("Dev2/ao0",)),
    )
    discover = lambda: (devices, None)  # noqa: E731
    system_config.hardware.nidaq_enabled = True
    system_config.nidaq_ports = NidaqPortConfiguration(cam_frames="Dev2/port0/line0")
    system_config.nidaq_stream = NidaqSignalStreamConfiguration(
        channels=(NidaqSignalChannelConfiguration("stim_readback", "Dev1/ai0"),),
        is_enabled=True)
    system_config.laser = LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(
            channel_id=LaserChannelId.LASER_1, analog_output="Dev1/ao0",
            diode_input="Dev1/ai0", shutter_output="Dev1/port0/line0"),),
        hardware_timed=True, sample_rate_hz=10_000.0)
    system_config.save_default(trainer_config_dir)
    monitor = _with_fake_worker(app_model)
    monitor._device_discovery = discover
    monkeypatch.setattr(app_model_module, "discover_nidaq_devices", discover)
    try:
        assert app_model.load_configuration() is True
        _settle(app_model, running=False)
        plan = monitor.timing_plan
        assert plan is not None and not plan.is_valid
        assert plan.master_device == "Dev2"

        assert app_model._start_nidaq_domain(restart=True) is False

        error = _nidaq_state(app_model).error
        assert "Dev2, which cannot clock digital input" in error
        assert "laser channel 1 output" in error and "is clocked from Dev2" in error
        assert "no explicit route" in error
    finally:
        _close_stream(app_model, monitor)


def test_an_idle_start_on_a_board_that_clocks_digital_input_is_unchanged(nidaq_app):
    monitor = nidaq_app.nidaq_signal_monitor
    monitor._device_discovery = _discover_dev1_lines(1_000_000.0)
    preflights = []
    monitor._exact_preflight = _counting_preflight(preflights, valid=True)

    assert nidaq_app.load_configuration() is True

    assert _settle(nidaq_app).is_running
    assert preflights == [True]
    assert _nidaq_state(nidaq_app).state is SubsystemState.READY


@pytest.mark.parametrize("discovery", ["a board", "no board"])
def test_an_idle_start_refuses_an_export_line_the_laser_takes_as_run_does(
    nidaq_app, system_config, trainer_config_dir, monkeypatch, discovery,
):
    # Run refused an export line on the laser's backplaneClockLine; the
    # stream's own start in Idle never asked, and drove the line the laser
    # drives its clock onto. The check needs no board, so a start that finds
    # none refuses it too, where Run's check of the boards is skipped.
    from tools.acquisition.model import app_model as app_model_module

    system_config.nidaq_ports = NidaqPortConfiguration(
        cam_frames="Dev1/port0/line0",
        timing=NidaqTimingConfiguration(sample_clock_export_terminal="PXI_Trig1"))
    system_config.save_default(trainer_config_dir)
    found = (nidaq_stream_fakes.discover_dev1 if discovery == "a board"
             else (lambda: ((), "NI-DAQmx reports no devices")))
    monitor = nidaq_app.nidaq_signal_monitor
    monitor._device_discovery = found
    monkeypatch.setattr(app_model_module, "discover_nidaq_devices", found)

    assert nidaq_app.load_configuration() is True
    _settle(nidaq_app, running=False)

    idle = _nidaq_state(nidaq_app)
    assert not monitor.is_running and monitor._process is None
    assert idle.state is SubsystemState.FAILED, idle
    assert "timing sampleClockExportTerminal is 'PXI_Trig1'" in idle.error
    assert "the laser backplaneClockLine" in idle.error

    assert nidaq_app._start_nidaq_domain(restart=True) is False
    run = _nidaq_state(nidaq_app)
    assert run.state is SubsystemState.FAILED
    assert "the laser backplaneClockLine" in run.error
    assert monitor._process is None


def test_the_daq_monitor_pauses_the_stream_and_resumes_it_when_closed(
    nidaq_app, monkeypatch,
):
    monkeypatch.setattr(nidaq_monitor_session, "NidaqSignalMonitorModel", _Stream)
    monkeypatch.setattr(nidaq_monitor_session, "discover_nidaq_devices",
                        lambda: ((), None))
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"

    session = NidaqMonitorSession(nidaq_app)
    try:
        assert not monitor.is_running
        assert "DAQ Monitor" in monitor.status_message
        # Nothing restarts it behind the monitor's back, a refresh included.
        nidaq_app._request_nidaq_stream("hardware refresh")
        _settle(nidaq_app, running=False, timeout=1.0)
        assert not monitor.is_running
        # System Mode is refused the stream too, rather than both reserving
        # the same lines.
        assert nidaq_app._start_nidaq_domain() is False
        blocked = _nidaq_state(nidaq_app)
        assert blocked.state is SubsystemState.BLOCKED
        assert "DAQ Monitor" in blocked.reason
    finally:
        session.close()

    assert _settle(nidaq_app).is_running
    session.close()  # a second close is harmless
    assert monitor.is_running


def test_the_stream_is_not_restarted_while_the_application_closes(nidaq_app):
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    monitor.stop()

    nidaq_app._prepare_application_shutdown()
    nidaq_app._request_nidaq_stream("acquisition stopped")

    assert not _settle(nidaq_app, running=True, timeout=1.0).is_running


def _launched_chunks(monkeypatch):
    """Record the read chunk each start hands its worker."""
    chunks = []
    resolve = nidaq_signal_monitor_model.resolve_nidaq_stream_configuration

    def recording(*args, **kwargs):
        configuration = resolve(*args, **kwargs)
        chunks.append(configuration.read_chunk_size)
        return configuration

    monkeypatch.setattr(nidaq_signal_monitor_model,
                        "resolve_nidaq_stream_configuration", recording)
    return chunks


def test_a_refresh_rate_change_leaves_a_running_session_stream_alone(
    nidaq_app, monkeypatch,
):
    # The Analysis view forwards the window's screen refresh rate, polled
    # once a second, and the read chunk follows it. Moving the window to a
    # screen with another rate while recording stopped the stream and started
    # a new one: the subsystem read STOPPED, the shared ring was reset and
    # the samples took a new timing anchor, so the recording carried on with
    # a silent gap. The new chunk now waits for the stream's next start.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    chunks = _launched_chunks(monkeypatch)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    try:
        assert nidaq_app.capture_start() is True
        assert monitor.is_running
        pid = _worker_pid(monitor)
        run_chunk = chunks[-1]
        before = _nidaq_state(nidaq_app)
        calls = []
        stop, start = monitor.stop, monitor.start
        monitor.stop = lambda: calls.append("stop") or stop()
        monitor.start = lambda: calls.append("start") or start()

        monitor.set_display_refresh_rate(2 * monitor.display_refresh_rate_hz)

        new_chunk = monitor.effective_read_chunk_size
        assert new_chunk != run_chunk
        assert calls == []
        assert monitor.is_running
        assert _worker_pid(monitor) == pid
        after = _nidaq_state(nidaq_app)
        assert (after.state, after.generation) == (before.state, before.generation)
    finally:
        nidaq_app.capture_stop()

    # Stop hands the stream back to Idle, and that start takes the new chunk.
    assert _settle(nidaq_app).is_running
    assert _worker_pid(monitor) not in (None, pid)
    assert chunks[-1] == new_chunk


def test_a_refresh_rate_change_in_idle_restarts_the_stream_with_its_chunk(
    nidaq_app, monkeypatch,
):
    chunks = _launched_chunks(monkeypatch)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    pid = _worker_pid(monitor)
    idle_chunk = chunks[-1]

    monitor.set_display_refresh_rate(2 * monitor.display_refresh_rate_hz)

    assert _settle(nidaq_app).is_running
    assert _worker_pid(monitor) not in (None, pid)
    assert monitor.effective_read_chunk_size != idle_chunk
    assert chunks[-1] == monitor.effective_read_chunk_size


@pytest.fixture
def independent_app(app_model, system_config, trainer_config_dir, monkeypatch):
    """Two boards on their own clocks, so the timing plan is "independent"."""
    system_config.hardware.nidaq_enabled = True
    system_config.nidaq_ports = NidaqPortConfiguration(
        cam_frames="Dev1/port0/line0",
        tone1="Dev2/port0/line0",
        timing=NidaqTimingConfiguration(
            sync_mode="independent", require_hardware_synchronization=False),
    )
    system_config.save_default(trainer_config_dir)
    monitor = _with_fake_worker(app_model)
    monitor._device_discovery = nidaq_stream_fakes.discover_dev1_and_dev2
    monkeypatch.setattr(app_model, "_require_valid_nidaq_configuration", lambda: None)
    try:
        yield app_model
    finally:
        rule = vars(app_model).get("_nidaq_stream_autostart")
        if rule is not None:
            rule.close()
        monitor.close()


def test_a_laser_on_independent_boards_does_not_fail_run_over_a_route_it_never_uses(
    app_model, system_config, trainer_config_dir, monkeypatch,
):
    # Independent boards share no clock, and the laser's output runs on its
    # own (NidaqLaserController gives it no shared clock). Fix round 1 gave a
    # valid plan with no shared clock a guessed one, and Run then failed
    # NI-DAQ over a route this mode never uses; it read "blocked", as a
    # diagnostic preview, before that, and does again. Here the validation
    # is the real one: the fixture above stubs it out.
    from tools.acquisition.model import app_model as app_model_module

    system_config.hardware.nidaq_enabled = True
    system_config.nidaq_ports = NidaqPortConfiguration(
        cam_frames="Dev1/port0/line0",
        tone1="Dev2/port0/line0",
        timing=NidaqTimingConfiguration(
            sync_mode="independent", require_hardware_synchronization=False),
    )
    system_config.laser = LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(
            channel_id=LaserChannelId.LASER_1, analog_output="Dev2/ao0",
            diode_input="Dev2/ai0", shutter_output="Dev2/port0/line1"),),
        hardware_timed=True, sample_rate_hz=10_000.0)
    system_config.save_default(trainer_config_dir)

    def board(name, **values):
        # On no shared bus: the boards can share no clock.
        return NidaqDevicePorts(
            name=name, counter_outputs=(f"{name}/ctr0",),
            digital_inputs=(f"{name}/port0/line0",),
            digital_input_max_rate=1_000_000.0, **values)

    devices = (board("Dev1"), board("Dev2", analog_outputs=("Dev2/ao0",)))
    discover = lambda: (devices, None)  # noqa: E731
    monitor = _with_fake_worker(app_model)
    monitor._device_discovery = discover
    monkeypatch.setattr(app_model_module, "discover_nidaq_devices", discover)
    try:
        assert app_model.load_configuration() is True
        assert _settle(app_model).timing_plan.resolved_mode == "independent"

        assert app_model.capture_start() is True

        status = _nidaq_state(app_model)
        assert status.state is SubsystemState.BLOCKED, status
        assert "independent device clocks" in status.reason
    finally:
        app_model.capture_stop()
        rule = vars(app_model).get("_nidaq_stream_autostart")
        if rule is not None:
            rule.close()
        monitor.close()


def test_a_run_on_independent_boards_is_blocked_from_recording(independent_app):
    # _start_nidaq_domain wrote BLOCKED with the generation it began, but the
    # stream's own start event had begun a newer one, so the registry dropped
    # the verdict as stale. The stream read READY and Record was allowed on
    # boards whose samples cannot be aligned.
    assert independent_app.load_configuration() is True
    monitor = _settle(independent_app)
    assert monitor.timing_plan.resolved_mode == "independent"
    try:
        assert independent_app.capture_start() is True

        assert monitor.is_running
        blocked = _nidaq_state(independent_app)
        assert blocked.state is SubsystemState.BLOCKED
        assert "independent device clocks" in blocked.reason
        assert any("independent device clocks" in blocker
                   for blocker in independent_app.recording_blockers)
    finally:
        independent_app.capture_stop()


def _hold_the_ready_announcement(monitor, until):
    """Settle "not starting", then wait for `until()` before telling anyone.

    The monitor settles both of its flags under its lock and then notifies,
    and a Run polls the flags without the lock, so it can act inside that
    moment. Held here, the moment is long every time.
    """
    set_starting = monitor._set_starting
    ready_pids = []

    def announce_late(value):
        if value or not monitor._is_starting:
            return set_starting(value)
        ready_pids.append(_worker_pid(monitor))
        monitor._is_starting = False
        deadline = time.monotonic() + 5.0
        while not until() and time.monotonic() < deadline:
            time.sleep(0.01)
        monitor._on_property_changed(monitor.IS_STARTING, False, True)

    monitor._set_starting = announce_late
    return ready_pids


def test_a_late_ready_from_the_stream_cannot_overwrite_the_runs_verdict(
    independent_app,
):
    # A Run can see the stream running and write its verdict, and finish,
    # before the monitor's READY event reaches the application; that READY
    # replaced BLOCKED.
    assert independent_app.load_configuration() is True
    monitor = _settle(independent_app)

    def the_run_has_finished_with_the_stream():
        return (_nidaq_state(independent_app).state is SubsystemState.BLOCKED
                and not independent_app._nidaq_domain_start_active)

    _hold_the_ready_announcement(
        monitor, until=the_run_has_finished_with_the_stream)
    try:
        assert independent_app.capture_start() is True
        assert _wait_until_running_is_announced(monitor)

        assert monitor.is_running
        assert _nidaq_state(independent_app).state is SubsystemState.BLOCKED
    finally:
        del monitor._set_starting
        independent_app.capture_stop()


def test_a_run_is_not_failed_in_the_moment_its_stream_turns_ready(
    nidaq_app, monkeypatch,
):
    # The monitor cleared "starting", and told its listeners (the views, on
    # the rig), before it set "running". A Run polling in that moment saw
    # neither, wrote "not ready within 12 seconds" and stopped a worker that
    # had just come up.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    first_asked = []

    def a_moment_later():
        first_asked.append(time.monotonic())
        return time.monotonic() - first_asked[0] >= 0.3

    ready_pids = _hold_the_ready_announcement(monitor, until=a_moment_later)
    try:
        assert nidaq_app.capture_start() is True

        state = _nidaq_state(nidaq_app)
        assert state.state is SubsystemState.READY, state
        assert _wait_until_running_is_announced(monitor)
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY
        assert monitor.is_running
        assert len(ready_pids) == 1
        assert _worker_pid(monitor) == ready_pids[0]
    finally:
        del monitor._set_starting
        nidaq_app.capture_stop()


def test_a_stream_that_dies_as_the_run_decides_reads_failed(nidaq_app, monkeypatch):
    # The Run saw the stream running, then wrote READY at its generation. A
    # worker that died between the two had already been written FAILED at
    # that same generation, and READY replaced it: a required subsystem read
    # ready on a dead stream.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    idle_pid = _worker_pid(monitor)
    timing_plan = nidaq_signal_monitor_model.NidaqSignalMonitorModel.timing_plan
    killed = []

    def die_as_the_run_decides(self):
        # The Run reads the plan right after it sees the stream running.
        if (self is monitor and not killed
                and threading.current_thread() is threading.main_thread()
                and self.is_running
                and _worker_pid(self) not in (None, idle_pid)):
            killed.append(_worker_pid(self))
            self._process.terminate()
            deadline = time.monotonic() + 5.0
            while (_nidaq_state(nidaq_app).state is not SubsystemState.FAILED
                   and time.monotonic() < deadline):
                time.sleep(0.02)
        return timing_plan.fget(self)

    monkeypatch.setattr(nidaq_signal_monitor_model.NidaqSignalMonitorModel,
                        "timing_plan", property(die_as_the_run_decides))
    try:
        assert nidaq_app.capture_start() is True

        assert killed, "the Run never read the timing plan"
        failed = _nidaq_state(nidaq_app)
        assert failed.state is SubsystemState.FAILED, failed
        assert "crashed" in failed.error
    finally:
        nidaq_app.capture_stop()


def test_a_crash_between_the_runs_last_two_looks_keeps_its_own_error(
    nidaq_app, monkeypatch,
):
    # After its verdict the Run reads the stream's error, then whether it
    # runs. A dying worker sets its error before it clears running, so a
    # crash between those two reads showed a stopped stream with no error,
    # and the Run wrote a generic FAILED over the crash's own.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    idle_pid = _worker_pid(monitor)
    error_message = nidaq_signal_monitor_model.NidaqSignalMonitorModel.error_message
    crashed = []

    def crash_just_after_this_read(self):
        value = error_message.fget(self)
        if (self is monitor and not crashed
                and threading.current_thread() is threading.main_thread()
                and _worker_pid(self) not in (None, idle_pid)
                and _nidaq_state(nidaq_app).reason
                == "synchronized NI-DAQ tasks running"):
            crashed.append(_worker_pid(self))
            self._process.terminate()
            deadline = time.monotonic() + 5.0
            while ((self._is_running
                    or _nidaq_state(nidaq_app).state is not SubsystemState.FAILED)
                   and time.monotonic() < deadline):
                time.sleep(0.02)
        return value

    monkeypatch.setattr(nidaq_signal_monitor_model.NidaqSignalMonitorModel,
                        "error_message", property(crash_just_after_this_read))
    try:
        assert nidaq_app.capture_start() is True

        assert crashed, "the Run never read the error after its verdict"
        failed = _nidaq_state(nidaq_app)
        assert failed.state is SubsystemState.FAILED, failed
        assert "crashed" in failed.error, failed
    finally:
        nidaq_app.capture_stop()


class _ReadyHeldBack:
    """A worker's message queue that holds back "ready" until released.

    With `dropped` set before the release, the "ready" is discarded instead,
    as if the worker had died before it could send it.
    """

    def __init__(self, message_queue, held, release, dropped):
        self._message_queue = message_queue
        self._held = held
        self._release = release
        self._dropped = dropped

    def get(self, timeout=None):
        message = self._message_queue.get(timeout=timeout)
        if message[0] == "ready" and not self._release.is_set():
            self._held.set()
            self._release.wait(10.0)
            if self._dropped.is_set():
                raise queue.Empty
        return message


def _hold_the_next_workers_ready(monitor):
    """Hold the next worker's "ready"; returns (held, release, dropped)."""
    held, release, dropped = threading.Event(), threading.Event(), threading.Event()
    run = monitor._run
    gated = []

    def run_with_its_ready_held(process, message_queue, stop_event, started):
        if not gated:
            gated.append(process)
            message_queue = _ReadyHeldBack(message_queue, held, release, dropped)
        return run(process, message_queue, stop_event, started)

    monitor._run = run_with_its_ready_held
    return held, release, dropped


def _nidaq_statuses_published(app_model, on_status=None):
    """Each NIDAQ state and reason the application publishes from now on."""
    seen = []

    def record(name, value, _previous):
        if name != app_model.Props.SUBSYSTEM_STATUSES:
            return
        status = value.get(SubsystemId.NIDAQ_STREAM.value)
        if status is None:
            return
        entry = (status.state, status.reason)
        if not seen or seen[-1] != entry:
            seen.append(entry)
        if on_status is not None:
            on_status(status)

    app_model.property_changed += record
    return seen


def test_an_idle_ready_inside_a_runs_start_does_not_stand_for_it(
    nidaq_app, monkeypatch,
):
    # An Idle start that began before the Run and turned ready after the
    # Run's NI-DAQ start had begun wrote READY, and it stood while the Run
    # stopped that worker and started a fresh one through discovery and
    # the preflight.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    monitor = nidaq_app.nidaq_signal_monitor
    monitor._startup_timeout_seconds = 30.0
    held, release, _dropped = _hold_the_next_workers_ready(monitor)

    def let_the_idle_ready_in(status):
        # Once the Run's NI-DAQ start has begun, and before it restarts.
        if (status.reason == "starting synchronized NI-DAQ tasks"
                and not release.is_set()):
            release.set()
            _wait_until_running_is_announced(monitor, timeout=5.0)

    try:
        assert nidaq_app.load_configuration() is True
        assert held.wait(10.0), "the stream did not start by itself"
        seen = _nidaq_statuses_published(nidaq_app, let_the_idle_ready_in)

        assert nidaq_app.capture_start() is True

        begun = seen.index(
            (SubsystemState.STARTING, "starting synchronized NI-DAQ tasks"))
        during = seen[begun:]
        assert during[-1] == (
            SubsystemState.READY, "synchronized NI-DAQ tasks running"), during
        assert SubsystemState.READY not in [s for s, _ in during[:-1]], during
        assert during[-2][0] is SubsystemState.STARTING, during
    finally:
        release.set()
        del monitor._run
        nidaq_app.capture_stop()


def test_an_idle_failure_inside_a_runs_start_gives_way_to_starting(
    nidaq_app, monkeypatch,
):
    # An Idle start that failed after the Run's NI-DAQ start had begun wrote
    # FAILED, and without the Run's own start saying "starting" again that
    # FAILED stood while System Mode started a fresh worker, until its verdict.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    monitor = nidaq_app.nidaq_signal_monitor
    monitor._startup_timeout_seconds = 30.0
    held, release, dropped = _hold_the_next_workers_ready(monitor)

    def fail_the_idle_start(status):
        # Once the Run's NI-DAQ start has begun: the Idle worker dies
        # instead of turning ready.
        if (status.reason == "starting synchronized NI-DAQ tasks"
                and not release.is_set()):
            monitor._process.terminate()
            dropped.set()
            release.set()
            deadline = time.monotonic() + 5.0
            while (_nidaq_state(nidaq_app).state is not SubsystemState.FAILED
                   and time.monotonic() < deadline):
                time.sleep(0.01)

    try:
        assert nidaq_app.load_configuration() is True
        assert held.wait(10.0), "the stream did not start by itself"
        seen = _nidaq_statuses_published(nidaq_app, fail_the_idle_start)

        assert nidaq_app.capture_start() is True

        begun = seen.index(
            (SubsystemState.STARTING, "starting synchronized NI-DAQ tasks"))
        assert seen[begun:] == [
            (SubsystemState.STARTING, "starting synchronized NI-DAQ tasks"),
            (SubsystemState.FAILED, ""),
            (SubsystemState.STARTING, "starting synchronized NI-DAQ tasks"),
            (SubsystemState.READY, "synchronized NI-DAQ tasks running"),
        ], seen[begun:]
        assert monitor.is_running
    finally:
        release.set()
        del monitor._run
        nidaq_app.capture_stop()


def test_a_run_whose_stream_never_becomes_ready_reads_failed(nidaq_app, monkeypatch):
    # The domain gave up and wrote FAILED with the generation it began; the
    # stream's start event had begun a newer one, so the FAILED was dropped
    # and the stream read "starting" with no error, after it had been stopped.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    # The next worker hangs, and the domain's bound is the shorter one.
    monitor._worker_target = nidaq_stream_fakes.silent_worker
    monitor._startup_timeout_seconds = 30.0

    assert nidaq_app._start_nidaq_domain(restart=True, timeout=1.0) is False

    assert not (monitor.is_running or monitor.is_starting)
    failed = _nidaq_state(nidaq_app)
    assert failed.state is SubsystemState.FAILED
    assert "not ready within 1 seconds" in failed.error


def test_a_stream_that_fails_under_a_run_still_reads_failed(nidaq_app, monkeypatch):
    # The Run's verdict is its own; a failure after it is the stream's.
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    try:
        assert nidaq_app.capture_start() is True
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY

        monitor._process.terminate()
        deadline = time.monotonic() + 5.0
        while (_nidaq_state(nidaq_app).state is not SubsystemState.FAILED
               and time.monotonic() < deadline):
            time.sleep(0.02)

        failed = _nidaq_state(nidaq_app)
        assert failed.state is SubsystemState.FAILED
        assert "crashed" in failed.error
    finally:
        nidaq_app.capture_stop()


# ------------------------------------------------ a bad NI line in the file


def _null_laser_configuration():
    return LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="Dev1/ao0",
                diode_input="Dev1/ai0",
                shutter_output="Dev1/port0/line2",
                command_copy_input="Dev1/ai1",
            ),
        ),
        backend="null",
    )


def _with_bad_nidaq_line(system_config, trainer_config_dir, **ports):
    from autotrainer.core import (
        NidaqSignalChannelConfiguration,
        NidaqSignalStreamConfiguration,
    )
    system_config.hardware.nidaq_enabled = True
    system_config.laser = _null_laser_configuration()
    system_config.nidaq_ports = NidaqPortConfiguration(
        cam_frames="Dev1/port0/line0", **ports)
    # An operator's own input, which nothing about the bad line should lose.
    system_config.nidaq_stream = NidaqSignalStreamConfiguration(
        channels=(NidaqSignalChannelConfiguration("stim_readback", "Dev1/ai5"),),
        is_enabled=True,
    )
    system_config.save_default(trainer_config_dir)


def _count_starts(monkeypatch, monitor):
    starts = []
    original = monitor.start

    def counting_start():
        starts.append(True)
        return original()

    monkeypatch.setattr(monitor, "start", counting_start)
    return starts


@pytest.mark.parametrize(
    ("ports", "reason"),
    [
        (dict(tone1="Dev1/port1/line0"), "tone1"),
        (dict(tone1="Dev1/port0/line0"), "assigned to both"),
    ],
    ids=["pfi_pin_tone", "duplicate_pin"],
)
def test_a_bad_ni_line_loads_everything_else_and_blocks_the_stream(
    app_model, system_config, trainer_config_dir, monkeypatch, caplog, ports, reason,
):
    # The plan was built after the lasers were applied and before the
    # configuration was kept, so a refused line left the load half done:
    # Edit DAQ Ports would not open, and the file had to be fixed by hand.
    _with_bad_nidaq_line(system_config, trainer_config_dir, **ports)
    monitor = _with_fake_worker(app_model)
    starts = _count_starts(monkeypatch, monitor)
    try:
        with caplog.at_level("ERROR"):
            assert app_model.load_configuration() is True
        _settle(app_model, running=False, timeout=2.0)

        assert app_model.loaded_configuration is not None
        assert app_model.laser.configuration == _null_laser_configuration()
        assert app_model.nidaq_ports.tone1 == ports["tone1"]
        status = _nidaq_state(app_model)
        assert status.state is SubsystemState.BLOCKED
        assert reason in status.reason
        assert starts == [] and not monitor.is_running
        refusals = [record.getMessage() for record in caplog.records
                    if "Edit DAQ Ports" in record.getMessage()]
        assert len(refusals) == 1 and reason in refusals[0]
        # A hardware refresh or Run does not start it either.
        app_model._request_nidaq_stream("test")
        _settle(app_model, running=False, timeout=2.0)
        assert starts == []
        assert app_model._start_nidaq_domain(restart=True) is False
        assert _nidaq_state(app_model).state is SubsystemState.BLOCKED
    finally:
        rule = vars(app_model).get("_nidaq_stream_autostart")
        if rule is not None:
            rule.close()
        monitor.close()


def test_an_exit_save_with_a_bad_ni_line_keeps_the_operators_values(
    app_model, system_config, trainer_config_dir,
):
    _with_bad_nidaq_line(system_config, trainer_config_dir, tone1="Dev1/port1/line0")
    monitor = _with_fake_worker(app_model)
    try:
        assert app_model.load_configuration() is True

        app_model.save_configuration()

        saved = app_model.get_config_from_location(app_model.get_config_location())
        assert saved.nidaq_ports.tone1 == "Dev1/port1/line0"
        assert saved.nidaq_ports.cam_frames == "Dev1/port0/line0"
        assert "stim_readback" in {channel.name for channel in saved.nidaq_stream.channels}
        assert saved.laser == _null_laser_configuration()
    finally:
        vars(app_model)["_nidaq_stream_autostart"].close()
        monitor.close()


def test_christielab10s_configuration_still_loads_exactly(
    app_model, system_config, trainer_config_dir,
):
    from nidaq_channel_plan_test import (
        CHRISTIELAB10_PORTS,
        _christielab10_lasers,
        _christielab10_stream,
    )
    system_config.hardware.nidaq_enabled = True
    system_config.nidaq_ports = CHRISTIELAB10_PORTS
    system_config.nidaq_stream = _christielab10_stream()
    system_config.laser = _christielab10_lasers()
    system_config.save_default(trainer_config_dir)
    monitor = _with_fake_worker(app_model)
    try:
        assert app_model.load_configuration() is True
        rule = vars(app_model)["_nidaq_stream_autostart"]
        assert rule.wait(10.0)

        assert monitor.configuration.channels == _christielab10_stream().channels
        assert monitor.configuration.display_channels == (
            _christielab10_stream().display_channels)
        assert app_model._nidaq_plan_error == ""
    finally:
        vars(app_model)["_nidaq_stream_autostart"].close()
        monitor.close()


def _load_with_a_bad_ni_line(app_model, system_config, trainer_config_dir, monkeypatch):
    """Load a PFI-pin tone1, with a stand-in worker and no board check."""
    _with_bad_nidaq_line(system_config, trainer_config_dir, tone1="Dev1/port1/line0")
    monitor = _with_fake_worker(app_model)
    monkeypatch.setattr(app_model, "_require_valid_nidaq_configuration", lambda: None)
    assert app_model.load_configuration() is True
    _settle(app_model, running=False, timeout=2.0)
    assert _nidaq_state(app_model).state is SubsystemState.BLOCKED
    return monitor


def _close_stream(app_model, monitor):
    rule = vars(app_model).get("_nidaq_stream_autostart")
    if rule is not None:
        rule.close()
    monitor.close()


def test_run_then_stop_keeps_a_bad_ni_line_blocked_with_its_reason(
    app_model, system_config, trainer_config_dir, monkeypatch,
):
    # Stop wrote DISABLED "NI-DAQ stream stopped" whenever the stream had no
    # channels, which it never has while a plan is refused: after one Run the
    # Hardware panel said NI-DAQ was disabled, and the reason was gone.
    monitor = _load_with_a_bad_ni_line(
        app_model, system_config, trainer_config_dir, monkeypatch)
    try:
        assert app_model.capture_start() is True
        assert _nidaq_state(app_model).state is SubsystemState.BLOCKED
        app_model.capture_stop()

        status = _nidaq_state(app_model)
        assert status.state is SubsystemState.BLOCKED
        assert "nidaqPorts.tone1 'Dev1/port1/line0'" in status.reason
        assert "Edit DAQ Ports" in status.reason
        assert status.required_for_recording
    finally:
        _close_stream(app_model, monitor)


def test_record_is_refused_while_a_bad_ni_line_is_held(
    app_model, system_config, trainer_config_dir, monkeypatch,
):
    from types import SimpleNamespace
    from unittest import mock

    monitor = _load_with_a_bad_ni_line(
        app_model, system_config, trainer_config_dir, monkeypatch)
    try:
        assert app_model.capture_start() is True
        assert any("nidaqPorts.tone1" in blocker
                   for blocker in app_model.recording_blockers)
        # Past the subject and camera checks, to the streams Record needs.
        app_model._selected_animal = object()
        monkeypatch.setattr(
            app_model, "_get_monitored_cams",
            lambda: (SimpleNamespace(is_recording_enabled=True),))
        with mock.patch.object(app_model, "on_error") as on_error:
            assert app_model.start_recording() is False
        title, message = on_error.call_args.args
        assert title == "Recording unavailable"
        assert "nidaqPorts.tone1 'Dev1/port1/line0'" in message
    finally:
        app_model._selected_animal = None
        app_model.capture_stop()
        _close_stream(app_model, monitor)


def test_a_good_second_load_clears_a_bad_ni_line(
    app_model, system_config, trainer_config_dir, monkeypatch,
):
    monitor = _load_with_a_bad_ni_line(
        app_model, system_config, trainer_config_dir, monkeypatch)
    try:
        system_config.nidaq_ports = NidaqPortConfiguration(
            cam_frames="Dev1/port0/line0", tone1="Dev1/port0/line3")
        system_config.save_default(trainer_config_dir)
        assert app_model.load_configuration() is True
        monitor = _settle(app_model)

        assert app_model.nidaq_plan_error == ""
        assert monitor.is_running
        assert _nidaq_state(app_model).state is SubsystemState.READY
        names = {channel.name for channel in monitor.configuration.channels}
        assert {"cam_frames", "tone1", "stim_readback"} <= names
    finally:
        _close_stream(app_model, monitor)


def test_a_bad_ni_line_is_what_the_stream_status_says(
    app_model, system_config, trainer_config_dir, monkeypatch,
):
    # The laser tabs and Analysis show the monitor's status message, which
    # said only that the stream was disabled.
    monitor = _load_with_a_bad_ni_line(
        app_model, system_config, trainer_config_dir, monkeypatch)
    try:
        assert "nidaqPorts.tone1 'Dev1/port1/line0'" in monitor.status_message
        assert monitor.status_message == _nidaq_state(app_model).reason
        assert monitor.status_message.startswith("NI-DAQ plan refused")
    finally:
        _close_stream(app_model, monitor)


def test_the_bad_ni_line_log_says_the_remedy_before_the_refusals(
    app_model, system_config, trainer_config_dir, caplog,
):
    # The status bar shows an error's first 320 characters. Three refusals
    # ran past that, and the remedy after them was cut off.
    _with_bad_nidaq_line(
        system_config, trainer_config_dir,
        tone1="Dev1/port1/line0", tone2="Dev1/port2/line1",
        tone3_r="Dev1/port1/line5")
    monitor = _with_fake_worker(app_model)
    try:
        with caplog.at_level("ERROR"):
            assert app_model.load_configuration() is True
        refusals = [record.getMessage() for record in caplog.records
                    if "Edit DAQ Ports" in record.getMessage()]
        assert len(refusals) == 1
        assert len(refusals[0]) > 320
        assert "Edit DAQ Ports" in refusals[0][:320]
        assert "nidaqPorts.tone1" in refusals[0][:320]
        reason = _nidaq_state(app_model).reason
        assert reason.index("Edit DAQ Ports") < reason.index("nidaqPorts.tone1")
    finally:
        _close_stream(app_model, monitor)


def test_the_fix_derives_the_stream_state_and_requirement_from_the_new_plan(
    app_model, system_config, trainer_config_dir,
):
    # The fix wrote STOPPED "configured; not started" and kept the refused
    # plan's requirement, whatever the fixed plan held.
    from autotrainer.core import NidaqSignalStreamConfiguration

    # A bad line, and no laser or input of the operator's own.
    system_config.hardware.nidaq_enabled = True
    system_config.laser = LaserSystemConfiguration()
    system_config.nidaq_ports = NidaqPortConfiguration(tone1="Dev1/port1/line0")
    system_config.nidaq_stream = NidaqSignalStreamConfiguration()
    system_config.save_default(trainer_config_dir)
    monitor = _with_fake_worker(app_model)
    try:
        assert app_model.load_configuration() is True
        status = _nidaq_state(app_model)
        assert status.state is SubsystemState.BLOCKED and status.required_for_recording

        # Fixed by clearing the line: nothing is left to acquire.
        app_model.update_daq_port_configuration(
            NidaqPortConfiguration(), LaserSystemConfiguration())
        _settle(app_model, running=False, timeout=2.0)

        status = _nidaq_state(app_model)
        assert app_model.nidaq_plan_error == ""
        assert not monitor.configuration.is_enabled
        assert status.state is SubsystemState.DISABLED
        assert not status.required_for_recording
    finally:
        _close_stream(app_model, monitor)


def test_headless_exits_1_naming_a_bad_ni_line(
    app_model, system_config, trainer_config_dir, config_file_path, tmp_path,
    monkeypatch, caplog,
):
    # Headless has no Edit DAQ Ports to fix the line in, so it refuses to run
    # on it, as every load did before the GUI learned to load and block.
    from types import SimpleNamespace

    import autotrainer.core.event as core_event
    import autotrainer.core.logging as core_logging
    from tools.acquisition import headless
    from tools.acquisition.model import app_model as app_model_module
    from tools.acquisition.model.app_model_status import AppModelStatus

    _with_bad_nidaq_line(system_config, trainer_config_dir, tone1="Dev1/port1/line0")
    monitor = _with_fake_worker(app_model)
    starts, closes = [], []
    monkeypatch.setattr(app_model_module, "AppModel", lambda _preferences: app_model)
    monkeypatch.setattr(core_event, "try_register_api_event_plugin",
                        lambda: SimpleNamespace(service=None))
    monkeypatch.setattr(core_logging, "get_console_handler",
                        lambda: SimpleNamespace(setLevel=lambda _level: None))
    monkeypatch.setattr(app_model, "capture_start",
                        lambda **kwargs: starts.append(kwargs) or True)
    monkeypatch.setattr(app_model, "on_close", lambda: closes.append(True))
    args = SimpleNamespace(
        configuration=str(config_file_path),
        preferences_file=tmp_path / "settings.ini",
        random_cameras=False,
        live_inference=None,
        start_mode=AppModelStatus.RUNNING,
    )
    try:
        with caplog.at_level("ERROR"):
            assert headless._exec_main(args) == 1

        assert starts == [] and closes == [True]
        assert any(
            "nidaqPorts.tone1 'Dev1/port1/line0'" in record.getMessage()
            and record.name == "autotrainer.headless"
            for record in caplog.records
        )
        # Headless has no Edit DAQ Ports: its load's own ERROR line said to
        # fix the line there. It names the file, and the refusal the field.
        messages = [record.getMessage() for record in caplog.records]
        assert not any("Edit DAQ Ports" in message for message in messages)
        load_error, = [message for message in messages
                       if message.startswith("NI-DAQ inputs are not acquired")]
        assert str(config_file_path) in load_error
        assert "nidaqPorts.tone1 'Dev1/port1/line0'" in load_error
    finally:
        _close_stream(app_model, monitor)


def test_releasing_a_hold_keeps_saying_why_a_bad_ni_line_blocks_the_stream(
    app_model, system_config, trainer_config_dir, monkeypatch,
):
    # Opening and closing the DAQ Monitor, or a refused DAQ ports save, pauses
    # and resumes the stream. The resume said "NI-DAQ signal stream
    # stopped", and the laser tabs and Analysis lost the plan's reason.
    monkeypatch.setattr(nidaq_monitor_session, "NidaqSignalMonitorModel", _Stream)
    monkeypatch.setattr(nidaq_monitor_session, "discover_nidaq_devices",
                        lambda: ((), None))
    monitor = _load_with_a_bad_ni_line(
        app_model, system_config, trainer_config_dir, monkeypatch)
    reason = _nidaq_state(app_model).reason
    try:
        session = NidaqMonitorSession(app_model)
        try:
            assert "DAQ Monitor" in monitor.status_message
        finally:
            session.close()

        assert monitor.status_message == reason
        assert _nidaq_state(app_model).state is SubsystemState.BLOCKED
        assert _nidaq_state(app_model).reason == reason

        # A DAQ ports save that its own plan refuses holds and releases too.
        with pytest.raises(ValueError):
            app_model.update_daq_port_configuration(
                NidaqPortConfiguration(
                    cam_frames="Dev1/port0/line0", tone1="Dev1/port2/line3"),
                _null_laser_configuration())
        assert monitor.status_message == reason
        assert _nidaq_state(app_model).state is SubsystemState.BLOCKED
    finally:
        _close_stream(app_model, monitor)
