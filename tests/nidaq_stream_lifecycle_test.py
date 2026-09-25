"""The shared NI-DAQ input stream runs by itself while reachAQ is idle.

It used to start only from a button, or with System Mode. These drive the
application model with the real stream model and a stand-in worker process,
so each start is a real one: a worker launched, a ready message received, a
process id to compare.
"""

import dataclasses
import threading
import time

import pytest

from autotrainer.core import NidaqPortConfiguration
from tools.acquisition.model import nidaq_monitor_session
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


def _settle(app_model, *, running=True, timeout=10.0):
    """Wait for any automatic start, then for the stream to reach `running`."""
    # vars(), not getattr: AppModel answers a missing attribute with an
    # EventsException rather than an AttributeError.
    rule = vars(app_model).get("_nidaq_stream_autostart")
    if rule is not None:
        assert rule.wait(timeout)
    monitor = app_model.nidaq_signal_monitor
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not monitor.is_starting and monitor.is_running == running:
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

        with pytest.raises(RuntimeError, match="System Mode"):
            NidaqMonitorSession(nidaq_app)

        assert monitor.is_running
        assert _worker_pid(monitor) == pid
        assert nidaq_app._nidaq_stream_autostart.pause_reasons == ()
        assert _nidaq_state(nidaq_app).state is SubsystemState.READY
    finally:
        nidaq_app.capture_stop()


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


def test_a_hardware_refresh_starts_a_stopped_stream(nidaq_app, monkeypatch):
    from hardware_status_content_test import _patch_hardware_scans

    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    monitor.stop()
    _patch_hardware_scans(nidaq_app, monkeypatch)

    nidaq_app.refresh_hardware_bindings()

    assert _settle(nidaq_app).is_running


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
