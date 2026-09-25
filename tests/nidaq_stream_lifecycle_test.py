"""The shared NI-DAQ input stream runs by itself while reachAQ is idle.

It used to start only from a button, or with System Mode. These drive the
application model with the real stream model and a stand-in worker process,
so each start is a real one: a worker launched, a ready message received, a
process id to compare.
"""

import dataclasses
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


def test_acquisition_start_keeps_the_running_stream(nidaq_app, monkeypatch):
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    pid = _worker_pid(monitor)
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    # A plot selection is not an acquisition change, and must not restart it.
    nidaq_app.update_nidaq_signal_stream_channels(())

    assert nidaq_app._start_nidaq_domain() is True

    assert _worker_pid(monitor) == pid
    assert _nidaq_state(nidaq_app).state is SubsystemState.READY


def test_acquisition_start_restarts_a_stream_whose_plan_changed(nidaq_app, monkeypatch):
    assert nidaq_app.load_configuration() is True
    monitor = _settle(nidaq_app)
    assert monitor.is_running, "the stream did not start by itself"
    pid = _worker_pid(monitor)
    monkeypatch.setattr(nidaq_app, "_require_valid_nidaq_configuration", lambda: None)
    # Nothing in the application changes the plan under a running stream
    # today; this stands in for a path that one day might.
    monitor._configuration = dataclasses.replace(
        monitor.configuration, sample_rate_hz=monitor.configuration.sample_rate_hz / 2)

    assert nidaq_app._start_nidaq_domain() is True

    assert monitor.is_running
    assert _worker_pid(monitor) != pid


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
