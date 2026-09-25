"""The laser calibration ramp runs in Idle, on the NI-DAQ lines it holds.

Run Ramp could never be pressed: it needed the laser controller, which opens
only while System Mode runs, and it was disabled whenever System Mode ran. In
System Mode the controller also refuses a ramp, because the shared input
stream feeds it. Ben chose Idle (2026-09-25): the ramp takes the NI-DAQ
stream through the same hold as the DAQ Monitor, opens a laser controller of
its own without the stream feeding it, and gives both back when it ends.

These drive the application model with the real stream model and the fake
stream worker, and a null laser controller for the ramp itself.
"""

import dataclasses
import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
)
from autotrainer.device import LaserCalibrationRamp, NullLaserController  # noqa: E402
from tools.acquisition.model import laser_model as laser_model_module  # noqa: E402
from tools.acquisition.model.app_model import AppModel  # noqa: E402
from tools.acquisition.model.app_model_status import SessionRecordingStatus  # noqa: E402
from tools.acquisition.model.laser_model import LaserModel  # noqa: E402
from tools.acquisition.model.nidaq_monitor_session import NidaqMonitorSession  # noqa: E402
from tools.acquisition.view.laser_control_content import LaserControlContent  # noqa: E402

from nidaq_stream_lifecycle_test import (  # noqa: E402,F401
    _settle,
    _worker_pid,
    nidaq_app,
)


RAMP_HOLD_REASON = "a laser calibration ramp is running"
RAMP = LaserCalibrationRamp(
    channel_id=LaserChannelId.LASER_1,
    start_volts=0.0,
    stop_volts=2.0,
    steps=3,
    samples_per_step=10,
)


def _null_lasers():
    return LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="Dev1/ao0",
                diode_input="Dev1/ai0",
                shutter_output="Dev1/port0/line2",
            ),
        ),
        backend="null",
        sample_rate_hz=1000.0,
    )


class _RampController(NullLaserController):
    """A null controller that can act, or fail, in the middle of its ramp."""

    def __init__(self, configuration, spy):
        super().__init__(configuration)
        self._spy = spy
        self.shutters_closed = False
        self.closed = False

    def run_calibration_ramp(self, ramp):
        if self._spy.during is not None:
            self._spy.during()
        if self._spy.error is not None:
            raise self._spy.error
        return super().run_calibration_ramp(ramp)

    def close_all_shutters(self):
        self.shutters_closed = True
        super().close_all_shutters()

    def close(self):
        self.closed = True
        super().close()


@pytest.fixture
def ramp_app(nidaq_app, monkeypatch):
    """Idle, the stream running by itself, and a null laser to calibrate."""
    assert nidaq_app.load_configuration() is True
    assert _settle(nidaq_app).is_running, "the stream did not start by itself"
    nidaq_app.laser.set_configuration_offline(_null_lasers())
    spy = SimpleNamespace(during=None, error=None, controllers=[], opened_with=[])

    def open_controller(configuration, **kwargs):
        spy.opened_with.append(kwargs)
        controller = _RampController(configuration, spy)
        spy.controllers.append(controller)
        return controller

    monkeypatch.setattr(nidaq_app.laser, "open_controller", open_controller)
    return nidaq_app, spy


def _stream_active(monitor):
    return monitor.is_running or monitor.is_starting


def test_a_ramp_holds_the_stream_and_hands_it_back(ramp_app):
    app, spy = ramp_app
    monitor = app.nidaq_signal_monitor
    idle_pid = _worker_pid(monitor)
    seen = {}

    def during():
        seen["stream active"] = _stream_active(monitor)
        seen["holders"] = app._nidaq_stream_autostart.pause_reasons
        seen["laser connected"] = app.laser.is_connected
        # A hardware refresh or a settings change asks for the stream while
        # the ramp runs; the hold must keep it off the lines the ramp uses.
        app._request_nidaq_stream("test: mid-ramp request")
        assert app._nidaq_stream_autostart.wait(10.0)
        seen["restarted mid-ramp"] = _stream_active(monitor)

    spy.during = during

    points = app.run_laser_calibration_ramp(RAMP)

    assert len(points) == 3
    assert seen == {
        "stream active": False,
        "holders": (RAMP_HOLD_REASON,),
        # Its own controller: the laser model stays as Idle has it.
        "laser connected": False,
        "restarted mid-ramp": False,
    }
    controller, = spy.controllers
    assert controller.shutters_closed and controller.closed
    assert app._nidaq_stream_autostart.pause_reasons == ()
    assert _settle(app).is_running
    assert _worker_pid(monitor) not in (None, idle_pid)
    assert app.laser_calibration_refusal() == ""


def test_a_failing_ramp_still_hands_the_stream_and_its_controller_back(ramp_app):
    app, spy = ramp_app
    spy.error = RuntimeError("DAQmx refused the calibration task")

    with pytest.raises(RuntimeError, match="refused the calibration task"):
        app.run_laser_calibration_ramp(RAMP)

    controller, = spy.controllers
    assert controller.shutters_closed and controller.closed
    assert app._nidaq_stream_autostart.pause_reasons == ()
    assert _settle(app).is_running
    assert app.laser_calibration_refusal() == ""


def test_run_the_daq_monitor_and_configuration_changes_are_refused_mid_ramp(ramp_app):
    app, spy = ramp_app
    laser_before = app.laser.configuration
    ports_before = app.nidaq_ports
    errors = []
    app.on_error += lambda _title, message: errors.append(message)
    outcomes = {}

    def during():
        outcomes["run"] = app.capture_start()
        outcomes["acquisition"] = (app._acquisition.starting, app._acquisition.started)
        for name, attempt in (
            ("DAQ Monitor", lambda: NidaqMonitorSession(app)),
            ("DAQ ports save", lambda: app.update_daq_port_configuration(
                dataclasses.replace(ports_before, tone1="Dev1/port0/line3"),
                laser_before,
            )),
            ("configuration load", app.load_configuration),
        ):
            try:
                attempt()
                outcomes[name] = "allowed"
            except RuntimeError as error:
                outcomes[name] = str(error)
        outcomes["holders"] = app._nidaq_stream_autostart.pause_reasons
        outcomes["stream active"] = _stream_active(app.nidaq_signal_monitor)

    spy.during = during

    app.run_laser_calibration_ramp(RAMP)

    assert outcomes["run"] is False
    assert outcomes["acquisition"] == (False, False)
    assert "calibration" in errors[-1]
    for name in ("DAQ Monitor", "DAQ ports save", "configuration load"):
        assert "calibration" in outcomes[name], (name, outcomes[name])
    assert outcomes["holders"] == (RAMP_HOLD_REASON,)
    assert outcomes["stream active"] is False
    assert app.laser.configuration == laser_before
    assert app.nidaq_ports == ports_before
    assert app._nidaq_stream_autostart.pause_reasons == ()
    assert _settle(app).is_running


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        ("run starting", "starting"),
        ("running", "System Mode is running"),
        ("recording", "recording"),
        ("DAQ Monitor open", "DAQ Monitor"),
    ],
)
def test_a_ramp_outside_idle_is_refused_and_leaves_the_stream_alone(
    ramp_app, state, reason,
):
    app, spy = ramp_app
    monitor = app.nidaq_signal_monitor
    pid = _worker_pid(monitor)
    session = None
    try:
        if state == "run starting":
            assert app._acquisition.begin_start()
        elif state == "running":
            app._acquisition.started = True
        elif state == "recording":
            app._acquisition.started = True
            app._set_session_recording_status(SessionRecordingStatus.RECORDING)
        else:
            session = NidaqMonitorSession(app)

        assert reason in app.laser_calibration_refusal()
        with pytest.raises(RuntimeError, match=reason):
            app.run_laser_calibration_ramp(RAMP)

        assert spy.controllers == []
        if session is None:
            assert app._nidaq_stream_autostart.pause_reasons == ()
            assert monitor.is_running and _worker_pid(monitor) == pid
    finally:
        if session is not None:
            session.close()
        if state == "recording":
            app._set_session_recording_status(SessionRecordingStatus.READY)
        app._acquisition.mark_stopped()


def test_the_ramp_controller_is_opened_without_the_stream_feeding_it(monkeypatch):
    # The controller refuses a ramp when the shared stream feeds it, which is
    # how System Mode always opened it. This one is opened for the ramp only,
    # and the laser model is not given it.
    built = []

    class _Nidaq:
        def __init__(self, configuration, *, feedback_reader=None, timing_plan=None):
            built.append((feedback_reader, timing_plan))
            self.configuration = configuration

    monkeypatch.setattr(laser_model_module, "NidaqLaserController", _Nidaq)
    configuration = dataclasses.replace(
        _null_lasers(), backend="nidaq", hardware_timed=True)
    laser = LaserModel()
    laser.set_configuration_offline(configuration)

    controller = laser.open_controller(configuration, timing_plan=None)

    assert isinstance(controller, _Nidaq)
    assert built == [(None, None)]
    assert not laser.is_connected


# ---------------------------------------------------------------- the button


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def idle_panel(qapp, app_model):
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    content = LaserControlContent(app_model)
    try:
        yield app_model, content, content._channel_tabs[0]
    finally:
        content.on_close()
        content.deleteLater()
        app_model.laser.close()
        app_model._acquisition.mark_stopped()


def _refresh(app_model, qapp):
    # What the model announces as a Run starts and as System Mode changes.
    app_model.property_changed(AppModel.Props.SUBSYSTEM_STATUSES, None, None)
    qapp.processEvents()


def test_run_ramp_is_enabled_in_idle(idle_panel):
    _app_model, _content, tab = idle_panel

    assert tab._run_ramp_button.isEnabled(), tab._run_ramp_button.toolTip()
    # Run Pulse still needs the controller System Mode opens.
    assert not tab._run_pulse_button.isEnabled()


def test_run_ramp_is_disabled_while_system_mode_runs(idle_panel, qapp):
    app_model, content, tab = idle_panel
    app_model._acquisition.started = True
    content.set_is_capture_active(True)

    assert not tab._run_ramp_button.isEnabled()
    assert "Idle" in tab._run_ramp_button.toolTip()


def test_run_ramp_is_disabled_while_a_session_records(idle_panel, qapp):
    app_model, content, tab = idle_panel
    app_model._acquisition.started = True
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)
    try:
        content.set_is_capture_active(True)

        assert not tab._run_ramp_button.isEnabled()
        assert "recording" in tab._run_ramp_button.toolTip()
    finally:
        app_model._set_session_recording_status(SessionRecordingStatus.READY)


def test_run_ramp_is_disabled_while_a_run_starts(idle_panel, qapp):
    # The laser connects part-way through a Run start, and System Mode reads
    # Running only at its end. Between the two the button was enabled, and a
    # ramp pressed there stopped System Mode's stream directly.
    app_model, content, tab = idle_panel
    assert app_model._acquisition.begin_start()
    _refresh(app_model, qapp)
    assert not tab._run_ramp_button.isEnabled()
    assert "starting" in tab._run_ramp_button.toolTip()

    app_model.laser.configure_null(_null_lasers())
    qapp.processEvents()

    rebuilt = content._channel_tabs[0]
    assert app_model.laser.is_connected
    assert not rebuilt._run_ramp_button.isEnabled()
    assert "starting" in rebuilt._run_ramp_button.toolTip()


def test_run_ramp_is_disabled_during_another_laser_operation(idle_panel):
    _app_model, content, tab = idle_panel

    content._set_running(True)
    assert not tab._run_ramp_button.isEnabled()
    assert "laser operation" in tab._run_ramp_button.toolTip()

    content._set_running(False)
    assert tab._run_ramp_button.isEnabled()


def test_run_ramp_runs_through_the_model_and_never_stops_the_stream_itself(
    idle_panel, qapp, monkeypatch,
):
    app_model, content, tab = idle_panel
    ramps = []

    def run_ramp(ramp):
        ramps.append(ramp)
        return NullLaserController(_null_lasers()).run_calibration_ramp(ramp)

    monitor_calls = []
    monkeypatch.setattr(app_model, "run_laser_calibration_ramp", run_ramp)
    monkeypatch.setattr(app_model.nidaq_signal_monitor, "stop",
                        lambda *args, **kwargs: monitor_calls.append("stop"))
    monkeypatch.setattr(app_model.nidaq_signal_monitor, "start",
                        lambda *args, **kwargs: monitor_calls.append("start") or True)
    tab._ramp_start.setValue(0.5)
    tab._ramp_stop.setValue(2.5)
    tab._ramp_steps.setValue(5)

    tab._run_ramp_button.click()
    deadline = time.monotonic() + 10.0
    while content._operation_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()

    ramp, = ramps
    assert (ramp.channel_id, ramp.start_volts, ramp.stop_volts, ramp.steps) == (
        LaserChannelId.LASER_1, 0.5, 2.5, 5)
    assert monitor_calls == []
    # The outcome stays on the status line; it was replaced at once by the
    # Run Pulse refusal, which is still true in Idle.
    assert content._status_label.text().startswith("Ramp complete: 5 points")
    assert tab._run_ramp_button.isEnabled()
