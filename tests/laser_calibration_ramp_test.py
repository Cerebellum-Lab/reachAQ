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
import threading
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


# ------------------------------------------------------- review fix round 1


def test_a_ramp_is_refused_once_closing_has_begun(ramp_app):
    app, spy = ramp_app
    app._closing_event.set()

    assert "closing" in app.laser_calibration_refusal()
    with pytest.raises(RuntimeError, match="closing"):
        app.run_laser_calibration_ramp(RAMP)
    assert spy.controllers == []
    assert app._nidaq_stream_autostart.pause_reasons == ()


def test_a_pause_that_fails_leaves_no_hold_behind(ramp_app, monkeypatch):
    # The hold is registered before the stream is stopped, and the pause sat
    # outside the try that lets go: a stop that raised left the ramp holding
    # the stream forever, and every later ramp and DAQ Monitor refused.
    app, spy = ramp_app
    monitor = app.nidaq_signal_monitor
    real_stop = monitor.stop
    calls = []

    def failing_stop(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("the NI-DAQ worker did not stop")
        return real_stop(*args, **kwargs)

    monkeypatch.setattr(monitor, "stop", failing_stop)

    with pytest.raises(RuntimeError, match="did not stop"):
        app.run_laser_calibration_ramp(RAMP)

    assert app._nidaq_stream_autostart.pause_reasons == ()
    assert app.laser_calibration_refusal() == ""
    assert spy.controllers == []
    assert len(app.run_laser_calibration_ramp(RAMP)) == 3
    assert _settle(app).is_running


def test_the_application_announces_a_ramp_starting_and_ending(ramp_app):
    app, _spy = ramp_app
    seen = []
    app.property_changed += lambda name, value, _old: (
        seen.append(value) if name == AppModel.Props.LASER_CALIBRATION_ACTIVE else None)

    app.run_laser_calibration_ramp(RAMP)

    assert seen == [True, False]
    assert app.laser_calibration_active is False


def test_a_nidaq_laser_without_hardware_timing_refuses_the_ramp(idle_panel, qapp):
    # The controller refuses a ramp without hardwareTimed and a sampleRateHz
    # (nidaq_laser.py); the button stayed enabled and the press failed.
    app_model, content, _tab = idle_panel
    app_model.laser.set_configuration_offline(dataclasses.replace(
        _null_lasers(), backend="nidaq", hardware_timed=False, sample_rate_hz=None))
    qapp.processEvents()

    refusal = app_model.laser_calibration_refusal()
    assert "hardwareTimed" in refusal and "sampleRateHz" in refusal
    tab = content._channel_tabs[0]
    assert not tab._run_ramp_button.isEnabled()
    assert "hardwareTimed" in tab._run_ramp_button.toolTip()


def test_the_footer_drops_its_own_refusal_once_a_ramp_can_run(idle_panel, qapp):
    # A rebuild during a hold, a configuration load's or a DAQ ports save's,
    # found nothing to fire and said "press Run first"; the refreshes after
    # it do not announce, so the line kept saying so in Idle, where a ramp
    # can run.
    app_model, content, _tab = idle_panel
    rule = app_model._nidaq_stream_autostart
    rule.pause("test hold", "the configuration is loading")
    try:
        content._refresh_from_model()
        assert "press Run first" in content._status_label.text()
    finally:
        rule.resume("test hold")
    qapp.processEvents()

    assert "press Run first" not in content._status_label.text()
    assert content._status_label.text().startswith("Ready: 1/4")


def test_a_failed_laser_operation_says_what_failed(idle_panel, qapp, caplog):
    # It said "Laser operation stopped", and the reason reached only the
    # log's traceback: a -89125 from the ramp's first press would not have
    # been seen. And nidaqmx's DaqError puts the status code last, so the
    # first line alone, which is all the footer and status bar show, lost it.
    _app_model, content, _tab = idle_panel

    def failing():
        # As nidaqmx.errors.DaqError formats itself: the driver's own text,
        # then the task, then the status code on the last line.
        raise RuntimeError(
            "The specified route cannot be satisfied, because it requires "
            "connecting the source and destination terminals using a trigger "
            "line, and no registered trigger lines could be found between the "
            "devices in the route.\n"
            "Source Device: PXI1Slot4\n"
            "Destination Device: PXI1Slot5\n"
            "\n"
            "Task Name: laser_1_calibration_ai\n"
            "\n"
            "Status Code: -89125")

    with caplog.at_level("ERROR"):
        content._start_operation("Running laser 1 calibration ramp", failing)
        deadline = time.monotonic() + 10.0
        while content._operation_thread is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        qapp.processEvents()

    footer = content._status_label.text()
    assert "-89125" in footer and "Task Name" not in footer
    assert footer.startswith("Laser operation failed: The specified route")
    # The status bar shows a record's first line.
    first_lines = [record.getMessage().splitlines()[0] for record in caplog.records
                   if record.getMessage().startswith("Laser operation failed")]
    assert first_lines and all("-89125" in line for line in first_lines)
    assert all(len(line) <= 320 for line in first_lines)


def test_the_panel_can_go_while_a_laser_operation_still_runs(qapp, app_model):
    # The operation thread was QThread(self). A panel destroyed while a
    # force-closed ramp still ran destroyed a running QThread with it, and
    # Qt aborts the process for that ("QThread: Destroyed while thread is
    # still running"), at the very moment reachAQ was closing.
    import shiboken6

    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    content = LaserControlContent(app_model)
    inside, release = threading.Event(), threading.Event()
    finished = []

    def held():
        inside.set()
        release.wait(10.0)
        finished.append(True)
        return "Ramp complete"

    try:
        content._start_operation("Running laser 1 calibration ramp", held)
        assert inside.wait(5.0)
        operation_thread = content._operation_thread
        content.on_close()
        shiboken6.delete(content)
        qapp.processEvents()
        release.set()
        operation_thread.join(5.0)
        # Its result, queued for a panel that has gone.
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.01)

        assert not operation_thread.is_alive()
        assert finished == [True]
    finally:
        release.set()
        app_model.laser.close()
        app_model._acquisition.mark_stopped()


def test_run_ramp_runs_on_the_operation_worker_off_the_qt_thread(ramp_app, qapp, monkeypatch):
    # The real path: the button's operation worker calls the application,
    # which pauses and stops the stream there, not on the Qt thread.
    app, spy = ramp_app
    monitor = app.nidaq_signal_monitor
    real_stop = monitor.stop
    stop_threads = []

    def recording_stop(*args, **kwargs):
        stop_threads.append(threading.current_thread())
        return real_stop(*args, **kwargs)

    monkeypatch.setattr(monitor, "stop", recording_stop)
    seen = {}

    def during():
        seen["thread"] = threading.current_thread()
        seen["holders"] = app._nidaq_stream_autostart.pause_reasons
        seen["stream active"] = _stream_active(monitor)

    spy.during = during
    content = LaserControlContent(app)
    try:
        tab = content._channel_tabs[0]
        assert tab._run_ramp_button.isEnabled(), tab._run_ramp_button.toolTip()

        tab._run_ramp_button.click()
        deadline = time.monotonic() + 15.0
        while content._operation_thread is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        qapp.processEvents()

        assert seen["thread"] is not threading.main_thread()
        assert seen["holders"] == (RAMP_HOLD_REASON,)
        assert seen["stream active"] is False
        assert stop_threads and all(
            thread is not threading.main_thread() for thread in stop_threads)
        controller, = spy.controllers
        assert controller.closed
        assert content._status_label.text().startswith("Ramp complete: 11 points")
        assert _settle(app).is_running
    finally:
        content.on_close()
        content.deleteLater()
