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


def _a_null_laser_to_calibrate(app, monkeypatch):
    """A null laser the ramp opens a _RampController for; the spy on it."""
    app.laser.set_configuration_offline(_null_lasers())
    spy = SimpleNamespace(during=None, error=None, controllers=[], opened_with=[])

    def open_controller(configuration, **kwargs):
        spy.opened_with.append(kwargs)
        controller = _RampController(configuration, spy)
        spy.controllers.append(controller)
        return controller

    monkeypatch.setattr(app.laser, "open_controller", open_controller)
    return spy


@pytest.fixture
def ramp_app(nidaq_app, monkeypatch):
    """Idle, the stream running by itself, and a null laser to calibrate."""
    assert nidaq_app.load_configuration() is True
    assert _settle(nidaq_app).is_running, "the stream did not start by itself"
    return nidaq_app, _a_null_laser_to_calibrate(nidaq_app, monkeypatch)


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
        def __init__(self, configuration, *, feedback_reader=None, timing_plan=None,
                     analog_terminal_config=None):
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


def test_run_pulse_is_refused_while_a_hung_laser_close_is_in_the_driver(
    idle_panel, qapp, monkeypatch,
):
    # The laser model still holds the controller whose close hung in the
    # driver; Run Pulse would call into it.
    from tools.acquisition.model import app_model as app_model_module

    app_model, content, _tab = idle_panel
    app_model.laser.configure_null(_null_lasers())
    _refresh(app_model, qapp)
    # Connecting rebuilt the tabs.
    tab = content._channel_tabs[0]
    assert tab._run_pulse_button.isEnabled(), tab._run_pulse_button.toolTip()

    monkeypatch.setattr(app_model, "laser_controller_close_refusal",
                        lambda: app_model_module._LASER_CLOSE_PENDING_REFUSAL)
    _refresh(app_model, qapp)

    assert tab is content._channel_tabs[0]
    assert not tab._run_pulse_button.isEnabled()
    assert tab._run_pulse_button.toolTip() == (
        "The laser controller is still closing after a driver hang; make the "
        "laser safe by hand, and restart reachAQ if this does not clear")
    assert not tab.stim_test_button.isEnabled()


def _type_into(spin_box, text, qapp, *, commit=True):
    """Replace a spin box's text by typing, then press Return, as an operator."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    spin_box.selectAll()
    QTest.keyClicks(spin_box, text)
    if commit:
        QTest.keyClick(spin_box, Qt.Key.Key_Return)
    qapp.processEvents()


def test_settle_is_a_time_with_a_fixed_default(idle_panel, qapp):
    # Settle followed a fifth of Samples/step until it was edited. It is a
    # time now, 600 µs, whatever Samples/step is: the laser, the diode and
    # the input take as long to settle however long the step is.
    _app_model, _content, tab = idle_panel

    assert tab._ramp_settle.value() == 600
    assert tab._ramp_settle.suffix() == " µs"
    _type_into(tab._ramp_samples_per_step, "50", qapp)
    assert tab._ramp_settle.value() == 600


def test_the_default_step_is_500_samples(idle_panel):
    # 5 ms at christielab10's 100 kHz: its slower diode was still rising
    # through the second half of a 1 ms step (H2b).
    _app_model, _content, tab = idle_panel

    assert tab._ramp_samples_per_step.value() == 500


def _ramps_run_from(tab, app_model, qapp, monkeypatch, content, run):
    ramps = []

    def run_ramp(ramp):
        ramps.append(ramp)
        return NullLaserController(_null_lasers()).run_calibration_ramp(ramp)

    monkeypatch.setattr(app_model, "run_laser_calibration_ramp", run_ramp)
    run()
    deadline = time.monotonic() + 10.0
    while content._operation_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    return ramps


def test_samples_per_step_typed_and_left_by_focus_is_what_the_ramp_runs(
    idle_panel, qapp, monkeypatch,
):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QFocusEvent
    from PySide6.QtWidgets import QApplication

    app_model, content, tab = idle_panel
    _type_into(tab._ramp_samples_per_step, "50", qapp, commit=False)
    assert tab._ramp_samples_per_step.value() == 500, "typing alone took it"
    QApplication.sendEvent(tab._ramp_samples_per_step, QFocusEvent(
        QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason))

    # Taken by the focus-out itself, before Run Ramp could take it.
    assert tab._ramp_samples_per_step.value() == 50
    ramp, = _ramps_run_from(tab, app_model, qapp, monkeypatch, content,
                            tab._run_ramp_button.click)

    assert (ramp.samples_per_step, ramp.settle_seconds) == (50, pytest.approx(600e-6))


def test_samples_per_step_typed_then_run_ramp_is_what_the_ramp_runs(
    idle_panel, qapp, monkeypatch,
):
    # Taken only when typing finishes, "50" was still unread when Run Ramp
    # was pressed without the field losing focus, as by its shortcut: the
    # ramp ran on the 500 before it.
    app_model, content, tab = idle_panel
    _type_into(tab._ramp_samples_per_step, "50", qapp, commit=False)

    ramp, = _ramps_run_from(tab, app_model, qapp, monkeypatch, content,
                            tab._run_ramp_button.click)

    assert ramp.samples_per_step == 50


def test_the_settle_tooltip_says_what_it_is_and_when_it_resets(idle_panel):
    _app_model, _content, tab = idle_panel
    tooltip = tab._ramp_settle.toolTip()

    assert "600 µs" in tooltip and "60 samples at 100 kHz" in tooltip
    assert "Run/Stop" in tooltip and "DAQ Ports save" in tooltip


def test_the_ramp_runs_with_the_settle_typed_in_microseconds(idle_panel, qapp, monkeypatch):
    app_model, content, tab = idle_panel
    _type_into(tab._ramp_settle, "250", qapp)

    ramp, = _ramps_run_from(tab, app_model, qapp, monkeypatch, content,
                            tab._run_ramp_button.click)

    assert ramp.settle_seconds == pytest.approx(250e-6)


def test_a_settle_that_leaves_no_sample_is_refused_before_the_ramp(
    idle_panel, qapp, monkeypatch, caplog,
):
    # At the null laser's 1 kHz, 5 ms is 5 samples: none of a 5-sample step.
    app_model, content, tab = idle_panel
    _type_into(tab._ramp_samples_per_step, "5", qapp)
    _type_into(tab._ramp_settle, "5000", qapp)

    ramps = _ramps_run_from(tab, app_model, qapp, monkeypatch, content,
                            tab._run_ramp_button.click)

    assert ramps == []
    # A refusal is logged as the panel's other refusals are.
    assert any("5 samples of each step" in record.getMessage()
               for record in caplog.records)


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
    # 600 us by default, whatever Samples/step is.
    assert tab._ramp_settle.value() == 600
    assert "settle" in tab._ramp_settle.toolTip().lower()
    tab._ramp_settle.setValue(70)

    tab._run_ramp_button.click()
    deadline = time.monotonic() + 10.0
    while content._operation_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()

    ramp, = ramps
    assert (ramp.channel_id, ramp.start_volts, ramp.stop_volts, ramp.steps) == (
        LaserChannelId.LASER_1, 0.5, 2.5, 5)
    assert ramp.settle_seconds == pytest.approx(70e-6)
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


def test_the_ramps_controller_references_its_inputs_as_the_stream_does(ramp_app):
    # The controller's own inputs took DAQmx's default: on christielab10's
    # 6221, ai3, ai4 and ai5 differential, where the stream reads them as RSE.
    app, spy = ramp_app

    app.run_laser_calibration_ramp(RAMP)

    stream = app.nidaq_signal_monitor.configuration.analog_terminal_config
    assert stream
    assert [kwargs.get("analog_terminal_config") for kwargs in spy.opened_with] == [stream]


def _laser_opened_as_a_run_opens_it(app):
    """The analogTerminalConfig System Mode gave the laser model as it opened it."""
    given = []
    load = app.laser.load_configuration

    def recorded(configuration, **kwargs):
        given.append(kwargs.get("analog_terminal_config"))
        return load(configuration, **kwargs)

    app.laser.load_configuration = recorded
    try:
        assert app._start_laser_domain()
    finally:
        del app.laser.load_configuration
        app.laser.stop_direct_trigger_receiver()
        app.laser.close()
    return given


def test_system_mode_opens_the_laser_with_the_streams_terminal_config(nidaq_app):
    assert nidaq_app.load_configuration() is True
    nidaq_app.laser.set_configuration_offline(_null_lasers())

    given = _laser_opened_as_a_run_opens_it(nidaq_app)

    stream = nidaq_app.nidaq_signal_monitor.configuration.analog_terminal_config
    assert stream
    assert given == [stream]


def test_with_a_refused_plan_the_laser_takes_the_stored_streams_terminal_config(
    nidaq_app, monkeypatch,
):
    # With a refused plan the stream runs no channels, on a default
    # configuration: the operator's own stream, as stored, is the source.
    assert nidaq_app.load_configuration() is True
    nidaq_app.laser.set_configuration_offline(_null_lasers())
    stored = nidaq_app._loaded_configuration
    monkeypatch.setattr(stored, "nidaq_stream", dataclasses.replace(
        stored.nidaq_stream, analog_terminal_config="nrse"))
    monkeypatch.setattr(nidaq_app, "_nidaq_plan_error", "test: a refused plan")

    given = _laser_opened_as_a_run_opens_it(nidaq_app)

    assert nidaq_app.nidaq_signal_monitor.configuration.analog_terminal_config != "nrse"
    assert given == ["nrse"]



# ---------------------------------------------------------------- the final fix round


def test_a_ramp_whose_controller_close_hangs_is_let_go_within_the_bound(
    ramp_app, monkeypatch, caplog,
):
    # The ramp closed its own controller in its finally with no bound: a
    # driver hung in that close kept the calibration active, the panel
    # stuck, and nothing was said until reachAQ closed.
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 0.3)
    app, spy = ramp_app
    release = threading.Event()

    def the_close_hangs():
        controller = spy.controllers[-1]
        close = controller.close

        def hung():
            release.wait(10.0)
            close()

        controller.close = hung

    spy.during = the_close_hangs
    outcome = []
    ramp = threading.Thread(
        target=lambda: outcome.append(app.run_laser_calibration_ramp(RAMP)), daemon=True)
    try:
        with caplog.at_level("CRITICAL"):
            ramp.start()
            ramp.join(5.0)
            assert not ramp.is_alive(), "the ramp waited for its hung close"

        points, = outcome
        assert len(points) == 3
        assert not app.laser_calibration_active
        critical, = [record.getMessage() for record in caplog.records
                     if record.levelname == "CRITICAL"]
        assert "laser 1" in critical and "did not close within 0.3 s" in critical
        assert "make the laser safe by hand" in critical
        refusal = app.laser_controller_close_refusal()
        assert "laser calibration controller is still closing" in refusal
        assert app.laser_calibration_refusal() == refusal
    finally:
        release.set()
    deadline = time.monotonic() + 5.0
    while app.laser_controller_close_refusal() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert app.laser_controller_close_refusal() == ""


def test_a_hardware_refresh_is_refused_mid_ramp_by_name(ramp_app, monkeypatch):
    # The docs said a refresh is refused while a ramp runs, and it was not:
    # only its request for the stream was declined by the ramp's hold.
    from hardware_status_content_test import _patch_hardware_scans

    app, spy = ramp_app
    _patch_hardware_scans(app, monkeypatch)
    seen = []

    def refresh():
        try:
            app.refresh_hardware_bindings()
        except RuntimeError as error:
            seen.append(str(error))
        else:
            seen.append("refreshed")

    spy.during = refresh

    app.run_laser_calibration_ramp(RAMP)

    assert seen == [
        "Hardware refresh is unavailable while a laser calibration ramp runs; "
        "wait for it to finish"]


# ---------------------------------------------- workstream D: the follow-ups


def _hang_the_ramps_close(spy, release):
    """During the ramp: its controller's close will hang until `release`."""
    def the_close_hangs():
        controller = spy.controllers[-1]
        close = controller.close

        def hung():
            release.wait(10.0)
            close()

        controller.close = hung

    spy.during = the_close_hangs


def _laser_status(app):
    from tools.acquisition.model.subsystem_status import SubsystemId

    return app.subsystem_statuses.get(SubsystemId.LASER.value)


def _ramp_whose_close_hangs(app, spy, monkeypatch):
    """Run the ramp, its close hung; the release that lets that close end."""
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 0.3)
    release = threading.Event()
    _hang_the_ramps_close(spy, release)
    ramp = threading.Thread(target=app.run_laser_calibration_ramp, args=(RAMP,), daemon=True)
    ramp.start()
    ramp.join(5.0)
    assert not ramp.is_alive(), "the ramp waited for its hung close"
    assert "laser calibration controller is still closing" in (
        app.laser_controller_close_refusal())
    return release


def _until_the_close_ends(app):
    deadline = time.monotonic() + 5.0
    while app.laser_controller_close_refusal() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert app.laser_controller_close_refusal() == ""


def test_a_ramps_hung_close_keeps_the_stream_stopped_until_it_ends(ramp_app, monkeypatch):
    # A close given up on let go of the ramp's hold on the stream, which
    # restarted while that close was still inside the driver, with the
    # ramp's inputs open on the same lines.
    app, spy = ramp_app
    monitor = app.nidaq_signal_monitor
    release = _ramp_whose_close_hangs(app, spy, monkeypatch)
    try:
        refusal = app.laser_controller_close_refusal()
        assert app._nidaq_stream_autostart.pause_reasons == ()
        assert app._nidaq_stream_autostart.wait(10.0)
        app._request_nidaq_stream("test: while the ramp's close hangs")
        assert app._nidaq_stream_autostart.wait(10.0)
        assert not _stream_active(monitor), "the stream restarted over a hung close"
        # And it says why.
        assert refusal in monitor.status_message
    finally:
        release.set()

    _until_the_close_ends(app)
    # It starts again by itself once the close has ended.
    assert _settle(app).is_running


def test_a_ramps_hung_close_fails_the_laser_until_it_ends(ramp_app, monkeypatch):
    # LASER said nothing of it, unlike a failed open's close still in the
    # driver: the refusal showed only once laser work was tried.
    from tools.acquisition.model.subsystem_status import SubsystemState

    app, spy = ramp_app
    release = _ramp_whose_close_hangs(app, spy, monkeypatch)
    try:
        laser = _laser_status(app)
        assert laser.state is SubsystemState.FAILED
        assert laser.error == app.laser_controller_close_refusal()
    finally:
        release.set()

    _until_the_close_ends(app)
    deadline = time.monotonic() + 5.0
    while _laser_status(app).state is SubsystemState.FAILED and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _laser_status(app).state is not SubsystemState.FAILED


def test_a_hung_close_says_nothing_of_holding_back_a_disabled_stream(
    nidaq_app, system_config, trainer_config_dir, monkeypatch,
):
    # With the stream disabled there is nothing to hold back. It said "held
    # back" all the same, and still said it once the close had ended.
    from autotrainer.core import NidaqPortConfiguration

    system_config.nidaq_ports = NidaqPortConfiguration()
    system_config.save_default(trainer_config_dir)
    assert nidaq_app.load_configuration() is True
    monitor = nidaq_app.nidaq_signal_monitor
    assert not monitor.configuration.is_enabled
    spy = _a_null_laser_to_calibrate(nidaq_app, monkeypatch)
    release = _ramp_whose_close_hangs(nidaq_app, spy, monkeypatch)
    try:
        assert "held back" not in monitor.status_message
    finally:
        release.set()

    _until_the_close_ends(nidaq_app)
    assert "held back" not in monitor.status_message


def test_the_held_back_text_goes_once_the_close_has_ended(ramp_app, monkeypatch):
    # The late finish asks for the stream; one that does not start then,
    # for whatever reason, still said it was held back by a close that had
    # ended.
    app, spy = ramp_app
    monitor = app.nidaq_signal_monitor
    release = _ramp_whose_close_hangs(app, spy, monkeypatch)
    try:
        assert app._nidaq_stream_autostart.wait(10.0)
        assert "held back" in monitor.status_message
        monkeypatch.setattr(app._nidaq_stream_autostart, "request", lambda: None)
    finally:
        release.set()

    _until_the_close_ends(app)
    deadline = time.monotonic() + 5.0
    while "held back" in monitor.status_message and time.monotonic() < deadline:
        time.sleep(0.02)
    assert monitor.status_message == "NI-DAQ signal stream stopped"
    assert not _stream_active(monitor)
