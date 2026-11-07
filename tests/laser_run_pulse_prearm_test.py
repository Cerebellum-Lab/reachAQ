"""Run Pulse is armed while it is pressed, and fired by its click (D2.4).

Ben, 2026-10-02 (latency-fix-plan.md, decisions 1 = D, 2 = (a), 3 = after):
the press arms the pulse and opens its shutter, and the click starts it, one
DAQmx start on the Qt thread, its edge about 0.6 ms later; arming, writing
and starting it after the click took 12-18 ms (session004; christielab10, M3
B). A press that ends with no click, or is held past its 2 s cap, fires
nothing and leaves the laser safe.

On the device tests' DAQmx fake: nothing here touches a driver or a board.
"""

import gc
import json
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QFocusEvent, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.device import (  # noqa: E402
    LaserChannelId,
    LaserOperationState,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser  # noqa: E402
from tools.acquisition.model.trial_action import LaserPulseProfile  # noqa: E402
from tools.acquisition.view import laser_control_content  # noqa: E402
from tools.acquisition.view.laser_control_content import (  # noqa: E402
    _ERROR_STATUS_COLOR,
    LaserControlContent,
)

PROFILE = LaserPulseProfile("short", 1, 1.0, 2.0)
AO = "laser_sync_pulse_ao"
AO_LINE = "PXI1Slot4/ao0"
SHUTTER_LINE = "PXI1Slot5/port0/line4"
OFF_THE_BUTTON = QPoint(-10, -10)


def _daqmx_fake():
    """The laser device tests' stand-in for NI-DAQmx, loaded from its file."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1].joinpath(
        "auto-trainer-device", "tests", "nidaq_daqmx_fake.py")
    spec = importlib.util.spec_from_file_location("nidaq_daqmx_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def fake(monkeypatch):
    module = _daqmx_fake()
    module.daq = module.FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: module.daq)
    return module


@pytest.fixture
def rig(qapp, app_model, fake, monkeypatch):
    """Laser Control on christielab10's laser 1 over the fake, a profile picked."""
    monkeypatch.setattr(type(app_model), "trial_protocol_state", property(
        lambda _self: {"laser_profiles": ({
            "profile_id": PROFILE.profile_id, "revision": PROFILE.revision,
            "summary": PROFILE.summary()},)}))
    monkeypatch.setattr(type(app_model), "laser_profile", lambda _self, profile_id: (
        PROFILE if profile_id == PROFILE.profile_id else None))
    # What earlier tests dropped is freed here, on the main thread, not by
    # the cyclic GC on whichever worker thread next crosses its threshold
    # (see the teardown).
    gc.collect()
    controller = NidaqLaserController(fake.rig_lasers())
    app_model.laser.set_controller(controller)
    panel = LaserControlContent(app_model)
    tab = panel._channel_tabs[0]
    panel._tabs.setCurrentWidget(tab)
    tab.stim_profile_selector.setCurrentIndex(tab.stim_profile_selector.findData("short"))
    qapp.processEvents()
    told = []
    app_model.laser.trace_received += told.append
    rig = SimpleNamespace(
        app_model=app_model, laser=app_model.laser, controller=controller,
        daq=fake.daq, panel=panel, tab=tab, button=tab._run_pulse_button,
        told=told, closed=False)
    try:
        yield rig
    finally:
        fake.daq.waits_released.set()
        fake.daq.before_start = None
        if not rig.closed:
            panel.on_close()
        app_model.laser.close()
        panel.deleteLater()
        qapp.processEvents()
        # Deleted now, not left for an event loop the tests never run: a
        # panel left alive is repolished by every later application font
        # change, which laser_control_layout_test makes around each test.
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        # Its Python side, pyqtgraph's parentless menus among it, is held by
        # reference cycles, and is freed here, on the main thread. Freed by
        # the cyclic GC on a worker thread, a later test's pulse thread, it
        # deadlocked the run about once in 17: PySide deleted the widgets on
        # the main thread and the other QObjects on the worker, and the two
        # held a Qt mutex and the GIL against each other (the review of task
        # 9, gdb on christielab10).
        rig.panel = rig.tab = rig.button = None
        del panel, tab
        gc.collect()
        assert controller.wait_for_work_left_running(5.0)


def _wait_until(qapp, condition, what, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"{what} did not happen"
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def _armed(rig):
    """The press's operation, once it is armed."""
    operations = list(rig.controller._live_operations.values())
    if len(operations) == 1 and operations[0].state is LaserOperationState.ARMED:
        return operations[0]
    return None


def _wait_for_the_arm(rig, qapp):
    _wait_until(qapp, lambda: _armed(rig) is not None, "the arm")
    return _armed(rig)


def _wait_for_the_operation(rig, qapp):
    _wait_until(qapp, lambda: rig.panel._operation_thread is None, "the operation's end", 10.0)


def _press(button):
    QTest.mousePress(button, Qt.MouseButton.LeftButton)


def _release(button, pos=None):
    QTest.mouseRelease(
        button, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        button.rect().center() if pos is None else pos)


def _drag_off(button):
    # QTest.mouseMove moves the cursor, which offscreen does not; a move
    # event with the button held is what Qt reads a drag from.
    QApplication.sendEvent(button, QMouseEvent(
        QEvent.Type.MouseMove, QPointF(OFF_THE_BUTTON),
        QPointF(button.mapToGlobal(OFF_THE_BUTTON)), Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))


def _manual_events(told):
    return [trace for trace in told if trace.source == "manual pulse"]


def _footer(rig):
    return rig.panel._status_label


def _shows_error(rig, text) -> bool:
    return _footer(rig).text() == text and _ERROR_STATUS_COLOR in _footer(rig).styleSheet()


def _left_safe(rig, before):
    """A press that fired nothing: the laser as it was, and the board free.

    The output never started and is at its minimum, the shutter closed, the
    output and the board let go of, and the model's record of the laser's
    last command what it was before the press.
    """
    daq = rig.daq
    assert AO not in daq.starts
    assert [write.data for write in daq.writes
            if write.channels == (AO_LINE,) and write.task != AO][-1] == 0.0
    assert daq.writes_to(SHUTTER_LINE)[-1] is False
    assert daq.reserved == {}
    assert rig.controller._live_operations == {}
    assert rig.laser.last_command_volts == before


def _no_driver_call_on_the_qt_thread(rig, since):
    """Nothing on the fake's timeline from the main thread since `since`.

    A press let go of is disarmed from the Qt thread, and its cancel, the
    shutter's write and the output's abort, is made on the thread that armed
    it: Ben accepted only the click's start on the Qt thread (the review of
    task 9, Important 1).
    """
    main = threading.main_thread()
    assert [(entry.event, entry.task) for entry in rig.daq.timeline[since:]
            if entry.thread is main] == []


# ------------------------------------------------ the plan's D2.4 tests


def test_a_press_arms_the_pulse_and_its_click_fires_it_on_the_main_thread(rig, qapp):
    starts, told_at = [], {}
    rig.daq.before_start = lambda task: starts.append(
        (task.label, threading.current_thread(), time.perf_counter()))
    rig.laser.trace_received += lambda trace: told_at.setdefault(
        trace.event, time.perf_counter())

    _press(rig.button)
    _wait_for_the_arm(rig, qapp)

    # Armed by the press, its shutter open (decision 2 = (a)), nothing
    # started and nothing told; the pressed button still enabled.
    assert rig.daq.starts == []
    assert rig.daq.writes_to(SHUTTER_LINE)[-1] is True
    assert rig.told == []
    assert rig.button.isDown() and rig.button.isEnabled()
    # Nothing runs yet, and the footer says so (the review of task 9, Minor 4).
    assert _footer(rig).text() == "Laser 1 armed: release to fire"

    _release(rig.button)

    assert _footer(rig).text() == "Running laser 1 pulse train"
    (label, thread, started_at), = starts
    assert (label, thread) == (AO, threading.main_thread())
    _wait_for_the_operation(rig, qapp)
    requested, completed = _manual_events(rig.told)
    assert (requested.event, completed.event) == ("requested", "completed")
    # Stamped before the start, told after it (decision 3).
    assert requested.origin_perf_time <= started_at <= told_at["requested"]
    # Fired by its click, not once armed: a start about 0.6 ms after it.
    context = json.loads(requested.context_json)
    assert context["fallback"] is False
    assert context["click_perf_time"] <= requested.origin_perf_time
    assert rig.daq.starts.count(AO) == 1
    assert rig.laser.last_command_volts == {1: 0.0}
    assert _footer(rig).text() == "Pulse complete: laser 1"


def test_a_press_dragged_off_the_button_is_disarmed(rig, qapp):
    before = rig.laser.last_command_volts
    _press(rig.button)
    operation = _wait_for_the_arm(rig, qapp)
    armed = len(rig.daq.timeline)

    # Qt's released, with no clicked.
    _drag_off(rig.button)
    _wait_until(qapp, lambda: operation.wait_until_finished(0), "the disarm")
    _release(rig.button, OFF_THE_BUTTON)
    _wait_for_the_operation(rig, qapp)

    assert operation.state is LaserOperationState.CANCELLED
    # Its cancel, shutter first, was made off the Qt thread.
    _no_driver_call_on_the_qt_thread(rig, armed)
    # No manual rows, and no waveform: nothing was asked for.
    assert rig.told == []
    _left_safe(rig, before)
    assert _footer(rig).text() == (
        "Run Pulse disarmed, nothing fired: laser 1's press ended without a click")


def test_a_press_held_past_its_cap_is_disarmed(rig, qapp, monkeypatch):
    # 0.5 s, not the 2 s cap, but wide enough that the arming thread is
    # never scheduled after the cap has run out (the review of task 9, Minor
    # 2): woken that late, its arm raised the cap's TimeoutError itself.
    monkeypatch.setattr(laser_control_content, "_RUN_PULSE_ARM_CAP_S", 0.5)
    before = rig.laser.last_command_volts
    _press(rig.button)
    operation = _wait_for_the_arm(rig, qapp)
    _wait_for_the_operation(rig, qapp)

    # Let go by its arm's start wait with the button still down; usable again.
    assert operation.state is LaserOperationState.FAILED
    assert isinstance(operation.error, TimeoutError)
    assert rig.button.isDown() and rig.button.isEnabled()
    assert rig.told == []
    _left_safe(rig, before)
    assert _footer(rig).text() == (
        "Run Pulse disarmed, nothing fired: laser 1's press was held over 0.5 s")

    # Its click, after the cap, fires nothing, and says so.
    _release(rig.button)
    qapp.processEvents()
    assert AO not in rig.daq.starts
    assert rig.told == []
    assert _shows_error(
        rig, "Laser 1: Run Pulse did not fire, its press was held over 0.5 s; click it again")

    # The next click fires.
    rig.button.click()
    _wait_for_the_operation(rig, qapp)
    assert rig.daq.starts.count(AO) == 1
    assert _footer(rig).text() == "Pulse complete: laser 1"


def _hold_the_output_write(fake, monkeypatch):
    """The arm held in its output's buffer write; (writing, release)."""
    writing, release = threading.Event(), threading.Event()
    write = fake.FakeTask.write

    def held_write(task, data, auto_start=False):
        if task.label == AO:
            writing.set()
            assert release.wait(5.0)
        return write(task, data, auto_start)

    monkeypatch.setattr(fake.FakeTask, "write", held_write)
    return writing, release


def test_a_click_before_its_arm_is_ready_fires_it_once_armed(rig, qapp, fake, monkeypatch, caplog):
    caplog.set_level("INFO", logger=laser_control_content.__name__)
    writing, release = _hold_the_output_write(fake, monkeypatch)
    starts = []
    rig.daq.before_start = lambda task: starts.append((task.label, threading.current_thread()))

    _press(rig.button)
    assert writing.wait(5.0)
    arming_thread = rig.panel._operation_thread
    _release(rig.button)
    assert starts == []
    release.set()
    _wait_for_the_operation(rig, qapp)

    # Once, on the thread that armed it.
    assert starts == [(AO, arming_thread)]
    fallback, = [record for record in caplog.records
                 if "fired once armed" in record.getMessage()]
    assert fallback.levelname == "INFO"
    requested, completed = _manual_events(rig.told)
    assert (requested.event, completed.event) == ("requested", "completed")
    # Marked as fired once armed, with its click's time, which is earlier
    # than its start: the data tells it from a sub-ms start (Minor 3).
    context = json.loads(requested.context_json)
    assert context["fallback"] is True
    assert context["click_perf_time"] < requested.origin_perf_time


def test_a_click_before_its_arm_whose_start_is_refused_says_so(
    rig, qapp, fake, monkeypatch, caplog,
):
    # The log line said "fired once armed" whenever trigger() returned, and a
    # start the driver refuses returns too, as the operation's failure
    # (the review of task 9, Minor 8).
    caplog.set_level("INFO", logger=laser_control_content.__name__)
    writing, release = _hold_the_output_write(fake, monkeypatch)
    rig.daq.failing_task = AO

    _press(rig.button)
    assert writing.wait(5.0)
    _release(rig.button)
    release.set()
    _wait_for_the_operation(rig, qapp)

    messages = [record.getMessage() for record in caplog.records
                if record.levelname == "INFO" and "Run Pulse was clicked" in record.getMessage()]
    assert len(messages) == 1
    assert "fired once armed" not in messages[0]
    assert "DAQmx refused to start laser_sync_pulse_ao" in messages[0]
    assert [trace.event for trace in _manual_events(rig.told)] == ["requested", "failed"]
    assert rig.daq.writes_to(SHUTTER_LINE)[-1] is False


def test_a_refused_arm_is_told_as_requested_then_refused_at_its_click(rig, qapp):
    # A trial's pulse holds the board, armed for its trigger.
    holder = rig.laser.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0),),
        wait=False, defer_start=True, timeout_seconds=30.0,
        operation_context={"logical_trial_id": 7}))
    try:
        rig.told.clear()
        _press(rig.button)
        _wait_for_the_operation(rig, qapp)

        refusal = _footer(rig).text()
        assert refusal.startswith(
            "Laser operation failed: Laser 1: refused while trial 7's pulse on laser 1")
        assert _shows_error(rig, refusal)
        # Nothing told until the click.
        assert rig.told == []

        _release(rig.button)

        requested, refused = _manual_events(rig.told)
        assert (requested.event, refused.event) == ("requested", "refused")
        assert refused.operation_id == requested.operation_id
        context = json.loads(refused.context_json)
        assert context["error_class"] == "LaserPulseRefused"
        # The trial's 2.5 V is still what the output may hold.
        assert context["last_command_volts"] == 2.5
        assert rig.laser.last_command_volts == {1: 2.5}
        assert _shows_error(rig, refusal)
        assert rig.daq.starts == []
    finally:
        holder.cancel()
        holder.wait_until_finished(5.0)


def test_space_fires_one_pulse_per_press_of_the_key(rig, qapp):
    starts = []
    rig.daq.before_start = lambda task: starts.append((task.label, threading.current_thread()))

    QTest.keyPress(rig.button, Qt.Key.Key_Space)
    operation = _wait_for_the_arm(rig, qapp)
    # Held, the key repeats, as X11 sends it; Qt takes none of it as a press.
    for kind in (QEvent.Type.KeyRelease, QEvent.Type.KeyPress):
        QApplication.sendEvent(rig.button, QKeyEvent(
            kind, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", True))
    qapp.processEvents()
    assert list(rig.controller._live_operations.values()) == [operation]
    QTest.keyRelease(rig.button, Qt.Key.Key_Space)
    _wait_for_the_operation(rig, qapp)

    assert starts == [(AO, threading.main_thread())]
    assert [trace.event for trace in _manual_events(rig.told)] == ["requested", "completed"]


def test_external_mode_is_not_armed_by_its_press(rig, qapp):
    rig.tab._trigger_mode.setCurrentText("external")
    rig.tab._trigger_source.setText("/PXI1Slot4/PFI0")

    _press(rig.button)
    qapp.processEvents()
    assert rig.panel._operation_thread is None
    assert not [task for task in rig.daq.tasks if task.label == AO]

    # Armed at its click and started by its edge, as before.
    _release(rig.button)
    _wait_for_the_operation(rig, qapp)
    assert rig.daq.task(AO).start_trigger == ("/PXI1Slot4/PFI0", "rising")
    requested, completed = _manual_events(rig.told)
    assert (requested.event, completed.event) == ("requested", "completed")
    assert json.loads(requested.context_json)["trigger_mode"] == "external"


# ------------------------------------------------ the safety invariants


def _stop(rig, _qapp):
    # What System Mode's Stop does to the laser (AppModel._close_laser_within_bound).
    rig.laser.close()


def _switch_laser_tab(rig, _qapp):
    rig.panel._tabs.setCurrentIndex(rig.panel._tabs.indexOf(rig.panel._builder))


def _switch_to_calibration(rig, _qapp):
    rig.tab._mode_tabs.setCurrentIndex(1)


def _close_the_panel(rig, _qapp):
    rig.panel.on_close()
    rig.closed = True


def _lose_focus(rig, _qapp):
    # As a dialog opening mid-press takes it.
    QApplication.sendEvent(rig.button, QFocusEvent(
        QEvent.Type.FocusOut, Qt.FocusReason.ActiveWindowFocusReason))


def _rebuild(rig, _qapp):
    # As a DAQ ports save or a configuration load rebuilds the laser tabs.
    rig.panel._refresh_from_model()


def _lock_the_panel(rig, _qapp):
    # The held button disabled under it: Qt emits released, with no click.
    rig.panel.set_is_editable(False)


@pytest.mark.parametrize("end_the_press", (
    _stop, _switch_laser_tab, _switch_to_calibration, _close_the_panel, _lose_focus,
    _rebuild, _lock_the_panel,
), ids=("stop", "laser_tab_switch", "calibration_tab", "panel_closed", "focus_lost",
        "rebuild", "button_disabled"))
def test_what_else_ends_a_press_fires_nothing_and_leaves_the_laser_safe(
    rig, qapp, end_the_press,
):
    before = rig.laser.last_command_volts
    _press(rig.button)
    operation = _wait_for_the_arm(rig, qapp)
    arming_thread = rig.panel._operation_thread
    armed = len(rig.daq.timeline)

    end_the_press(rig, qapp)
    _wait_until(qapp, lambda: operation.wait_until_finished(0), "the pulse's end")
    arming_thread.join(5.0)
    assert not arming_thread.is_alive()
    # A release on the button after it fires nothing either.
    _release(rig.button)
    qapp.processEvents()

    assert operation.state is LaserOperationState.CANCELLED
    if end_the_press is not _stop:
        # The test's Stop closes the controller on this thread itself; the
        # application's runs on a thread of its own.
        _no_driver_call_on_the_qt_thread(rig, armed)
    assert rig.told == []
    _left_safe(rig, before)


def test_a_press_whose_operation_is_refused_arms_nothing(rig, qapp, caplog):
    # Another laser operation still holds the panel, with Run Pulse enabled
    # meanwhile, as before the panel's refresh has run: the press is refused
    # as any second operation is, and its click fires and records nothing
    # (the review of task 9, Minor 9).
    before = rig.laser.last_command_volts
    release = threading.Event()
    rig.panel._start_operation(
        "Running another laser operation", lambda: release.wait(10.0) and "done")
    rig.button.setEnabled(True)
    try:
        with caplog.at_level("ERROR"):
            _press(rig.button)
            _release(rig.button)
            qapp.processEvents()

        assert _shows_error(rig, "Laser operation already in progress")
        assert [record.getMessage() for record in caplog.records
                if "already in progress" in record.getMessage()] == [
                    "Laser control operation rejected: Laser operation already in progress"]
        assert not [task for task in rig.daq.tasks if task.label == AO]
        assert rig.told == []
        assert rig.laser.last_command_volts == before
    finally:
        release.set()
        _wait_for_the_operation(rig, qapp)


def test_a_stop_while_the_press_arms_fires_nothing_and_leaves_the_laser_safe(
    rig, qapp, fake, monkeypatch,
):
    # The arm is still being made: the cancel fails it before it is armed,
    # and its amplitude's record is taken back all the same.
    before = rig.laser.last_command_volts
    writing, release = _hold_the_output_write(fake, monkeypatch)
    _press(rig.button)
    assert writing.wait(5.0)
    operation, = rig.controller._live_operations.values()
    arming_thread = rig.panel._operation_thread
    stop = threading.Thread(target=rig.laser.close, daemon=True)
    stop.start()
    _wait_until(qapp, lambda: operation.state is LaserOperationState.CANCELLED, "the cancel")
    release.set()
    stop.join(10.0)
    arming_thread.join(5.0)
    assert not stop.is_alive() and not arming_thread.is_alive()
    _release(rig.button)
    qapp.processEvents()

    assert rig.told == []
    _left_safe(rig, before)


def test_a_stop_during_the_fired_pulse_cancels_it_and_says_so(rig, qapp, caplog):
    # Preserved from the waited-for Run Pulse: recorded as cancelled, and
    # its failure says the controller was closed.
    rig.daq.block_wait = rig.daq.hold_waits = True
    _press(rig.button)
    operation = _wait_for_the_arm(rig, qapp)
    _release(rig.button)
    assert operation.state is LaserOperationState.TRIGGERED

    with caplog.at_level("ERROR"):
        rig.laser.close()
        _wait_for_the_operation(rig, qapp)

    assert operation.state is LaserOperationState.CANCELLED
    requested, cancelled = _manual_events(rig.told)
    assert (requested.event, cancelled.event) == ("requested", "cancelled")
    context = json.loads(cancelled.context_json)
    assert context["error_class"] == "LaserPulseCancelled"
    reason = (f"Laser operation {operation.operation_id} was cancelled: the laser "
              "controller was closed while it ran")
    assert context["error"] == reason
    # The record that puts it on the status bar, and on the footer until
    # the panel's next status.
    assert [record.getMessage() for record in caplog.records
            if record.getMessage().startswith("Laser operation failed")] == [
                f"Laser operation failed: {reason}"]
    assert rig.daq.writes_to(SHUTTER_LINE)[-1] is False


def test_a_recorder_that_fails_at_the_request_no_longer_stops_the_pulse(rig, qapp, caplog):
    # Decision 3's trade: the row is told after the start, so a failure to
    # tell it is logged, with the pulse's id, and the pulse runs and is told
    # as it ended.
    def failing(trace):
        if trace.event == "requested":
            raise RuntimeError("the recorder failed")

    rig.laser.trace_received += failing
    _press(rig.button)
    _wait_for_the_arm(rig, qapp)
    with caplog.at_level("ERROR"):
        _release(rig.button)
        _wait_for_the_operation(rig, qapp)

    assert rig.daq.starts.count(AO) == 1
    requested, completed = _manual_events(rig.told)
    assert (requested.event, completed.event) == ("requested", "completed")
    logged, = [record for record in caplog.records
               if requested.operation_id in record.getMessage()]
    assert logged.levelname == "ERROR"
    assert _footer(rig).text() == "Pulse complete: laser 1"


def test_a_pulse_whose_record_cannot_be_made_is_not_started(rig, qapp, monkeypatch):
    # Made after the start, a record that failed left a pulse that ran with
    # nothing told of it, and its amplitude's record taken back (the review
    # of task 9, Minor 6). Made before it, the click starts nothing, and the
    # arm is let go by its start wait.
    monkeypatch.setattr(laser_control_content, "_RUN_PULSE_ARM_CAP_S", 0.5)

    def cannot(*_args):
        raise RuntimeError("the record could not be made")

    monkeypatch.setattr(rig.laser, "_manual_pulse", cannot)
    before = rig.laser.last_command_volts
    _press(rig.button)
    _wait_for_the_arm(rig, qapp)
    _release(rig.button)
    _wait_for_the_operation(rig, qapp)

    assert rig.told == []
    _left_safe(rig, before)


def test_an_arm_that_returns_no_operation_takes_its_amplitude_back(rig, monkeypatch):
    # Every other arm failure took it back; this defensive one raised with
    # the amplitude still recorded as what the output may hold (Minor 6).
    before = rig.laser.last_command_volts
    pulse_train, manual_context = rig.tab._validated_pulse()
    monkeypatch.setattr(rig.controller, "run_synchronized_pulse_train", lambda _train: None)

    with pytest.raises(RuntimeError, match="did not arm"):
        rig.laser.arm_manual_pulse(
            pulse_train, manual_context=manual_context, start_wait_seconds=2.0)

    assert rig.laser.last_command_volts == before
