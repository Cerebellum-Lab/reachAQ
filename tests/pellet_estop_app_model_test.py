"""The application's side of the pellet ESTOP: the session keeps it, and closing stops the stage.

Every ESTOP during a session is written to its metadata (recording.estops,
with time, reason and axes). Closing the app stops a stage still moving before
anything that can wait; an idle stage is left alone. A stopped stage is not
homed automatically at the end of a session: the operator homes it.
"""

import json
import os
import threading
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from reachaq.core import Motor, SystemCommandKind
from reachaq.core.interfaces import RecordingEndingReason
from reachaq.app.model.recording_session_controller import RecordingSessionController
from reachaq.app.model.session_boundary import SessionBoundary
from reachaq.app.model.subsystem_status import SubsystemId, SubsystemState
from reachaq.app.view import main_window as main_window_module
from reachaq.app.view.main_window import MainWindow

_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)


def test_the_session_keeps_its_estops_and_record_resets_them():
    controller = RecordingSessionController()
    record = {"wallTime": 1.0, "perfTime": 2.0, "reason": "r", "axes": ["X", "Y", "Z"]}

    controller.add_estop(record)
    assert controller.estops == (record,)

    controller.prepare_record({})
    assert controller.estops == ()


def test_an_estop_is_written_to_the_sessions_metadata(app_model, tmp_path):
    app_model.hardware._on_device_estop("MOVE_Z was not acknowledged within 3 s", _XYZ, True)
    project = app_model.project.to_local_value()
    app_model._recording_session.boundary = SessionBoundary(
        session_id=project.short_id, primary_camera="left", primary_frame_id=42,
        start_perf_time=10.0, start_wall_time=1_800_000_000.25, camera_when=1_000_000.0,
        end_perf_time=12.0, end_wall_time=1_800_000_002.25,
    )
    app_model._on_session_capture_ended(RecordingEndingReason.MANUAL_STOP)
    output = tmp_path / "session003" / "metadata"
    (tmp_path / "session003" / "streams").mkdir(parents=True)

    app_model._save_metadata(project, project.when, str(output), project.session)

    saved = json.loads(output.with_suffix(".json").read_text())
    (estop,) = saved["recording"]["estops"]
    assert estop["reason"] == "MOVE_Z was not acknowledged within 3 s"
    assert estop["axes"] == ["X", "Y", "Z"]
    assert isinstance(estop["wallTime"], float) and isinstance(estop["perfTime"], float)
    assert estop["operator"] is False and estop["sent"] is True


class _StoppedBoard:
    """CanDevice and connection fakes: one log of releases and sends; a home is acked at once."""

    def __init__(self, hardware):
        self.log = []
        self._hardware = hardware
        self.motion_possibly_in_flight = False

    def estop(self, reason):
        self.log.append(("estop", reason))
        self._hardware._on_device_estop(reason, _XYZ, True)
        return True

    def release_estop(self):
        self.log.append(("release",))

    def send_message(self, kind, data=None, context=None):
        self.log.append(("send", kind))
        if kind is SystemCommandKind.SEND_HOME:
            threading.Timer(0.05, self._hardware._ack_received, args=(context,)).start()


def _stopped_app(app_model):
    hardware = app_model.hardware
    board = _StoppedBoard(hardware)
    hardware._can_device = board
    hardware._device_conn = board
    hardware.estop("MOVE_Z was not acknowledged within 3 s")
    return hardware, board


def test_the_session_end_home_after_a_stop_releases_the_latch_homes_and_clears_it(app_model, caplog):
    import logging

    app_model._set_subsystem_status(SubsystemId.CAN_PELLET, SubsystemState.READY)
    hardware, board = _stopped_app(app_model)
    try:
        with mock.patch.object(type(hardware), "connected", new_callable=mock.PropertyMock,
                               return_value=True), caplog.at_level(logging.INFO):
            app_model._request_session_end_home("stop")

        assert board.log[1:] == [("release",), ("send", SystemCommandKind.SEND_HOME)]
        assert not hardware.estop_active
        assert hardware.pellet_axes_needing_home == ()
        action = app_model._recording_session.end_actions[0]
        assert action["status"] == "completed"
        assert any("Session-end pellet home after a stop" in r.getMessage() for r in caplog.records)
    finally:
        hardware._can_device = hardware._device_conn = None


def test_a_pellet_machine_auto_home_is_still_refused_while_stopped(app_model):
    hardware, board = _stopped_app(app_model)
    try:
        pellet = app_model.behavior.system_machine.pellet
        try:
            pellet.move_home(force=True)
        except Exception:
            pass  # PelletDeviceCommandFailed: the home was not queued

        assert ("send", SystemCommandKind.SEND_HOME) not in board.log
        assert hardware.estop_active
    finally:
        hardware._can_device = hardware._device_conn = None


class _Stop(Exception):
    pass


def test_closing_stops_a_moving_stage_before_anything_else(app_model):
    order = []

    def shutdown():
        order.append(("shutdown",))
        raise _Stop  # the rest of the close is not this test's

    with mock.patch.object(app_model.hardware, "estop_if_moving",
                           side_effect=lambda reason: order.append(("estop", reason))), \
            mock.patch.object(app_model, "_prepare_application_shutdown", side_effect=shutdown), \
            mock.patch.object(app_model, "save_session_before_stop",
                              side_effect=lambda *a, **k: order.append(("save",))):
        with pytest.raises(_Stop):
            app_model.on_close()

    assert order[0] == ("estop", "reachAQ is closing")


def test_closing_the_window_stops_a_moving_stage_before_the_capture_stop(monkeypatch):
    order = []

    class _NowThread:
        def __init__(self, target):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(main_window_module, "threading",
                        SimpleNamespace(Thread=_NowThread, current_thread=threading.current_thread))
    monkeypatch.setattr(main_window_module, "InvokeMethod", lambda *args: None)
    hardware = SimpleNamespace(estop_if_moving=lambda reason: order.append(("estop", reason)))
    window = SimpleNamespace(
        _app_model=SimpleNamespace(app_lock=threading.RLock(), hardware=hardware,
                                   on_close=lambda: order.append(("on_close",))),
        _closing=False,
        _close_event=None,
        _confirm_close_with_session=lambda: True,
        _on_capture_start_stop=lambda started: order.append(("capture_stop",)),
        _stop_capture_thread=None,
        _timer_calibrate_diamond_triangle=SimpleNamespace(cancel=lambda: None),
        _finish_close=lambda: None,
    )

    MainWindow.close(window)

    assert order == [("estop", "reachAQ is closing"), ("capture_stop",), ("on_close",)]


def test_the_session_end_home_after_an_operator_stop_does_not_release_it(app_model):
    # Ben, 2026-11-24: after an operator STOP only an operator Home or a reconnect releases.
    from reachaq.app.model.hardware_model import OPERATOR_STOP_REASON

    app_model._set_subsystem_status(SubsystemId.CAN_PELLET, SubsystemState.READY)
    hardware = app_model.hardware
    board = _StoppedBoard(hardware)
    hardware._can_device = hardware._device_conn = board
    hardware.estop(OPERATOR_STOP_REASON)
    try:
        with mock.patch.object(type(hardware), "connected", new_callable=mock.PropertyMock,
                               return_value=True):
            app_model._request_session_end_home("stop")

        assert ("release",) not in board.log
        assert ("send", SystemCommandKind.SEND_HOME) not in board.log
        assert hardware.estop_active
        action = app_model._recording_session.end_actions[0]
        assert action["status"] == "skipped"
        assert "operator STOP" in action["error"]
    finally:
        hardware._can_device = hardware._device_conn = None


def test_the_session_end_home_after_an_operator_stop_and_a_failure_stop_does_not_release(app_model):
    # Review I-C, end to end: STOP pressed, then a failure stop; the session end is skipped.
    from reachaq.app.model.hardware_model import OPERATOR_STOP_REASON

    app_model._set_subsystem_status(SubsystemId.CAN_PELLET, SubsystemState.READY)
    hardware = app_model.hardware
    board = _StoppedBoard(hardware)
    hardware._can_device = hardware._device_conn = board
    hardware.estop(OPERATOR_STOP_REASON)
    hardware.estop("calibration waypoint 3 was not reached within 0.88 s")
    try:
        with mock.patch.object(type(hardware), "connected", new_callable=mock.PropertyMock,
                               return_value=True):
            app_model._request_session_end_home("stop")

        assert ("release",) not in board.log
        assert app_model._recording_session.end_actions[0]["status"] == "skipped"
    finally:
        hardware._can_device = hardware._device_conn = None


def test_each_estop_is_its_own_entry_in_the_session_metadata(app_model):
    from reachaq.app.model.hardware_model import OPERATOR_STOP_REASON

    hardware = app_model.hardware
    hardware._on_device_estop(OPERATOR_STOP_REASON, _XYZ, True)
    hardware._on_device_estop("a later failure", _XYZ, False)

    reasons = [(e["reason"], e["operator"], e["kind"], e["sent"]) for e in app_model._recording_session.estops]
    assert reasons == [(OPERATOR_STOP_REASON, True, "operator", True),
                       ("a later failure", False, "operator", False)]


def _limit_fault():
    from reachaq.device import StepperFault, Target

    return StepperFault(target=Target.PELLET_DEVICE, motor=Motor.PELLET_Z_MOTOR, reason=2, flags=0x02,
                        pulses_issued=571, pulses_planned=1142, believed_position_turns=2.5,
                        believed_position_mm=6.09, pulses_per_turn=192.0, uuid=12)


def test_the_session_keeps_its_stepper_faults_and_record_resets_them():
    controller = RecordingSessionController()
    record = {"axis": "Z", "reasonName": "LIMIT"}

    controller.add_stepper_fault(record)
    assert controller.stepper_faults == (record,)

    controller.prepare_record({})
    assert controller.stepper_faults == ()


def test_a_board_fault_is_written_to_the_sessions_metadata(app_model, tmp_path):
    app_model.hardware._on_stepper_fault(_limit_fault())
    project = app_model.project.to_local_value()
    app_model._recording_session.boundary = SessionBoundary(
        session_id=project.short_id, primary_camera="left", primary_frame_id=42,
        start_perf_time=10.0, start_wall_time=1_800_000_000.25, camera_when=1_000_000.0,
        end_perf_time=12.0, end_wall_time=1_800_000_002.25,
    )
    app_model._on_session_capture_ended(RecordingEndingReason.MANUAL_STOP)
    output = tmp_path / "session003" / "metadata"
    (tmp_path / "session003" / "streams").mkdir(parents=True)

    app_model._save_metadata(project, project.when, str(output), project.session)

    saved = json.loads(output.with_suffix(".json").read_text())
    (fault,) = saved["recording"]["stepperFaults"]
    assert fault["axis"] == "Z" and fault["reasonName"] == "LIMIT" and fault["blocking"] is True
    assert (fault["pulsesIssued"], fault["pulsesPlanned"]) == (571, 1142)
    assert isinstance(fault["wallTime"], float)


def test_a_board_fault_holds_automatic_motion_until_the_axis_is_homed(app_model):
    app_model.hardware._on_stepper_fault(_limit_fault())

    assert app_model.hardware.pellet_axes_needing_home == ("Z",)
    assert not app_model._pellet_motion_readiness()
