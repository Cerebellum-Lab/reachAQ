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

from autotrainer.core import Motor
from autotrainer.core.interfaces import RecordingEndingReason
from tools.acquisition.model.recording_session_controller import RecordingSessionController
from tools.acquisition.model.session_boundary import SessionBoundary
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState
from tools.acquisition.view import main_window as main_window_module
from tools.acquisition.view.main_window import MainWindow

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


def test_a_stopped_stage_is_not_homed_at_the_end_of_the_session(app_model):
    app_model._set_subsystem_status(SubsystemId.CAN_PELLET, SubsystemState.READY)
    app_model.hardware._on_device_estop("a test stop", _XYZ, True)
    with mock.patch.object(app_model.hardware, "send_home_and_wait") as send_home:
        app_model._request_session_end_home("stop")

    send_home.assert_not_called()
    action = app_model._recording_session.end_actions[0]
    assert action["status"] == "skipped"
    assert "stopped (a test stop)" in action["error"]


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
