"""A pose process that dies on its own stops live inference, and says why.

On the rig a pose process raised on its first preview frame and exited, and the
app never noticed. LIVE_INFERENCE had been marked READY when the process was
spawned, the Terminated message was a no-op, and the pose watchdog is only
registered once the first pose comes back. Record was allowed, and a 120 s
recording posed none of its frames.

A model that fails to load sends no Terminated at all, and neither does a
process that is killed outright, so the exit is detected from the process
itself. The message only supplies the reason, when there is one to give.
"""

import queue
import threading
from types import SimpleNamespace

import pytest

from autotrainer.inference import InferenceStatus, InferenceStatusMessageKind
from autotrainer.inference import pose_process
from autotrainer.inference.pose_process import PoseProcess
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.inference_model import InferenceModel
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState


def _exited(code=0):
    return SimpleNamespace(exitcode=code)


def _running():
    # Also what a Process reports before start() has been called on it.
    return SimpleNamespace(exitcode=None)


def _inference(status, process):
    model = InferenceModel.__new__(InferenceModel)
    model._status = status
    model._pose_process = process
    model._pose_process_error = None
    model.stop_calls = 0

    def stop():
        model.stop_calls += 1

    model.stop = stop
    return model


# -- detecting the exit ---------------------------------------------------

@pytest.mark.parametrize("status", [
    InferenceStatus.loading,
    InferenceStatus.waiting,
    InferenceStatus.live,
    InferenceStatus.intersession,
])
def test_a_process_that_exited_on_its_own_stops_inference(status):
    model = _inference(status, _exited())

    model._check_pose_process_exited()

    assert model.stop_calls == 1


@pytest.mark.parametrize("status", [InferenceStatus.stopping, InferenceStatus.stopped])
def test_a_requested_stop_is_not_mistaken_for_a_crash(status):
    model = _inference(status, _exited())

    model._check_pose_process_exited()

    assert model.stop_calls == 0


@pytest.mark.parametrize("process", [None, _running()])
def test_a_running_or_absent_process_is_left_alone(process):
    model = _inference(InferenceStatus.live, process)

    model._check_pose_process_exited()

    assert model.stop_calls == 0


def test_the_message_loop_checks_for_the_exit_while_idle():
    model = _inference(InferenceStatus.waiting, _exited())
    model._notif_msg_queue = queue.Queue()
    stopped = threading.Event()
    model.stop = stopped.set

    loop = threading.Thread(target=model._monitor_msg_queue, daemon=True)
    loop.start()
    try:
        assert stopped.wait(2.0), "an exited pose process went unnoticed"
    finally:
        model._notif_msg_queue.put(None)  # the loop's exit sentinel
        loop.join(2.0)


# -- carrying the reason --------------------------------------------------

def test_the_reason_a_process_gives_on_its_way_out_is_kept():
    model = _inference(InferenceStatus.live, _running())
    model._notif_msg_queue = queue.Queue()
    model._notif_msg_queue.put(
        (InferenceStatusMessageKind.Terminated, "ValueError: boom"))
    model._notif_msg_queue.put(None)

    model._monitor_msg_queue()

    assert model.pose_process_error == "ValueError: boom"


def test_a_pose_process_that_cannot_load_says_why(monkeypatch):
    sent = []
    process = PoseProcess.__new__(PoseProcess)
    process._msg_queue = SimpleNamespace(put=sent.append)
    monkeypatch.setattr(pose_process.signal, "signal", lambda *args: None)
    monkeypatch.setattr(pose_process, "setup_logging", lambda: None)

    def fail_to_load():
        raise ValueError("best.pt predicts 17 keypoints but yolo_pose.yaml names 14")

    monkeypatch.setattr(process, "_PoseProcess__do_run", fail_to_load)

    process._do_run(log_dict_config=None)

    assert sent[-1] == (
        InferenceStatusMessageKind.Terminated,
        "ValueError: best.pt predicts 17 keypoints but yolo_pose.yaml names 14",
    )


def _app(cause):
    calls = []
    app = SimpleNamespace(
        _acquisition=SimpleNamespace(
            started=True,
            stopping=False,
            subsystems={SubsystemId.LIVE_INFERENCE:
                        SimpleNamespace(state=SubsystemState.READY)},
        ),
        _inference=SimpleNamespace(pose_process_error=cause),
        _set_subsystem_status=lambda *args, **kwargs: calls.append(("status", args, kwargs)),
        _handle_recording_subsystem_failure=lambda *args: calls.append(("recording", args)),
        _analysis=SimpleNamespace(
            watchdog_monitor=SimpleNamespace(unregister_watchdog=lambda key: None)),
        _reach_cameras=[],
        _update_status_text_overlay=lambda: None,
    )
    return app, calls


def test_the_failed_subsystem_names_the_cause():
    """What the status panel, the degraded tooltip and the Record refusal show."""
    app, calls = _app("RuntimeError: CUDA error: out of memory")

    AppModel._on_inference_property_changed(
        app, InferenceModel.STATUS, InferenceStatus.stopped, InferenceStatus.live)

    expected = "live inference stopped unexpectedly: RuntimeError: CUDA error: out of memory"
    assert calls == [
        ("status", (SubsystemId.LIVE_INFERENCE, SubsystemState.FAILED), {"error": expected}),
        ("recording", (SubsystemId.LIVE_INFERENCE, expected)),
    ]


def test_a_process_that_gave_no_reason_still_fails_the_subsystem():
    app, calls = _app(None)

    AppModel._on_inference_property_changed(
        app, InferenceModel.STATUS, InferenceStatus.stopped, InferenceStatus.loading)

    assert calls[0] == ("status", (SubsystemId.LIVE_INFERENCE, SubsystemState.FAILED),
                        {"error": "live inference stopped unexpectedly"})
