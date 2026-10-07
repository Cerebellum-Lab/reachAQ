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

import math
import queue
import threading
from types import SimpleNamespace

import pytest

from autotrainer.inference import (
    InferenceCommandMessageKind, InferenceStatus, InferenceStatusMessageKind,
)
from autotrainer.inference import pose_process
from autotrainer.inference.pose_process import PoseProcess
from tools.acquisition.model import inference_model as inference_model_module
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


@pytest.mark.parametrize("code, reason", [
    (3, "pose process exited with code 3"),
    (-9, "pose process was killed by SIGKILL"),
    (-11, "pose process was killed by SIGSEGV"),
])
def test_a_process_that_gave_no_reason_is_described_by_how_it_ended(code, reason):
    """A killed process, or one that failed while still importing, says nothing.
    The kernel's OOM killer sends SIGKILL, and a CUDA fault is a SIGSEGV."""
    model = _inference(InferenceStatus.loading, _exited(code))

    model._check_pose_process_exited()

    assert model.pose_process_error == reason


def test_a_reason_the_process_gave_is_not_replaced_by_its_exit_code():
    model = _inference(InferenceStatus.live, _exited(0))
    model._pose_process_error = "ValueError: boom"

    model._check_pose_process_exited()

    assert model.pose_process_error == "ValueError: boom"


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


# -- before the model reports Loading -------------------------------------
#
# Spawning the pose process and importing and building its model take seconds
# before the child's first Loading message. Status used to stay `stopped` all
# that time, so a death there went unnoticed, and a stop() there returned at
# once and left the process loading a model for a capture that had ended.

class _FakePoseProcess:
    def __init__(self, *args, **kwargs):
        self.started = False
        self.joined = False
        self.exitcode = None

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and self.exitcode is None

    def join(self, timeout=None):
        self.joined = True


class _CommandQueue(queue.Queue):
    """Delivers Terminate to the fake process at once, and acknowledges it."""

    def __init__(self, ack):
        super().__init__()
        self._ack = ack
        self.process = None

    def put(self, item, *args, **kwargs):
        super().put(item, *args, **kwargs)
        kind, _context = item
        if kind == InferenceCommandMessageKind.Terminate and self.process is not None:
            self.process.exitcode = 0
        self._ack.set()


def _spawning_inference(monkeypatch):
    """The real start() and stop(), with the processes, pool and thread faked."""
    monkeypatch.setattr(inference_model_module, "PoseProcess", _FakePoseProcess)
    model = InferenceModel.__new__(InferenceModel)
    model._on_property_changed = lambda name, new, old: None
    model._status = InferenceStatus.stopped
    model._model_location = "/unused"
    model.can_start_live_inference = lambda: True
    ack = threading.Event()
    model._cmd_queue_ack = ack
    model._cmd_queue = _CommandQueue(ack)
    model._cmd_queue_lock = threading.Lock()
    model._output_data_queue = queue.Queue()
    model._notif_msg_queue = queue.Queue()
    model._data_monitor_cmd_queue = queue.Queue()
    model._record_stop_sema = None
    model._mp_manager = SimpleNamespace(Event=threading.Event)
    model._process_pool = SimpleNamespace(
        close=lambda: None, terminate=lambda: None, join=lambda: None)
    model._msg_thread = object()  # already running; start() must not add one
    model._data_monitor_proc = SimpleNamespace(
        stop_recorded=threading.Event(), is_alive=lambda: True,
        join=lambda timeout=None: None, exitcode=0, pid=0)
    model._pose_process = None
    model._pose_process_error = None
    model._pose_process_watchdog_perf_c = SimpleNamespace(value=math.nan)
    model._offline_segmentation_thread = None
    model._offline_analysis_thread = None
    model._intersession_block = None
    model._intersession_detection = None
    return model


_LIVE_QUEUE = SimpleNamespace(shape=(256, 256), frames_per_camera=1)


def test_inference_is_loading_from_the_moment_its_process_is_spawned(monkeypatch):
    model = _spawning_inference(monkeypatch)

    assert model.start(_LIVE_QUEUE)

    assert model._pose_process.started
    assert model.status == InferenceStatus.loading


def test_a_stop_before_the_model_reports_loading_still_stops_the_process(monkeypatch):
    model = _spawning_inference(monkeypatch)
    model.start(_LIVE_QUEUE)
    process = model._pose_process
    model._cmd_queue.process = process

    model.stop()

    assert process.exitcode == 0, "the process was never asked to terminate"
    assert process.joined
    assert model._pose_process is None
    assert model.status == InferenceStatus.stopped


@pytest.mark.parametrize("status", [InferenceStatus.stopping, InferenceStatus.stopped])
@pytest.mark.parametrize("message", [
    (InferenceStatusMessageKind.Loading, None),
    (InferenceStatusMessageKind.Initialized, ["Nose"]),
    (InferenceStatusMessageKind.Running, 0),
])
def test_a_process_being_stopped_cannot_make_inference_active_again(status, message):
    """Stopped mid-load, the process keeps loading until it reads Terminate,
    and reports Loading and Initialized on the way. On the rig, acting on them
    moved the stopping inference back to loading and then waiting, and stalled
    the message loop 3 s on the acknowledgement of a Start that could not come."""
    model = _inference(status, _running())
    calls = []
    model._set_status = calls.append
    model._send_message = lambda kind, context=None: calls.append(kind)
    model._notif_msg_queue = queue.Queue()
    model._notif_msg_queue.put(message)
    model._notif_msg_queue.put(None)

    model._monitor_msg_queue()

    assert calls == []
    assert model.status == status


def test_a_process_that_dies_before_the_model_reports_loading_is_caught(monkeypatch):
    model = _spawning_inference(monkeypatch)
    model.start(_LIVE_QUEUE)
    process = model._pose_process
    process.exitcode = 1  # an import failed in the child, say

    model._check_pose_process_exited()

    assert process.joined
    assert model._pose_process is None
    assert model.status == InferenceStatus.stopped
    assert model.pose_process_error == "pose process exited with code 1"


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


def test_a_grouped_tensorflow_error_is_described_by_its_first_cause():
    """TensorFlow leads with a count of its errors; the cause is on the next line."""
    error = RuntimeError(
        "2 root error(s) found.\n"
        "  (0) RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[6,256,256,3]\n"
        "  (1) RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[6,256,256,3]")

    assert pose_process._describe_error(error) == (
        "RuntimeError: (0) RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[6,256,256,3]")


def _app(cause, inference=None):
    calls = []
    app = SimpleNamespace(
        _acquisition=SimpleNamespace(
            started=True,
            stopping=False,
            subsystems={SubsystemId.LIVE_INFERENCE:
                        SimpleNamespace(state=SubsystemState.READY)},
        ),
        _inference=(SimpleNamespace(pose_process_error=cause)
                    if inference is None else inference),
        _set_subsystem_status=lambda *args, **kwargs: calls.append(("status", args, kwargs)),
        _handle_recording_subsystem_failure=lambda *args: calls.append(("recording", args)),
        on_error=lambda title, text: calls.append(("alert", title, text)),
        _analysis=SimpleNamespace(
            watchdog_monitor=SimpleNamespace(unregister_watchdog=lambda key: None)),
        _reach_cameras=[],
        _update_status_text_overlay=lambda: None,
    )
    return app, calls


def test_the_failed_subsystem_names_the_cause_and_the_operator_is_told():
    """The status panel, the degraded tooltip and the Record refusal show the
    reason, and the operator is alerted as a watchdog timeout would alert them:
    the exit is now caught before that watchdog can fire."""
    app, calls = _app("RuntimeError: CUDA error: out of memory")

    AppModel._on_inference_property_changed(
        app, InferenceModel.STATUS, InferenceStatus.stopped, InferenceStatus.live)

    expected = "live inference stopped unexpectedly: RuntimeError: CUDA error: out of memory"
    assert calls == [
        ("status", (SubsystemId.LIVE_INFERENCE, SubsystemState.FAILED), {"error": expected}),
        ("recording", (SubsystemId.LIVE_INFERENCE, expected)),
        ("alert", "Hardware subsystem failure",
         f"{expected}. Unrelated hardware remains running; Record is blocked "
         "until the required subsystem is ready."),
    ]


def test_a_process_that_gave_no_reason_still_fails_the_subsystem():
    app, calls = _app(None)

    AppModel._on_inference_property_changed(
        app, InferenceModel.STATUS, InferenceStatus.stopped, InferenceStatus.loading)

    assert calls[0] == ("status", (SubsystemId.LIVE_INFERENCE, SubsystemState.FAILED),
                        {"error": "live inference stopped unexpectedly"})


def test_an_inference_that_keeps_no_pose_process_reason_still_fails_the_subsystem():
    """Like the VoidInference test fixture, which has no pose process at all."""
    app, calls = _app(None, inference=SimpleNamespace())

    AppModel._on_inference_property_changed(
        app, InferenceModel.STATUS, InferenceStatus.stopped, InferenceStatus.live)

    assert calls[0] == ("status", (SubsystemId.LIVE_INFERENCE, SubsystemState.FAILED),
                        {"error": "live inference stopped unexpectedly"})
