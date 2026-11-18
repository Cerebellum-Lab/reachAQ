"""Pose training runs in the app: started only while no session records,
paused before one starts, and deploys that can be undone.

The child process is replaced by a thread (multiprocessing.dummy) running a
fake target, so these tests exercise the run manager's states and the
recording gate without torch or a GPU.
"""

import multiprocessing.dummy
import os
import time

import pytest
import yaml

from tools.acquisition.model.pose_training_model import (
    DEPLOY_HISTORY,
    PoseTrainingModel,
    TrainingState,
    last_deploy,
    record_deploy,
)


def _fake_target(dataset_path, run_dir, epochs, resume, messages, stop_event):
    messages.put(("started", {"resume": resume}))
    epoch = 0
    while not stop_event.is_set():
        epoch += 1
        messages.put(("progress", (epoch, epochs, {"epoch": epoch})))
        if epoch >= epochs:
            messages.put(("done", {"status": "finished", "epoch": epoch}))
            return
        time.sleep(0.01)
    messages.put(("done", {"status": "stopped", "epoch": epoch}))


def _failing_target(dataset_path, run_dir, epochs, resume, messages, stop_event):
    messages.put(("error", "ValueError: there are no labelled training frames"))


def _model(tmp_path, ready=True, target=_fake_target):
    gate = {"ready": ready}
    model = PoseTrainingModel(str(tmp_path / "models"), is_ready=lambda: gate["ready"],
                              context=multiprocessing.dummy, target=target)
    return model, gate


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_start_is_refused_while_a_session_records(tmp_path):
    model, _ = _model(tmp_path, ready=False)
    with pytest.raises(RuntimeError, match="recording"):
        model.start(str(tmp_path / "data"), epochs=5)
    assert model.state is TrainingState.IDLE


def test_a_run_finishes_and_reports_progress(tmp_path):
    model, _ = _model(tmp_path)
    seen = []
    model.property_changed += lambda name, new, old: seen.append(name)
    model.start(str(tmp_path / "data"), epochs=3)
    assert _wait(lambda: model.state is TrainingState.FINISHED)
    assert model.progress[0] == 3
    assert PoseTrainingModel.PROGRESS in seen
    assert model.run_dir.startswith(str(tmp_path / "models"))


def test_pause_stops_the_run_and_resume_continues_it(tmp_path):
    model, _ = _model(tmp_path)
    model.start(str(tmp_path / "data"), epochs=10_000)
    assert _wait(lambda: model.progress[0] >= 2)
    assert model.pause() is True
    assert model.state is TrainingState.PAUSED
    run_dir = model.run_dir
    model.resume()
    assert model.run_dir == run_dir
    assert _wait(lambda: model.last_started.get("resume") is True)
    model.pause()


def test_resume_is_refused_while_a_session_records(tmp_path):
    model, gate = _model(tmp_path)
    model.start(str(tmp_path / "data"), epochs=10_000)
    model.pause()
    gate["ready"] = False
    with pytest.raises(RuntimeError, match="recording"):
        model.resume()
    assert model.state is TrainingState.PAUSED


def test_a_failure_is_reported_with_its_reason(tmp_path):
    model, _ = _model(tmp_path, target=_failing_target)
    model.start(str(tmp_path / "data"), epochs=3)
    assert _wait(lambda: model.state is TrainingState.FAILED)
    assert "no labelled training frames" in model.error


def test_pause_when_idle_is_a_no_op(tmp_path):
    model, _ = _model(tmp_path)
    assert model.pause() is True
    assert model.state is TrainingState.IDLE


def test_cancel_discards_a_paused_run(tmp_path):
    model, _ = _model(tmp_path)
    model.start(str(tmp_path / "data"), epochs=10_000)
    model.pause()
    os.makedirs(model.run_dir, exist_ok=True)
    run_dir = model.run_dir
    model.cancel()
    assert model.state is TrainingState.IDLE
    assert not os.path.exists(run_dir)


def test_deploy_history_records_and_returns_the_previous_model(tmp_path):
    root = str(tmp_path / "models")
    assert last_deploy(root) is None
    record_deploy(root, previous="/old", deployed="/new")
    record_deploy(root, previous="/new", deployed="/newer")
    assert last_deploy(root)["previous"] == "/new"
    with open(os.path.join(root, DEPLOY_HISTORY), encoding="utf-8") as handle:
        assert len(yaml.safe_load(handle)) == 2


# -- AppModel wiring ----------------------------------------------------------

def test_arming_a_recording_pauses_training(app_model, tmp_path):
    from tools.acquisition.model.app_model_status import SessionRecordingStatus
    training = app_model.pose_training
    training.models_root = str(tmp_path / "models")
    training._context = multiprocessing.dummy
    training._target = _fake_target
    training.start(str(tmp_path / "data"), epochs=10_000)
    assert _wait(lambda: training.progress[0] >= 1)
    app_model._publish_session_recording_transition(
        SessionRecordingStatus.READY, SessionRecordingStatus.ARMING)
    assert training.state is TrainingState.PAUSED


def test_deploy_is_refused_while_not_ready_and_writes_no_history(app_model, tmp_path):
    from tools.acquisition.model.app_model_status import SessionRecordingStatus
    training = app_model.pose_training
    root = training.models_root = str(tmp_path / "models")
    app_model._recording_session.status = SessionRecordingStatus.RECORDING
    with pytest.raises(RuntimeError):
        app_model.deploy_pose_model(str(tmp_path / "model"))
    assert not os.path.exists(os.path.join(root, DEPLOY_HISTORY))


def test_the_real_child_trains_a_dataset_into_a_live_model_directory(tmp_path, monkeypatch):
    """Spawned process, real dataset and trainer, one epoch on the CPU."""
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from autotrainer.inference import pose_plugins
    from autotrainer.inference.keypoint.dataset import FRAME_SIZE, LABELLED, PoseDataset

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    data = PoseDataset.create(str(tmp_path / "sets"), "tiny", ["a", "b"])
    for session in ("s1_left", "s2_left"):
        data.add_image(numpy.full((FRAME_SIZE, FRAME_SIZE), 90, "uint8"),
                               session[:2], session, status=LABELLED,
                               points=numpy.array([[40, 50], [numpy.nan, numpy.nan]]))
    data.set_val("s2", True)
    data.save()

    model = PoseTrainingModel(str(tmp_path / "models"))
    model.start(data.path, epochs=1)
    assert _wait(lambda: model.state in (TrainingState.FINISHED, TrainingState.FAILED),
                 timeout=120)
    assert model.state is TrainingState.FINISHED, model.error + "\n".join(model.log_lines)
    assert pose_plugins.KEYPOINT_PLUGIN.claims(model.run_dir)
    with open(os.path.join(model.run_dir, "keypoint_pose.yaml"), encoding="utf-8") as handle:
        training = yaml.safe_load(handle)["training"]
    assert training["content_hash"] == data.content_hash()
    assert training["val_sessions"] == ["s2"]


def _ignores_stop(dataset_path, run_dir, epochs, resume, messages, stop_event):
    stop_event.wait()
    time.sleep(0.05)


def test_a_child_that_exits_after_a_stop_request_is_paused_not_failed(tmp_path):
    model, _ = _model(tmp_path, target=_ignores_stop)
    model.start(str(tmp_path / "data"), epochs=5)
    assert model.pause() is True
    time.sleep(0.3)
    assert model.state is TrainingState.PAUSED


def test_a_stop_before_training_begins_exits_without_training(tmp_path):
    import queue
    import threading
    pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    from autotrainer.inference.keypoint.dataset import FRAME_SIZE, LABELLED, PoseDataset
    from tools.acquisition.model.pose_training_model import run_training
    data = PoseDataset.create(str(tmp_path / "sets"), "tiny", ["a"])
    data.add_image(numpy.zeros((FRAME_SIZE, FRAME_SIZE), "uint8"), "s1", "s1_left",
                   points=numpy.array([[10, 10]]), status=LABELLED)
    data.save()
    messages, stop = queue.Queue(), threading.Event()
    stop.set()
    run_training(data.path, str(tmp_path / "run"), 5, False, messages, stop)
    kinds = []
    while not messages.empty():
        kinds.append(messages.get())
    assert ("done", {"status": "stopped", "epoch": 0, "seconds": 0.0}) in kinds
    assert not os.path.exists(tmp_path / "run" / "weights.pt")


def test_a_gpu_job_is_stopped_when_training_pauses_for_a_recording(tmp_path):
    import threading
    from tools.acquisition.model.pose_training_model import JobCancelled
    model = PoseTrainingModel(str(tmp_path / "models"))
    outcome = {}

    def job():
        try:
            model.run_job(time.sleep, 60)
            outcome["result"] = "finished"
        except JobCancelled:
            outcome["result"] = "cancelled"

    thread = threading.Thread(target=job)
    started = time.monotonic()
    thread.start()
    assert _wait(lambda: model.job_running, timeout=30)
    assert model.pause() is True
    thread.join(30)
    assert outcome.get("result") == "cancelled"
    assert time.monotonic() - started < 45


def test_resume_is_blocked_with_a_reason_once_labels_change(tmp_path):
    numpy = pytest.importorskip("numpy")
    from autotrainer.inference.keypoint.dataset import FRAME_SIZE, LABELLED, PoseDataset
    from tools.acquisition.model.pose_training_model import RUN_FILE
    data = PoseDataset.create(str(tmp_path / "sets"), "tiny", ["a"])
    frame = data.add_image(numpy.zeros((FRAME_SIZE, FRAME_SIZE), "uint8"), "s1", "s1_left",
                           points=numpy.array([[10, 10]]), status=LABELLED)
    data.save()
    model, _ = _model(tmp_path)
    model.start(data.path, epochs=10_000)
    model.pause()
    os.makedirs(model.run_dir, exist_ok=True)
    with open(os.path.join(model.run_dir, RUN_FILE), "w", encoding="utf-8") as handle:
        yaml.safe_dump({"content_hash": data.content_hash(), "val_sessions": []}, handle)
    assert model.resume_blocker() == ""
    data.set_points(frame, numpy.array([[11, 10]]))
    data.save()
    assert "changed" in model.resume_blocker()
