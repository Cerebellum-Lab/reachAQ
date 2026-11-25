"""Training a live pose model from inside the app, without disturbing a recording.

Training runs in a spawned child process, so a crash in torch or the GPU
driver cannot take the acquisition process with it, and the GPU memory it held
is returned when the process exits. A thread drains the child's messages and
publishes them as property changes; views must marshal those to the Qt thread.

Training must never share the GPU with a recording. Start and Resume are
refused unless the session is READY, and AppModel pauses a running training
when a recording arms, before the cameras start. Pausing asks the child to
checkpoint and exit (keypoint.train discards the partial epoch, so the resumed
run ends where an uninterrupted one would) and waits for it to be gone.

A run directory is a model directory: once finished it holds weights.pt and
keypoint_pose.yaml and the keypoint plugin claims it.
"""

import datetime
import enum
import os
import queue
import shutil
import threading
import traceback
import typing

import yaml

from reachaq.core import ObservableObject
from reachaq.core.logging import get_verbose_logger
from reachaq.core.multiproc import get_mp_ctx

logger = get_verbose_logger(__name__)

POSE_TRAINING_ROOT = os.path.join(os.path.expanduser("~"), "reachAQ", "pose_training")
DATASETS_ROOT = os.path.join(POSE_TRAINING_ROOT, "datasets")
MODELS_ROOT = os.path.join(POSE_TRAINING_ROOT, "models")
DEPLOY_HISTORY = "deploy_history.yaml"
RUN_FILE = "run.yaml"
DEFAULT_EPOCHS = 300
# How long a pause waits for the child after asking it to stop. A step on the
# T1000 takes well under a second; this only bounds a hung child.
PAUSE_TIMEOUT_S = 30.0


class TrainingState(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"


# -- the child --------------------------------------------------------------

def run_training(dataset_path: str, run_dir: str, epochs: int, resume: bool,
                 messages, stop_event) -> None:
    """Child process entry point: train one run and report through `messages`.

    Messages are (kind, payload): ("started", {...}), ("progress", (epoch,
    epochs, line)), ("log", text), ("done", result) or ("error", text).

    A stop requested before training begins - a recording arming while this
    process is still importing torch or reading frames - ends the run here,
    before CUDA is initialised, rather than at the first training step.
    """
    stopped = ("done", {"status": "stopped", "epoch": 0, "seconds": 0.0})
    if stop_event.is_set():
        messages.put(stopped)
        return
    try:
        import torch

        from reachaq.inference.keypoint.dataset import PoseDataset
        from reachaq.inference.keypoint.train import train_arrays

        dataset = PoseDataset.open(dataset_path)
        content_hash = dataset.content_hash()
        run_file = os.path.join(run_dir, RUN_FILE)
        if resume and os.path.isfile(run_file):
            with open(run_file, encoding="utf-8") as handle:
                recorded = yaml.safe_load(handle) or {}
            if (recorded.get("content_hash") != content_hash
                    or recorded.get("val_sessions") != dataset.val_sessions):
                raise ValueError("the labels or the held-out sessions changed since this "
                                 "run was paused; start a new run to train on them")
        provenance = {"dataset": dataset.name, "dataset_path": dataset.path,
                      "content_hash": content_hash, "val_sessions": dataset.val_sessions,
                      "trained": datetime.datetime.now().isoformat(timespec="seconds")}
        os.makedirs(run_dir, exist_ok=True)
        with open(run_file, "w", encoding="utf-8") as handle:
            yaml.safe_dump({**provenance, "epochs": epochs}, handle, sort_keys=False)

        train_frames, train_points = dataset.arrays("train")
        val_frames, val_points = dataset.arrays("val")
        if stop_event.is_set():
            messages.put(stopped)
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        messages.put(("started", {"resume": resume, "device": device,
                                  "train": len(train_frames), "val": len(val_frames)}))

        def log(text):
            messages.put(("log", str(text)))

        result = train_arrays(
            train_frames, train_points, val_frames, val_points, dataset.bodyparts, run_dir,
            epochs=epochs, device=device, resume=resume, provenance=provenance,
            progress=lambda epoch, total, line: messages.put(("progress", (epoch, total, line))),
            should_stop=stop_event.is_set, log=log)
        messages.put(("done", {"status": result["status"], "epoch": result["epoch"],
                               "seconds": result["seconds"]}))
    except Exception as err:  # noqa: BLE001 - reported to the operator, not raised
        messages.put(("error", f"{type(err).__name__}: {err}"))
        messages.put(("log", traceback.format_exc()))


# -- one-off GPU jobs --------------------------------------------------------
#
# Pre-labelling, evaluation and the speed check load a model onto the GPU. They
# run in a spawned process like training does, so CUDA is never initialised in
# the acquisition process and the memory is returned when the job ends.

def prelabel_frames(model_path: str, dataset_path: str,
                    frames: typing.Sequence[str]) -> typing.Dict[str, list]:
    """Live-model predictions for these frames, parts below the gate absent."""
    import numpy

    from reachaq.inference.keypoint.dataset import PoseDataset
    from reachaq.inference.keypoint.evaluate import match_parts, open_model, predict_with

    dataset = PoseDataset.open(dataset_path)
    model, gate = open_model(model_path, batch_size=1)
    images = numpy.stack([dataset.image(frame) for frame in frames])
    predicted, _ = match_parts(predict_with(model, images), model.body_parts, dataset.bodyparts)
    points = predicted[..., :2].copy()
    points[predicted[..., 2] < gate] = numpy.nan
    return {frame: points[i].tolist() for i, frame in enumerate(frames)}


def evaluate_models(dataset_path: str, model_paths: typing.Sequence[str]) -> typing.List[dict]:
    """Score each model on the dataset's held-out frames; a failure is reported, not raised."""
    from reachaq.inference.keypoint.dataset import PoseDataset
    from reachaq.inference.keypoint.evaluate import evaluate_model

    dataset = PoseDataset.open(dataset_path)
    results = []
    for path in model_paths:
        try:
            results.append(evaluate_model(path, dataset))
        except Exception as err:  # noqa: BLE001 - shown in the table
            results.append({"model": path, "error": f"{type(err).__name__}: {err}"})
    return results


def measure_model_speed(model_path: str) -> dict:
    from reachaq.inference.keypoint.evaluate import measure_speed
    return measure_speed(model_path)


class JobCancelled(RuntimeError):
    """A GPU job was stopped because a recording armed."""


def _job_entry(function, args, messages) -> None:
    try:
        messages.put(("ok", function(*args)))
    except BaseException as err:  # noqa: BLE001 - reported to the caller
        messages.put(("error", f"{type(err).__name__}: {err}"))


# -- deploy history ---------------------------------------------------------

def _read_history(models_root: str) -> typing.List[dict]:
    path = os.path.join(models_root, DEPLOY_HISTORY)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return list(yaml.safe_load(handle) or [])


def record_deploy(models_root: str, previous: str, deployed: str) -> None:
    history = _read_history(models_root)
    history.append({"previous": previous, "deployed": deployed,
                    "when": datetime.datetime.now().isoformat(timespec="seconds")})
    os.makedirs(models_root, exist_ok=True)
    path = os.path.join(models_root, DEPLOY_HISTORY)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        yaml.safe_dump(history, handle, sort_keys=False)
    os.replace(path + ".tmp", path)


def last_deploy(models_root: str) -> typing.Optional[dict]:
    history = _read_history(models_root)
    return history[-1] if history else None


# -- the run manager --------------------------------------------------------

class PoseTrainingModel(ObservableObject):
    STATE = "state"
    PROGRESS = "progress"
    LOG = "log"

    def __init__(self, models_root: str = MODELS_ROOT,
                 is_ready: typing.Callable[[], bool] = lambda: True,
                 context=None, target=run_training):
        super().__init__()
        self.models_root = models_root
        self.datasets_root = DATASETS_ROOT
        self._is_ready = is_ready
        self._context = context
        self._target = target
        self._lock = threading.RLock()
        self._state = TrainingState.IDLE
        self._progress: typing.Tuple[int, int, dict] = (0, 0, {})
        self._process = None
        self._stop_event = None
        self._drain_thread = None
        self._job = None
        self._job_cancelled = False
        self._dataset_path = ""
        self._epochs = DEFAULT_EPOCHS
        self.run_dir = ""
        self.error = ""
        self.result: dict = {}
        self.last_started: dict = {}
        self.log_lines: typing.List[str] = []

    def is_ready(self) -> bool:
        """Whether the GPU may be used now: no session is recording."""
        return bool(self._is_ready())

    @property
    def job_running(self) -> bool:
        job = self._job
        return job is not None and job.is_alive()

    def run_job(self, function, *args):
        """Run one of the GPU jobs above in a spawned process; blocks for its result.

        Refused unless the session is READY. pause() - so a recording arming -
        terminates a running job, which then raises JobCancelled here.
        """
        with self._lock:
            if not self._is_ready():
                raise RuntimeError("this is unavailable while a session is recording")
            if self.job_running:
                raise RuntimeError("another job is still running")
            context = self._context or get_mp_ctx()
            messages = context.Queue()
            job = context.Process(target=_job_entry, args=(function, args, messages),
                                  name="pose-training-job")
            job.daemon = True
            self._job_cancelled = False
            job.start()
            self._job = job
        try:
            while True:
                try:
                    kind, payload = messages.get(timeout=0.2)
                    break
                except queue.Empty:
                    if job.is_alive():
                        continue
                try:
                    kind, payload = messages.get(timeout=0.5)
                    break
                except queue.Empty:
                    if self._job_cancelled:
                        raise JobCancelled("stopped because a recording started") from None
                    raise RuntimeError("the job process exited unexpectedly") from None
        finally:
            job.join(5)
            with self._lock:
                if self._job is job:
                    self._job = None
        if kind == "error":
            raise RuntimeError(payload)
        return payload

    def cancel_job(self) -> None:
        with self._lock:
            job = self._job
            if job is None or not job.is_alive():
                return
            self._job_cancelled = True
            logger.info("stopping a pose training GPU job")
            if hasattr(job, "terminate"):
                job.terminate()
            job.join(10)

    def resume_blocker(self) -> str:
        """Why the paused run cannot resume, or "" if it can.

        A run resumes only on the labels and held-out sessions it started with,
        so its model stays true to what its sidecar says it was trained on.
        """
        if self._state is not TrainingState.PAUSED or not self.run_dir:
            return ""
        run_file = os.path.join(self.run_dir, RUN_FILE)
        if not os.path.isfile(run_file):
            return ""
        from reachaq.inference.keypoint.dataset import PoseDataset
        with open(run_file, encoding="utf-8") as handle:
            recorded = yaml.safe_load(handle) or {}
        try:
            dataset = PoseDataset.open(self._dataset_path)
            unchanged = (recorded.get("content_hash") == dataset.content_hash()
                         and recorded.get("val_sessions") == dataset.val_sessions)
        except (OSError, ValueError) as err:
            return f"the dataset cannot be read: {err}"
        if unchanged:
            return ""
        return ("the labels or held-out sessions have changed since this run was paused, "
                "so it cannot resume; cancel it and start a new run to train on them")

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def progress(self) -> typing.Tuple[int, int, dict]:
        return self._progress

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    def _set_state(self, value: TrainingState) -> None:
        prev, self._state = self._state, value
        self._on_property_changed(self.STATE, value, prev)

    # -- commands ---------------------------------------------------------

    def start(self, dataset_path: str, epochs: int = DEFAULT_EPOCHS) -> None:
        with self._lock:
            if self._state in (TrainingState.RUNNING, TrainingState.PAUSED):
                raise RuntimeError("a training run is already open; cancel it first")
            name = os.path.basename(os.path.normpath(dataset_path))
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_dir = os.path.join(self.models_root, name, stamp)
            self._dataset_path = dataset_path
            self._epochs = int(epochs)
            self._launch(resume=False)

    def resume(self) -> None:
        with self._lock:
            if self._state is not TrainingState.PAUSED:
                raise RuntimeError("there is no paused training run to resume")
            self._launch(resume=True)

    def _launch(self, resume: bool) -> None:
        if not self._is_ready():
            raise RuntimeError("training is unavailable while a session is recording")
        context = self._context or get_mp_ctx()
        messages = context.Queue()
        self._stop_event = context.Event()
        self.error = ""
        self.result = {}
        self._process = context.Process(
            target=self._target, name="pose-training",
            args=(self._dataset_path, self.run_dir, self._epochs, resume,
                  messages, self._stop_event))
        self._process.daemon = True
        self._process.start()
        self._drain_thread = threading.Thread(
            target=self._drain, args=(self._process, messages), name="pose-training-drain",
            daemon=True)
        self._drain_thread.start()
        self._set_state(TrainingState.RUNNING)
        logger.info("pose training %s: %s (%s epochs)", "resumed" if resume else "started",
                    self.run_dir, self._epochs)

    def pause(self, timeout: float = PAUSE_TIMEOUT_S) -> bool:
        """Stop a running training at its checkpoint, and any GPU job; True once
        nothing uses the GPU."""
        with self._lock:
            self.cancel_job()
            if self._state is not TrainingState.RUNNING:
                return True
            stopped = self._stop_child(timeout)
            if self._state is TrainingState.RUNNING:
                self._set_state(TrainingState.PAUSED)
            return stopped

    def cancel(self, timeout: float = PAUSE_TIMEOUT_S) -> None:
        """Stop and discard the open run."""
        with self._lock:
            if self._state is TrainingState.RUNNING:
                self._stop_child(timeout)
            if self._state in (TrainingState.RUNNING, TrainingState.PAUSED,
                               TrainingState.FAILED) and self.run_dir:
                shutil.rmtree(self.run_dir, ignore_errors=True)
            self.run_dir = ""
            self._set_state(TrainingState.IDLE)

    def _stop_child(self, timeout: float) -> bool:
        process, drain = self._process, self._drain_thread
        if process is None:
            return True
        self._stop_event.set()
        process.join(timeout)
        if process.is_alive():
            logger.warning("pose training did not stop within %.0f s; terminating it", timeout)
            if hasattr(process, "terminate"):
                process.terminate()
                process.join(5)
        if drain is not None:
            drain.join(timeout)
        return not process.is_alive()

    # -- messages ---------------------------------------------------------

    def _drain(self, process, messages) -> None:
        finished = False
        while not finished:
            try:
                kind, payload = messages.get(timeout=0.2)
            except queue.Empty:
                if not process.is_alive():
                    # Anything put just before the exit is still readable.
                    try:
                        kind, payload = messages.get(timeout=0.5)
                    except queue.Empty:
                        break
                else:
                    continue
            finished = self._handle(kind, payload)
        if not finished and self._process is process:
            if (self._stop_event is not None and self._stop_event.is_set()
                    and self._state is not TrainingState.FAILED):
                # Asked to stop and gone without saying so (terminated after a
                # pause timed out): its last periodic checkpoint still resumes.
                self._set_state(TrainingState.PAUSED)
                return
            self.error = self.error or "the training process exited unexpectedly"
            self._set_state(TrainingState.FAILED)

    def _handle(self, kind: str, payload) -> bool:
        if kind == "started":
            self.last_started = dict(payload)
        elif kind == "progress":
            prev, self._progress = self._progress, tuple(payload)
            self._on_property_changed(self.PROGRESS, self._progress, prev)
        elif kind == "log":
            self.log_lines.append(payload)
            del self.log_lines[:-500]
            self._on_property_changed(self.LOG, payload, None)
        elif kind == "done":
            self.result = dict(payload)
            if payload.get("status") == "finished":
                self._set_state(TrainingState.FINISHED)
            else:
                self._set_state(TrainingState.PAUSED)
            return True
        elif kind == "error":
            self.error = str(payload)
            logger.error("pose training failed: %s", payload)
            self._set_state(TrainingState.FAILED)
            # The traceback follows as a log message; keep draining for it.
        return False
