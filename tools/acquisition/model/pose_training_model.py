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

from autotrainer.core import ObservableObject
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.multiproc import get_mp_ctx

logger = get_verbose_logger(__name__)

POSE_TRAINING_ROOT = os.path.join(os.path.expanduser("~"), "Autotrainer", "pose_training")
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
    """
    try:
        import torch

        from autotrainer.inference.keypoint.dataset import PoseDataset
        from autotrainer.inference.keypoint.train import train_arrays

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
        self._dataset_path = ""
        self._epochs = DEFAULT_EPOCHS
        self.run_dir = ""
        self.error = ""
        self.result: dict = {}
        self.last_started: dict = {}
        self.log_lines: typing.List[str] = []

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
        """Stop a running training at its checkpoint; True once nothing trains."""
        with self._lock:
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
