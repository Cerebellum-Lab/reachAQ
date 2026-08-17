"""Low-latency stim-camera detector and batched evidence persistence."""

from __future__ import annotations

import dataclasses
import json
import queue
import threading
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy


STIM_EVIDENCE_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class StimRoiDefinition:
    name: str = "first_reach"
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    threshold: float = 0.0
    hysteresis: float = 0.0
    version: int = 1
    runnable: bool = True

    def __post_init__(self):
        if self.name not in {"first_reach", "roi_1", "roi_2"}:
            raise ValueError(f"Unknown stim ROI role: {self.name}")
        if min(self.x, self.y, self.width, self.height) < 0:
            raise ValueError("Stim ROI coordinates cannot be negative")
        if not numpy.isfinite(self.threshold) or not numpy.isfinite(self.hysteresis):
            raise ValueError("Stim ROI threshold and hysteresis must be finite")
        if self.hysteresis < 0:
            raise ValueError("Stim ROI hysteresis cannot be negative")
        if self.name in {"roi_1", "roi_2"} and self.runnable:
            raise ValueError("ROI1/ROI2 detector roles are reserved for a future release")


@dataclasses.dataclass(frozen=True)
class StimCameraDetectionConfiguration:
    enabled: bool = False
    target_fps: float = 900.0
    roi: StimRoiDefinition = dataclasses.field(default_factory=StimRoiDefinition)
    evidence_batch_size: int = 1024
    evidence_queue_batches: int = 32
    pre_event_frames: int = 90
    post_event_frames: int = 180
    preview_fps: float = 15.0

    def __post_init__(self):
        if self.enabled and self.target_fps != 900.0:
            raise ValueError("First Reach stim-camera mode requires 900 Hz")
        if self.evidence_batch_size < 1 or self.evidence_queue_batches < 1:
            raise ValueError("Stim evidence buffers must be positive")
        if self.pre_event_frames < 0 or self.post_event_frames < 0:
            raise ValueError("Stim clip bounds cannot be negative")
        if not 0 < self.preview_fps <= 60:
            raise ValueError("Stim preview rate must be within 0..60 Hz")


@dataclasses.dataclass(frozen=True)
class StimDetectorArm:
    session_generation: int
    operation_id: str
    logical_trial_id: int
    attempt_id: int
    nonce: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]):
        return cls(
            session_generation=int(value["session_generation"]),
            operation_id=str(value["operation_id"]),
            logical_trial_id=int(value["logical_trial_id"]),
            attempt_id=int(value["attempt_id"]),
            nonce=str(value["nonce"]),
        )

    def to_record(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StimTriggerDecision:
    arm: StimDetectorArm
    stim_frame_id: int
    camera_timestamp_ns: int
    frame_perf_time: float
    decision_perf_time: float
    detector_value: float
    threshold: float

    def to_record(self):
        return {
            **self.arm.to_record(),
            "stim_frame_id": self.stim_frame_id,
            "camera_timestamp_ns": self.camera_timestamp_ns,
            "frame_perf_time": self.frame_perf_time,
            "decision_perf_time": self.decision_perf_time,
            "detector_value": self.detector_value,
            "threshold": self.threshold,
            "decision_latency_seconds": self.decision_perf_time - self.frame_perf_time,
        }


class StimCameraDetector:
    """One-shot legacy First Reach transition detector."""

    def __init__(self, configuration: StimCameraDetectionConfiguration):
        self.configuration = configuration
        self._lock = threading.RLock()
        self._arm: Optional[StimDetectorArm] = None
        self._observed_below = False
        self._triggered = False

    @property
    def arm_context(self):
        with self._lock:
            return self._arm

    def arm(self, context: Mapping[str, object]):
        arm = StimDetectorArm.from_mapping(context)
        with self._lock:
            self._arm = arm
            self._observed_below = False
            self._triggered = False
        return arm

    def disarm(self, operation_id=None):
        with self._lock:
            if (
                operation_id is not None
                and self._arm is not None
                and self._arm.operation_id != str(operation_id)
            ):
                return False
            self._arm = None
            self._observed_below = False
            self._triggered = False
            return True

    def process(self, frame, frame_id, camera_timestamp_ns, frame_perf_time):
        value = self.detector_value(frame)
        with self._lock:
            arm = self._arm
            if arm is None or self._triggered:
                return value, None
            threshold = self.configuration.roi.threshold
            if value <= threshold - self.configuration.roi.hysteresis:
                self._observed_below = True
            if not self._observed_below or value <= threshold:
                return value, None
            self._triggered = True
            decision = StimTriggerDecision(
                arm=arm,
                stim_frame_id=int(frame_id),
                camera_timestamp_ns=int(camera_timestamp_ns),
                frame_perf_time=float(frame_perf_time),
                decision_perf_time=time.perf_counter(),
                detector_value=float(value),
                threshold=float(threshold),
            )
            return value, decision

    def detector_value(self, frame):
        roi = self.configuration.roi
        array = numpy.asarray(frame)
        if array.ndim == 3:
            array = array[:, :, 0]
        if roi.width and roi.height:
            array = array[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
        if array.size == 0:
            raise ValueError("Configured stim-camera ROI is empty")
        # Exact retained First Reach metric: sum vertically, average the first
        # five columns of the already-cropped stim frame/ROI.
        columns = numpy.sum(array, axis=0)
        return float(numpy.mean(columns[:min(5, columns.size)]))


_EVIDENCE_DTYPE = numpy.dtype([
    ("frame_id", "<i8"),
    ("camera_timestamp_ns", "<i8"),
    ("frame_perf_time", "<f8"),
    ("decision_perf_time", "<f8"),
    ("detector_value", "<f8"),
    ("threshold", "<f8"),
    ("session_generation", "<i8"),
    ("logical_trial_id", "<i8"),
    ("attempt_id", "<i8"),
    ("armed", "?"),
    ("triggered", "?"),
    ("gap", "?"),
])


class StimEvidenceWriter:
    """Preallocated capture-side buffer with a bounded HDF5 writer queue."""

    def __init__(self, path: Path, configuration: StimCameraDetectionConfiguration):
        self.path = Path(path)
        self.configuration = configuration
        self._buffer = numpy.empty(configuration.evidence_batch_size, dtype=_EVIDENCE_DTYPE)
        self._buffer_count = 0
        self._queue = queue.Queue(maxsize=configuration.evidence_queue_batches)
        self._lock = threading.RLock()
        self._closed = False
        self._drops = 0
        self._clip_drops = 0
        self._clips_written = 0
        self._high_water = 0
        self._write_latency_max = 0.0
        self._thread = threading.Thread(target=self._run, name="StimEvidenceWriter", daemon=True)
        self._thread.start()

    @property
    def diagnostics(self):
        return {
            "dropped_batches": self._drops,
            "dropped_clips": self._clip_drops,
            "clips_written": self._clips_written,
            "queue_high_water": self._high_water,
            "write_latency_max_seconds": self._write_latency_max,
        }

    def append(self, *, frame_id, camera_timestamp_ns, frame_perf_time, value, threshold, arm, decision, gap=False):
        with self._lock:
            if self._closed:
                return
            index = self._buffer_count
            self._buffer[index] = (
                int(frame_id), int(camera_timestamp_ns), float(frame_perf_time),
                float("nan") if decision is None else decision.decision_perf_time,
                float(value), float(threshold),
                0 if arm is None else arm.session_generation,
                0 if arm is None else arm.logical_trial_id,
                0 if arm is None else arm.attempt_id,
                arm is not None,
                decision is not None,
                bool(gap),
            )
            self._buffer_count += 1
            if self._buffer_count == len(self._buffer):
                self._flush_buffer()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._flush_buffer()
        self._queue.put(None)
        self._thread.join(10)
        if self._thread.is_alive():
            raise TimeoutError("Stim evidence writer did not stop")
        if self._drops or self._clip_drops:
            raise RuntimeError(
                "Stim evidence queue overflow: "
                f"batches={self._drops} clips={self._clip_drops}"
            )

    def queue_clip(self, decision: StimTriggerDecision, frames) -> bool:
        """Queue one bounded clip without performing file I/O on capture."""
        payload = {
            "decision": decision.to_record(),
            "frames": tuple(frames),
        }
        try:
            self._queue.put_nowait(("clip", payload))
            self._high_water = max(self._high_water, self._queue.qsize())
            return True
        except queue.Full:
            self._clip_drops += 1
            return False

    def _flush_buffer(self):
        if not self._buffer_count:
            return
        batch = self._buffer[:self._buffer_count].copy()
        self._buffer_count = 0
        try:
            self._queue.put_nowait(("evidence", batch))
            self._high_water = max(self._high_water, self._queue.qsize())
        except queue.Full:
            self._drops += 1

    def _run(self):
        import h5py
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "w") as store:
            store.attrs["schema_version"] = STIM_EVIDENCE_SCHEMA_VERSION
            store.attrs["target_fps"] = self.configuration.target_fps
            store.attrs["roi"] = json.dumps(
                dataclasses.asdict(self.configuration.roi), sort_keys=True,
            )
            dataset = store.create_dataset(
                "evidence", shape=(0,), maxshape=(None,), dtype=_EVIDENCE_DTYPE,
                chunks=(self.configuration.evidence_batch_size,),
            )
            count = 0
            while True:
                item = self._queue.get()
                if item is None:
                    break
                started = time.perf_counter()
                kind, payload = item
                if kind == "evidence":
                    batch = payload
                    dataset.resize((count + len(batch),))
                    dataset[count:count + len(batch)] = batch
                    count += len(batch)
                elif kind == "clip":
                    self._write_clip(store, payload)
                else:
                    raise RuntimeError(f"Unknown stim evidence item: {kind}")
                self._write_latency_max = max(
                    self._write_latency_max, time.perf_counter() - started
                )
            store.attrs.update(self.diagnostics)
            store.flush()

    def _write_clip(self, store, payload):
        decision = payload["decision"]
        group = store.require_group("clips").create_group(decision["operation_id"])
        group.attrs["decision"] = json.dumps(decision, sort_keys=True)
        frames = payload["frames"]
        group.create_dataset(
            "frame_id", data=numpy.asarray([item[0] for item in frames], dtype="<i8"),
        )
        group.create_dataset(
            "camera_timestamp_ns",
            data=numpy.asarray([item[1] for item in frames], dtype="<i8"),
        )
        group.create_dataset(
            "frame_perf_time",
            data=numpy.asarray([item[2] for item in frames], dtype="<f8"),
        )
        if frames:
            group.create_dataset(
                "frames",
                data=numpy.stack([item[3] for item in frames]),
                chunks=True,
                compression="lzf",
            )
        self._clips_written += 1
