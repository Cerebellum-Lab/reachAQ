from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional
from uuid import UUID

import h5py
import numpy as np

from autotrainer.core import ProjectInfo
from tools.acquisition.model.session_boundary import SessionBoundary


class _SessionLogHandler(logging.Handler):
    def __init__(self, recorder: "SessionDataRecorder"):
        super().__init__(logging.NOTSET)
        self._recorder = recorder
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._recorder.add_current_log(record.created, self.format(record))
        except Exception:
            self.handleError(record)


class SessionDataRecorder:
    """Capture auxiliary streams against the recorded primary-camera boundary."""

    def __init__(
        self,
        analysis,
        nidaq_monitor,
        laser_model,
        *,
        system_message_handler=None,
        hardware_model=None,
        device_event_capacity: int = 100_000,
    ):
        self._analysis = analysis
        self._nidaq_monitor = nidaq_monitor
        self._laser_model = laser_model
        self._lock = threading.RLock()
        self._armed = False
        self._project: Optional[ProjectInfo] = None
        self._start_perf: Optional[float] = None
        self._start_wall: Optional[float] = None
        self._boundary: Optional[SessionBoundary] = None
        self._device_rows = deque(maxlen=device_event_capacity)
        self._device_event_capacity = int(device_event_capacity)
        self._device_event_overruns = 0
        self._device_events_since_start = 0
        self._laser_rows = []
        self._log_rows = []
        self._nidaq_chunks = []
        self._nidaq_last_index = None
        self._nidaq_stop = threading.Event()
        self._nidaq_thread: Optional[threading.Thread] = None
        self._source_manifest = ()
        self._source_results = {}

        self._system_message_handler = system_message_handler
        self._hardware_model = hardware_model
        if system_message_handler is not None:
            system_message_handler.decoded_message_received += self._on_device_message
        if hardware_model is not None:
            hardware_model.device_event += self._on_hardware_device_event
        laser_model.trace_received += self._on_laser_trace
        laser_model.property_changed += self._on_laser_property_changed
        self._log_handler = _SessionLogHandler(self)
        logging.getLogger().addHandler(self._log_handler)

    def arm(self, project: ProjectInfo, *, source_manifest=()) -> None:
        self.abort()
        with self._lock:
            self._armed = True
            # BehaviorAlgorithm assigns the next session index immediately
            # after arming and before enabling the shared camera trigger.
            # Keep this object reference so the recorder follows that atomic
            # session assignment without racing the first camera frame.
            self._project = project
            self._device_events_since_start = 0
            self._laser_rows = []
            self._log_rows = []
            self._nidaq_chunks = []
            self._nidaq_last_index = None
            self._source_manifest = tuple(source_manifest)
            self._source_results = {}
            self._start_perf = None
            self._start_wall = None
            self._boundary = None
            self._nidaq_stop.clear()
        thread = threading.Thread(
            target=self._poll_nidaq,
            name="session-nidaq-recorder",
            daemon=True,
        )
        self._nidaq_thread = thread
        thread.start()

    def commit_start(
        self,
        perf_time: float,
        wall_time: float,
        *,
        boundary: Optional[SessionBoundary] = None,
    ) -> None:
        with self._lock:
            if not self._armed:
                return
            self._start_perf = float(perf_time)
            self._start_wall = float(wall_time)
            self._boundary = boundary
            self._device_events_since_start = sum(
                1
                for row in self._device_rows
                if row[0] >= self._start_perf
            )

    def stop(self, end_perf: float):
        self._stop_nidaq_thread()
        with self._lock:
            if not self._armed or self._project is None or self._start_perf is None:
                self._clear_locked()
                return None
            project = self._project
            start_perf = self._start_perf
            start_wall = self._start_wall
            end_perf = max(start_perf, float(end_perf))
            boundary = self._boundary
            if boundary is not None:
                boundary = boundary.with_end(end_perf)
            device_rows = tuple(self._device_rows)
            device_event_overruns = max(
                0,
                self._device_events_since_start - self._device_event_capacity,
            )
            laser_rows = tuple(self._laser_rows)
            log_rows = tuple(self._log_rows)
            nidaq_chunks = tuple(self._nidaq_chunks)
            timing_plan = self._nidaq_monitor.timing_plan
            source_manifest = self._source_manifest
            source_results = dict(self._source_results)
            self._clear_locked()
        return self._write_session(
            project,
            start_perf,
            start_wall,
            end_perf,
            device_rows,
            laser_rows,
            log_rows,
            nidaq_chunks,
            timing_plan,
            device_event_overruns=device_event_overruns,
            source_manifest=source_manifest,
            source_results=source_results,
            boundary=boundary,
        )

    def abort(self) -> None:
        self._stop_nidaq_thread()
        with self._lock:
            self._clear_locked()

    def close(self) -> None:
        self.abort()
        if self._system_message_handler is not None:
            self._system_message_handler.decoded_message_received -= self._on_device_message
        if self._hardware_model is not None:
            self._hardware_model.device_event -= self._on_hardware_device_event
        logging.getLogger().removeHandler(self._log_handler)
        with self._lock:
            self._device_rows.clear()

    def add_log(self, perf_time: float, wall_time: float, message: str) -> None:
        with self._lock:
            if self._armed:
                self._log_rows.append((float(perf_time), float(wall_time), message))

    def add_current_log(self, wall_time: float, message: str) -> None:
        """Timestamp a log only when a session is actively collecting it."""
        with self._lock:
            if not self._armed:
                return
            self._log_rows.append((
                time.perf_counter(),
                float(wall_time),
                message,
            ))

    def set_source_result(
        self,
        source_id: str,
        *,
        sample_count: Optional[int] = None,
        path: Optional[str] = None,
        failure: str = "",
    ) -> None:
        with self._lock:
            if not self._armed:
                return
            self._source_results[str(source_id)] = {
                "sampleCount": (
                    None if sample_count is None else int(sample_count)
                ),
                "path": path,
                "failure": str(failure or ""),
            }

    def _clear_locked(self) -> None:
        self._armed = False
        self._project = None
        self._start_perf = None
        self._start_wall = None
        self._boundary = None
        self._device_events_since_start = 0
        self._laser_rows = []
        self._log_rows = []
        self._nidaq_chunks = []
        self._nidaq_last_index = None
        self._source_manifest = ()
        self._source_results = {}

    def _on_device_message(
        self,
        kind,
        data,
        perf_time: float,
        wall_time: float,
    ) -> None:
        context = None
        if (
            getattr(kind, "name", None) == "ACKNOWLEDGE"
            and isinstance(data, (tuple, list))
            and data
        ):
            context = data[0]
        self._append_device_event(
            perf_time,
            wall_time,
            "inbound",
            kind,
            data,
            context,
            None,
        )

    def _on_hardware_device_event(
        self,
        direction,
        kind,
        data,
        context,
        target,
        perf_time,
        wall_time,
    ) -> None:
        self._append_device_event(
            perf_time,
            wall_time,
            direction,
            kind,
            data,
            context,
            target,
        )

    def _append_device_event(
        self,
        perf_time,
        wall_time,
        direction,
        kind,
        data,
        context,
        target,
    ) -> None:
        kind_name = kind.name if isinstance(kind, Enum) else str(kind)
        target_name = target.name if isinstance(target, Enum) else target
        device_timestamp = getattr(data, "timestamp", None)
        device_index = getattr(data, "index", None)
        payload_json = self._payload_json(data)
        with self._lock:
            if len(self._device_rows) == self._device_rows.maxlen:
                self._device_event_overruns += 1
            if (
                self._start_perf is not None
                and float(perf_time) >= self._start_perf
            ):
                self._device_events_since_start += 1
            self._device_rows.append((
                float(perf_time),
                float(wall_time),
                str(direction),
                kind_name,
                "" if target_name is None else str(target_name),
                "" if context is None else str(context),
                device_timestamp,
                device_index,
                payload_json,
            ))

    @classmethod
    def _payload_json(cls, value) -> str:
        return json.dumps(
            cls._json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _json_safe(cls, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, UUID):
            return str(value)
        if dataclasses.is_dataclass(value):
            return {
                field.name: cls._json_safe(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if hasattr(value, "__dict__"):
            return {
                str(key): cls._json_safe(item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        return str(value)

    def _on_laser_trace(self, trace) -> None:
        perf_time = time.perf_counter()
        wall_time = time.time()
        x_values = trace.x_values or (0.0,)
        count = max(
            len(x_values),
            len(trace.command_volts),
            len(trace.diode_volts),
            len(trace.command_copy_volts),
            1,
        )
        with self._lock:
            if not self._armed:
                return
            for index in range(count):
                offset = float(x_values[index]) if index < len(x_values) else 0.0
                self._laser_rows.append((
                    perf_time + offset,
                    wall_time + offset,
                    "trace",
                    trace.channel_id.value,
                    trace.source,
                    self._value_at(trace.command_volts, index),
                    self._value_at(trace.diode_volts, index),
                    self._value_at(trace.command_copy_volts, index),
                    trace.output_name,
                    trace.output_value,
                ))

    def _on_laser_property_changed(self, name, value, _) -> None:
        if name != self._laser_model.LAST_FEEDBACK_SAMPLE or value is None:
            return
        with self._lock:
            if self._armed:
                self._laser_rows.append((
                    time.perf_counter(),
                    time.time(),
                    "feedback",
                    value.channel_id.value,
                    "",
                    value.command_volts,
                    value.diode_volts,
                    value.command_copy_volts,
                    "",
                    None,
                ))

    @staticmethod
    def _value_at(values, index):
        return values[index] if index < len(values) else math.nan

    def _poll_nidaq(self) -> None:
        while not self._nidaq_stop.wait(0.01):
            self._copy_nidaq_once()
        self._copy_nidaq_once()

    def _copy_nidaq_once(self) -> None:
        try:
            ring = self._nidaq_monitor.sample_ring
            destination = np.empty(
                (max(1, len(ring.channel_names)), ring.capacity),
                dtype=np.float32,
            )
            read = ring.copy_since(self._nidaq_last_index, destination)
            if read is None:
                return
            self._nidaq_last_index = read.end_sample_index
            if read.sample_count <= 0:
                return
            values = destination[:len(ring.channel_names), :read.sample_count].copy()
            indices = np.arange(
                read.start_sample_index,
                read.end_sample_index,
                dtype=np.int64,
            )
            # One task-start anchor defines the whole hardware epoch. Block
            # publication timestamps remain diagnostics and never re-anchor the
            # sample timeline.
            perf_times = read.perf_times(indices)
            wall_times = read.wall_times(indices)
            with self._lock:
                if self._armed:
                    self._nidaq_chunks.append((
                        indices,
                        perf_times,
                        wall_times,
                        values,
                        tuple(ring.channel_names),
                        read.sample_rate_hz,
                        read.epoch,
                        read.gap_count,
                        read.overrun_samples,
                        read.source_perf_time,
                        read.source_wall_time,
                    ))
        except Exception:
            logging.getLogger(__name__).exception("Unable to collect NI-DAQ session samples")

    def _stop_nidaq_thread(self) -> None:
        thread = self._nidaq_thread
        if thread is None:
            return
        self._nidaq_stop.set()
        if thread is not threading.current_thread():
            thread.join()
        self._nidaq_thread = None
        self._nidaq_stop.clear()

    @staticmethod
    def _write_session(
        project,
        start_perf,
        start_wall,
        end_perf,
        device_rows,
        laser_rows,
        log_rows,
        nidaq_chunks,
        timing_plan=None,
        *,
        device_event_overruns=0,
        source_manifest=(),
        source_results=None,
        boundary: Optional[SessionBoundary] = None,
    ):
        session_dir = Path(project.get_session_path().location)
        streams_dir = session_dir / "streams"
        logs_dir = session_dir / "logs"
        streams_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        device_rows = tuple(
            row for row in device_rows if start_perf <= row[0] <= end_perf
        )
        device_rows = tuple(
            SessionDataRecorder._normalize_device_row(row)
            for row in device_rows
        )
        laser_rows = tuple(
            row for row in laser_rows if start_perf <= row[0] <= end_perf
        )
        log_rows = tuple(
            row for row in log_rows if start_perf <= row[0] <= end_perf
        )
        nidaq_perf = tuple(
            float(sample_perf)
            for chunk in nidaq_chunks
            for sample_perf in chunk[1]
            if start_perf <= sample_perf <= end_perf
        )
        camera_nidaq_alignment = SessionDataRecorder._match_camera_nidaq_edge(
            boundary,
            start_perf,
            nidaq_chunks,
        )
        tone_confirmation = SessionDataRecorder._correlate_tone_confirmations(
            device_rows,
            nidaq_chunks,
        )

        SessionDataRecorder._write_csv(
            streams_dir / "device.csv",
            (
                "perf_time",
                "offset_seconds",
                "wall_time",
                "direction",
                "kind",
                "target",
                "context",
                "device_timestamp",
                "device_index",
                "payload_json",
            ),
            (
                (
                    perf,
                    perf - start_perf,
                    wall,
                    direction,
                    kind,
                    target,
                    context,
                    device_timestamp,
                    device_index,
                    payload,
                )
                for (
                    perf,
                    wall,
                    direction,
                    kind,
                    target,
                    context,
                    device_timestamp,
                    device_index,
                    payload,
                ) in device_rows
            ),
        )
        SessionDataRecorder._write_csv(
            streams_dir / "laser.csv",
            ("perf_time", "offset_seconds", "wall_time", "event", "channel", "source",
             "command_volts", "diode_volts", "command_copy_volts", "output_name",
             "output_value"),
            (
                (
                    perf,
                    perf - start_perf,
                    wall,
                    event,
                    channel,
                    source,
                    command,
                    diode,
                    copy,
                    output_name,
                    output_value,
                )
                for (
                    perf,
                    wall,
                    event,
                    channel,
                    source,
                    command,
                    diode,
                    copy,
                    output_name,
                    output_value,
                ) in (
                    SessionDataRecorder._normalize_laser_row(row)
                    for row in laser_rows
                )
            ),
        )

        with (logs_dir / "session.log").open("w", encoding="utf-8") as stream:
            stream.write(
                f"# recording_start_perf={start_perf:.9f} "
                f"recording_start_wall={start_wall:.9f} "
                f"recording_end_perf={end_perf:.9f}\n"
            )
            for perf, _, message in log_rows:
                stream.write(f"[+{perf - start_perf:.6f}s] {message}\n")

        SessionDataRecorder._write_nidaq(
            streams_dir / "nidaq.h5",
            start_perf,
            start_wall,
            end_perf,
            nidaq_chunks,
            timing_plan,
        )
        finalized_sources = SessionDataRecorder._finalize_source_manifest(
            session_dir,
            source_manifest,
            {} if source_results is None else source_results,
            start_perf=start_perf,
            device_perf=tuple(row[0] for row in device_rows),
            laser_perf=tuple(row[0] for row in laser_rows),
            log_perf=tuple(row[0] for row in log_rows),
            nidaq_perf=nidaq_perf,
            nidaq_chunks=nidaq_chunks,
            device_event_overruns=device_event_overruns,
        )
        incomplete_reasons = []
        if device_event_overruns:
            incomplete_reasons.append(
                f"decoded device event ring overran by {device_event_overruns} event(s)"
            )
        for source in finalized_sources:
            source_id = source.get("id", "unknown")
            if source["persistenceStatus"] != "written":
                incomplete_reasons.append(
                    f"{source_id} persistence {source['persistenceStatus']}"
                    + (
                        ""
                        if not source.get("failure")
                        else f": {source['failure']}"
                    )
                )
            if source["gapCount"]:
                incomplete_reasons.append(
                    f"{source_id} reported {source['gapCount']} acquisition gap(s)"
                )
            if source["overrunCount"]:
                incomplete_reasons.append(
                    f"{source_id} overran by {source['overrunCount']} sample/event(s)"
                )
        alignment = {
            "schemaVersion": 1,
            "canonicalBoundary": {
                "source": "primary_camera_recorded_frames",
                "clock": "time.perf_counter",
                "startPerfTime": start_perf,
                "endPerfTime": end_perf,
                "startWallTime": start_wall,
                "endWallTime": start_wall + (end_perf - start_perf),
                "primaryCamera": (
                    None if boundary is None else boundary.primary_camera
                ),
                "primaryFrameId": (
                    None if boundary is None else boundary.primary_frame_id
                ),
                "cameraWhen": None if boundary is None else boundary.camera_when,
                "nidaqSampleIndex": camera_nidaq_alignment.get(
                    "matchedSampleIndex"
                ),
            },
            "streams": {
                "nidaq": SessionDataRecorder._alignment_entry(
                    "nidaq.h5",
                    nidaq_perf,
                    start_perf,
                    "NI-DAQ ring reconstructed on time.perf_counter",
                ),
                "device": SessionDataRecorder._alignment_entry(
                    "device.csv",
                    tuple(row[0] for row in device_rows),
                    start_perf,
                    "device monotonic timestamp",
                ),
                "laser": SessionDataRecorder._alignment_entry(
                    "laser.csv",
                    tuple(row[0] for row in laser_rows),
                    start_perf,
                    "time.perf_counter",
                ),
                "logs": SessionDataRecorder._alignment_entry(
                    "../logs/session.log",
                    tuple(row[0] for row in log_rows),
                    start_perf,
                    "time.perf_counter",
                ),
            },
            "nidaqTiming": (
                None if timing_plan is None else dataclasses.asdict(timing_plan)
            ),
            "deviceEventOverruns": int(device_event_overruns),
            "sessionComplete": not incomplete_reasons,
            "incompleteReasons": incomplete_reasons,
            "enabledSources": finalized_sources,
            "cameraNidaqAlignment": camera_nidaq_alignment,
            "toneConfirmation": tone_confirmation,
        }
        with (streams_dir / "alignment.json").open("w", encoding="utf-8") as stream:
            json.dump(alignment, stream, indent=2)
            stream.write("\n")
        return {
            "cameraNidaqAlignment": camera_nidaq_alignment,
            "toneConfirmation": tone_confirmation,
            "deviceEventOverruns": int(device_event_overruns),
            "sessionComplete": not incomplete_reasons,
            "incompleteReasons": tuple(incomplete_reasons),
            "enabledSources": finalized_sources,
        }

    @staticmethod
    def _finalize_source_manifest(
        session_dir,
        source_manifest,
        source_results,
        *,
        start_perf,
        device_perf,
        laser_perf,
        log_perf,
        nidaq_perf,
        nidaq_chunks,
        device_event_overruns,
    ):
        gap_count = max(
            (int(chunk[7]) for chunk in nidaq_chunks),
            default=0,
        )
        nidaq_overruns = sum(
            int(chunk[8]) for chunk in nidaq_chunks
        )
        stream_stats = {
            "device": (
                device_perf,
                int(device_event_overruns),
                0,
            ),
            "laser_outputs": (laser_perf, 0, 0),
            "session_logs": (log_perf, 0, 0),
        }
        finalized = []
        for raw_source in source_manifest:
            source = dict(raw_source)
            source_id = str(source.get("id", ""))
            result = source_results.get(source_id, {})
            perf_times = ()
            overrun_count = 0
            source_gap_count = 0
            if source_id.startswith("nidaq."):
                perf_times = nidaq_perf
                overrun_count = nidaq_overruns
                source_gap_count = gap_count
            elif source_id in stream_stats:
                (
                    perf_times,
                    overrun_count,
                    source_gap_count,
                ) = stream_stats[source_id]

            if source_id == "pose":
                pose_paths = sorted(session_dir.glob("*_raw2D_live.h5"))
                source["paths"] = [
                    path.relative_to(session_dir).as_posix()
                    for path in pose_paths
                ]
                sample_count = sum(
                    SessionDataRecorder._h5_primary_row_count(path)
                    for path in pose_paths
                )
                path_exists = bool(pose_paths)
            else:
                result_path = result.get("path")
                if result_path:
                    source["path"] = result_path
                path_text = source.get("path")
                path = (
                    None
                    if not path_text
                    else session_dir / str(path_text)
                )
                path_exists = path is not None and path.exists()
                sample_count = result.get("sampleCount")
                if sample_count is None:
                    sample_count = len(perf_times)

            failure = result.get("failure", "")
            source["sampleCount"] = int(sample_count or 0)
            source["firstOffsetSeconds"] = (
                None
                if not perf_times
                else float(perf_times[0] - start_perf)
            )
            source["lastOffsetSeconds"] = (
                None
                if not perf_times
                else float(perf_times[-1] - start_perf)
            )
            source["gapCount"] = int(source_gap_count)
            source["overrunCount"] = int(overrun_count)
            source["failure"] = failure or None
            source["persistenceStatus"] = (
                "failed"
                if failure
                else "written"
                if path_exists
                else "missing"
            )
            finalized.append(source)
        return finalized

    @staticmethod
    def _h5_primary_row_count(path: Path) -> int:
        try:
            with h5py.File(path, "r") as stream:
                counts = []
                stream.visititems(
                    lambda _name, item: (
                        counts.append(int(item.shape[0]))
                        if isinstance(item, h5py.Dataset) and item.shape
                        else None
                    )
                )
                return max(counts, default=0)
        except (OSError, ValueError):
            return 0

    @staticmethod
    def _nidaq_arrays(chunks):
        names = next((tuple(chunk[4]) for chunk in chunks if chunk[4]), tuple())
        selected = tuple(
            (chunk[0], chunk[1], chunk[3])
            for chunk in chunks
            if len(chunk[0]) and len(chunk[1])
        )
        if not selected:
            return (
                names,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty((len(names), 0), dtype=np.float32),
                None,
            )
        rate = float(next(chunk[5] for chunk in chunks if len(chunk[0])))
        return (
            names,
            np.concatenate([item[0] for item in selected]),
            np.concatenate([item[1] for item in selected]),
            np.concatenate([item[2] for item in selected], axis=1),
            rate,
        )

    @staticmethod
    def _rising_edge_positions(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return np.empty(0, dtype=np.int64)
        high = np.isfinite(values) & (values > 0.5)
        return np.flatnonzero(high & np.concatenate(([False], ~high[:-1])))

    @staticmethod
    def _match_camera_nidaq_edge(boundary, start_perf, chunks) -> dict:
        names, indices, perf, values, rate = SessionDataRecorder._nidaq_arrays(
            chunks
        )
        resolution = None if rate is None else 1.0 / rate
        base = {
            "cameraFrameId": (
                None if boundary is None else boundary.primary_frame_id
            ),
            "cameraPerfTime": float(start_perf),
            "cameraTimestamp": (
                None if boundary is None else boundary.camera_when
            ),
            "channel": "cam_frames",
            "matchedSampleIndex": None,
            "matchedNidaqPerfTime": None,
            "signedOffsetSeconds": None,
            "resolutionSeconds": resolution,
            "ambiguitySeconds": None,
        }
        if indices.size == 0 or rate is None:
            return {
                **base,
                "status": "unavailable",
                "confidence": "none",
                "reason": "No NI-DAQ samples were saved for the session",
            }
        if "cam_frames" not in names:
            nearest = int(np.argmin(np.abs(perf - start_perf)))
            return {
                **base,
                "status": "host_estimated",
                "confidence": "host_estimated",
                "matchedSampleIndex": int(indices[nearest]),
                "matchedNidaqPerfTime": float(perf[nearest]),
                "signedOffsetSeconds": float(perf[nearest] - start_perf),
                "reason": (
                    "cam_frames was not configured; alignment uses the "
                    "single NI-DAQ epoch anchor"
                ),
            }

        channel_values = values[names.index("cam_frames")]
        edge_positions = SessionDataRecorder._rising_edge_positions(
            channel_values
        )
        if edge_positions.size == 0:
            return {
                **base,
                "status": "unmatched",
                "confidence": "none",
                "reason": "cam_frames contained no rising edge",
            }
        distances = np.abs(perf[edge_positions] - start_perf)
        order = np.argsort(distances)
        best_position = int(edge_positions[order[0]])
        second_distance = (
            None if order.size < 2 else float(distances[order[1]])
        )
        edge_intervals = np.diff(perf[edge_positions])
        half_frame_period = (
            0.05
            if edge_intervals.size == 0
            else max(2.0 / rate, float(np.median(edge_intervals)) / 2.0)
        )
        best_distance = float(distances[order[0]])
        if best_distance > half_frame_period:
            return {
                **base,
                "status": "unmatched",
                "confidence": "none",
                "reason": (
                    "Nearest cam_frames edge was outside half of the "
                    "observed frame period"
                ),
                "ambiguitySeconds": second_distance,
            }
        ambiguity = (
            None
            if second_distance is None
            else second_distance - best_distance
        )
        is_ambiguous = ambiguity is not None and ambiguity <= 1.0 / rate
        return {
            **base,
            "status": "matched",
            "confidence": (
                "ambiguous_hardware_edge"
                if is_ambiguous
                else "hardware_edge"
            ),
            "matchedSampleIndex": int(indices[best_position]),
            "matchedNidaqPerfTime": float(perf[best_position]),
            "signedOffsetSeconds": float(perf[best_position] - start_perf),
            "ambiguitySeconds": ambiguity,
            "reason": (
                "Nearest rising cam_frames edge on the NI-DAQ sample timeline"
            ),
        }

    @staticmethod
    def _correlate_tone_confirmations(device_rows, chunks) -> dict:
        names, indices, perf, values, rate = SessionDataRecorder._nidaq_arrays(
            chunks
        )
        tone_channels = tuple(
            name
            for name in ("tone1", "tone2", "tone3_r", "tone3_l")
            if name in names
        )
        edges = {
            channel: [
                {
                    "sampleIndex": int(indices[position]),
                    "perfTime": float(perf[position]),
                }
                for position in SessionDataRecorder._rising_edge_positions(
                    values[names.index(channel)]
                )
            ]
            for channel in tone_channels
        }
        events = []
        previous_states = {channel: False for channel in tone_channels}
        for row in sorted(device_rows, key=lambda item: item[0]):
            perf_time, _, direction, kind, target, context, _, _, payload = row
            try:
                decoded = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if kind == "STIMULUS_INPUTS" and isinstance(decoded, dict):
                for channel in tone_channels:
                    if channel not in decoded:
                        continue
                    state = bool(decoded[channel])
                    if state and not previous_states[channel]:
                        events.append({
                            "channel": channel,
                            "eventPerfTime": float(perf_time),
                            "direction": direction,
                            "kind": kind,
                            "target": target,
                            "context": context,
                        })
                    previous_states[channel] = state
            elif kind == "PLAY_TONE":
                events.append({
                    "channel": None,
                    "eventPerfTime": float(perf_time),
                    "direction": direction,
                    "kind": kind,
                    "target": target,
                    "context": context,
                    "payload": decoded,
                })

        claimed = {channel: set() for channel in tone_channels}
        matched = []
        unmatched = []
        for event in events:
            channel = event["channel"]
            if channel is None:
                unmatched.append({
                    **event,
                    "reason": "PLAY_TONE does not identify a confirmation line",
                })
                continue
            candidates = [
                (edge_index, edge)
                for edge_index, edge in enumerate(edges[channel])
                if edge_index not in claimed[channel]
            ]
            if not candidates:
                unmatched.append({
                    **event,
                    "reason": f"{channel} contained no unclaimed rising edge",
                })
                continue
            edge_index, edge = min(
                candidates,
                key=lambda item: abs(
                    item[1]["perfTime"] - event["eventPerfTime"]
                ),
            )
            latency = edge["perfTime"] - event["eventPerfTime"]
            if abs(latency) > 0.25:
                unmatched.append({
                    **event,
                    "nearestEdgePerfTime": edge["perfTime"],
                    "reason": "Nearest electrical edge was more than 250 ms away",
                })
                continue
            claimed[channel].add(edge_index)
            matched.append({
                **event,
                "sampleIndex": edge["sampleIndex"],
                "edgePerfTime": edge["perfTime"],
                "latencySeconds": latency,
                "resolutionSeconds": None if rate is None else 1.0 / rate,
            })

        unmatched_edges = [
            {
                "channel": channel,
                **edge,
            }
            for channel in tone_channels
            for edge_index, edge in enumerate(edges[channel])
            if edge_index not in claimed[channel]
        ]
        return {
            "status": (
                "unavailable"
                if not tone_channels
                else "complete"
                if not unmatched and not unmatched_edges
                else "partial"
            ),
            "channels": list(tone_channels),
            "matched": matched,
            "unmatchedEvents": unmatched,
            "unmatchedEdges": unmatched_edges,
            "resolutionSeconds": None if rate is None else 1.0 / rate,
        }

    @staticmethod
    def _alignment_entry(path, perf_times, start_perf, clock):
        if perf_times:
            first_perf = float(perf_times[0])
            last_perf = float(perf_times[-1])
            first_offset = first_perf - start_perf
            last_offset = last_perf - start_perf
        else:
            first_perf = last_perf = first_offset = last_offset = None
        return {
            "path": path,
            "clock": clock,
            "sampleCount": len(perf_times),
            "firstPerfTime": first_perf,
            "lastPerfTime": last_perf,
            "firstOffsetSeconds": first_offset,
            "lastOffsetSeconds": last_offset,
        }

    @staticmethod
    def _normalize_device_row(row):
        if len(row) == 9:
            return row
        if len(row) == 6:
            perf, wall, switch, pressure, temperature, humidity = row
            return (
                perf,
                wall,
                "inbound",
                "MEASUREMENT_SAMPLE",
                "",
                "",
                None,
                None,
                SessionDataRecorder._payload_json({
                    "switch": switch,
                    "pressure": pressure,
                    "temperature": temperature,
                    "humidity": humidity,
                }),
            )
        raise ValueError(f"Unsupported structured device row with {len(row)} fields")

    @staticmethod
    def _normalize_laser_row(row):
        if len(row) == 10:
            return row
        if len(row) == 8:
            return (*row, "", None)
        raise ValueError(f"Unsupported laser event row with {len(row)} fields")

    @staticmethod
    def _write_csv(path: Path, header, rows) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _write_nidaq(
        path,
        start_perf,
        start_wall,
        end_perf,
        chunks,
        timing_plan=None,
    ) -> None:
        channel_names = next((chunk[4] for chunk in chunks if chunk[4]), tuple())
        selected = []
        for chunk in chunks:
            (
                indices,
                perf,
                wall,
                values,
                names,
                rate,
                epoch,
                gaps,
                overrun,
                *observation,
            ) = chunk
            observation_perf = (
                float(observation[0]) if len(observation) > 0 else math.nan
            )
            observation_wall = (
                float(observation[1]) if len(observation) > 1 else math.nan
            )
            mask = (perf >= start_perf) & (perf <= end_perf)
            if np.any(mask):
                selected.append((
                    indices[mask],
                    perf[mask],
                    wall[mask],
                    values[:, mask],
                    rate,
                    epoch,
                    gaps,
                    overrun,
                    np.full(np.count_nonzero(mask), epoch, dtype=np.uint64),
                    np.full(
                        np.count_nonzero(mask),
                        observation_perf,
                        dtype=np.float64,
                    ),
                    np.full(
                        np.count_nonzero(mask),
                        observation_wall,
                        dtype=np.float64,
                    ),
                ))
        with h5py.File(path, "w") as output:
            output.attrs["recording_start_perf"] = start_perf
            output.attrs["recording_start_wall"] = start_wall
            output.attrs["recording_end_perf"] = end_perf
            output.attrs["alignment"] = "first sample at or after primary camera first frame"
            output.attrs["source_clock"] = "NI-DAQ ring reconstructed on time.perf_counter"
            output.attrs["channel_names"] = channel_names
            output.attrs["timing_plan_json"] = json.dumps(
                None if timing_plan is None else dataclasses.asdict(timing_plan),
                sort_keys=True,
            )
            if not selected:
                output.create_dataset("sample_index", data=np.empty(0, dtype=np.int64))
                output.create_dataset("perf_time", data=np.empty(0, dtype=np.float64))
                output.create_dataset("offset_seconds", data=np.empty(0, dtype=np.float64))
                output.create_dataset("epoch", data=np.empty(0, dtype=np.uint64))
                output.create_dataset(
                    "block_observation_perf_time",
                    data=np.empty(0, dtype=np.float64),
                )
                output.create_dataset(
                    "block_observation_wall_time",
                    data=np.empty(0, dtype=np.float64),
                )
                output.create_dataset(
                    "values",
                    data=np.empty((len(channel_names), 0), dtype=np.float32),
                )
                return
            indices = np.concatenate([item[0] for item in selected])
            perf = np.concatenate([item[1] for item in selected])
            values = np.concatenate([item[3] for item in selected], axis=1)
            epochs = np.concatenate([item[8] for item in selected])
            observation_perf = np.concatenate([item[9] for item in selected])
            observation_wall = np.concatenate([item[10] for item in selected])
            if indices.size > 1 and not np.all(np.diff(indices) == 1):
                raise RuntimeError(
                    "NI-DAQ session sample indices are not strictly consecutive"
                )
            if perf.size > 1 and not np.all(np.diff(perf) > 0):
                raise RuntimeError(
                    "NI-DAQ session timestamps are not strictly increasing"
                )
            output.create_dataset("sample_index", data=indices, compression="gzip")
            output.create_dataset("perf_time", data=perf, compression="gzip")
            output.create_dataset("offset_seconds", data=perf - start_perf, compression="gzip")
            output.create_dataset("epoch", data=epochs, compression="gzip")
            output.create_dataset(
                "block_observation_perf_time",
                data=observation_perf,
                compression="gzip",
            )
            output.create_dataset(
                "block_observation_wall_time",
                data=observation_wall,
                compression="gzip",
            )
            output.create_dataset(
                "wall_time",
                data=start_wall + (perf - start_perf),
                compression="gzip",
            )
            output.create_dataset("values", data=values, compression="gzip")
            output.attrs["first_perf_time"] = perf[0]
            output.attrs["last_perf_time"] = perf[-1]
            output.attrs["first_offset_seconds"] = perf[0] - start_perf
            output.attrs["last_offset_seconds"] = perf[-1] - start_perf
            output.attrs["sample_rate_hz"] = selected[0][4]
            output.attrs["epoch"] = selected[-1][5]
            output.attrs["gap_count"] = selected[-1][6]
            output.attrs["overrun_samples"] = sum(item[7] for item in selected)
