from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Tuple
from uuid import UUID

import h5py
import numpy as np

from autotrainer.core import ProjectInfo
from tools.acquisition.model.session_boundary import SessionBoundary
from tools.acquisition.model.atomic_session_io import (
    atomic_publish_file,
    atomic_write_json,
    file_manifest_entry,
    remove_empty_staging_generation,
)


@dataclass(frozen=True)
class _NidaqSpoolSnapshot:
    path: Path
    channel_names: Tuple[str, ...]
    sample_rate_hz: float
    sample_count: int
    first_sample_index: Optional[int]
    last_sample_index: Optional[int]
    first_perf_time: Optional[float]
    last_perf_time: Optional[float]
    gap_count: int
    overrun_samples: int
    first_error: Optional[str]
    last_error: Optional[str]
    error_count: int
    worker_failed: bool


@dataclass(frozen=True)
class _NidaqPerfSummary:
    count: int
    first: Optional[float]
    last: Optional[float]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not self.count:
            raise IndexError(index)
        if index in (0, -self.count):
            return self.first
        if index in (-1, self.count - 1):
            return self.last
        raise IndexError(
            "bounded NI-DAQ timeline exposes only its first and last sample"
        )


_PELLET_STIMULUS_CHANNEL_NAMES = ("tone1", "tone2", "stim2", "stim3")
_LIVE_TONE_CHANNEL_NAMES = ("tone1", "tone2")
_MINIMUM_VALID_TONE_PULSE_SECONDS = 0.002


class _SessionLogHandler(logging.Handler):
    def __init__(self, recorder: "SessionDataRecorder"):
        super().__init__(logging.NOTSET)
        self._recorder = recorder
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        # Raw python-can frame dumps can arrive hundreds of times per second.
        # They remain available in the acquisition-wide diagnostic log, while
        # the session log keeps warnings/errors and higher-level decoded events.
        if record.name.startswith("can.bus") and record.levelno < logging.WARNING:
            return
        try:
            self._recorder.add_current_log(record.created, self.format(record))
        except Exception:
            self.handleError(record)


class SessionDataRecorder:
    """Capture auxiliary streams against the recorded primary-camera boundary."""

    def __init__(
        self,
        nidaq_monitor,
        laser_model,
        *,
        system_message_handler=None,
        hardware_model=None,
        nidaq_tone_edge_callback=None,
        device_event_capacity: int = 100_000,
    ):
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
        self._nidaq_chunks = []  # compatibility-only; production uses a spool
        self._nidaq_last_index = None
        self._nidaq_scratch = None
        self._nidaq_spool = None
        self._nidaq_spool_path: Optional[Path] = None
        self._nidaq_stats = self._new_nidaq_stats()
        self._nidaq_stop = threading.Event()
        self._nidaq_thread: Optional[threading.Thread] = None
        self._source_manifest = ()
        self._source_results = {}
        self._trial_records = ()
        self._trial_summary = {}
        self._metadata_generation_id: Optional[str] = None
        self._pending_finalization: Optional[dict] = None
        self._pending_stop_end_perf: Optional[float] = None
        self._nidaq_tone_edge_callback = nidaq_tone_edge_callback
        self._live_tone_states = {}

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

    def arm(
        self,
        project: ProjectInfo,
        *,
        source_manifest=(),
        metadata_generation_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            if self._armed:
                raise RuntimeError("session recorder is already armed")
            if self._pending_finalization is not None:
                raise RuntimeError(
                    "previous session finalization is still pending; retry or abort it"
                )
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
            self._nidaq_scratch = None
            self._nidaq_stats = self._new_nidaq_stats()
            self._live_tone_states = {}
            self._source_manifest = tuple(source_manifest)
            self._source_results = {}
            self._trial_records = ()
            self._trial_summary = {}
            self._start_perf = None
            self._start_wall = None
            self._boundary = None
            self._metadata_generation_id = str(
                metadata_generation_id
                or f"{project.short_id}-g{getattr(project, 'session_generation', 0)}"
            )
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
            if self._nidaq_spool is None:
                self._open_nidaq_spool_locked()
            self._device_events_since_start = sum(
                1
                for row in self._device_rows
                if row[0] >= self._start_perf
            )

    def stop(self, end_perf: float):
        with self._lock:
            self._pending_stop_end_perf = float(end_perf)
        self._stop_nidaq_thread()
        with self._lock:
            if not self._armed or self._project is None or self._start_perf is None:
                self._pending_stop_end_perf = None
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
            nidaq_chunks = self._snapshot_nidaq_locked()
            timing_plan = self._nidaq_monitor.timing_plan
            source_manifest = self._source_manifest
            source_results = dict(self._source_results)
            if self._laser_model.configuration.backend != "disabled":
                laser_result = dict(source_results.get("laser_outputs", {}))
                laser_result["diagnostics"] = {
                    **dict(laser_result.get("diagnostics", {})),
                    "timing": self._laser_model.timing_status,
                }
                source_results["laser_outputs"] = laser_result
            trial_records = tuple(self._trial_records)
            trial_summary = dict(self._trial_summary)
            snapshot = {
                "project": project,
                "start_perf": start_perf,
                "start_wall": start_wall,
                "end_perf": end_perf,
                "device_rows": device_rows,
                "laser_rows": laser_rows,
                "log_rows": log_rows,
                "nidaq_chunks": nidaq_chunks,
                "timing_plan": timing_plan,
                "device_event_overruns": device_event_overruns,
                "source_manifest": source_manifest,
                "source_results": source_results,
                "boundary": boundary,
                "trial_records": trial_records,
                "trial_summary": trial_summary,
                "metadata_generation_id": self._metadata_generation_id,
            }
            self._armed = False
            self._pending_stop_end_perf = None
            self._pending_finalization = snapshot
        return self._publish_pending_finalization(max_attempts=2)

    def retry_pending_finalization(self):
        """Retry a retained auxiliary snapshot without reacquiring any data."""
        with self._lock:
            pending_stop_end_perf = self._pending_stop_end_perf
        if pending_stop_end_perf is not None:
            return self.stop(pending_stop_end_perf)
        return self._publish_pending_finalization(max_attempts=1)

    @property
    def has_pending_finalization(self) -> bool:
        with self._lock:
            return (
                self._pending_stop_end_perf is not None
                or self._pending_finalization is not None
            )

    @property
    def pending_finalization_session_id(self) -> Optional[str]:
        with self._lock:
            snapshot = self._pending_finalization
            project = (
                None if snapshot is None else snapshot["project"]
            ) or self._project
            return None if project is None else project.short_id

    @property
    def pending_finalization_project(self) -> Optional[ProjectInfo]:
        """Return an isolated project snapshot for metadata republication."""
        with self._lock:
            snapshot = self._pending_finalization
            project = (
                None if snapshot is None else snapshot["project"]
            ) or self._project
            return None if project is None else project.to_local_value()

    def _publish_pending_finalization(self, *, max_attempts: int):
        with self._lock:
            snapshot = self._pending_finalization
        if snapshot is None:
            return None
        first_error = None
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            try:
                result = self._write_session(**snapshot)
            except Exception as error:
                first_error = first_error or error
                logging.getLogger(__name__).exception(
                    "Session auxiliary finalization attempt %d/%d failed",
                    attempt,
                    max_attempts,
                )
                continue
            with self._lock:
                if self._pending_finalization is snapshot:
                    nidaq_source = snapshot.get("nidaq_chunks")
                    if isinstance(nidaq_source, _NidaqSpoolSnapshot):
                        nidaq_source.path.unlink(missing_ok=True)
                    remove_empty_staging_generation(
                        Path(snapshot["project"].get_session_path().location),
                        str(snapshot["metadata_generation_id"]),
                    )
                    self._pending_finalization = None
                    self._clear_locked()
            return result
        raise RuntimeError(
            "Session auxiliary finalization failed; staged files and the recorder "
            "snapshot were retained for retry"
        ) from first_error

    def abort(self) -> None:
        self._stop_nidaq_thread()
        with self._lock:
            spool_path = self._nidaq_spool_path
            pending = self._pending_finalization
            if pending is not None:
                pending_source = pending.get("nidaq_chunks")
                if isinstance(pending_source, _NidaqSpoolSnapshot):
                    spool_path = pending_source.path
            self._pending_finalization = None
            self._pending_stop_end_perf = None
            self._clear_locked()
        if spool_path is not None:
            spool_path.unlink(missing_ok=True)

    def request_abort(self) -> None:
        """Request recorder shutdown without blocking the operator/UI thread."""
        self._nidaq_stop.set()

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
        warnings: Tuple[str, ...] = (),
        diagnostics: Optional[dict] = None,
    ) -> None:
        with self._lock:
            if not self._armed:
                return
            source_id = str(source_id)
            previous = self._source_results.get(source_id, {})
            previous_warnings = tuple(previous.get("warnings", ()))
            combined_warnings = tuple(
                dict.fromkeys((*previous_warnings, *(str(item) for item in warnings)))
            )
            combined_diagnostics = dict(previous.get("diagnostics", {}))
            combined_diagnostics.update(diagnostics or {})
            self._source_results[source_id] = {
                "sampleCount": (
                    previous.get("sampleCount")
                    if sample_count is None
                    else int(sample_count)
                ),
                "path": previous.get("path") if path is None else path,
                "failure": str(failure or previous.get("failure", "")),
                "warnings": combined_warnings,
                "diagnostics": combined_diagnostics,
            }

    def set_trial_ledger(self, records, summary) -> None:
        """Snapshot the authoritative trial ledger for the current session."""
        with self._lock:
            if not self._armed:
                return
            self._trial_records = tuple(dict(record) for record in records)
            self._trial_summary = dict(summary)

    def persist_trial_ledger(self, project, records, summary) -> None:
        """Publish live trial progress, or update the closed-session ledger."""
        self.set_trial_ledger(records, summary)
        streams_dir = Path(project.get_session_path().location) / "streams"
        alignment_path = streams_dir / "alignment.json"
        if alignment_path.is_file():
            self.update_persisted_trial_ledger(project, records, summary)
            return
        with self._lock:
            if not self._armed or self._start_perf is None:
                return
            start_perf = self._start_perf
            generation_id = (
                self._metadata_generation_id or f"{project.short_id}-live"
            )
        session_dir = Path(project.get_session_path().location)
        streams_dir.mkdir(parents=True, exist_ok=True)
        live_records = []
        for original in records:
            record = dict(original)
            if float(record["send_perf_time"]) < start_perf:
                continue
            record["metadata_generation_id"] = generation_id
            record["send_offset_seconds"] = (
                float(record["send_perf_time"]) - start_perf
            )
            live_records.append(record)
        self._atomic_write_json_lines(
            streams_dir / "trials.jsonl",
            session_dir,
            generation_id,
            live_records,
        )
        live_summary = dict(summary)
        live_summary["metadata_generation_id"] = generation_id
        atomic_write_json(
            streams_dir / "trial_summary.json",
            live_summary,
            session_dir=session_dir,
            generation_id=generation_id,
        )

    def trial_stream_references(self, start_perf: float, end_perf: float):
        """Return decoded tone and laser records inside one pellet window."""
        with self._lock:
            tones = tuple(
                {
                    "perf_time": row[0],
                    "wall_time": row[1],
                    "kind": row[3],
                    "context": row[5],
                    "payload_json": row[8],
                }
                for row in self._device_rows
                if start_perf <= row[0] <= end_perf
                and row[3] == "STIMULUS_INPUTS"
            )
            lasers = tuple(
                {
                    "perf_time": row[0],
                    "wall_time": row[1],
                    "record_type": row[2],
                    "channel": row[3],
                    "source": row[4],
                    "command_volts": row[5],
                    "diode_volts": row[6],
                    "command_copy_volts": row[7],
                    "output_name": row[8],
                    "output_value": row[9],
                }
                for row in self._laser_rows
                if start_perf <= row[0] <= end_perf
            )
        return tones, lasers

    def persist_trial_tracking(self, project, request, result=None) -> Path:
        """Atomically retain the exact live data used for one trial result."""
        from tools.acquisition.model.intertrial_analysis import (
            tracking_request_record,
        )

        session_dir = Path(project.get_session_path().location)
        tracking_dir = session_dir / "streams" / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        trial = "unindexed" if request.trial_id is None else f"{request.trial_id:06d}"
        path = tracking_dir / f"trial_{trial}_attempt_{request.attempt_id:03d}.json"
        generation_id = self._metadata_generation_id
        alignment_path = session_dir / "streams" / "alignment.json"
        if alignment_path.is_file():
            try:
                with alignment_path.open("r", encoding="utf-8") as stream:
                    generation_id = json.load(stream).get("metadataGenerationId")
            except (OSError, ValueError, TypeError):
                pass
        generation_id = generation_id or f"{project.short_id}-live"
        atomic_write_json(
            path,
            tracking_request_record(
                request,
                result,
                metadata_generation_id=generation_id,
            ),
            session_dir=session_dir,
            generation_id=generation_id,
        )
        return path

    @staticmethod
    def load_trial_tracking_requests(project):
        """Read final-validation inputs; isolate one corrupt attempt file."""
        from tools.acquisition.model.intertrial_analysis import (
            tracking_request_from_record,
        )

        tracking_dir = (
            Path(project.get_session_path().location) / "streams" / "tracking"
        )
        alignment_path = tracking_dir.parent / "alignment.json"
        expected_generation_id = None
        try:
            with alignment_path.open("r", encoding="utf-8") as stream:
                expected_generation_id = json.load(stream).get(
                    "metadataGenerationId"
                )
        except (OSError, ValueError, TypeError):
            pass
        requests = []
        errors = []
        for path in sorted(tracking_dir.glob("trial_*_attempt_*.json")):
            try:
                with path.open("r", encoding="utf-8") as stream:
                    requests.append(tracking_request_from_record(
                        json.load(stream),
                        expected_metadata_generation_id=expected_generation_id,
                    ))
            except Exception as error:
                errors.append(f"{path.name}: {error}")
        return tuple(requests), tuple(errors)

    @staticmethod
    def update_persisted_trial_ledger(project, records, summary) -> None:
        """Atomically replace the post-analysis trial ledger and summary."""
        streams_dir = Path(project.get_session_path().location) / "streams"
        alignment_path = streams_dir / "alignment.json"
        with alignment_path.open("r", encoding="utf-8") as stream:
            alignment = json.load(stream)
        boundary = alignment["canonicalBoundary"]
        metadata_generation_id = str(
            alignment.get("metadataGenerationId", f"{project.short_id}-legacy")
        )
        start_perf = float(boundary["startPerfTime"])
        end_perf = float(boundary["endPerfTime"])
        records = tuple(
            dict(record)
            for record in records
            if start_perf <= float(record["send_perf_time"]) <= end_perf
        )
        for record in records:
            record["metadata_generation_id"] = metadata_generation_id
            record["send_offset_seconds"] = (
                float(record["send_perf_time"]) - start_perf
            )

        trial_path = streams_dir / "trials.jsonl"
        SessionDataRecorder._atomic_write_json_lines(
            trial_path,
            Path(project.get_session_path().location),
            metadata_generation_id,
            records,
        )

        summary_path = streams_dir / "trial_summary.json"
        summary = dict(summary)
        summary["metadata_generation_id"] = metadata_generation_id
        atomic_write_json(
            summary_path,
            summary,
            session_dir=Path(project.get_session_path().location),
            generation_id=metadata_generation_id,
        )
        manifest_path = streams_dir / "stream_manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            manifest_paths = (
                streams_dir / "device.csv",
                trial_path,
                summary_path,
                streams_dir / "laser.csv",
                streams_dir.parent / "logs" / "session.log",
                alignment_path,
                *sorted((streams_dir / "tracking").glob("*.json")),
            )
            manifest["files"] = [
                file_manifest_entry(path, relative_to=streams_dir.parent)
                for path in manifest_paths
                if path.is_file()
            ]
            atomic_write_json(
                manifest_path,
                manifest,
                session_dir=streams_dir.parent,
                generation_id=metadata_generation_id,
            )

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
        self._nidaq_scratch = None
        self._nidaq_spool = None
        self._nidaq_spool_path = None
        self._nidaq_stats = self._new_nidaq_stats()
        self._live_tone_states = {}
        self._source_manifest = ()
        self._source_results = {}
        self._trial_records = ()
        self._trial_summary = {}
        self._metadata_generation_id = None
        self._pending_stop_end_perf = None

    def _on_device_message(
        self,
        kind,
        data,
        perf_time: float,
        wall_time: float,
    ) -> None:
        source_index = getattr(data, "index", None)
        source_timestamp_ns = getattr(data, "timestamp_ns", None)
        if isinstance(source_index, int) and source_index > 0:
            perf_time = source_index / 1e9
        if isinstance(source_timestamp_ns, int) and source_timestamp_ns > 0:
            wall_time = source_timestamp_ns / 1e9
        context = None
        if (
            getattr(kind, "name", None) == "STIMULUS_INPUTS"
            and isinstance(data, (tuple, list))
        ):
            data = {
                name: bool(value)
                for name, value in zip(_PELLET_STIMULUS_CHANNEL_NAMES, data)
            }
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
        device_timestamp = getattr(
            data, "timestamp_ns", getattr(data, "timestamp", None),
        )
        device_index = getattr(data, "index", None)
        # Containers such as list expose ``index`` as a method. Only a concrete
        # device-provided value belongs in the CSV column.
        if callable(device_index):
            device_index = None
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
        tone_edges = ()
        try:
            ring = self._nidaq_monitor.sample_ring
            destination = self._nidaq_scratch
            expected_shape = (max(1, len(ring.channel_names)), ring.capacity)
            if destination is None or destination.shape != expected_shape:
                destination = np.empty(expected_shape, dtype=np.float32)
                self._nidaq_scratch = destination
            read = ring.copy_since(self._nidaq_last_index, destination)
            if read is None:
                return
            if read.sample_count <= 0:
                return
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
                if self._armed and self._nidaq_spool is not None:
                    if (
                        read.overrun_samples
                        or (
                            self._nidaq_last_index is not None
                            and read.start_sample_index != self._nidaq_last_index
                        )
                    ):
                        self._live_tone_states = {}
                    self._append_nidaq_spool_locked(
                        indices,
                        perf_times,
                        wall_times,
                        destination[:len(ring.channel_names), :read.sample_count],
                        tuple(ring.channel_names),
                        read,
                    )
                    tone_edges = self._validated_live_tone_edges(
                        indices,
                        perf_times,
                        wall_times,
                        destination[:len(ring.channel_names), :read.sample_count],
                        tuple(ring.channel_names),
                        float(ring.sample_rate_hz),
                    )
                    self._nidaq_last_index = read.end_sample_index
            callback = self._nidaq_tone_edge_callback
            if callback is not None:
                for edge in tone_edges:
                    try:
                        callback(**edge)
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Unable to publish validated NI-DAQ tone edge",
                        )
        except Exception as error:
            with self._lock:
                message = f"{type(error).__name__}: {error}"
                stats = self._nidaq_stats
                stats["first_error"] = stats["first_error"] or message
                stats["last_error"] = message
                stats["error_count"] += 1
            logging.getLogger(__name__).exception("Unable to collect NI-DAQ session samples")

    def _validated_live_tone_edges(
        self, indices, perf_times, wall_times, values, channel_names, sample_rate,
    ):
        """Return exact onsets after rejecting sub-2 ms digital glitches."""
        minimum_samples = max(
            2,
            int(math.ceil(sample_rate * _MINIMUM_VALID_TONE_PULSE_SECONDS)),
        )
        start_perf = self._start_perf
        stop_perf = self._pending_stop_end_perf
        edges = []
        for channel in _LIVE_TONE_CHANNEL_NAMES:
            if channel not in channel_names:
                continue
            state = self._live_tone_states.setdefault(channel, {
                "high": False,
                "count": 0,
                "emitted": False,
                "sample_index": None,
                "perf_time": None,
                "wall_time": None,
            })
            samples = np.asarray(
                values[channel_names.index(channel)], dtype=np.float32,
            ) >= 0.5
            for position, high in enumerate(samples):
                if not high:
                    state.update(high=False, count=0, emitted=False)
                    continue
                if not state["high"]:
                    state.update(
                        high=True,
                        count=0,
                        emitted=False,
                        sample_index=int(indices[position]),
                        perf_time=float(perf_times[position]),
                        wall_time=float(wall_times[position]),
                    )
                state["count"] += 1
                if state["emitted"] or state["count"] < minimum_samples:
                    continue
                state["emitted"] = True
                edge_perf = state["perf_time"]
                if start_perf is not None and edge_perf < start_perf:
                    continue
                if stop_perf is not None and edge_perf > stop_perf:
                    continue
                edges.append({
                    "channel": channel,
                    "perf_time": edge_perf,
                    "wall_time": state["wall_time"],
                    "sample_index": state["sample_index"],
                })
        return tuple(edges)

    def _stop_nidaq_thread(self) -> None:
        thread = self._nidaq_thread
        if thread is None:
            return
        self._nidaq_stop.set()
        if thread is not threading.current_thread():
            thread.join(10.0)
        if thread.is_alive():
            with self._lock:
                message = "NI-DAQ recorder thread did not stop within 10 seconds"
                self._nidaq_stats["worker_failed"] = True
                self._nidaq_stats["last_error"] = message
                self._nidaq_stats["first_error"] = (
                    self._nidaq_stats["first_error"] or message
                )
            raise RuntimeError(message)
        with self._lock:
            self._close_nidaq_spool_locked()
        self._nidaq_thread = None
        self._nidaq_stop.clear()

    @staticmethod
    def _new_nidaq_stats():
        return {
            "sample_count": 0,
            "first_sample_index": None,
            "last_sample_index": None,
            "first_perf_time": None,
            "last_perf_time": None,
            "gap_count": 0,
            "overrun_samples": 0,
            "first_error": None,
            "last_error": None,
            "error_count": 0,
            "worker_failed": False,
        }

    def _open_nidaq_spool_locked(self) -> None:
        ring = self._nidaq_monitor.sample_ring
        if not tuple(ring.channel_names):
            return
        if self._project is None or self._metadata_generation_id is None:
            return
        session_dir = Path(self._project.get_session_path().location)
        path = (
            session_dir / ".staging" / self._metadata_generation_id
            / "streams" / "nidaq_capture.h5"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        output = h5py.File(path, "w")
        channel_count = len(ring.channel_names)
        for name, dtype in (
            ("sample_index", np.int64),
            ("perf_time", np.float64),
            ("wall_time", np.float64),
            ("epoch", np.uint64),
            ("block_observation_perf_time", np.float64),
            ("block_observation_wall_time", np.float64),
        ):
            output.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                  chunks=True)
        output.create_dataset(
            "values",
            shape=(channel_count, 0),
            maxshape=(channel_count, None),
            dtype=np.float32,
            chunks=(channel_count, max(1, min(ring.capacity, 65536))),
        )
        output.attrs["channel_names"] = tuple(ring.channel_names)
        output.attrs["sample_rate_hz"] = float(ring.sample_rate_hz)
        self._nidaq_spool = output
        self._nidaq_spool_path = path

    def _append_nidaq_spool_locked(
        self, indices, perf_times, wall_times, values, channel_names, read,
    ) -> None:
        output = self._nidaq_spool
        if output is None:
            return
        count = len(indices)
        start = int(output["sample_index"].shape[0])
        end = start + count
        observations = {
            "sample_index": indices,
            "perf_time": perf_times,
            "wall_time": wall_times,
            "epoch": np.full(count, read.epoch, dtype=np.uint64),
            "block_observation_perf_time": np.full(
                count, read.source_perf_time, dtype=np.float64,
            ),
            "block_observation_wall_time": np.full(
                count, read.source_wall_time, dtype=np.float64,
            ),
        }
        dataset_names = (*observations, "values")
        try:
            for name, data in observations.items():
                dataset = output[name]
                dataset.resize((end,))
                dataset[start:end] = data
            dataset = output["values"]
            dataset.resize((len(channel_names), end))
            dataset[:, start:end] = values
            output.flush()
        except Exception:
            # All datasets share one committed length. Restore it before the
            # caller retries the same ring range.
            for name in dataset_names:
                dataset = output[name]
                shape = (len(channel_names), start) if name == "values" else (start,)
                dataset.resize(shape)
            output.flush()
            raise
        stats = self._nidaq_stats
        stats["sample_count"] += count
        stats["first_sample_index"] = (
            int(indices[0]) if stats["first_sample_index"] is None
            else stats["first_sample_index"]
        )
        stats["last_sample_index"] = int(indices[-1])
        stats["first_perf_time"] = (
            float(perf_times[0]) if stats["first_perf_time"] is None
            else stats["first_perf_time"]
        )
        stats["last_perf_time"] = float(perf_times[-1])
        stats["gap_count"] = max(stats["gap_count"], int(read.gap_count))
        stats["overrun_samples"] += int(read.overrun_samples)

    def _close_nidaq_spool_locked(self) -> None:
        output, self._nidaq_spool = self._nidaq_spool, None
        if output is not None:
            output.flush()
            output.close()

    def _snapshot_nidaq_locked(self):
        if self._nidaq_spool_path is None:
            return tuple(self._nidaq_chunks)
        ring = self._nidaq_monitor.sample_ring
        stats = self._nidaq_stats
        return _NidaqSpoolSnapshot(
            path=self._nidaq_spool_path,
            channel_names=tuple(ring.channel_names),
            sample_rate_hz=float(ring.sample_rate_hz),
            **stats,
        )

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
        trial_records=(),
        trial_summary=None,
        metadata_generation_id=None,
    ):
        session_dir = Path(project.get_session_path().location)
        streams_dir = session_dir / "streams"
        logs_dir = session_dir / "logs"
        streams_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        metadata_generation_id = str(
            metadata_generation_id or f"{project.short_id}-legacy"
        )

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
        nidaq_perf = SessionDataRecorder._nidaq_perf_values(
            nidaq_chunks, start_perf, end_perf,
        )
        camera_nidaq_alignment = SessionDataRecorder._match_camera_nidaq_edge(
            boundary,
            start_perf,
            nidaq_chunks,
        )
        tone_confirmation = SessionDataRecorder._correlate_tone_confirmations(
            device_rows,
            nidaq_chunks,
            start_perf=start_perf,
            end_perf=end_perf,
        )
        trial_records = tuple(
            dict(record)
            for record in trial_records
            if start_perf <= float(record["send_perf_time"]) <= end_perf
        )
        for record in trial_records:
            record["metadata_generation_id"] = metadata_generation_id
            record["send_offset_seconds"] = (
                float(record["send_perf_time"]) - start_perf
            )
        trial_perf = tuple(
            float(record["send_perf_time"])
            for record in trial_records
        )

        SessionDataRecorder._atomic_write_csv(
            streams_dir / "device.csv", session_dir, metadata_generation_id,
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
        SessionDataRecorder._atomic_write_json_lines(
            streams_dir / "trials.jsonl", session_dir, metadata_generation_id,
            trial_records,
        )
        trial_summary = {} if trial_summary is None else dict(trial_summary)
        trial_summary["metadata_generation_id"] = metadata_generation_id
        atomic_write_json(
            streams_dir / "trial_summary.json",
            trial_summary,
            session_dir=session_dir,
            generation_id=metadata_generation_id,
        )
        SessionDataRecorder._atomic_write_csv(
            streams_dir / "laser.csv", session_dir, metadata_generation_id,
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

        def write_log(path):
            with path.open("w", encoding="utf-8") as stream:
                stream.write(
                    f"# metadata_generation_id={metadata_generation_id}\n"
                    f"# recording_start_perf={start_perf:.9f} "
                    f"recording_start_wall={start_wall:.9f} "
                    f"recording_end_perf={end_perf:.9f}\n"
                )
                for perf, _, message in log_rows:
                    stream.write(f"[+{perf - start_perf:.6f}s] {message}\n")

        atomic_publish_file(
            logs_dir / "session.log",
            write_log,
            session_dir=session_dir,
            generation_id=metadata_generation_id,
        )

        atomic_publish_file(
            streams_dir / "nidaq.h5",
            lambda path: SessionDataRecorder._write_nidaq(
                path, start_perf, start_wall, end_perf, nidaq_chunks, timing_plan,
            ),
            session_dir=session_dir,
            generation_id=metadata_generation_id,
            validate=SessionDataRecorder._validate_h5,
        )
        source_results = {} if source_results is None else dict(source_results)
        nidaq_health = SessionDataRecorder._nidaq_stream_health(
            nidaq_chunks,
            start_perf=start_perf,
            end_perf=end_perf,
            saved_perf=nidaq_perf,
        )
        for source in source_manifest:
            source_id = str(source.get("id", ""))
            if not source_id.startswith("nidaq."):
                continue
            existing = dict(source_results.get(source_id, {}))
            existing_warnings = tuple(existing.get("warnings", ()))
            existing["warnings"] = tuple(dict.fromkeys(
                (*existing_warnings, *nidaq_health["warnings"])
            ))
            existing["diagnostics"] = {
                **nidaq_health["diagnostics"],
                **dict(existing.get("diagnostics", {})),
            }
            if not existing.get("failure"):
                existing["failure"] = nidaq_health["failure"]
            source_results[source_id] = existing
        finalized_sources = SessionDataRecorder._finalize_source_manifest(
            session_dir,
            source_manifest,
            source_results,
            start_perf=start_perf,
            end_perf=end_perf,
            device_perf=tuple(row[0] for row in device_rows),
            laser_perf=tuple(row[0] for row in laser_rows),
            log_perf=tuple(row[0] for row in log_rows),
            trial_perf=trial_perf,
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
            if source["gapCount"] and not str(source_id).startswith("nidaq."):
                incomplete_reasons.append(
                    f"{source_id} reported {source['gapCount']} acquisition gap(s)"
                )
            if source["overrunCount"] and not str(source_id).startswith("nidaq."):
                incomplete_reasons.append(
                    f"{source_id} overran by {source['overrunCount']} sample/event(s)"
                )
        alignment = {
            "schemaVersion": 1,
            "metadataGenerationId": metadata_generation_id,
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
                "trials": SessionDataRecorder._alignment_entry(
                    "trials.jsonl",
                    trial_perf,
                    start_perf,
                    "pellet send dispatch on time.perf_counter",
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
        atomic_write_json(
            streams_dir / "alignment.json",
            alignment,
            session_dir=session_dir,
            generation_id=metadata_generation_id,
        )
        critical_paths = (
            streams_dir / "device.csv",
            streams_dir / "trials.jsonl",
            streams_dir / "trial_summary.json",
            streams_dir / "laser.csv",
            logs_dir / "session.log",
            streams_dir / "alignment.json",
            streams_dir / "camera_alignment.json",
            Path(project.get_frame_timing_path()),
            streams_dir / "nidaq.h5",
            *sorted((streams_dir / "tracking").glob("*.json")),
        )
        stream_manifest = {
            "schemaVersion": 1,
            "metadataGenerationId": metadata_generation_id,
            "sessionId": project.short_id,
            "sessionComplete": not incomplete_reasons,
            "enabledSources": finalized_sources,
            "files": [
                file_manifest_entry(path, relative_to=session_dir)
                for path in critical_paths
                if path.is_file()
            ],
        }
        atomic_write_json(
            streams_dir / "stream_manifest.json",
            stream_manifest,
            session_dir=session_dir,
            generation_id=metadata_generation_id,
        )
        return {
            "metadataGenerationId": metadata_generation_id,
            "cameraNidaqAlignment": camera_nidaq_alignment,
            "toneConfirmation": tone_confirmation,
            "deviceEventOverruns": int(device_event_overruns),
            "sessionComplete": not incomplete_reasons,
            "incompleteReasons": tuple(incomplete_reasons),
            "enabledSources": finalized_sources,
        }

    @staticmethod
    def _nidaq_stream_health(chunks, *, start_perf, end_perf, saved_perf):
        diagnostics = {
            "expectedStartPerfTime": float(start_perf),
            "expectedEndPerfTime": float(end_perf),
            "savedSampleCount": int(len(saved_perf)),
        }
        warnings = []
        failure = ""
        if isinstance(chunks, _NidaqSpoolSnapshot):
            diagnostics.update({
                "collectionSampleCount": chunks.sample_count,
                "firstSampleIndex": chunks.first_sample_index,
                "lastSampleIndex": chunks.last_sample_index,
                "firstPerfTime": chunks.first_perf_time,
                "lastPerfTime": chunks.last_perf_time,
                "collectionErrorCount": chunks.error_count,
                "firstCollectionError": chunks.first_error,
                "lastCollectionError": chunks.last_error,
                "workerFailed": chunks.worker_failed,
            })
            if chunks.error_count:
                warnings.append(
                    f"NI-DAQ recovered from {chunks.error_count} collection error(s); "
                    f"first: {chunks.first_error}"
                )
            if chunks.worker_failed:
                failure = chunks.last_error or "NI-DAQ session recorder failed"
        if not len(saved_perf):
            failure = failure or "enabled NI-DAQ stream saved zero samples"
            return {
                "failure": failure,
                "warnings": tuple(warnings),
                "diagnostics": diagnostics,
            }
        first_missing = max(0.0, float(saved_perf[0]) - float(start_perf))
        last_missing = max(0.0, float(end_perf) - float(saved_perf[-1]))
        diagnostics["startCoverageMissingSeconds"] = first_missing
        diagnostics["endCoverageMissingSeconds"] = last_missing
        for edge, missing in (("start", first_missing), ("end", last_missing)):
            if missing <= 0:
                continue
            message = f"NI-DAQ {edge} boundary coverage missing by {missing:.6f} seconds"
            if missing > 5.0:
                failure = failure or message
            else:
                warnings.append(message)
        gap_count, overrun_samples = SessionDataRecorder._nidaq_diagnostics(chunks)
        diagnostics["gapCount"] = gap_count
        diagnostics["overrunSamples"] = overrun_samples
        if gap_count:
            warnings.append(f"NI-DAQ reported {gap_count} recovered acquisition gap(s)")
        if overrun_samples:
            warnings.append(
                f"NI-DAQ session reader overran by {overrun_samples} sample(s)"
            )
        return {
            "failure": failure,
            "warnings": tuple(warnings),
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _finalize_source_manifest(
        session_dir,
        source_manifest,
        source_results,
        *,
        start_perf,
        end_perf,
        device_perf,
        laser_perf,
        log_perf,
        trial_perf,
        nidaq_perf,
        nidaq_chunks,
        device_event_overruns,
    ):
        gap_count = max(
            SessionDataRecorder._nidaq_diagnostics(nidaq_chunks)[0],
            0,
        )
        nidaq_overruns = SessionDataRecorder._nidaq_diagnostics(nidaq_chunks)[1]
        stream_stats = {
            "device": (
                device_perf,
                int(device_event_overruns),
                0,
            ),
            "laser_outputs": (laser_perf, 0, 0),
            "session_logs": (log_perf, 0, 0),
            "trials": (trial_perf, 0, 0),
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
            source["warnings"] = list(result.get("warnings", ()))
            source["diagnostics"] = dict(result.get("diagnostics", {}))
            source["sampleCount"] = int(sample_count or 0)
            if (
                (source_id.startswith("camera.") or source_id == "pose")
                and source["sampleCount"] > 0
                and len(perf_times) == 0
            ):
                perf_times = (float(start_perf), float(end_perf))
                source["timingSource"] = "canonical_camera_boundary"
            source["firstOffsetSeconds"] = (
                None
                if len(perf_times) == 0
                else float(perf_times[0] - start_perf)
            )
            source["lastOffsetSeconds"] = (
                None
                if len(perf_times) == 0
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
    def _iter_nidaq_chunks(chunks, *, block_size=65536):
        if not isinstance(chunks, _NidaqSpoolSnapshot):
            yield from chunks
            return
        with h5py.File(chunks.path, "r") as source:
            count = int(source["sample_index"].shape[0])
            for start in range(0, count, block_size):
                end = min(count, start + block_size)
                yield (
                    source["sample_index"][start:end],
                    source["perf_time"][start:end],
                    source["wall_time"][start:end],
                    source["values"][:, start:end],
                    chunks.channel_names,
                    chunks.sample_rate_hz,
                    int(source["epoch"][end - 1]) if end > start else 0,
                    chunks.gap_count,
                    chunks.overrun_samples,
                    float(source["block_observation_perf_time"][end - 1]),
                    float(source["block_observation_wall_time"][end - 1]),
                )

    @staticmethod
    def _nidaq_perf_values(chunks, start_perf, end_perf):
        if isinstance(chunks, _NidaqSpoolSnapshot):
            with h5py.File(chunks.path, "r") as source:
                perf = source["perf_time"]
                first_index = SessionDataRecorder._h5_searchsorted(
                    perf, start_perf, side="left",
                )
                end_index = SessionDataRecorder._h5_searchsorted(
                    perf, end_perf, side="right",
                )
                count = max(0, end_index - first_index)
                return _NidaqPerfSummary(
                    count=count,
                    first=(None if not count else float(perf[first_index])),
                    last=(None if not count else float(perf[end_index - 1])),
                )
        return np.asarray(tuple(
            float(sample_perf)
            for chunk in chunks
            for sample_perf in chunk[1]
            if start_perf <= sample_perf <= end_perf
        ), dtype=np.float64)

    @staticmethod
    def _h5_searchsorted(dataset, value, *, side):
        low, high = 0, int(dataset.shape[0])
        while low < high:
            middle = (low + high) // 2
            observed = float(dataset[middle])
            if observed < value or (side == "right" and observed == value):
                low = middle + 1
            else:
                high = middle
        return low

    @staticmethod
    def _nidaq_diagnostics(chunks):
        if isinstance(chunks, _NidaqSpoolSnapshot):
            return chunks.gap_count, chunks.overrun_samples
        return (
            max((int(chunk[7]) for chunk in chunks), default=0),
            sum(int(chunk[8]) for chunk in chunks),
        )

    @staticmethod
    def _nidaq_arrays(chunks, requested_names=None):
        if isinstance(chunks, _NidaqSpoolSnapshot):
            names = chunks.channel_names
            selected_names = (
                names if requested_names is None
                else tuple(name for name in requested_names if name in names)
            )
            rows = [names.index(name) for name in selected_names]
            with h5py.File(chunks.path, "r") as source:
                indices = source["sample_index"][:]
                perf = source["perf_time"][:]
                values = (
                    source["values"][rows, :]
                    if rows
                    else np.empty((0, len(indices)), dtype=np.float32)
                )
            return selected_names, indices, perf, values, chunks.sample_rate_hz
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
    def _transition_positions(values: np.ndarray) -> np.ndarray:
        """Return observed digital transitions, excluding the initial state."""
        values = np.asarray(values, dtype=np.float64)
        if values.size < 2:
            return np.empty(0, dtype=np.int64)
        high = np.isfinite(values) & (values > 0.5)
        return np.flatnonzero(high[1:] != high[:-1]) + 1

    @staticmethod
    def _match_camera_nidaq_edge(boundary, start_perf, chunks) -> dict:
        names, indices, perf, values, rate = SessionDataRecorder._nidaq_arrays(
            chunks, requested_names=("cam_frames",),
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
            "edgePolarity": None,
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
        edge_positions = SessionDataRecorder._transition_positions(
            channel_values
        )
        if edge_positions.size == 0:
            return {
                **base,
                "status": "unmatched",
                "confidence": "none",
                "reason": "cam_frames contained no observed transition",
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
            "edgePolarity": (
                "rising" if channel_values[best_position] > 0.5 else "falling"
            ),
            "reason": (
                "Nearest cam_frames transition on the NI-DAQ sample timeline"
            ),
        }

    @staticmethod
    def _correlate_tone_confirmations(
        device_rows, chunks, *, start_perf=-math.inf, end_perf=math.inf,
    ) -> dict:
        names, indices, perf, values, rate = SessionDataRecorder._nidaq_arrays(
            chunks,
            requested_names=("tone1", "tone2", "tone3_r", "tone3_l"),
        )
        tone_channels = tuple(
            name
            for name in ("tone1", "tone2", "tone3_r", "tone3_l")
            if name in names
        )
        minimum_samples = (
            2
            if rate is None
            else max(2, int(math.ceil(
                rate * _MINIMUM_VALID_TONE_PULSE_SECONDS,
            )))
        )
        pulses = {
            channel: SessionDataRecorder._digital_pulses(
                values[names.index(channel)], indices, perf,
                start_perf=start_perf,
                end_perf=end_perf,
                minimum_samples=minimum_samples,
                sample_rate=rate,
            )
            for channel in tone_channels
        }
        events = []
        previous_states = {channel: False for channel in tone_channels}
        tone_status_states = {channel: False for channel in tone_channels}
        for row_index, row in enumerate(sorted(device_rows, key=lambda item: item[0])):
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
                            "rowIndex": row_index,
                        })
                    previous_states[channel] = state
            elif kind == "TONE_STATUS" and isinstance(decoded, dict):
                frequency = int(decoded.get("frequency_hz", 0))
                remaining_ms = int(decoded.get("time_remaining_ms", 0))
                if remaining_ms <= 0:
                    # The board reports an idle generator as frequency zero, so
                    # this single status ends whichever named tone was active.
                    for known_channel in tone_status_states:
                        tone_status_states[known_channel] = False
                    continue
                channel = {5000: "tone1", 6000: "tone2"}.get(frequency)
                if channel not in tone_channels:
                    continue
                state = True
                if state and not tone_status_states[channel]:
                    events.append({
                        "channel": channel,
                        "eventPerfTime": float(perf_time),
                        "direction": direction,
                        "kind": kind,
                        "target": target,
                        "context": context,
                        "rowIndex": row_index,
                    })
                tone_status_states[channel] = state
            elif kind == "PLAY_TONE":
                frequency = None
                if isinstance(decoded, (tuple, list)) and decoded:
                    frequency = int(decoded[0])
                elif isinstance(decoded, dict):
                    frequency = int(decoded.get("frequency_hz", 0))
                channel = {5000: "tone1", 6000: "tone2"}.get(frequency)
                events.append({
                    "channel": channel,
                    "eventPerfTime": float(perf_time),
                    "direction": direction,
                    "kind": kind,
                    "target": target,
                    "context": context,
                    "payload": decoded,
                    "rowIndex": row_index,
                })

        matched_events = {channel: {} for channel in tone_channels}
        unmatched_events = []
        for event in events:
            channel = event["channel"]
            if channel is None:
                unmatched_events.append({
                    **event,
                    "reason": "Tone frequency does not identify a confirmation line",
                })
                continue
            candidates = [
                (pulse_index, pulse)
                for pulse_index, pulse in enumerate(pulses[channel])
                if pulse["valid"]
            ]
            if not candidates:
                unmatched_events.append({
                    **event,
                    "reason": f"{channel} contained no valid pulse in the session",
                })
                continue
            pulse_index, pulse = min(
                candidates,
                key=lambda item: abs(
                    item[1]["perfTime"] - event["eventPerfTime"]
                ),
            )
            latency = pulse["perfTime"] - event["eventPerfTime"]
            if abs(latency) > 0.25:
                unmatched_events.append({
                    **event,
                    "nearestEdgePerfTime": pulse["perfTime"],
                    "reason": "Nearest electrical edge was more than 250 ms away",
                })
                continue
            matched_events[channel].setdefault(pulse_index, []).append({
                **event,
                "latencySeconds": latency,
            })

        preference = {"TONE_STATUS": 0, "PLAY_TONE": 1, "STIMULUS_INPUTS": 2}
        matched = []
        for channel in tone_channels:
            for pulse_index, observations in matched_events[channel].items():
                pulse = pulses[channel][pulse_index]
                observations = sorted(
                    observations,
                    key=lambda item: (
                        preference.get(item["kind"], 99),
                        abs(item["latencySeconds"]),
                    ),
                )
                primary = observations[0]
                matched.append({
                    **primary,
                    "sampleIndex": pulse["sampleIndex"],
                    "edgePerfTime": pulse["perfTime"],
                    "alignedEventPerfTime": pulse["perfTime"],
                    "pulseDurationSeconds": pulse["durationSeconds"],
                    "resolutionSeconds": None if rate is None else 1.0 / rate,
                    "observations": observations,
                })

        unmatched_edges = [
            {
                "channel": channel,
                **pulse,
            }
            for channel in tone_channels
            for pulse_index, pulse in enumerate(pulses[channel])
            if pulse["valid"] and pulse_index not in matched_events[channel]
        ]
        artifacts = [
            {"channel": channel, **pulse}
            for channel in tone_channels
            for pulse in pulses[channel]
            if not pulse["valid"]
        ]
        return {
            "status": (
                "unavailable"
                if not tone_channels
                else "complete"
                if not unmatched_events and not unmatched_edges
                else "partial"
            ),
            "signalQuality": "artifacts_detected" if artifacts else "clean",
            "channels": list(tone_channels),
            "matched": matched,
            "unmatchedEvents": unmatched_events,
            "unmatchedEdges": unmatched_edges,
            "artifacts": artifacts,
            "artifactCount": len(artifacts),
            "resolutionSeconds": None if rate is None else 1.0 / rate,
        }

    @staticmethod
    def _digital_pulses(
        values,
        indices,
        perf,
        *,
        start_perf,
        end_perf,
        minimum_samples,
        sample_rate,
    ):
        binary = np.asarray(values, dtype=np.float32) >= 0.5
        if not binary.size:
            return []
        starts = list(np.flatnonzero((~binary[:-1]) & binary[1:]) + 1)
        if binary[0]:
            starts.insert(0, 0)
        ends = list(np.flatnonzero(binary[:-1] & (~binary[1:])) + 1)
        if binary[-1]:
            ends.append(len(binary))
        pulses = []
        for position, end_position in zip(starts, ends):
            pulse_perf = float(perf[position])
            if not start_perf <= pulse_perf <= end_perf:
                continue
            sample_count = int(end_position - position)
            pulses.append({
                "sampleIndex": int(indices[position]),
                "perfTime": pulse_perf,
                "sampleCount": sample_count,
                "durationSeconds": (
                    None
                    if sample_rate is None
                    else sample_count / sample_rate
                ),
                "valid": sample_count >= minimum_samples,
                "classification": (
                    "valid_pulse"
                    if sample_count >= minimum_samples
                    else "short_pulse_artifact"
                ),
            })
        return pulses

    @staticmethod
    def _alignment_entry(path, perf_times, start_perf, clock):
        if len(perf_times):
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
    def _write_json_lines(path: Path, records) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                json.dump(record, stream, sort_keys=True)
                stream.write("\n")

    @staticmethod
    def _atomic_write_csv(path, session_dir, generation_id, header, rows) -> None:
        rows = tuple(rows)
        atomic_publish_file(
            path,
            lambda staged: SessionDataRecorder._write_csv(staged, header, rows),
            session_dir=session_dir,
            generation_id=generation_id,
        )

    @staticmethod
    def _atomic_write_json_lines(
        path, session_dir, generation_id, records,
    ) -> None:
        records = tuple(records)

        def validate(staged):
            with staged.open("r", encoding="utf-8") as stream:
                for line in stream:
                    json.loads(line)

        atomic_publish_file(
            path,
            lambda staged: SessionDataRecorder._write_json_lines(staged, records),
            session_dir=session_dir,
            generation_id=generation_id,
            validate=validate,
        )

    @staticmethod
    def _validate_h5(path: Path) -> None:
        with h5py.File(path, "r") as stream:
            for required in ("sample_index", "perf_time", "values"):
                if required not in stream:
                    raise RuntimeError(f"NI-DAQ output lacks required dataset {required}")

    @staticmethod
    def _write_nidaq(
        path,
        start_perf,
        start_wall,
        end_perf,
        chunks,
        timing_plan=None,
    ) -> None:
        if isinstance(chunks, _NidaqSpoolSnapshot):
            SessionDataRecorder._write_nidaq_spool_selection(
                path,
                start_perf,
                start_wall,
                end_perf,
                chunks,
                timing_plan,
            )
            return
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

    @staticmethod
    def _write_nidaq_spool_selection(
        path,
        start_perf,
        start_wall,
        end_perf,
        snapshot: _NidaqSpoolSnapshot,
        timing_plan=None,
    ) -> None:
        with h5py.File(snapshot.path, "r") as source, h5py.File(path, "w") as output:
            perf_source = source["perf_time"]
            count = int(perf_source.shape[0])
            if count:
                # The sample-index reconstruction is strictly monotonic, so
                # binary search finds the camera-boundary slice without loading
                # the complete values matrix into Python memory.
                first = SessionDataRecorder._h5_searchsorted(
                    perf_source, start_perf, side="left",
                )
                last = SessionDataRecorder._h5_searchsorted(
                    perf_source, end_perf, side="right",
                )
            else:
                first = last = 0
            selected_count = max(0, last - first)
            output.attrs["recording_start_perf"] = start_perf
            output.attrs["recording_start_wall"] = start_wall
            output.attrs["recording_end_perf"] = end_perf
            output.attrs["alignment"] = (
                "first sample at or after primary camera first frame"
            )
            output.attrs["source_clock"] = (
                "NI-DAQ ring reconstructed on time.perf_counter"
            )
            output.attrs["channel_names"] = snapshot.channel_names
            output.attrs["timing_plan_json"] = json.dumps(
                None if timing_plan is None else dataclasses.asdict(timing_plan),
                sort_keys=True,
            )
            output.attrs["collection_error_count"] = snapshot.error_count
            output.attrs["collection_first_error"] = snapshot.first_error or ""
            output.attrs["collection_last_error"] = snapshot.last_error or ""
            output.attrs["collection_worker_failed"] = snapshot.worker_failed
            output.attrs["gap_count"] = snapshot.gap_count
            output.attrs["overrun_samples"] = snapshot.overrun_samples

            one_dimensional = (
                "sample_index",
                "perf_time",
                "epoch",
                "block_observation_perf_time",
                "block_observation_wall_time",
            )
            for name in one_dimensional:
                source_dataset = source[name]
                output_dataset = output.create_dataset(
                    name,
                    shape=(selected_count,),
                    dtype=source_dataset.dtype,
                    compression="gzip" if selected_count else None,
                )
                for relative in range(0, selected_count, 65536):
                    amount = min(65536, selected_count - relative)
                    output_dataset[relative:relative + amount] = source_dataset[
                        first + relative:first + relative + amount
                    ]
            values = output.create_dataset(
                "values",
                shape=(len(snapshot.channel_names), selected_count),
                dtype=np.float32,
                compression="gzip" if selected_count else None,
            )
            for relative in range(0, selected_count, 65536):
                amount = min(65536, selected_count - relative)
                values[:, relative:relative + amount] = source["values"][
                    :, first + relative:first + relative + amount
                ]
            perf_dataset = output["perf_time"]
            offsets = output.create_dataset(
                "offset_seconds",
                shape=(selected_count,),
                dtype=np.float64,
                compression="gzip" if selected_count else None,
            )
            wall = output.create_dataset(
                "wall_time",
                shape=(selected_count,),
                dtype=np.float64,
                compression="gzip" if selected_count else None,
            )
            for relative in range(0, selected_count, 65536):
                amount = min(65536, selected_count - relative)
                selected_perf = perf_dataset[relative:relative + amount]
                offsets[relative:relative + amount] = selected_perf - start_perf
                wall[relative:relative + amount] = (
                    start_wall + selected_perf - start_perf
                )
            if selected_count:
                first_perf = float(perf_dataset[0])
                last_perf = float(perf_dataset[-1])
                output.attrs["first_perf_time"] = first_perf
                output.attrs["last_perf_time"] = last_perf
                output.attrs["first_offset_seconds"] = first_perf - start_perf
                output.attrs["last_offset_seconds"] = last_perf - start_perf
                output.attrs["sample_rate_hz"] = snapshot.sample_rate_hz
                output.attrs["epoch"] = int(output["epoch"][-1])
