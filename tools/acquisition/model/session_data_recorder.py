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


class _SessionLogHandler(logging.Handler):
    def __init__(self, recorder: "SessionDataRecorder"):
        super().__init__(logging.NOTSET)
        self._recorder = recorder
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._recorder.add_log(
                time.perf_counter(),
                record.created,
                self.format(record),
            )
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
        self._device_rows = deque(maxlen=device_event_capacity)
        self._device_event_capacity = int(device_event_capacity)
        self._device_event_overruns = 0
        self._laser_rows = []
        self._log_rows = []
        self._nidaq_chunks = []
        self._nidaq_last_index = None
        self._nidaq_stop = threading.Event()
        self._nidaq_thread: Optional[threading.Thread] = None

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

    def arm(self, project: ProjectInfo) -> None:
        self.abort()
        with self._lock:
            self._armed = True
            # BehaviorAlgorithm assigns the next session index immediately
            # after arming and before enabling the shared camera trigger.
            # Keep this object reference so the recorder follows that atomic
            # session assignment without racing the first camera frame.
            self._project = project
            self._device_rows = deque(maxlen=self._device_event_capacity)
            self._device_event_overruns = 0
            self._laser_rows = []
            self._log_rows = []
            self._nidaq_chunks = []
            self._nidaq_last_index = None
            self._start_perf = None
            self._start_wall = None
            self._nidaq_stop.clear()
        thread = threading.Thread(
            target=self._poll_nidaq,
            name="session-nidaq-recorder",
            daemon=True,
        )
        self._nidaq_thread = thread
        thread.start()

    def commit_start(self, perf_time: float, wall_time: float) -> None:
        with self._lock:
            if not self._armed:
                return
            self._start_perf = float(perf_time)
            self._start_wall = float(wall_time)

    def stop(self, end_perf: float) -> None:
        self._stop_nidaq_thread()
        with self._lock:
            if not self._armed or self._project is None or self._start_perf is None:
                self._clear_locked()
                return
            project = self._project
            start_perf = self._start_perf
            start_wall = self._start_wall
            end_perf = max(start_perf, float(end_perf))
            device_rows = tuple(self._device_rows)
            device_event_overruns = self._device_event_overruns
            laser_rows = tuple(self._laser_rows)
            log_rows = tuple(self._log_rows)
            nidaq_chunks = tuple(self._nidaq_chunks)
            timing_plan = self._nidaq_monitor.timing_plan
            self._clear_locked()
        self._write_session(
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

    def add_log(self, perf_time: float, wall_time: float, message: str) -> None:
        with self._lock:
            if self._armed:
                self._log_rows.append((float(perf_time), float(wall_time), message))

    def _clear_locked(self) -> None:
        self._armed = False
        self._project = None
        self._start_perf = None
        self._start_wall = None
        self._device_rows = deque(maxlen=self._device_event_capacity)
        self._device_event_overruns = 0
        self._laser_rows = []
        self._log_rows = []
        self._nidaq_chunks = []
        self._nidaq_last_index = None

    def _on_device_message(
        self,
        kind,
        data,
        perf_time: float,
        wall_time: float,
    ) -> None:
        self._append_device_event(
            perf_time,
            wall_time,
            "inbound",
            kind,
            data,
            None,
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
            if not self._armed:
                return
            if len(self._device_rows) == self._device_rows.maxlen:
                self._device_event_overruns += 1
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
    ) -> None:
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
             "command_volts", "diode_volts", "command_copy_volts"),
            (
                (perf, perf - start_perf, wall, event, channel, source, command, diode, copy)
                for perf, wall, event, channel, source, command, diode, copy in laser_rows
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
        alignment = {
            "schemaVersion": 1,
            "canonicalBoundary": {
                "source": "primary_camera_recorded_frames",
                "clock": "time.perf_counter",
                "startPerfTime": start_perf,
                "endPerfTime": end_perf,
                "startWallTime": start_wall,
                "endWallTime": start_wall + (end_perf - start_perf),
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
        }
        with (streams_dir / "alignment.json").open("w", encoding="utf-8") as stream:
            json.dump(alignment, stream, indent=2)
            stream.write("\n")

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
