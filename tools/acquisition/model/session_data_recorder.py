from __future__ import annotations

import csv
import logging
import math
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

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

    def __init__(self, analysis, nidaq_monitor, laser_model):
        self._analysis = analysis
        self._nidaq_monitor = nidaq_monitor
        self._laser_model = laser_model
        self._lock = threading.RLock()
        self._armed = False
        self._project: Optional[ProjectInfo] = None
        self._start_perf: Optional[float] = None
        self._start_wall: Optional[float] = None
        self._device_rows = []
        self._laser_rows = []
        self._log_rows = []
        self._nidaq_chunks = []
        self._nidaq_last_index = None
        self._nidaq_stop = threading.Event()
        self._nidaq_thread: Optional[threading.Thread] = None

        analysis.measurements_sampled += self._on_measurements
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
            self._device_rows = []
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
            laser_rows = tuple(self._laser_rows)
            log_rows = tuple(self._log_rows)
            nidaq_chunks = tuple(self._nidaq_chunks)
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
        )

    def abort(self) -> None:
        self._stop_nidaq_thread()
        with self._lock:
            self._clear_locked()

    def close(self) -> None:
        self.abort()
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
        self._device_rows = []
        self._laser_rows = []
        self._log_rows = []
        self._nidaq_chunks = []
        self._nidaq_last_index = None

    def _on_measurements(self, measurements: Iterable) -> None:
        with self._lock:
            if not self._armed:
                return
            for sample in measurements:
                self._device_rows.append((
                    float(sample.timestamp) / 1e9,
                    float(sample.when),
                    int(bool(sample.switch)),
                    float(sample.pressure),
                    float(sample.temperature),
                    float(sample.humidity),
                ))

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
            # source_perf_time/source_wall_time are stamped when the block is
            # published. Reconstruct each sample working backward from its end.
            sample_lag = (read.end_sample_index - indices) / read.sample_rate_hz
            perf_times = read.source_perf_time - sample_lag
            wall_times = read.source_wall_time - sample_lag
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
    ) -> None:
        session_dir = Path(project.get_session_path().location)
        streams_dir = session_dir / "streams"
        logs_dir = session_dir / "logs"
        streams_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        SessionDataRecorder._write_csv(
            streams_dir / "device.csv",
            ("perf_time", "offset_seconds", "wall_time", "switch", "pressure",
             "temperature", "humidity"),
            (
                (perf, perf - start_perf, wall, switch, pressure, temperature, humidity)
                for perf, wall, switch, pressure, temperature, humidity in device_rows
                if start_perf <= perf <= end_perf
            ),
        )
        SessionDataRecorder._write_csv(
            streams_dir / "laser.csv",
            ("perf_time", "offset_seconds", "wall_time", "event", "channel", "source",
             "command_volts", "diode_volts", "command_copy_volts"),
            (
                (perf, perf - start_perf, wall, event, channel, source, command, diode, copy)
                for perf, wall, event, channel, source, command, diode, copy in laser_rows
                if start_perf <= perf <= end_perf
            ),
        )

        with (logs_dir / "session.log").open("w", encoding="utf-8") as stream:
            stream.write(
                f"# recording_start_perf={start_perf:.9f} "
                f"recording_start_wall={start_wall:.9f} "
                f"recording_end_perf={end_perf:.9f}\n"
            )
            for perf, _, message in log_rows:
                if start_perf <= perf <= end_perf:
                    stream.write(f"[+{perf - start_perf:.6f}s] {message}\n")

        SessionDataRecorder._write_nidaq(
            streams_dir / "nidaq.h5",
            start_perf,
            start_wall,
            end_perf,
            nidaq_chunks,
        )

    @staticmethod
    def _write_csv(path: Path, header, rows) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _write_nidaq(path, start_perf, start_wall, end_perf, chunks) -> None:
        channel_names = next((chunk[4] for chunk in chunks if chunk[4]), tuple())
        selected = []
        for indices, perf, wall, values, names, rate, epoch, gaps, overrun in chunks:
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
                ))
        with h5py.File(path, "w") as output:
            output.attrs["recording_start_perf"] = start_perf
            output.attrs["recording_start_wall"] = start_wall
            output.attrs["recording_end_perf"] = end_perf
            output.attrs["alignment"] = "first sample at or after primary camera first frame"
            output.attrs["channel_names"] = channel_names
            if not selected:
                output.create_dataset("sample_index", data=np.empty(0, dtype=np.int64))
                output.create_dataset("perf_time", data=np.empty(0, dtype=np.float64))
                output.create_dataset("offset_seconds", data=np.empty(0, dtype=np.float64))
                output.create_dataset(
                    "values",
                    data=np.empty((len(channel_names), 0), dtype=np.float32),
                )
                return
            indices = np.concatenate([item[0] for item in selected])
            perf = np.concatenate([item[1] for item in selected])
            values = np.concatenate([item[3] for item in selected], axis=1)
            output.create_dataset("sample_index", data=indices, compression="gzip")
            output.create_dataset("perf_time", data=perf, compression="gzip")
            output.create_dataset("offset_seconds", data=perf - start_perf, compression="gzip")
            output.create_dataset(
                "wall_time",
                data=start_wall + (perf - start_perf),
                compression="gzip",
            )
            output.create_dataset("values", data=values, compression="gzip")
            output.attrs["sample_rate_hz"] = selected[0][4]
            output.attrs["epoch"] = selected[-1][5]
            output.attrs["gap_count"] = selected[-1][6]
            output.attrs["overrun_samples"] = sum(item[7] for item in selected)
