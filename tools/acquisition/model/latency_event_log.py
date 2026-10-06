"""Latency rows the GUI process owns, kept in memory and written at stop.

The GUI process is where the Python lock is scarce (latency-report.md §2), so
nothing here does I/O while recording: a row is one assignment into a
preallocated block. The session recorder writes the blocks to
streams/latency/events.h5 as it finalizes, before the latency finalizer reads
them.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Dict, List, Mapping

import numpy

from autotrainer.core.latency import latency_recording_enabled
from autotrainer.core.latency.schema import EVENT_TABLES, LATENCY_SCHEMA_VERSION

logger = logging.getLogger(__name__)

_INT16 = numpy.iinfo(numpy.int16)


class _BlockTable:
    """Rows in fixed-size blocks, so growth never copies what is already kept."""

    def __init__(self, dtype: numpy.dtype, block_rows: int):
        self._dtype = numpy.dtype(dtype)
        self._block_rows = int(block_rows)
        self._blocks: List[numpy.ndarray] = []
        self._count = 0  # rows used in the last block

    def clear(self) -> None:
        self._blocks = []
        self._count = 0

    def append(self, row) -> None:
        if not self._blocks or self._count == self._block_rows:
            self._blocks.append(numpy.empty(self._block_rows, dtype=self._dtype))
            self._count = 0
        self._blocks[-1][self._count] = row
        self._count += 1

    def snapshot(self) -> numpy.ndarray:
        if not self._blocks:
            return numpy.empty(0, dtype=self._dtype)
        return numpy.concatenate([*self._blocks[:-1], self._blocks[-1][:self._count]])


def _text(value, size: int) -> bytes:
    return str("" if value is None else value).encode("utf-8")[:size]


def _perf(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _flag(value) -> int:
    return -1 if value is None else int(bool(value))


class LatencyEventLog:
    """Per-recording GUI-process latency rows; inactive outside a recording.

    The record_* methods never raise. They run on the pose-receive thread, the
    reach gate, the stim paths and the CAN path, and a diagnostic row must not
    be able to break any of them. A row that cannot be built or stored is
    counted in rows_rejected, and the first one of each recording is logged.
    """

    def __init__(self, *, block_rows: int = 16384):
        self._lock = threading.Lock()
        self._active = False
        self._enabled = False
        self._rows_rejected = 0
        self._reject_logged = False
        self._tables: Dict[str, _BlockTable] = {
            name: _BlockTable(dtype, block_rows) for name, dtype in EVENT_TABLES.items()
        }

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def enabled(self) -> bool:
        """Whether the latest begin() found the latency record switched on.

        Unlike is_active it stays set after end(), so the session recorder can
        tell a recording with the switch off (nothing to write) from one that
        simply produced no rows."""
        return self._enabled

    @property
    def rows_rejected(self) -> int:
        """Rows of the latest recording that could not be built or stored."""
        return self._rows_rejected

    def begin(self) -> None:
        with self._lock:
            for table in self._tables.values():
                table.clear()
            self._rows_rejected = 0
            self._reject_logged = False
            self._enabled = latency_recording_enabled()
            self._active = self._enabled

    def end(self) -> Dict[str, numpy.ndarray]:
        with self._lock:
            self._active = False
            return {name: table.snapshot() for name, table in self._tables.items()}

    def _append(self, name: str, row) -> None:
        try:
            with self._lock:
                if self._active:
                    self._tables[name].append(row)
        except Exception:
            self._reject(name)

    def _reject(self, name: str) -> None:
        """Count a row that could not be recorded; call from an except block.

        The log line is written after the lock is released: the session log
        handler takes the recorder's lock, and the recorder holds that lock
        while it calls begin() and end() here."""
        with self._lock:
            self._rows_rejected += 1
            first = not self._reject_logged
            self._reject_logged = True
        if first:
            logger.error(
                "A latency row for %s could not be recorded and was dropped; "
                "later rejects this recording are counted, not logged",
                name, exc_info=True)

    def record_live_pose(self, response, gui_recv_perf: float) -> None:
        if not self._active:
            return
        try:
            row = (
                int(response.sequence),
                _perf(getattr(response, "live_recv_perf_c", math.nan)),
                _perf(response.perf_c),
                _perf(getattr(response, "live_put_perf_c", math.nan)),
                _perf(gui_recv_perf),
            )
        except Exception:
            self._reject("live_poses")
            return
        self._append("live_poses", row)

    def record_gate_observation(self, observe_perf: float, sample, reaching) -> None:
        if not self._active:
            return
        try:
            row = (
                _perf(observe_perf),
                -1 if sample is None else int(sample.sequence),
                math.nan if sample is None else _perf(sample.processing_perf),
                _flag(reaching),
            )
        except Exception:
            self._reject("gate_observations")
            return
        self._append("gate_observations", row)

    def record_stim_dispatch(self, payload: Mapping[str, object], *, gui_recv_perf=None) -> None:
        """One stim trigger. The direct route's payload already carries its IPC
        stamps; the STIM3 route passes the GUI receive time it took itself."""
        if not self._active:
            return
        try:
            send = payload.get("ipc_send_perf_time", payload.get("msg_send_perf_time"))
            received = payload.get("ipc_receive_perf_time") if gui_recv_perf is None else gui_recv_perf
            row = (
                _text(payload.get("operation_id"), 64),
                _text(payload.get("trigger_route"), 24),
                int(payload.get("stim_frame_id", -1)),
                _perf(payload.get("frame_perf_time")),
                _perf(payload.get("decision_perf_time")),
                _perf(payload.get("evidence_done_perf_time")),
                _perf(payload.get("clip_done_perf_time")),
                _perf(send),
                _perf(received),
                _perf(payload.get("validated_perf_time")),
                _perf(payload.get("daqmx_start_entry_perf_time")),
                _perf(payload.get("daqmx_start_return_perf_time")),
                _flag(payload.get("accepted")),
            )
        except Exception:
            self._reject("stim_dispatch")
            return
        self._append("stim_dispatch", row)

    def record_stim3_pulse(self, operation_id, token, call_perf, return_perf) -> None:
        if not self._active:
            return
        try:
            row = (
                _text(operation_id, 64), _text(token, 36), _perf(call_perf), _perf(return_perf),
            )
        except Exception:
            self._reject("stim3_pulses")
            return
        self._append("stim3_pulses", row)

    def record_can(self, stage: str, fields: Mapping[str, object]) -> None:
        if not self._active:
            return
        try:
            uuid = fields.get("can_uuid")
            uuid = -1 if uuid is None else int(uuid)
            # The column is int16. numpy 2 raises on an out-of-range Python int
            # but numpy 1.x wraps it with a DeprecationWarning, which would
            # store a wrong id that joins to the wrong command.
            if not _INT16.min <= uuid <= _INT16.max:
                raise OverflowError(f"can_uuid {uuid} does not fit the stored int16")
            row = (
                _text(stage, 16),
                _text(fields.get("token"), 36),
                _text(fields.get("kind"), 40),
                uuid,
                _perf(fields.get("perf")),
                _perf(fields.get("perf_end")),
                _perf(fields.get("kernel_wall")),
            )
        except Exception:
            self._reject("can_events")
            return
        self._append("can_events", row)

    @staticmethod
    def write(path, tables: Mapping[str, numpy.ndarray]) -> None:
        import h5py
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as store:
            store.attrs["schema_version"] = LATENCY_SCHEMA_VERSION
            for name, dtype in EVENT_TABLES.items():
                rows = tables.get(name)
                if rows is None:
                    rows = numpy.empty(0, dtype=dtype)
                options = {"compression": "lzf", "shuffle": True} if len(rows) else {}
                store.create_dataset(name, data=numpy.asarray(rows, dtype=dtype), **options)
