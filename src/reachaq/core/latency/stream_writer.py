"""Bounded, non-blocking HDF5 writer for one process's latency rows.

The rows are recorded on paths that must never wait: the 900 Hz stim capture
loop runs at real-time priority, and a camera loop that stalls loses frames.
So append() only copies a row into a preallocated batch under a short lock;
full batches go to a bounded queue, and a daemon thread does all file I/O.
When the queue is full the batch is dropped and counted, never waited for:
latency data is diagnostic, and a gap in it is reported rather than paid for
with acquisition time.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy

from .schema import CLOCK_PAIR_DTYPE, CLOCK_PAIRS, LATENCY_SCHEMA_VERSION

logger = logging.getLogger(__name__)

LATENCY_RECORD_ENV_VAR = "REACHAQ_LATENCY_RECORD"


def latency_recording_enabled() -> bool:
    """False when REACHAQ_LATENCY_RECORD is 0/false/no/off: the overhead A/B switch.

    Read when each recording starts, so a child process sees the value it was
    spawned with.
    """
    value = os.environ.get(LATENCY_RECORD_ENV_VAR, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def latency_stream_path(project, name: str) -> Path:
    """Where a raw latency stream for the project's current session lives."""
    return (
        Path(project.get_session_path().location) / "streams" / "latency" / f"{name}.h5"
    )


class LatencyStreamWriter:
    """One HDF5 file of compound datasets, appended in batches by a daemon thread.

    append() never raises and never waits: invalid dataset names and bad row data
    (wrong arity, type error, overflow) are rejected and counted in
    stats["rowsRejected"], and a full queue drops the batch and counts it in
    stats["rowsDropped"], with no log. The writer thread always stops within the
    caller-supplied timeout, even if writes fail.
    """

    def __init__(
        self,
        path,
        datasets: Mapping[str, numpy.dtype],
        *,
        attrs: Optional[Mapping[str, object]] = None,
        batch_rows: int = 512,
        queue_batches: int = 64,
        clock_pair_period: float = 1.0,
    ):
        if batch_rows < 1 or queue_batches < 1:
            raise ValueError("latency writer buffers must be positive")
        if CLOCK_PAIRS in datasets:
            raise ValueError(f"{CLOCK_PAIRS!r} is written by the writer itself")
        self.path = Path(path)
        self._dtypes: Dict[str, numpy.dtype] = {
            **{name: numpy.dtype(dtype) for name, dtype in datasets.items()},
            CLOCK_PAIRS: CLOCK_PAIR_DTYPE,
        }
        self._attrs = dict(attrs or {})
        self._batch_rows = int(batch_rows)
        self._buffers = {
            name: numpy.empty(self._batch_rows, dtype=dtype)
            for name, dtype in self._dtypes.items()
        }
        self._counts = {name: 0 for name in self._dtypes}
        self._rows_written = {name: 0 for name in self._dtypes}
        self._rows_dropped = {name: 0 for name in self._dtypes}
        self._rows_rejected = 0
        self._rejection_logged = False
        self._queue: "queue.Queue" = queue.Queue(maxsize=int(queue_batches))
        self._lock = threading.Lock()
        self._closed = False
        self._failed = False
        self._first_error = ""
        self._high_water = 0
        self._stop_event = threading.Event()
        self._clock_pair_period = float(clock_pair_period)
        self._thread = threading.Thread(
            target=self._run, name=f"LatencyWriter-{self.path.stem}", daemon=True,
        )
        self._thread.start()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "rowsWritten": dict(self._rows_written),
                "rowsDropped": dict(self._rows_dropped),
                "rowsRejected": self._rows_rejected,
                "queueHighWater": self._high_water,
                "failed": self._failed,
                "firstError": self._first_error or None,
            }

    def append(self, dataset: str, row) -> None:
        """Copy one row into its batch. Never waits on I/O and never raises.

        Invalid dataset names and bad row data (wrong arity, type error, overflow, NaN)
        result in silent rejection, a count in stats["rowsRejected"], and a single ERROR
        log per writer. A full queue silently drops the batch and counts it in
        stats["rowsDropped"], with no log.
        """
        with self._lock:
            if self._closed:
                return
            try:
                index = self._counts[dataset]
                buffer = self._buffers[dataset]
                buffer[index] = row
            except Exception as error:
                # Log once per writer on first rejection.
                self._rows_rejected += 1
                if not self._rejection_logged:
                    self._rejection_logged = True
                    logger.error("Latency stream %s rejected a row (%s, dataset=%s): %s",
                                 self.path, type(error).__name__, dataset, error)
                return
            index += 1
            if index == self._batch_rows:
                self._counts[dataset] = 0
                self._enqueue_locked(dataset, buffer.copy())
            else:
                self._counts[dataset] = index

    def close(self, timeout: float = 10.0) -> dict:
        """Flush partial batches, signal the thread to stop, and return the final stats.

        Always returns within the caller's deadline, even if the writer thread fails.
        Queued batches are still written before the thread exits.
        """
        deadline = time.perf_counter() + timeout
        with self._lock:
            closed_already = self._closed
            if not closed_already:
                self._closed = True
                for name, count in self._counts.items():
                    if count:
                        self._enqueue_locked(name, self._buffers[name][:count].copy())
                        self._counts[name] = 0
        if not closed_already:
            self._stop_event.set()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass  # The stop event covers the full queue case.
        remaining = max(0.0, deadline - time.perf_counter())
        self._thread.join(remaining)
        if self._thread.is_alive():
            self._fail(TimeoutError("latency writer thread did not stop"))
        return self.stats

    def _enqueue_locked(self, dataset: str, batch: numpy.ndarray) -> None:
        if self._failed:
            self._rows_dropped[dataset] += len(batch)
            return
        try:
            self._queue.put_nowait((dataset, batch))
        except queue.Full:
            self._rows_dropped[dataset] += len(batch)
            return
        self._high_water = max(self._high_water, self._queue.qsize())

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            first = not self._failed
            self._failed = True
            if not self._first_error:
                self._first_error = f"{type(error).__name__}: {error}"
        if first:
            logger.error("Latency stream %s failed; its rows are dropped from now on: %s",
                         self.path, error)

    def _clock_pair(self) -> numpy.ndarray:
        # Bracket the wall read with two perf reads and keep their midpoint, so
        # the pair is off by half the bracket at most rather than a whole read.
        before = time.perf_counter()
        wall = time.time()
        after = time.perf_counter()
        return numpy.array([(wall, (before + after) / 2)], dtype=CLOCK_PAIR_DTYPE)

    def _run(self) -> None:
        try:
            import h5py
            self.path.parent.mkdir(parents=True, exist_ok=True)
            store = h5py.File(self.path, "w")
        except Exception as error:
            self._fail(error)
            self._drain_discarding()
            return
        try:
            store.attrs["schema_version"] = LATENCY_SCHEMA_VERSION
            for key, value in self._attrs.items():
                store.attrs[key] = value
            datasets = {
                name: store.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype,
                    chunks=(self._batch_rows,), compression="lzf", shuffle=True,
                )
                for name, dtype in self._dtypes.items()
            }
            next_pair = 0.0
            while True:
                now = time.perf_counter()
                if now >= next_pair:
                    self._write(datasets, CLOCK_PAIRS, self._clock_pair())
                    next_pair = now + self._clock_pair_period
                try:
                    # Once stop is signaled, use a short timeout like the drain path does,
                    # so we notice the stop event promptly even if the queue fills.
                    now = time.perf_counter()
                    time_to_pair = max(0.0, next_pair - now)
                    timeout = min(time_to_pair, 0.05) if self._stop_event.is_set() else time_to_pair
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    # No data and no stop signal: keep polling for clock pairs.
                    if self._stop_event.is_set() and self._queue.empty():
                        break
                    continue
                if item is None:
                    # Received the stop sentinel: exit after writing queued batches.
                    break
                name, batch = item
                self._write(datasets, name, batch)
            self._write(datasets, CLOCK_PAIRS, self._clock_pair())
        except Exception as error:
            self._fail(error)
            self._drain_discarding()
        finally:
            try:
                store.attrs["writer_stats"] = json.dumps(self.stats, sort_keys=True)
                store.close()
            except Exception as error:
                self._fail(error)

    def _write(self, datasets, name: str, batch: numpy.ndarray) -> None:
        with self._lock:
            failed = self._failed
            if failed:
                self._rows_dropped[name] += len(batch)
        if failed:
            return
        dataset = datasets[name]
        start = dataset.shape[0]
        dataset.resize((start + len(batch),))
        dataset[start:] = batch
        with self._lock:
            self._rows_written[name] += len(batch)

    def _drain_discarding(self) -> None:
        """Keep consuming after a failure, so close() never waits on a full queue.

        Polls with timeout and respects the stop event, so the thread always exits
        even if the sentinel was never delivered or if a write fails.
        """
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                # No data available. Exit if the stop event is set and queue is empty.
                if self._stop_event.is_set() and self._queue.empty():
                    return
                continue
            if item is None:
                return
            name, batch = item
            with self._lock:
                self._rows_dropped[name] += len(batch)
