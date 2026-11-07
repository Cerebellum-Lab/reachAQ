"""Session-output write preflight, capacity estimation, and low-rate telemetry."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class StorageSnapshot:
    sampled_wall_time: float
    free_bytes: int
    used_session_bytes: int
    estimated_bytes_per_second: float
    observed_bytes_per_second: Optional[float]
    projected_remaining_minutes: Optional[float]


@dataclass(frozen=True)
class StoragePreflight:
    target_directory: str
    free_bytes: int
    estimated_bytes_per_second: float
    projected_maximum_minutes: Optional[float]
    configured_duration_seconds: Optional[float]
    duration_fits_projection: bool


def _projected_minutes(free_bytes: int, bytes_per_second: float) -> Optional[float]:
    if bytes_per_second <= 0:
        return None
    return max(0.0, float(free_bytes) / float(bytes_per_second) / 60.0)


def _directory_size(path: Path) -> int:
    total = 0
    try:
        entries = tuple(path.rglob("*")) if path.exists() else ()
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def preflight_storage(
    target_directory: Path,
    *,
    estimated_bytes_per_second: float,
    configured_duration_seconds: Optional[float],
) -> StoragePreflight:
    """Prove a real durable write on the filesystem used by the session."""

    target = Path(target_directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    fd, probe_name = tempfile.mkstemp(prefix=".reachaq-write-probe-", dir=str(target))
    try:
        with os.fdopen(fd, "wb") as probe:
            probe.write(os.urandom(4096))
            probe.flush()
            os.fsync(probe.fileno())
    finally:
        try:
            os.unlink(probe_name)
        except FileNotFoundError:
            pass
    usage = shutil.disk_usage(target)
    estimate = max(0.0, float(estimated_bytes_per_second))
    projected = _projected_minutes(usage.free, estimate)
    duration_fits = (
        configured_duration_seconds is None
        or projected is None
        or float(configured_duration_seconds) <= projected * 60.0
    )
    return StoragePreflight(
        target_directory=str(target.resolve()),
        free_bytes=int(usage.free),
        estimated_bytes_per_second=estimate,
        projected_maximum_minutes=projected,
        configured_duration_seconds=configured_duration_seconds,
        duration_fits_projection=duration_fits,
    )


class RecordingStorageMonitor:
    """Poll storage away from frame writers and report threshold crossings once."""

    def __init__(self, *, interval_seconds: float = 30.0):
        self._interval_seconds = max(1.0, float(interval_seconds))
        self._lock = threading.RLock()
        self._timer: Optional[threading.Timer] = None
        self._generation = 0
        self._target: Optional[Path] = None
        self._estimated_rate = 0.0
        self._last_sample: Optional[StorageSnapshot] = None
        self._samples: Tuple[StorageSnapshot, ...] = ()
        self._warnings: Tuple[dict, ...] = ()
        self._crossed = set()
        self._callback: Optional[Callable[[StorageSnapshot, Optional[int]], None]] = None

    def start(
        self,
        target_directory: Path,
        *,
        estimated_bytes_per_second: float,
        callback: Callable[[StorageSnapshot, Optional[int]], None],
    ) -> None:
        self.stop()
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._target = Path(target_directory)
            self._estimated_rate = max(0.0, float(estimated_bytes_per_second))
            self._last_sample = None
            self._samples = ()
            self._warnings = ()
            self._crossed = set()
            self._callback = callback
        self._sample(generation, reschedule=True)

    def stop(self) -> dict:
        with self._lock:
            self._generation += 1
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        # Capture an end sample without scheduling another timer.
        self._sample(self._generation, reschedule=False, allow_inactive=True)
        return self.snapshot()

    def abort(self) -> None:
        with self._lock:
            self._generation += 1
            timer, self._timer = self._timer, None
            self._target = None
            self._last_sample = None
            self._samples = ()
            self._warnings = ()
            self._crossed = set()
            self._callback = None
        if timer is not None:
            timer.cancel()

    def sample_now(self) -> Optional[StorageSnapshot]:
        with self._lock:
            generation = self._generation
        return self._sample(generation, reschedule=False)

    def snapshot(self) -> dict:
        with self._lock:
            samples = tuple(self._samples)
            warnings = tuple(self._warnings)
            estimated_rate = self._estimated_rate
            target = None if self._target is None else str(self._target)
        observed = tuple(
            sample.observed_bytes_per_second
            for sample in samples
            if sample.observed_bytes_per_second is not None
        )
        return {
            "targetDirectory": target,
            "estimatedBytesPerSecond": estimated_rate,
            "observedBytesPerSecond": (None if not observed else max(observed)),
            "startFreeBytes": (None if not samples else samples[0].free_bytes),
            "endFreeBytes": (None if not samples else samples[-1].free_bytes),
            "projectedRemainingMinutes": (
                None if not samples else samples[-1].projected_remaining_minutes
            ),
            "sampleCount": len(samples),
            "warnings": [dict(item) for item in warnings],
        }

    def _sample(
        self,
        generation: int,
        *,
        reschedule: bool,
        allow_inactive: bool = False,
    ) -> Optional[StorageSnapshot]:
        with self._lock:
            if generation != self._generation and not allow_inactive:
                return None
            target = self._target
            estimated_rate = self._estimated_rate
            previous = self._last_sample
            callback = self._callback
        if target is None:
            return None
        now = time.time()
        free = int(shutil.disk_usage(target).free)
        used = _directory_size(target)
        observed = None
        if previous is not None and now > previous.sampled_wall_time:
            observed = max(
                0.0,
                (used - previous.used_session_bytes)
                / (now - previous.sampled_wall_time),
            )
        effective_rate = max(estimated_rate, observed or 0.0)
        snapshot = StorageSnapshot(
            sampled_wall_time=now,
            free_bytes=free,
            used_session_bytes=used,
            estimated_bytes_per_second=estimated_rate,
            observed_bytes_per_second=observed,
            projected_remaining_minutes=_projected_minutes(free, effective_rate),
        )
        crossed = None
        with self._lock:
            if generation != self._generation and not allow_inactive:
                return None
            self._last_sample = snapshot
            self._samples = (*self._samples, snapshot)
            remaining = snapshot.projected_remaining_minutes
            if remaining is not None:
                for threshold in (30, 10, 1):
                    if remaining < threshold and threshold not in self._crossed:
                        self._crossed.add(threshold)
                        self._warnings = (*self._warnings, {
                            "thresholdMinutes": threshold,
                            "projectedRemainingMinutes": remaining,
                            "freeBytes": free,
                            "sampledWallTime": now,
                        })
                        crossed = threshold
            if reschedule and generation == self._generation:
                timer = threading.Timer(
                    self._interval_seconds,
                    self._sample,
                    args=(generation,),
                    kwargs={"reschedule": True},
                )
                timer.daemon = True
                self._timer = timer
                timer.start()
        if callback is not None:
            callback(snapshot, crossed)
        return snapshot


def preflight_as_dict(preflight: StoragePreflight) -> dict:
    return asdict(preflight)
