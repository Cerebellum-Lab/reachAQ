"""Bounded, acquisition-timed storage for live pose results."""

from __future__ import annotations

import dataclasses
import math
import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class FrameTimelineAnchor:
    """Map primary-camera frame IDs onto the canonical session clock."""

    primary_frame_id: int
    start_perf_time: float
    frame_rate: float

    def __post_init__(self):
        if not math.isfinite(self.start_perf_time):
            raise ValueError("Frame timeline requires a finite start time")
        if not math.isfinite(self.frame_rate) or self.frame_rate <= 0:
            raise ValueError("Frame timeline requires a positive frame rate")

    def perf_time(self, frame_id: int) -> float:
        return self.start_perf_time + (
            int(frame_id) - self.primary_frame_id
        ) / self.frame_rate


@dataclass(frozen=True)
class TrackingLocation:
    name: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TrackingOffset:
    origin: str
    target: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class LiveTrackingSample:
    """Immutable subset of one live pose response needed after a pellet cycle."""

    sequence: int
    primary_frame_ids: Tuple[int, ...]
    primary_frame_perf_times: Tuple[float, ...]
    source_start_perf: float
    source_end_perf: float
    processing_perf: float
    pellet_seen: bool
    locations_3d: Tuple[TrackingLocation, ...]
    offsets_3d: Tuple[TrackingOffset, ...]

    @property
    def representative_frame_id(self) -> int:
        return self.primary_frame_ids[len(self.primary_frame_ids) // 2]

    @property
    def source_perf(self) -> float:
        return (self.source_start_perf + self.source_end_perf) / 2.0

    def location(self, name: str) -> Optional[TrackingLocation]:
        return next((item for item in self.locations_3d if item.name == name), None)

    def to_record(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_pose_response(cls, response, anchor: FrameTimelineAnchor):
        source_ids = tuple(getattr(response, "source_frame_ids", ()))
        primary_ids = tuple(int(value) for value in source_ids[0]) if source_ids else ()
        if not primary_ids:
            return None
        if any(current <= previous for previous, current in zip(primary_ids, primary_ids[1:])):
            raise ValueError("Live primary source frame IDs must be strictly increasing")

        def xyz(value):
            return float(value[0]), float(value[1]), float(value[2])

        locations = tuple(
            TrackingLocation(str(name), *xyz(value))
            for name, value in sorted(response.locations_3d.items(), key=lambda item: str(item[0]))
        )
        offsets = tuple(
            TrackingOffset(str(origin), str(target), *xyz(value))
            for origin, targets in sorted(
                response.parts_3d_offsets.items(), key=lambda item: str(item[0])
            )
            for target, value in sorted(targets.items(), key=lambda item: str(item[0]))
        )
        return cls(
            sequence=int(response.sequence),
            primary_frame_ids=primary_ids,
            primary_frame_perf_times=tuple(
                anchor.perf_time(frame_id) for frame_id in primary_ids
            ),
            source_start_perf=anchor.perf_time(primary_ids[0]),
            source_end_perf=anchor.perf_time(primary_ids[-1]),
            processing_perf=float(response.perf_c),
            pellet_seen=bool(response.pellet_seen),
            locations_3d=locations,
            offsets_3d=offsets,
        )


@dataclass(frozen=True)
class TrackingWindow:
    start_perf: float
    end_perf: float
    samples: Tuple[LiveTrackingSample, ...]
    expected_frames: int
    observed_frames: int
    missing_frame_ids: Tuple[int, ...]
    duplicate_frame_ids: Tuple[int, ...]
    monotonic: bool
    requested_start_frame_id: Optional[int] = None
    requested_end_frame_id: Optional[int] = None

    @property
    def coverage(self) -> float:
        if self.expected_frames <= 0:
            return 0.0
        return min(1.0, self.observed_frames / self.expected_frames)

    @property
    def complete(self) -> bool:
        return (
            bool(self.samples)
            and self.monotonic
            and not self.missing_frame_ids
            and not self.duplicate_frame_ids
        )


class LiveTrackingBuffer:
    """Thread-safe bounded ring of live tracking samples.

    Capacity is expressed in pose responses, not camera frames. Appending never
    blocks acquisition; an active pellet window that outlives the configured
    history is reported as incomplete rather than growing memory without bound.
    """

    def __init__(self, capacity: int = 18_000):
        if capacity <= 0:
            raise ValueError("Live tracking capacity must be positive")
        self._samples = deque(maxlen=int(capacity))
        self._lock = threading.RLock()
        self._dropped_samples = 0

    @property
    def dropped_samples(self) -> int:
        with self._lock:
            return self._dropped_samples

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._dropped_samples = 0

    def append(self, sample: LiveTrackingSample) -> None:
        with self._lock:
            if len(self._samples) == self._samples.maxlen:
                self._dropped_samples += 1
            self._samples.append(sample)

    def snapshot(self) -> Tuple[LiveTrackingSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def window(
        self,
        start_perf: float,
        end_perf: float,
        *,
        expected_start_frame_id: Optional[int] = None,
        expected_end_frame_id: Optional[int] = None,
    ) -> TrackingWindow:
        start_perf = float(start_perf)
        end_perf = float(end_perf)
        if end_perf < start_perf:
            raise ValueError("Tracking window end precedes its start")
        with self._lock:
            selected = tuple(
                sample
                for sample in self._samples
                if sample.source_end_perf >= start_perf
                and sample.source_start_perf <= end_perf
            )
        all_ids = tuple(
            frame_id
            for sample in selected
            for frame_id, perf_time in zip(
                sample.primary_frame_ids,
                sample.primary_frame_perf_times,
            )
            if start_perf <= perf_time <= end_perf
        )
        counts = Counter(all_ids)
        duplicates = tuple(sorted(value for value, count in counts.items() if count > 1))
        unique = tuple(dict.fromkeys(all_ids))
        monotonic = all(current > previous for previous, current in zip(unique, unique[1:]))
        missing = ()
        expected = 0
        if (
            expected_start_frame_id is None
            and expected_end_frame_id is None
            and unique
        ):
            expected_start_frame_id = min(unique)
            expected_end_frame_id = max(unique)
        if (
            expected_start_frame_id is not None
            or expected_end_frame_id is not None
        ):
            if expected_start_frame_id is None or expected_end_frame_id is None:
                raise ValueError("Both expected tracking-window frame bounds are required")
            expected_start_frame_id = int(expected_start_frame_id)
            expected_end_frame_id = int(expected_end_frame_id)
            if expected_end_frame_id < expected_start_frame_id:
                raise ValueError("Expected tracking-window frame bounds are reversed")
            expected_ids = set(range(
                expected_start_frame_id,
                expected_end_frame_id + 1,
            ))
            missing = tuple(sorted(expected_ids - set(unique)))
            expected = len(expected_ids)
        return TrackingWindow(
            start_perf=start_perf,
            end_perf=end_perf,
            samples=selected,
            expected_frames=expected,
            observed_frames=len(set(all_ids)),
            missing_frame_ids=missing,
            duplicate_frame_ids=duplicates,
            monotonic=monotonic,
            requested_start_frame_id=expected_start_frame_id,
            requested_end_frame_id=expected_end_frame_id,
        )
