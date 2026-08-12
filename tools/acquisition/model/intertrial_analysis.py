"""Asynchronous analysis of one completed pellet-cycle tracking window."""

from __future__ import annotations

import enum
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from autotrainer.behavior import TrialOutcome
from autotrainer.core.pose_elements import SceneElement
from tools.acquisition.model.live_tracking_buffer import (
    LiveTrackingSample,
    TrackingLocation,
    TrackingOffset,
    TrackingWindow,
)


logger = logging.getLogger(__name__)


class AnalysisProgressionMode(str, enum.Enum):
    CONTINUE = "continue"
    WAIT = "wait"


class PelletPresence(str, enum.Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"


class PelletMisplacement(str, enum.Enum):
    NOT_MISPLACED = "not_misplaced"
    MISPLACED = "misplaced"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PelletStateEvidence:
    presence: PelletPresence
    misplacement: PelletMisplacement
    observed_samples: int
    pellet_seen_samples: int
    triangle_pellet_samples: int
    median_triangle_pellet_distance: Optional[float]


@dataclass(frozen=True)
class IntertrialAnalysisRequest:
    generation: int
    session_id: str
    trial_id: Optional[int]
    attempt_id: int
    operation_id: str
    window: TrackingWindow
    pellet_state: PelletStateEvidence

    @property
    def attempt_label(self) -> str:
        return f"{self.trial_id}.{self.attempt_id}"


@dataclass(frozen=True)
class ReachTrajectory:
    start_perf: float
    closest_perf: float
    end_perf: float
    closest_offset: Tuple[float, float, float]
    consumed: bool


@dataclass(frozen=True)
class IntertrialAnalysisResult:
    request: IntertrialAnalysisRequest
    outcome: TrialOutcome
    reaches: Tuple[ReachTrajectory, ...]
    reach_count: int
    success_count: int
    consumption_count: int
    recommended_shift: Optional[Tuple[float, float, float]]
    interpolated_points: int
    long_gap_count: int
    analysis_seconds: float
    error: str = ""

    @property
    def video_seconds(self) -> float:
        return max(0.0, self.request.window.end_perf - self.request.window.start_perf)


@dataclass(frozen=True)
class AnalysisTimingEstimate:
    completed_windows: int
    mean_realtime_factor: Optional[float]
    projected_seconds: Optional[float]

    @property
    def display_text(self) -> str:
        if self.projected_seconds is None or self.mean_realtime_factor is None:
            return "estimating"
        return (
            f"{self.projected_seconds:.2f}s projected "
            f"({self.mean_realtime_factor:.2f}x realtime)"
        )


def classify_pellet_state(
    window: TrackingWindow,
    *,
    expected_triangle_pellet_distance: float,
    misplacement_threshold: float,
    minimum_missing_samples: int = 5,
) -> PelletStateEvidence:
    """Finalize live pellet evidence without waiting for trajectory analysis."""

    samples = window.samples
    pellet_seen = sum(sample.pellet_seen for sample in samples)
    if not samples:
        presence = PelletPresence.UNKNOWN
    elif pellet_seen:
        presence = PelletPresence.PRESENT
    elif len(samples) >= minimum_missing_samples:
        presence = PelletPresence.MISSING
    else:
        presence = PelletPresence.UNKNOWN

    distances = []
    for sample in samples:
        offset = next(
            (
                value
                for value in sample.offsets_3d
                if value.origin == str(SceneElement.Triangle)
                and value.target == str(SceneElement.Pellet)
            ),
            None,
        )
        if offset is not None:
            distances.append(math.sqrt(offset.x ** 2 + offset.y ** 2 + offset.z ** 2))
    median_distance = None
    if distances:
        ordered = sorted(distances)
        midpoint = len(ordered) // 2
        median_distance = (
            ordered[midpoint]
            if len(ordered) % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
        )
        misplaced = (
            PelletMisplacement.MISPLACED
            if abs(median_distance - expected_triangle_pellet_distance)
            > misplacement_threshold
            else PelletMisplacement.NOT_MISPLACED
        )
    else:
        misplaced = PelletMisplacement.UNKNOWN
    return PelletStateEvidence(
        presence=presence,
        misplacement=misplaced,
        observed_samples=len(samples),
        pellet_seen_samples=pellet_seen,
        triangle_pellet_samples=len(distances),
        median_triangle_pellet_distance=median_distance,
    )


def analyze_tracking_window(
    request: IntertrialAnalysisRequest,
    *,
    hand_distance_threshold_mm: float = 15.0,
    max_interpolation_gap_seconds: float = 0.1,
) -> IntertrialAnalysisResult:
    """Classify reaches from existing live tracking; no frame inference occurs."""

    started = time.perf_counter()
    samples, interpolated, long_gaps = _interpolate_short_gaps(
        request.window.samples,
        max_gap_seconds=max_interpolation_gap_seconds,
    )
    trajectories = []
    active = []
    was_near = False
    prior_distance = None
    approached = False
    pellet_was_seen = False
    pellet_lost = False

    def finish_active():
        nonlocal active, approached, pellet_lost
        if not active or not approached:
            active = []
            approached = False
            pellet_lost = False
            return
        closest = min(active, key=lambda item: item[1])
        trajectories.append(ReachTrajectory(
            start_perf=active[0][0],
            closest_perf=closest[0],
            end_perf=active[-1][0],
            closest_offset=closest[2],
            consumed=pellet_lost,
        ))
        active = []
        approached = False
        pellet_lost = False

    for sample in samples:
        pellet = sample.location(str(SceneElement.Pellet))
        hand = sample.location(str(SceneElement.R_Hand))
        if sample.pellet_seen:
            pellet_was_seen = True
        elif pellet_was_seen and active:
            pellet_lost = True
        if pellet is None or hand is None:
            continue
        offset = (hand.x - pellet.x, hand.y - pellet.y, hand.z - pellet.z)
        distance = math.sqrt(sum(value * value for value in offset))
        near = distance <= hand_distance_threshold_mm
        if prior_distance is not None and distance < prior_distance:
            approached = True
        if near or active:
            active.append((sample.source_perf, distance, offset))
        if active and was_near and not near:
            finish_active()
        was_near = near
        prior_distance = distance
    finish_active()

    consumed = sum(item.consumed for item in trajectories)
    successes = consumed
    reach_count = len(trajectories)
    if request.pellet_state.presence is PelletPresence.MISSING:
        outcome = TrialOutcome.PELLET_MISSING
        error = "live tracking confirmed pellet absence"
    elif consumed:
        outcome = TrialOutcome.SUCCESS
        error = ""
    elif reach_count:
        outcome = TrialOutcome.FAILURE
        error = ""
    else:
        outcome = TrialOutcome.NO_REACH
        error = "" if samples else "live tracking unavailable"

    failed_offsets = tuple(
        trajectory.closest_offset
        for trajectory in trajectories
        if not trajectory.consumed
    )
    recommended_shift = None
    if failed_offsets:
        recommended_shift = tuple(
            sum(values) / len(values)
            for values in zip(*failed_offsets)
        )
    return IntertrialAnalysisResult(
        request=request,
        outcome=outcome,
        reaches=tuple(trajectories),
        reach_count=reach_count,
        success_count=successes,
        consumption_count=consumed,
        recommended_shift=recommended_shift,
        interpolated_points=interpolated,
        long_gap_count=long_gaps,
        analysis_seconds=time.perf_counter() - started,
        error=error,
    )


def _interpolate_short_gaps(samples, *, max_gap_seconds):
    """Linearly fill only short internal gaps; retain long gaps as diagnostics."""

    if len(samples) < 2:
        return tuple(samples), 0, 0
    result = []
    interpolated = 0
    long_gaps = 0
    for left, right in zip(samples, samples[1:]):
        result.append(left)
        if not left.primary_frame_ids or not right.primary_frame_ids:
            continue
        missing = right.primary_frame_ids[0] - left.primary_frame_ids[-1] - 1
        duration = right.source_start_perf - left.source_end_perf
        if missing <= 0:
            continue
        if duration > max_gap_seconds:
            long_gaps += 1
            continue
        for index in range(1, missing + 1):
            fraction = index / (missing + 1)
            result.append(_interpolated_sample(left, right, fraction))
            interpolated += 1
    result.append(samples[-1])
    return tuple(result), interpolated, long_gaps


def _interpolated_sample(left, right, fraction):
    right_locations = {item.name: item for item in right.locations_3d}
    locations = []
    for first in left.locations_3d:
        second = right_locations.get(first.name)
        if second is None:
            continue
        locations.append(TrackingLocation(
            first.name,
            first.x + (second.x - first.x) * fraction,
            first.y + (second.y - first.y) * fraction,
            first.z + (second.z - first.z) * fraction,
        ))
    perf_time = left.source_end_perf + (
        right.source_start_perf - left.source_end_perf
    ) * fraction
    return LiveTrackingSample(
        sequence=left.sequence,
        primary_frame_ids=(),
        primary_frame_perf_times=(),
        source_start_perf=perf_time,
        source_end_perf=perf_time,
        processing_perf=max(left.processing_perf, right.processing_perf),
        pellet_seen=left.pellet_seen and right.pellet_seen,
        locations_3d=tuple(locations),
        offsets_3d=(),
    )


class IntertrialAnalysisCoordinator:
    """Bounded single-worker coordinator with session-generation ownership."""

    def __init__(
        self,
        result_callback: Callable[[IntertrialAnalysisResult], None],
        *,
        queue_capacity: int = 8,
        analyzer: Callable[[IntertrialAnalysisRequest], IntertrialAnalysisResult] = analyze_tracking_window,
    ):
        if queue_capacity <= 0:
            raise ValueError("Analysis queue capacity must be positive")
        self._callback = result_callback
        self._analyzer = analyzer
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._generation = 0
        self._session_id = ""
        self._pending = 0
        self._closed = False
        self._ratios = []
        self._worker = threading.Thread(
            target=self._run,
            name="IntertrialAnalysis",
            daemon=True,
        )
        self._worker.start()

    def begin_session(self, generation: int, session_id: str) -> None:
        with self._lock:
            self._generation = int(generation)
            self._session_id = str(session_id)
            self._ratios.clear()
            self._discard_queued_unlocked()

    def submit(self, request: IntertrialAnalysisRequest) -> bool:
        with self._lock:
            if self._closed or not self._is_current_unlocked(request):
                return False
            try:
                self._queue.put_nowait(request)
            except queue.Full:
                return False
            self._pending += 1
            self._idle.notify_all()
            return True

    def is_idle(self) -> bool:
        with self._lock:
            return self._pending == 0

    def wait_for_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._idle:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def cancel_session(self) -> None:
        with self._lock:
            self._generation += 1
            self._session_id = ""
            self._discard_queued_unlocked()
            self._idle.notify_all()

    def timing_estimate(self, video_seconds: float) -> AnalysisTimingEstimate:
        with self._lock:
            if not self._ratios:
                factor = None
                projected = None
            else:
                factor = sum(self._ratios) / len(self._ratios)
                projected = factor * max(0.0, float(video_seconds))
            return AnalysisTimingEstimate(len(self._ratios), factor, projected)

    def close(self, timeout: float = 2.0) -> bool:
        with self._lock:
            self._closed = True
            self._generation += 1
            self._session_id = ""
            self._discard_queued_unlocked()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout)
        return not self._worker.is_alive()

    def _run(self):
        while True:
            request = self._queue.get()
            if request is None:
                return
            with self._lock:
                current_before = self._is_current_unlocked(request)
            result = None
            if current_before:
                try:
                    result = self._analyzer(request)
                except Exception as error:  # preserve a per-trial failure result
                    result = IntertrialAnalysisResult(
                        request=request,
                        outcome=TrialOutcome.INCOMPLETE,
                        reaches=(),
                        reach_count=0,
                        success_count=0,
                        consumption_count=0,
                        recommended_shift=None,
                        interpolated_points=0,
                        long_gap_count=0,
                        analysis_seconds=0.0,
                        error=f"intertrial analysis failed: {error}",
                    )
            with self._lock:
                current_after = self._is_current_unlocked(request)
            if result is not None and current_after:
                try:
                    self._callback(result)
                except Exception:
                    logger.exception(
                        "Intertrial result callback failed for %s",
                        request.attempt_label,
                    )
            with self._lock:
                self._pending = max(0, self._pending - 1)
                if result is not None and result.video_seconds > 0:
                    self._ratios.append(
                        result.analysis_seconds / result.video_seconds
                    )
                    self._ratios = self._ratios[-20:]
                self._idle.notify_all()

    def _is_current_unlocked(self, request):
        return (
            request.generation == self._generation
            and request.session_id == self._session_id
        )

    def _discard_queued_unlocked(self):
        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                discarded += 1
        self._pending = max(0, self._pending - discarded)


def tracking_request_record(
    request: IntertrialAnalysisRequest,
    result: Optional[IntertrialAnalysisResult] = None,
    *,
    metadata_generation_id: Optional[str] = None,
) -> dict:
    """Create the non-executable record used for final validation/repair."""
    record = {
        "schemaVersion": 1,
        "metadataGenerationId": metadata_generation_id,
        "identity": {
            "generation": request.generation,
            "sessionId": request.session_id,
            "trialId": request.trial_id,
            "attemptId": request.attempt_id,
            "operationId": request.operation_id,
        },
        "window": {
            "startPerf": request.window.start_perf,
            "endPerf": request.window.end_perf,
            "expectedFrames": request.window.expected_frames,
            "observedFrames": request.window.observed_frames,
            "missingFrameIds": request.window.missing_frame_ids,
            "duplicateFrameIds": request.window.duplicate_frame_ids,
            "monotonic": request.window.monotonic,
            "samples": [sample.to_record() for sample in request.window.samples],
        },
        "pelletState": {
            "presence": request.pellet_state.presence.value,
            "misplacement": request.pellet_state.misplacement.value,
            "observedSamples": request.pellet_state.observed_samples,
            "pelletSeenSamples": request.pellet_state.pellet_seen_samples,
            "trianglePelletSamples": request.pellet_state.triangle_pellet_samples,
            "medianTrianglePelletDistance": (
                request.pellet_state.median_triangle_pellet_distance
            ),
        },
    }
    if result is not None:
        record["analysis"] = {
            "outcome": result.outcome.value,
            "reachCount": result.reach_count,
            "successCount": result.success_count,
            "consumptionCount": result.consumption_count,
            "recommendedShift": result.recommended_shift,
            "interpolatedPoints": result.interpolated_points,
            "longGapCount": result.long_gap_count,
            "analysisSeconds": result.analysis_seconds,
            "error": result.error,
        }
    return record


def tracking_request_from_record(
    record: dict,
    *,
    expected_metadata_generation_id: Optional[str] = None,
) -> IntertrialAnalysisRequest:
    if int(record.get("schemaVersion", 0)) != 1:
        raise ValueError("Unsupported trial tracking schema")
    if (
        expected_metadata_generation_id is not None
        and record.get("metadataGenerationId")
        != expected_metadata_generation_id
    ):
        raise ValueError(
            "Trial tracking metadata generation does not match alignment: "
            f"{record.get('metadataGenerationId')!r} != "
            f"{expected_metadata_generation_id!r}"
        )
    identity = record["identity"]
    window_record = record["window"]
    samples = tuple(
        LiveTrackingSample(
            sequence=int(item["sequence"]),
            primary_frame_ids=tuple(item["primary_frame_ids"]),
            primary_frame_perf_times=tuple(item["primary_frame_perf_times"]),
            source_start_perf=float(item["source_start_perf"]),
            source_end_perf=float(item["source_end_perf"]),
            processing_perf=float(item["processing_perf"]),
            pellet_seen=bool(item["pellet_seen"]),
            locations_3d=tuple(
                TrackingLocation(**location) for location in item["locations_3d"]
            ),
            offsets_3d=tuple(
                TrackingOffset(**offset) for offset in item["offsets_3d"]
            ),
        )
        for item in window_record["samples"]
    )
    tracking_window = TrackingWindow(
        start_perf=float(window_record["startPerf"]),
        end_perf=float(window_record["endPerf"]),
        samples=samples,
        expected_frames=int(window_record["expectedFrames"]),
        observed_frames=int(window_record["observedFrames"]),
        missing_frame_ids=tuple(window_record["missingFrameIds"]),
        duplicate_frame_ids=tuple(window_record["duplicateFrameIds"]),
        monotonic=bool(window_record["monotonic"]),
    )
    pellet = record["pelletState"]
    return IntertrialAnalysisRequest(
        generation=int(identity["generation"]),
        session_id=str(identity["sessionId"]),
        trial_id=(
            None if identity["trialId"] is None else int(identity["trialId"])
        ),
        attempt_id=int(identity["attemptId"]),
        operation_id=str(identity["operationId"]),
        window=tracking_window,
        pellet_state=PelletStateEvidence(
            presence=PelletPresence(pellet["presence"]),
            misplacement=PelletMisplacement(pellet["misplacement"]),
            observed_samples=int(pellet["observedSamples"]),
            pellet_seen_samples=int(pellet["pelletSeenSamples"]),
            triangle_pellet_samples=int(pellet["trianglePelletSamples"]),
            median_triangle_pellet_distance=(
                None
                if pellet["medianTrianglePelletDistance"] is None
                else float(pellet["medianTrianglePelletDistance"])
            ),
        ),
    )
