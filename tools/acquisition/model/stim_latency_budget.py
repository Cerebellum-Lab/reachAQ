"""Tier 1 stim-loop latency budget aggregation.

The direct stim-to-NI trigger path already stamps every stage boundary it
crosses: the capture thread records frame arrival and detector decision, the
IPC hop records send and receive, and the trigger receiver records DAQmx start
entry and return. Nothing aggregated those stamps, so the closed-loop budget
could only be answered by post-hoc analysis of session evidence files.

This module turns that per-event record stream into a rolling per-stage
percentile summary. It is deliberately passive: it consumes the records the
trigger receiver already emits to its observer and never touches the capture
thread, the detector, or the DAQ output path.

Percentiles, not means, are the acceptance figure. A stim loop with a 2 ms mean
and a 15 ms p99 is not a 5 ms loop.

One subtlety in reading the output: the first stamp, `frame_perf_time`, is the
camera driver's estimate of frame capture time mapped into the perf_counter
domain, not the moment the frame reached the host. Spans measured from it
therefore already carry sensor readout and transport, and inherit the accuracy
of that clock mapping.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections import deque
from typing import Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from autotrainer.core.logging import get_verbose_logger


logger = get_verbose_logger(__name__)


DEFAULT_BUDGET_SECONDS = 0.005
"""Working Tier 1 target. Recorded in planning.md as a p99 figure, not a mean."""

DEFAULT_WINDOW = 512


@dataclasses.dataclass(frozen=True)
class StimLatencyStage:
    """One measurable span between two perf-counter stamps on the trigger path."""

    key: str
    start_field: str
    end_field: str
    description: str


# Ordered so that the reported breakdown reads in the direction the trigger
# travels. Every field below is already present on the trigger record; this
# module adds no new instrumentation to the hot path.
STIM_LATENCY_STAGES: Tuple[StimLatencyStage, ...] = (
    StimLatencyStage(
        "capture_to_decision",
        "frame_perf_time",
        "decision_perf_time",
        "estimated frame capture to detector decision; already includes sensor "
        "readout, transport, and host handoff",
    ),
    StimLatencyStage(
        "queue_put",
        "decision_perf_time",
        "ipc_send_perf_time",
        "decision to trigger-queue put, in the capture thread",
    ),
    StimLatencyStage(
        "ipc",
        "ipc_send_perf_time",
        "ipc_receive_perf_time",
        "cross-process queue transit to the trigger receiver",
    ),
    StimLatencyStage(
        "daq_dispatch",
        "ipc_receive_perf_time",
        "daqmx_start_entry_perf_time",
        "receiver-side context validation before the DAQmx call",
    ),
    StimLatencyStage(
        "daq_start",
        "daqmx_start_entry_perf_time",
        "daqmx_start_return_perf_time",
        "DAQmx start call duration",
    ),
)

TOTAL_STAGE = StimLatencyStage(
    "total",
    "frame_perf_time",
    "daqmx_start_return_perf_time",
    "estimated frame capture to DAQmx start return; excludes exposure and "
    "physical output settling",
)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile over an already-sorted-or-not sequence.

    Implemented locally rather than via numpy so this stays usable from the
    trigger observer thread without importing an array stack, and so the
    interpolation rule is explicit in review.
    """

    if len(values) == 0:
        return math.nan
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"percentile fraction out of range: {fraction!r}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    return lower + (upper - lower) * weight


@dataclasses.dataclass(frozen=True)
class StageSummary:
    """Percentile summary for one stage, in seconds."""

    key: str
    count: int
    p50: float
    p95: float
    p99: float
    maximum: float

    def as_record(self) -> Dict[str, object]:
        return {
            "stage": self.key,
            "count": self.count,
            "p50_seconds": self.p50,
            "p95_seconds": self.p95,
            "p99_seconds": self.p99,
            "max_seconds": self.maximum,
        }


@dataclasses.dataclass(frozen=True)
class StimLatencySummary:
    """Rolling-window view of the Tier 1 budget."""

    window: int
    accepted: int
    rejected: int
    incomplete: int
    budget_seconds: float
    over_budget: int
    stages: Tuple[StageSummary, ...]
    total: Optional[StageSummary]
    camera_clock_offset_span_seconds: float

    @property
    def meets_budget(self) -> bool:
        """True only when a p99 exists and sits inside the budget."""

        if self.total is None or not math.isfinite(self.total.p99):
            return False
        return self.total.p99 <= self.budget_seconds

    def as_record(self) -> Dict[str, object]:
        return {
            "window": self.window,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "incomplete": self.incomplete,
            "budget_seconds": self.budget_seconds,
            "over_budget": self.over_budget,
            "meets_budget": self.meets_budget,
            "camera_clock_offset_span_seconds": self.camera_clock_offset_span_seconds,
            "total": None if self.total is None else self.total.as_record(),
            "stages": [stage.as_record() for stage in self.stages],
        }


class StimLatencyBudget:
    """Accumulates direct stim-trigger records into a per-stage percentile view.

    Records arrive on the trigger-receiver thread, so every mutation is guarded.
    The window is bounded, so memory does not grow across a session.

    Rejected triggers are counted but excluded from the latency window: a
    rejected trigger never reached the DAQmx call, so including it would report
    a validation failure as though it were a fast loop.
    """

    def __init__(
        self,
        *,
        budget_seconds: float = DEFAULT_BUDGET_SECONDS,
        window: int = DEFAULT_WINDOW,
    ):
        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("Stim latency budget must be a positive, finite duration")
        if window < 1:
            raise ValueError("Stim latency window must hold at least one record")
        self.budget_seconds = float(budget_seconds)
        self.window = int(window)
        self._lock = threading.RLock()
        self._spans: Dict[str, Deque[float]] = {
            stage.key: deque(maxlen=self.window)
            for stage in (*STIM_LATENCY_STAGES, TOTAL_STAGE)
        }
        # `frame_perf_time` already arrives mapped into the perf_counter domain
        # by the camera driver. This series tracks that mapping's own stability:
        # the driver only recalibrates its camera-to-perf offset when an acquire
        # retries, so a drifting or stale offset silently biases every span
        # measured from frame capture. Retained as a health signal, never
        # reported as latency.
        self._camera_clock_offsets: Deque[float] = deque(maxlen=self.window)
        self._accepted = 0
        self._rejected = 0
        self._incomplete = 0
        self._over_budget = 0

    def reset(self) -> None:
        with self._lock:
            for spans in self._spans.values():
                spans.clear()
            self._camera_clock_offsets.clear()
            self._accepted = 0
            self._rejected = 0
            self._incomplete = 0
            self._over_budget = 0

    def observe(self, record: Mapping[str, object]) -> None:
        """Record one direct stim-trigger result.

        Never raises: this runs on the trigger path's observer callback, and a
        metrics failure must not affect stimulus delivery.
        """

        try:
            self._observe(record)
        except Exception:
            logger.exception("Stim latency budget rejected a trigger record")

    def _observe(self, record: Mapping[str, object]) -> None:
        accepted = bool(record.get("accepted"))
        with self._lock:
            if not accepted:
                self._rejected += 1
                return
            self._accepted += 1

            complete = True
            for stage in STIM_LATENCY_STAGES:
                span = _span(record, stage)
                if span is None:
                    complete = False
                    continue
                self._spans[stage.key].append(span)

            total = _span(record, TOTAL_STAGE)
            if total is None:
                complete = False
            else:
                self._spans[TOTAL_STAGE.key].append(total)
                if total > self.budget_seconds:
                    self._over_budget += 1

            if not complete:
                self._incomplete += 1

            offset = _camera_clock_offset(record)
            if offset is not None:
                self._camera_clock_offsets.append(offset)

    def summary(self) -> StimLatencySummary:
        with self._lock:
            stages = tuple(
                _summarize(stage.key, self._spans[stage.key])
                for stage in STIM_LATENCY_STAGES
            )
            total_spans = self._spans[TOTAL_STAGE.key]
            total = (
                _summarize(TOTAL_STAGE.key, total_spans)
                if len(total_spans) > 0
                else None
            )
            offsets = tuple(self._camera_clock_offsets)
            return StimLatencySummary(
                window=self.window,
                accepted=self._accepted,
                rejected=self._rejected,
                incomplete=self._incomplete,
                budget_seconds=self.budget_seconds,
                over_budget=self._over_budget,
                stages=stages,
                total=total,
                camera_clock_offset_span_seconds=_span_range(offsets),
            )

    def format_report(self) -> str:
        """Operator/log-readable breakdown in milliseconds."""

        summary = self.summary()
        lines = [
            "Tier 1 stim loop latency"
            f" | accepted={summary.accepted} rejected={summary.rejected}"
            f" incomplete={summary.incomplete}"
            f" | budget={summary.budget_seconds * 1e3:.3f}ms"
            f" over_budget={summary.over_budget}",
        ]
        for stage in (*summary.stages, *( (summary.total,) if summary.total else () )):
            lines.append(
                f"  {stage.key:<13}"
                f" n={stage.count:<6}"
                f" p50={stage.p50 * 1e3:8.3f}ms"
                f" p95={stage.p95 * 1e3:8.3f}ms"
                f" p99={stage.p99 * 1e3:8.3f}ms"
                f" max={stage.maximum * 1e3:8.3f}ms"
            )
        if summary.total is None:
            lines.append("  total         unavailable: no complete trigger record yet")
        else:
            lines.append(
                "  verdict       "
                + ("within budget" if summary.meets_budget else "OVER BUDGET at p99")
            )
        if math.isfinite(summary.camera_clock_offset_span_seconds):
            lines.append(
                "  camera-clock offset span"
                f" {summary.camera_clock_offset_span_seconds * 1e3:.3f}ms"
                " (drift indicator only, not latency)"
            )
        return "\n".join(lines)


def _span(record: Mapping[str, object], stage: StimLatencyStage) -> Optional[float]:
    start = _perf_value(record.get(stage.start_field))
    end = _perf_value(record.get(stage.end_field))
    if start is None or end is None:
        return None
    span = end - start
    if not math.isfinite(span) or span < 0:
        # A negative span means the stamps did not come from one perf_counter
        # domain, which is a wiring defect rather than a fast loop.
        logger.warning(
            "Discarding non-monotonic stim latency span: stage=%s start=%r end=%r",
            stage.key, start, end,
        )
        return None
    return span


def _perf_value(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _camera_clock_offset(record: Mapping[str, object]) -> Optional[float]:
    camera_timestamp_ns = _perf_value(record.get("camera_timestamp_ns"))
    frame_perf_time = _perf_value(record.get("frame_perf_time"))
    if camera_timestamp_ns is None or frame_perf_time is None:
        return None
    if camera_timestamp_ns <= 0:
        return None
    return frame_perf_time - (camera_timestamp_ns * 1e-9)


def _span_range(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if len(ordered) < 2:
        return math.nan
    return ordered[-1] - ordered[0]


def _summarize(key: str, values: Sequence[float]) -> StageSummary:
    collected = tuple(values)
    return StageSummary(
        key=key,
        count=len(collected),
        p50=percentile(collected, 0.50),
        p95=percentile(collected, 0.95),
        p99=percentile(collected, 0.99),
        maximum=max(collected) if len(collected) > 0 else math.nan,
    )
