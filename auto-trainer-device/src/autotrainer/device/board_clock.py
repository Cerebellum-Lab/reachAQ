"""Board-to-host clock fitting without changing raw CAN timestamp evidence."""

from __future__ import annotations

import dataclasses
import math
import uuid
from collections import deque
from typing import Optional


@dataclasses.dataclass(frozen=True)
class BoardClockObservation:
    boot_id: int
    request_id: int
    board_midpoint_seconds: float
    host_midpoint_seconds: float
    effective_round_trip_seconds: float


@dataclasses.dataclass(frozen=True)
class BoardClockEstimate:
    model_id: str
    boot_id: int
    generation: int
    observation_count: int
    slope: float
    intercept_seconds: float
    first_board_seconds: float
    last_board_seconds: float
    residual_seconds: float
    round_trip_bound_seconds: float
    uncertainty_seconds: float

    def map_board_time(self, board_time_us: int) -> float:
        return self.intercept_seconds + self.slope * (int(board_time_us) / 1e6)


class BoardClockModel:
    """Fit an affine board-time mapping from bounded NTP-style exchanges."""

    def __init__(self, *, capacity=64, max_extrapolation_seconds=60.0):
        self._observations = deque(maxlen=max(4, int(capacity)))
        self._max_extrapolation = float(max_extrapolation_seconds)
        self._boot_id: Optional[int] = None
        self._generation = 0
        self._estimate: Optional[BoardClockEstimate] = None

    @property
    def estimate(self):
        return self._estimate

    def reset(self, boot_id: Optional[int] = None):
        self._observations.clear()
        self._boot_id = None if boot_id is None else int(boot_id)
        self._generation += 1
        self._estimate = None

    def observe(
        self,
        *,
        request_id: int,
        host_send_perf_ns: int,
        board_receive_time_us: int,
        board_send_time_us: int,
        host_receive_perf_ns: int,
        boot_id: int,
    ) -> BoardClockEstimate:
        host_send = int(host_send_perf_ns) / 1e9
        host_receive = int(host_receive_perf_ns) / 1e9
        board_receive = int(board_receive_time_us) / 1e6
        board_send = int(board_send_time_us) / 1e6
        values = (host_send, host_receive, board_receive, board_send)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Clock synchronization timestamps must be finite")
        if host_receive < host_send or board_send < board_receive:
            raise ValueError("Clock synchronization timestamps are out of order")
        boot_id = int(boot_id)
        if self._boot_id != boot_id:
            self.reset(boot_id)
        observation = BoardClockObservation(
            boot_id=boot_id,
            request_id=int(request_id),
            board_midpoint_seconds=(board_receive + board_send) / 2.0,
            host_midpoint_seconds=(host_send + host_receive) / 2.0,
            effective_round_trip_seconds=max(
                0.0,
                (host_receive - host_send) - (board_send - board_receive),
            ),
        )
        self._observations.append(observation)
        self._estimate = self._fit()
        return self._estimate

    def align(self, board_time_us: int):
        estimate = self._estimate
        if estimate is None:
            return None
        board_seconds = int(board_time_us) / 1e6
        if (
            board_seconds < estimate.first_board_seconds - self._max_extrapolation
            or board_seconds > estimate.last_board_seconds + self._max_extrapolation
        ):
            return None
        return {
            "perf_time": estimate.map_board_time(board_time_us),
            "model_id": estimate.model_id,
            "uncertainty_seconds": estimate.uncertainty_seconds,
            "generation": estimate.generation,
        }

    def _fit(self):
        observations = tuple(self._observations)
        minimum_rtt = min(item.effective_round_trip_seconds for item in observations)
        selected = tuple(
            item for item in observations
            if item.effective_round_trip_seconds <= minimum_rtt + 0.002
        )
        if len(selected) < min(4, len(observations)):
            selected = tuple(sorted(
                observations, key=lambda item: item.effective_round_trip_seconds,
            )[:min(4, len(observations))])
        x = [item.board_midpoint_seconds for item in selected]
        y = [item.host_midpoint_seconds for item in selected]
        if len(selected) < 2 or max(x) == min(x):
            slope = 1.0
        else:
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)
            slope = sum(
                (x_i - x_mean) * (y_i - y_mean)
                for x_i, y_i in zip(x, y)
            ) / sum((x_i - x_mean) ** 2 for x_i in x)
            if not 0.999 <= slope <= 1.001:
                raise ValueError(f"Board clock drift estimate is implausible: {slope}")
        intercept = sum(y_i - slope * x_i for x_i, y_i in zip(x, y)) / len(x)
        residual = math.sqrt(sum(
            (y_i - (intercept + slope * x_i)) ** 2
            for x_i, y_i in zip(x, y)
        ) / len(x))
        rtt_bound = max(item.effective_round_trip_seconds for item in selected)
        return BoardClockEstimate(
            model_id=f"board-clock-{self._generation}-{uuid.uuid4().hex[:12]}",
            boot_id=int(self._boot_id),
            generation=self._generation,
            observation_count=len(observations),
            slope=slope,
            intercept_seconds=intercept,
            first_board_seconds=min(x),
            last_board_seconds=max(x),
            residual_seconds=residual,
            round_trip_bound_seconds=rtt_bound,
            uncertainty_seconds=max(residual, rtt_bound / 2.0),
        )


class BoardSequenceTracker:
    """Detect reboot, duplicate, backward, and missing board message sequences."""

    def __init__(self):
        self._boot_id = None
        self._sequence = None

    def observe(self, boot_id: int, sequence: int) -> dict:
        boot_id, sequence = int(boot_id), int(sequence)
        reboot = self._boot_id is not None and boot_id != self._boot_id
        previous = None if reboot else self._sequence
        duplicate = previous is not None and sequence == previous
        backward = previous is not None and sequence < previous
        gap = (
            0 if previous is None or sequence <= previous
            else max(0, sequence - previous - 1)
        )
        self._boot_id = boot_id
        self._sequence = sequence
        return {
            "reboot": reboot,
            "duplicate": duplicate,
            "backward": backward,
            "gap": gap,
        }
