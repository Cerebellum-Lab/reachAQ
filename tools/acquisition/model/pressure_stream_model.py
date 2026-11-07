from __future__ import annotations

import dataclasses
import math
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from autotrainer.core import SystemStatusMessageKind
from tools.acquisition.view.rolling_stream_buffer import RollingStreamBuffer


# The pellet board samples pressure on a 12-bit ADC referenced to 3V3.
ADC_FULL_SCALE = 4095
REFERENCE_VOLTS = 3.3


@dataclasses.dataclass(frozen=True)
class PressureStreamFrame:
    """One repaint's worth of pressure history, shared across both graphs."""

    latest_perf_time: float
    series: Dict[int, Tuple[np.ndarray, np.ndarray]]


class PressureStreamModel:
    """Buffers pellet-board FSR samples for the Analysis panel's pressure graphs.

    Samples arrive on the system message handler's thread at roughly 83 Hz per
    sensor.  They are appended under a lock and drained by the UI timer, so no
    per-sample marshalling to the Qt thread is needed.
    """

    INSTANCES: Tuple[int, ...] = (0, 1)
    WINDOW_SECONDS = 10.0
    # The board sends ~83 Hz per sensor; the buffer is sized well above that so a
    # burst cannot evict samples the window should still be showing.
    _MAXIMUM_EXPECTED_RATE_HZ = 250.0

    def __init__(self, system_message_handler, *, window_seconds: float = WINDOW_SECONDS):
        self._system_message_handler = system_message_handler
        self._window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        capacity = int(math.ceil(self._window_seconds * self._MAXIMUM_EXPECTED_RATE_HZ))
        self._buffers: Dict[int, RollingStreamBuffer] = {
            instance: RollingStreamBuffer(capacity) for instance in self.INSTANCES
        }
        system_message_handler.decoded_message_received += self._decoded_message_received

    def clear(self) -> None:
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()

    def close(self) -> None:
        self._system_message_handler.decoded_message_received -= self._decoded_message_received

    def _decoded_message_received(self, kind, data, perf_time, _wall_time) -> None:
        if kind != SystemStatusMessageKind.PRESSURE_READING:
            return
        sample_time = _sample_perf_time(data, perf_time)
        with self._lock:
            buffer = self._buffers.get(int(data.instance))
            if buffer is None:
                return
            buffer.append((sample_time,), (float(data.pressure),))

    def snapshot(self) -> Optional[PressureStreamFrame]:
        with self._lock:
            series = {
                instance: buffer.ordered()
                for instance, buffer in self._buffers.items()
            }
        latest = max(
            (x[-1] for x, _ in series.values() if x.size),
            default=None,
        )
        if latest is None:
            return None
        oldest_visible = latest - self._window_seconds
        return PressureStreamFrame(
            latest_perf_time=float(latest),
            series={
                instance: _within_window(x, y, latest, oldest_visible)
                for instance, (x, y) in series.items()
            },
        )


def counts_to_volts(counts: np.ndarray) -> np.ndarray:
    """Convert raw 12-bit ADC counts to volts across the board's 3V3 reference."""
    return np.asarray(counts, dtype=np.float64) / ADC_FULL_SCALE * REFERENCE_VOLTS


def _sample_perf_time(reading, host_receive_perf_time: float) -> float:
    """Prefer the board's event time, matching how session rows are timestamped."""
    event_perf_time = getattr(reading, "event_perf_time", None)
    try:
        event_perf_time = float(event_perf_time)
    except (TypeError, ValueError):
        return float(host_receive_perf_time)
    if not math.isfinite(event_perf_time):
        return float(host_receive_perf_time)
    return event_perf_time


def _within_window(
    x: np.ndarray,
    y: np.ndarray,
    latest: float,
    oldest_visible: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim to the visible window and re-origin so the newest sample sits at zero.

    The buffer is trimmed by time rather than by capacity because the window must
    stay correct whatever rate the board actually sends at.
    """
    first_visible = int(np.searchsorted(x, oldest_visible, side="left"))
    return x[first_visible:] - latest, y[first_visible:]
