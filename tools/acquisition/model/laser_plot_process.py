from __future__ import annotations

import dataclasses
import math
import queue
import signal
import time
from typing import Dict, Optional, Tuple

import numpy as np

from autotrainer.core.multiproc import get_mp_ctx
from tools.acquisition.model.laser_model import LaserTraceBlock
from tools.acquisition.model.nidaq_sample_ring import SharedNidaqSampleRing
from tools.acquisition.view.rolling_stream_buffer import RollingStreamBuffer


_CONTROL_CONFIGURE = "configure"
_CONTROL_STREAMING = "streaming"
_CONTROL_TRACE = "trace"
_CONTROL_CLEAR = "clear"
_CONTROL_STOP = "stop"
_CHANNEL_COUNT = 4
_CURVE_NAMES = ("command", "diode", "copy")
_CURVES_PER_CHANNEL = len(_CURVE_NAMES)
_CURVE_COUNT = _CHANNEL_COUNT * _CURVES_PER_CHANNEL
_MAX_POINTS = 4096


def _curve_slot(channel_id: int, curve_name: str) -> int:
    return (int(channel_id) - 1) * _CURVES_PER_CHANNEL + _CURVE_NAMES.index(curve_name)


@dataclasses.dataclass(frozen=True)
class LaserPlotFrame:
    sequence: int
    point_counts: Tuple[int, ...]
    latest_x_values: Tuple[float, ...]
    window_seconds: Tuple[float, ...]
    raw_generation: int
    overrun_count: int
    gap_count: int
    source_latency_ms: float


@dataclasses.dataclass
class _LaserChannelPlotState:
    window_seconds: float
    pixel_width: int
    diode_name: Optional[str]
    copy_name: Optional[str]
    buffers: Dict[str, RollingStreamBuffer]
    scratch: Dict[str, Tuple[np.ndarray, np.ndarray]]
    streaming: bool = False


def _new_buffers(sample_rate_hz: float, window_seconds: float) -> Dict[str, RollingStreamBuffer]:
    capacity = max(1, int(math.ceil(sample_rate_hz * window_seconds)))
    return {
        curve_name: RollingStreamBuffer(capacity)
        for curve_name in _CURVE_NAMES
    }


def _new_scratch(
    sample_rate_hz: float,
    window_seconds: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    capacity = max(1, int(math.ceil(sample_rate_hz * window_seconds)))
    return {
        curve_name: (
            np.empty(capacity, dtype=np.float64),
            np.empty(capacity, dtype=np.float64),
        )
        for curve_name in _CURVE_NAMES
    }


def _copy_analog_lod(
    buffer: RollingStreamBuffer,
    x_scratch: np.ndarray,
    y_scratch: np.ndarray,
    output_x: np.ndarray,
    output_y: np.ndarray,
    max_points: int,
) -> int:
    """Copy a bounded time-ordered min/max representation without full-array allocation."""
    sample_count = buffer.copy_ordered_into(x_scratch, y_scratch)
    if sample_count <= 0:
        return 0
    max_points = max(2, min(int(max_points), output_x.size, output_y.size))
    if sample_count <= max_points:
        output_x[:sample_count] = x_scratch[:sample_count]
        output_y[:sample_count] = y_scratch[:sample_count]
        return sample_count

    bucket_count = max(1, max_points // 2)
    bucket_size = int(math.ceil(sample_count / bucket_count))
    usable_size = (sample_count // bucket_size) * bucket_size
    if usable_size < bucket_size:
        output_x[:max_points] = x_scratch[sample_count - max_points:sample_count]
        output_y[:max_points] = y_scratch[sample_count - max_points:sample_count]
        return max_points
    x_buckets = x_scratch[sample_count - usable_size:sample_count].reshape(
        -1, bucket_size,
    )
    y_buckets = y_scratch[sample_count - usable_size:sample_count].reshape(
        -1, bucket_size,
    )
    minimum_indices = np.argmin(y_buckets, axis=1)
    maximum_indices = np.argmax(y_buckets, axis=1)
    first_indices = np.minimum(minimum_indices, maximum_indices)
    second_indices = np.maximum(minimum_indices, maximum_indices)
    rows = np.arange(y_buckets.shape[0])
    output_count = y_buckets.shape[0] * 2
    output_x[:output_count:2] = x_buckets[rows, first_indices]
    output_x[1:output_count:2] = x_buckets[rows, second_indices]
    output_y[:output_count:2] = y_buckets[rows, first_indices]
    output_y[1:output_count:2] = y_buckets[rows, second_indices]
    return output_count


def _trace_gap(x_values: Tuple[float, ...]) -> float:
    positive_steps = [
        second - first
        for first, second in zip(x_values, x_values[1:])
        if second > first
    ]
    return min(positive_steps) if positive_steps else 0.001


def _append_trace(state: _LaserChannelPlotState, trace: LaserTraceBlock) -> None:
    if trace.replace:
        for buffer in state.buffers.values():
            buffer.clear()
    if not trace.x_values:
        return
    if trace.replace:
        x_values = np.asarray(trace.x_values, dtype=np.float64)
    else:
        latest_values = [
            buffer.latest_x
            for buffer in state.buffers.values()
            if buffer.latest_x is not None
        ]
        append_offset = (
            max(latest_values) + _trace_gap(trace.x_values)
            if latest_values
            else 0.0
        )
        x_values = (
            append_offset
            + np.asarray(trace.x_values, dtype=np.float64)
            - trace.x_values[0]
        )
    for curve_name, values in (
        ("command", trace.command_volts),
        ("diode", trace.diode_volts),
        ("copy", trace.command_copy_volts),
    ):
        if values:
            state.buffers[curve_name].append(x_values[:len(values)], values)


def _laser_plot_worker(
    control_queue,
    raw_ring,
    shared_x_values,
    shared_y_values,
    active_buffer,
    reader_buffer,
    sequence,
    point_count_values,
    latest_x_values,
    window_values,
    raw_generation_values,
    overrun_count_values,
    gap_count_values,
    source_perf_time_values,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    shared_x = np.frombuffer(shared_x_values, dtype=np.float32).reshape(
        2, _CURVE_COUNT, _MAX_POINTS,
    )
    shared_y = np.frombuffer(shared_y_values, dtype=np.float32).reshape(
        2, _CURVE_COUNT, _MAX_POINTS,
    )
    point_counts = np.frombuffer(point_count_values, dtype=np.int32).reshape(
        2, _CURVE_COUNT,
    )
    latest_x_output = np.frombuffer(latest_x_values, dtype=np.float64).reshape(
        2, _CHANNEL_COUNT,
    )
    window_output = np.frombuffer(window_values, dtype=np.float64).reshape(
        2, _CHANNEL_COUNT,
    )
    raw_scratch = np.empty(
        (raw_ring.channel_count, raw_ring.capacity), dtype=np.float32,
    )
    ring_channel_indices = {
        name: index for index, name in enumerate(raw_ring.channel_names)
    }
    shared_x.fill(np.nan)
    shared_y.fill(np.nan)
    point_counts.fill(0)
    latest_x_output.fill(0.0)
    window_output.fill(10.0)

    states: Dict[int, _LaserChannelPlotState] = {}
    last_raw_sample_index = None
    last_raw_epoch = None
    last_gap_count = None
    raw_generation = 0
    overrun_count = 0
    gap_count = 0
    source_perf_time = 0.0
    dirty = True
    stopped = False

    while not stopped:
        while True:
            try:
                kind, payload = control_queue.get_nowait()
            except queue.Empty:
                break
            if kind == _CONTROL_STOP:
                stopped = True
                break
            if kind == _CONTROL_CONFIGURE:
                channel_id, window_seconds, pixel_width, diode_name, copy_name = payload
                channel_id = int(channel_id)
                window_seconds = max(0.1, min(60.0, float(window_seconds)))
                existing = states.get(channel_id)
                if existing is None:
                    states[channel_id] = _LaserChannelPlotState(
                        window_seconds=window_seconds,
                        pixel_width=max(1, int(pixel_width)),
                        diode_name=diode_name,
                        copy_name=copy_name,
                        buffers=_new_buffers(raw_ring.sample_rate_hz, window_seconds),
                        scratch=_new_scratch(raw_ring.sample_rate_hz, window_seconds),
                    )
                else:
                    if existing.diode_name != diode_name:
                        existing.buffers["diode"].clear()
                    if existing.copy_name != copy_name:
                        existing.buffers["copy"].clear()
                    existing.window_seconds = window_seconds
                    existing.pixel_width = max(1, int(pixel_width))
                    existing.diode_name = diode_name
                    existing.copy_name = copy_name
                    capacity = max(1, int(math.ceil(raw_ring.sample_rate_hz * window_seconds)))
                    if next(iter(existing.buffers.values())).capacity != capacity:
                        for buffer in existing.buffers.values():
                            buffer.resize(capacity)
                        existing.scratch = _new_scratch(
                            raw_ring.sample_rate_hz,
                            window_seconds,
                        )
                dirty = True
            elif kind == _CONTROL_STREAMING:
                channel_id, is_streaming = payload
                state = states.get(int(channel_id))
                if state is not None:
                    state.streaming = bool(is_streaming)
            elif kind == _CONTROL_TRACE:
                trace = payload
                state = states.get(int(trace.channel_id))
                if state is not None:
                    if trace.source == "calibration":
                        state.streaming = True
                    if state.streaming:
                        _append_trace(state, trace)
                        dirty = True
            elif kind == _CONTROL_CLEAR:
                state = states.get(int(payload))
                if state is not None:
                    for buffer in state.buffers.values():
                        buffer.clear()
                    dirty = True

        if stopped:
            break

        sample_read = (
            raw_ring.copy_since(last_raw_sample_index, raw_scratch)
            if states
            else None
        )
        if sample_read is not None:
            if last_raw_epoch is not None and sample_read.epoch != last_raw_epoch:
                for state in states.values():
                    state.buffers["diode"].clear()
                    state.buffers["copy"].clear()
                last_raw_sample_index = None
                last_raw_epoch = sample_read.epoch
                last_gap_count = sample_read.gap_count
                dirty = True
                continue
            last_raw_epoch = sample_read.epoch
            raw_generation = sample_read.generation
            gap_count = sample_read.gap_count
            if last_gap_count is not None and gap_count != last_gap_count:
                for state in states.values():
                    state.buffers["diode"].clear()
                    state.buffers["copy"].clear()
                last_raw_sample_index = None
                last_gap_count = gap_count
                dirty = True
                continue
            last_gap_count = gap_count
            if sample_read.overrun_samples:
                overrun_count += 1
                for state in states.values():
                    state.buffers["diode"].clear()
                    state.buffers["copy"].clear()
            if sample_read.source_perf_time > 0:
                source_perf_time = sample_read.source_perf_time
            if sample_read.sample_count:
                x_values = np.arange(
                    sample_read.start_sample_index,
                    sample_read.end_sample_index,
                    dtype=np.float64,
                ) / sample_read.sample_rate_hz
                for state in states.values():
                    if not state.streaming:
                        continue
                    for curve_name, stream_name in (
                        ("diode", state.diode_name),
                        ("copy", state.copy_name),
                    ):
                        ring_index = ring_channel_indices.get(stream_name)
                        if ring_index is not None:
                            state.buffers[curve_name].append(
                                x_values,
                                raw_scratch[ring_index, :sample_read.sample_count],
                            )
                            dirty = True
            last_raw_sample_index = sample_read.end_sample_index

        if dirty:
            inactive = 1 - int(active_buffer.value)
            if int(reader_buffer.value) != inactive:
                point_counts[inactive].fill(0)
                latest_x_output[inactive].fill(0.0)
                for channel_id, state in states.items():
                    channel_index = channel_id - 1
                    window_output[inactive, channel_index] = state.window_seconds
                    latest_values = [
                        buffer.latest_x
                        for buffer in state.buffers.values()
                        if buffer.latest_x is not None
                    ]
                    if latest_values:
                        latest_x_output[inactive, channel_index] = max(latest_values)
                    channel_latest_x = latest_x_output[inactive, channel_index]
                    point_limit = max(256, min(_MAX_POINTS, state.pixel_width))
                    for curve_name, buffer in state.buffers.items():
                        slot = _curve_slot(channel_id, curve_name)
                        channel_scratch = state.scratch[curve_name]
                        count = _copy_analog_lod(
                            buffer,
                            channel_scratch[0],
                            channel_scratch[1],
                            shared_x[inactive, slot],
                            shared_y[inactive, slot],
                            point_limit,
                        )
                        if count:
                            shared_x[inactive, slot, :count] -= channel_latest_x
                        point_counts[inactive, slot] = count
                raw_generation_values[inactive] = raw_generation
                overrun_count_values[inactive] = overrun_count
                gap_count_values[inactive] = gap_count
                source_perf_time_values[inactive] = source_perf_time
                active_buffer.value = inactive
                sequence.value += 1
                dirty = False

        time.sleep(0.001)


class LaserPlotProcess:
    """Shared-memory plot preparation for every Laser Control stream graph."""

    CURVE_NAMES = _CURVE_NAMES
    CHANNEL_COUNT = _CHANNEL_COUNT
    MAX_POINTS = _MAX_POINTS

    def __init__(self, raw_ring: SharedNidaqSampleRing, *, mp_ctx=None):
        self._mp_ctx = get_mp_ctx() if mp_ctx is None else mp_ctx
        self.raw_ring = raw_ring
        self._control_queue = self._mp_ctx.Queue()
        self._shared_x_values = self._mp_ctx.RawArray(
            "f", 2 * _CURVE_COUNT * _MAX_POINTS,
        )
        self._shared_y_values = self._mp_ctx.RawArray(
            "f", 2 * _CURVE_COUNT * _MAX_POINTS,
        )
        self._active_buffer = self._mp_ctx.RawValue("i", 0)
        self._reader_buffer = self._mp_ctx.RawValue("i", -1)
        self._sequence = self._mp_ctx.RawValue("Q", 0)
        self._point_count_values = self._mp_ctx.RawArray("i", 2 * _CURVE_COUNT)
        self._latest_x_values = self._mp_ctx.RawArray("d", 2 * _CHANNEL_COUNT)
        self._window_values = self._mp_ctx.RawArray("d", 2 * _CHANNEL_COUNT)
        self._raw_generation_values = self._mp_ctx.RawArray("Q", 2)
        self._overrun_count_values = self._mp_ctx.RawArray("Q", 2)
        self._gap_count_values = self._mp_ctx.RawArray("Q", 2)
        self._source_perf_time_values = self._mp_ctx.RawArray("d", 2)
        self._shared_x = np.frombuffer(self._shared_x_values, dtype=np.float32).reshape(
            2, _CURVE_COUNT, _MAX_POINTS,
        )
        self._shared_y = np.frombuffer(self._shared_y_values, dtype=np.float32).reshape(
            2, _CURVE_COUNT, _MAX_POINTS,
        )
        self._point_counts = np.frombuffer(
            self._point_count_values, dtype=np.int32,
        ).reshape(2, _CURVE_COUNT)
        self._latest_x = np.frombuffer(
            self._latest_x_values, dtype=np.float64,
        ).reshape(2, _CHANNEL_COUNT)
        self._windows = np.frombuffer(
            self._window_values, dtype=np.float64,
        ).reshape(2, _CHANNEL_COUNT)
        self._last_sequence = 0
        self._process = self._mp_ctx.Process(
            target=_laser_plot_worker,
            name="laser-plot-data",
            args=(
                self._control_queue,
                self.raw_ring,
                self._shared_x_values,
                self._shared_y_values,
                self._active_buffer,
                self._reader_buffer,
                self._sequence,
                self._point_count_values,
                self._latest_x_values,
                self._window_values,
                self._raw_generation_values,
                self._overrun_count_values,
                self._gap_count_values,
                self._source_perf_time_values,
            ),
            daemon=True,
        )
        self._process.start()

    @property
    def pid(self) -> Optional[int]:
        return None if self._process is None else self._process.pid

    @staticmethod
    def curve_slot(channel_id: int, curve_name: str) -> int:
        return _curve_slot(channel_id, curve_name)

    def configure_channel(
        self,
        channel_id: int,
        *,
        window_seconds: float,
        pixel_width: int,
        diode_name: Optional[str],
        copy_name: Optional[str],
    ) -> None:
        self._control_queue.put((
            _CONTROL_CONFIGURE,
            (channel_id, window_seconds, pixel_width, diode_name, copy_name),
        ))

    def set_streaming(self, channel_id: int, is_streaming: bool) -> None:
        self._control_queue.put((_CONTROL_STREAMING, (channel_id, is_streaming)))

    def submit_trace(self, trace: LaserTraceBlock) -> None:
        self._control_queue.put((_CONTROL_TRACE, trace))

    def clear(self, channel_id: int) -> None:
        self._control_queue.put((_CONTROL_CLEAR, channel_id))

    def copy_latest_into(
        self,
        x_destinations: Dict[Tuple[int, str], np.ndarray],
        y_destinations: Dict[Tuple[int, str], np.ndarray],
    ) -> Optional[LaserPlotFrame]:
        current_sequence = int(self._sequence.value)
        if current_sequence == self._last_sequence:
            return None
        buffer_index = int(self._active_buffer.value)
        self._reader_buffer.value = buffer_index
        if (
            buffer_index != int(self._active_buffer.value)
            or current_sequence != int(self._sequence.value)
        ):
            self._reader_buffer.value = -1
            return None
        counts = tuple(int(value) for value in self._point_counts[buffer_index])
        try:
            for channel_id in range(1, _CHANNEL_COUNT + 1):
                for curve_name in _CURVE_NAMES:
                    slot = _curve_slot(channel_id, curve_name)
                    count = counts[slot]
                    key = (channel_id, curve_name)
                    destination_x = x_destinations.get(key)
                    destination_y = y_destinations.get(key)
                    if destination_x is None or destination_y is None:
                        continue
                    np.copyto(
                        destination_x[:count],
                        self._shared_x[buffer_index, slot, :count],
                    )
                    np.copyto(
                        destination_y[:count],
                        self._shared_y[buffer_index, slot, :count],
                    )
            latest_x_values = tuple(float(value) for value in self._latest_x[buffer_index])
            windows = tuple(float(value) for value in self._windows[buffer_index])
            raw_generation = int(self._raw_generation_values[buffer_index])
            overrun_count = int(self._overrun_count_values[buffer_index])
            gap_count = int(self._gap_count_values[buffer_index])
            source_perf_time = float(self._source_perf_time_values[buffer_index])
        finally:
            self._reader_buffer.value = -1
        self._last_sequence = current_sequence
        latency_ms = (
            max(0.0, (time.perf_counter() - source_perf_time) * 1000.0)
            if source_perf_time > 0
            else 0.0
        )
        return LaserPlotFrame(
            sequence=current_sequence,
            point_counts=counts,
            latest_x_values=latest_x_values,
            window_seconds=windows,
            raw_generation=raw_generation,
            overrun_count=overrun_count,
            gap_count=gap_count,
            source_latency_ms=latency_ms,
        )

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        self._control_queue.put((_CONTROL_STOP, None))
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)
