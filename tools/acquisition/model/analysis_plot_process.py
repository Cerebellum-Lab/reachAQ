from __future__ import annotations

import dataclasses
import math
import queue
import signal
import time
from typing import Dict, Optional, Tuple

import numpy as np

from autotrainer.core import NidaqSignalStreamConfiguration
from autotrainer.core.multiproc import get_mp_ctx
from tools.acquisition.model.nidaq_sample_ring import SharedNidaqSampleRing
from tools.acquisition.view.rolling_stream_buffer import RollingStreamBuffer


_CONTROL_CONFIGURE = "configure"
_CONTROL_WINDOW = "window"
_CONTROL_PIXELS = "pixels"
_CONTROL_CLEAR = "clear"
_CONTROL_STOP = "stop"
_MAX_CHANNELS = 32
_MAX_PIXEL_WIDTH = 4096
_POINTS_PER_PIXEL = 8
_MAX_POINTS = _MAX_PIXEL_WIDTH * _POINTS_PER_PIXEL


@dataclasses.dataclass(frozen=True)
class AnalysisPlotFrame:
    sequence: int
    latest_x: float
    pixel_width: int
    window_seconds: float
    point_counts: Tuple[int, ...]
    raw_generation: int
    overrun_count: int
    gap_count: int
    source_latency_ms: float


def _copy_digital_steps(
    buffer,
    x_scratch,
    y_scratch,
    output_x,
    output_y,
    pixel_width: int,
    window_seconds: float,
    latest_x: float,
) -> int:
    """Write timestamp-correct, bounded digital step vertices."""
    sample_count = buffer.copy_ordered_into(x_scratch, y_scratch)
    if sample_count <= 0:
        return 0

    first = int(np.searchsorted(
        x_scratch[:sample_count],
        latest_x - window_seconds,
        side="left",
    ))
    x_values = x_scratch[first:sample_count]
    y_values = y_scratch[first:sample_count]
    if x_values.size <= 0:
        return 0
    relative_x = x_values - latest_x
    changes = np.flatnonzero(y_values[1:] != y_values[:-1]) + 1
    required_points = 2 * changes.size + 2
    if required_points <= output_x.size:
        output_x[0] = relative_x[0]
        output_y[0] = y_values[0]
        if changes.size:
            stop = 1 + 2 * changes.size
            edge_x = relative_x[changes]
            output_x[1:stop:2] = edge_x
            output_x[2:stop:2] = edge_x
            output_y[1:stop:2] = y_values[changes - 1]
            output_y[2:stop:2] = y_values[changes]
        else:
            stop = 1
        output_x[stop] = relative_x[-1]
        output_y[stop] = y_values[-1]
        return required_points

    # More transitions exist than can be represented individually. Preserve
    # both states in each physical-pixel interval as isolated rectangular
    # occupancy envelopes, avoiding diagonal analog-style interpolation.
    pixel_width = max(1, min(pixel_width, output_x.size // 5))
    bin_indices = np.floor(
        (relative_x + window_seconds) * pixel_width / window_seconds,
    ).astype(np.int64)
    np.clip(bin_indices, 0, pixel_width - 1, out=bin_indices)
    minimums = np.full(pixel_width, np.inf, dtype=np.float64)
    maximums = np.full(pixel_width, -np.inf, dtype=np.float64)
    np.minimum.at(minimums, bin_indices, y_values)
    np.maximum.at(maximums, bin_indices, y_values)
    populated = np.flatnonzero(np.isfinite(minimums))
    point_count = 5 * populated.size
    rectangles_x = output_x[:point_count].reshape(-1, 5)
    rectangles_y = output_y[:point_count].reshape(-1, 5)
    pixel_seconds = window_seconds / pixel_width
    left = -window_seconds + populated * pixel_seconds
    right = left + pixel_seconds
    rectangles_x[:, 0] = left
    rectangles_x[:, 1] = left
    rectangles_x[:, 2] = right
    rectangles_x[:, 3] = right
    rectangles_x[:, 4] = np.nan
    rectangles_y[:, 0] = minimums[populated]
    rectangles_y[:, 1] = maximums[populated]
    rectangles_y[:, 2] = maximums[populated]
    rectangles_y[:, 3] = minimums[populated]
    rectangles_y[:, 4] = np.nan
    return point_count


def _analysis_plot_worker(
    control_queue,
    raw_ring,
    shared_x_values,
    shared_y_values,
    active_buffer,
    reader_buffer,
    sequence,
    latest_x_values,
    pixel_width_values,
    window_values,
    point_count_values,
    raw_generation_values,
    overrun_count_values,
    gap_count_values,
    source_perf_time_values,
) -> None:
    """Maintain rolling data and publish fixed shared-memory plot buffers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    shared_x = np.frombuffer(shared_x_values, dtype=np.float32).reshape(
        2, _MAX_CHANNELS, _MAX_POINTS,
    )
    shared_y = np.frombuffer(shared_y_values, dtype=np.float32).reshape(
        2, _MAX_CHANNELS, _MAX_POINTS,
    )
    point_counts = np.frombuffer(point_count_values, dtype=np.int32).reshape(
        2, _MAX_CHANNELS,
    )
    raw_scratch = np.empty(
        (raw_ring.channel_count, raw_ring.capacity), dtype=np.float32,
    )
    shared_x.fill(np.nan)
    shared_y.fill(np.nan)
    point_counts.fill(0)
    ring_channel_indices = {
        name: index for index, name in enumerate(raw_ring.channel_names)
    }
    configuration = NidaqSignalStreamConfiguration()
    window_seconds = configuration.rolling_window_seconds
    pixel_width = 1
    buffers: Dict[str, RollingStreamBuffer] = {}
    scratch: Dict[str, np.ndarray] = {}
    latest_x = 0.0
    last_raw_sample_index = None
    last_raw_epoch = None
    last_gap_count = None
    raw_generation = 0
    overrun_count = 0
    gap_count = 0
    source_perf_time = 0.0
    dirty = False
    stopped = False

    def configure_buffers() -> None:
        nonlocal buffers, scratch, latest_x, last_raw_sample_index
        nonlocal last_raw_epoch, last_gap_count, dirty
        capacity = max(
            configuration.read_chunk_size,
            int(math.ceil(configuration.sample_rate_hz * window_seconds)),
        )
        buffers = {
            channel.name: RollingStreamBuffer(capacity)
            for channel in configuration.channels[:_MAX_CHANNELS]
        }
        scratch = {
            channel.name: (
                np.empty(capacity, dtype=np.float64),
                np.empty(capacity, dtype=np.float64),
            )
            for channel in configuration.channels[:_MAX_CHANNELS]
        }
        latest_x = 0.0
        last_raw_sample_index = None
        last_raw_epoch = None
        last_gap_count = None
        dirty = True

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
                configuration, requested_pixels = payload
                window_seconds = configuration.rolling_window_seconds
                pixel_width = max(1, min(_MAX_PIXEL_WIDTH, int(requested_pixels)))
                configure_buffers()
            elif kind == _CONTROL_WINDOW:
                window_seconds = max(0.1, min(60.0, float(payload)))
                configure_buffers()
            elif kind == _CONTROL_PIXELS:
                requested_pixels = max(1, min(_MAX_PIXEL_WIDTH, int(payload)))
                if requested_pixels != pixel_width:
                    pixel_width = requested_pixels
                    dirty = True
            elif kind == _CONTROL_CLEAR:
                for buffer in buffers.values():
                    buffer.clear()
                latest_x = 0.0
                last_raw_sample_index = raw_ring.current_end_sample_index()
                dirty = True

        if stopped:
            break

        sample_read = raw_ring.copy_since(last_raw_sample_index, raw_scratch)
        if sample_read is not None:
            if last_raw_epoch is not None and sample_read.epoch != last_raw_epoch:
                for buffer in buffers.values():
                    buffer.clear()
                latest_x = 0.0
                last_raw_sample_index = None
                last_raw_epoch = sample_read.epoch
                last_gap_count = sample_read.gap_count
                dirty = True
                continue
            last_raw_epoch = sample_read.epoch
            raw_generation = sample_read.generation
            gap_count = sample_read.gap_count
            if last_gap_count is not None and gap_count != last_gap_count:
                for buffer in buffers.values():
                    buffer.clear()
                latest_x = 0.0
                last_raw_sample_index = None
                last_gap_count = gap_count
                dirty = True
                continue
            last_gap_count = gap_count
            if sample_read.overrun_samples:
                overrun_count += 1
                for buffer in buffers.values():
                    buffer.clear()
                latest_x = 0.0
            if sample_read.source_perf_time > 0:
                source_perf_time = sample_read.source_perf_time
            sample_count = sample_read.sample_count
            if sample_count > 0:
                sample_indices = np.arange(
                    sample_read.start_sample_index,
                    sample_read.end_sample_index,
                    dtype=np.float64,
                )
                x_values = sample_indices / sample_read.sample_rate_hz
                for channel in configuration.channels[:_MAX_CHANNELS]:
                    buffer = buffers.get(channel.name)
                    ring_index = ring_channel_indices.get(channel.name)
                    if buffer is None or ring_index is None:
                        continue
                    buffer.append(x_values, raw_scratch[ring_index, :sample_count])
                latest_x = float(x_values[-1])
                dirty = True
            last_raw_sample_index = sample_read.end_sample_index

        if dirty:
            inactive = 1 - int(active_buffer.value)
            if int(reader_buffer.value) != inactive:
                target_x = shared_x[inactive]
                target_y = shared_y[inactive]
                point_counts[inactive].fill(0)
                for channel_index, channel in enumerate(configuration.channels[:_MAX_CHANNELS]):
                    buffer = buffers.get(channel.name)
                    channel_scratch = scratch.get(channel.name)
                    if buffer is None or channel_scratch is None:
                        continue
                    point_counts[inactive, channel_index] = _copy_digital_steps(
                        buffer,
                        channel_scratch[0],
                        channel_scratch[1],
                        target_x[channel_index],
                        target_y[channel_index],
                        pixel_width,
                        window_seconds,
                        latest_x,
                    )
                latest_x_values[inactive] = latest_x
                pixel_width_values[inactive] = pixel_width
                window_values[inactive] = window_seconds
                raw_generation_values[inactive] = raw_generation
                overrun_count_values[inactive] = overrun_count
                gap_count_values[inactive] = gap_count
                source_perf_time_values[inactive] = source_perf_time
                active_buffer.value = inactive
                sequence.value += 1
                dirty = False

        time.sleep(0.001)


class AnalysisPlotProcess:
    """Shared-memory facade for the visualization-only plot worker."""

    MAX_PIXEL_WIDTH = _MAX_PIXEL_WIDTH
    MAX_POINTS = _MAX_POINTS

    def __init__(self, raw_ring: SharedNidaqSampleRing, *, mp_ctx=None):
        self._mp_ctx = get_mp_ctx() if mp_ctx is None else mp_ctx
        self.raw_ring = raw_ring
        self._control_queue = self._mp_ctx.Queue()
        self._shared_x_values = self._mp_ctx.RawArray(
            "f", 2 * _MAX_CHANNELS * _MAX_POINTS,
        )
        self._shared_y_values = self._mp_ctx.RawArray(
            "f", 2 * _MAX_CHANNELS * _MAX_POINTS,
        )
        self._active_buffer = self._mp_ctx.RawValue("i", 0)
        self._reader_buffer = self._mp_ctx.RawValue("i", -1)
        self._sequence = self._mp_ctx.RawValue("Q", 0)
        self._latest_x_values = self._mp_ctx.RawArray("d", 2)
        self._pixel_width_values = self._mp_ctx.RawArray("i", 2)
        self._window_values = self._mp_ctx.RawArray("d", 2)
        self._point_count_values = self._mp_ctx.RawArray("i", 2 * _MAX_CHANNELS)
        self._raw_generation_values = self._mp_ctx.RawArray("Q", 2)
        self._overrun_count_values = self._mp_ctx.RawArray("Q", 2)
        self._gap_count_values = self._mp_ctx.RawArray("Q", 2)
        self._source_perf_time_values = self._mp_ctx.RawArray("d", 2)
        self._shared_x = np.frombuffer(self._shared_x_values, dtype=np.float32).reshape(
            2, _MAX_CHANNELS, _MAX_POINTS,
        )
        self._shared_y = np.frombuffer(self._shared_y_values, dtype=np.float32).reshape(
            2, _MAX_CHANNELS, _MAX_POINTS,
        )
        self._point_counts = np.frombuffer(self._point_count_values, dtype=np.int32).reshape(
            2, _MAX_CHANNELS,
        )
        self._channel_names: Tuple[str, ...] = tuple()
        self._requested_pixel_width = 1
        self._last_sequence = 0
        self._process = self._mp_ctx.Process(
            target=_analysis_plot_worker,
            name="analysis-plot-data",
            args=(
                self._control_queue,
                self.raw_ring,
                self._shared_x_values,
                self._shared_y_values,
                self._active_buffer,
                self._reader_buffer,
                self._sequence,
                self._latest_x_values,
                self._pixel_width_values,
                self._window_values,
                self._point_count_values,
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
        return self._process.pid

    def configure(self, configuration: NidaqSignalStreamConfiguration, pixel_width: int) -> None:
        if len(configuration.channels) > _MAX_CHANNELS:
            raise ValueError(f"analysis plot supports at most {_MAX_CHANNELS} channels")
        self._channel_names = tuple(channel.name for channel in configuration.channels)
        self._requested_pixel_width = max(1, min(_MAX_PIXEL_WIDTH, int(pixel_width)))
        self._last_sequence = int(self._sequence.value)
        self._control_queue.put(
            (_CONTROL_CONFIGURE, (configuration, self._requested_pixel_width)),
        )

    def set_time_window(self, seconds: float) -> None:
        self._control_queue.put((_CONTROL_WINDOW, float(seconds)))

    def set_pixel_width(self, pixel_width: int) -> None:
        self._requested_pixel_width = max(1, min(_MAX_PIXEL_WIDTH, int(pixel_width)))
        self._control_queue.put((_CONTROL_PIXELS, self._requested_pixel_width))

    def clear(self) -> None:
        self._control_queue.put((_CONTROL_CLEAR, None))

    def copy_latest_into(
        self,
        x_destinations: Dict[str, np.ndarray],
        y_destinations: Dict[str, np.ndarray],
    ) -> Optional[AnalysisPlotFrame]:
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
        pixel_width = int(self._pixel_width_values[buffer_index])
        if pixel_width != self._requested_pixel_width:
            self._reader_buffer.value = -1
            self._last_sequence = current_sequence
            return None
        channel_point_counts = tuple(
            int(value)
            for value in self._point_counts[buffer_index, :len(self._channel_names)]
        )
        try:
            for channel_index, channel_name in enumerate(self._channel_names):
                point_count = channel_point_counts[channel_index]
                x_destination = x_destinations.get(channel_name)
                y_destination = y_destinations.get(channel_name)
                if (
                    x_destination is None
                    or y_destination is None
                    or x_destination.size < point_count
                    or y_destination.size < point_count
                ):
                    continue
                np.copyto(
                    x_destination[:point_count],
                    self._shared_x[buffer_index, channel_index, :point_count],
                )
                np.copyto(
                    y_destination[:point_count],
                    self._shared_y[buffer_index, channel_index, :point_count],
                )
            latest_x = float(self._latest_x_values[buffer_index])
            window_seconds = float(self._window_values[buffer_index])
            raw_generation = int(self._raw_generation_values[buffer_index])
            overrun_count = int(self._overrun_count_values[buffer_index])
            gap_count = int(self._gap_count_values[buffer_index])
            source_perf_time = float(self._source_perf_time_values[buffer_index])
            source_latency_ms = (
                max(0.0, (time.perf_counter() - source_perf_time) * 1000.0)
                if source_perf_time > 0
                else 0.0
            )
        finally:
            self._reader_buffer.value = -1
        self._last_sequence = current_sequence
        return AnalysisPlotFrame(
            sequence=current_sequence,
            latest_x=latest_x,
            pixel_width=pixel_width,
            window_seconds=window_seconds,
            point_counts=channel_point_counts,
            raw_generation=raw_generation,
            overrun_count=overrun_count,
            gap_count=gap_count,
            source_latency_ms=source_latency_ms,
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
