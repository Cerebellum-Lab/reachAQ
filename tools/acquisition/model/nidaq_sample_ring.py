from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional, Tuple

import numpy as np

from autotrainer.core import NidaqSignalStreamConfiguration
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.device import NidaqSignalSampleBlock


@dataclasses.dataclass(frozen=True)
class SharedSampleRead:
    start_sample_index: int
    end_sample_index: int
    sample_rate_hz: float
    epoch: int
    generation: int
    source_perf_time: float
    source_wall_time: float
    overrun_samples: int
    gap_count: int

    @property
    def sample_count(self) -> int:
        return self.end_sample_index - self.start_sample_index


class SharedNidaqSampleRing:
    """Single-producer, multi-reader ring for raw NI-DAQ sample blocks.

    The NI worker is the only writer. Readers use the sequence counter as a
    seqlock: an odd value means a write is in progress, and a changed value
    means their snapshot must be retried. Sample indices are absolute within a
    worker run, so timestamps never depend on queue delivery cadence.
    """

    def __init__(
        self,
        configuration: NidaqSignalStreamConfiguration,
        *,
        mp_ctx=None,
        capacity: Optional[int] = None,
    ):
        self._mp_ctx = get_mp_ctx() if mp_ctx is None else mp_ctx
        self.channel_names: Tuple[str, ...] = tuple(
            channel.name for channel in configuration.channels
        )
        self.sample_rate_hz = float(configuration.sample_rate_hz)
        minimum_capacity = max(
            configuration.read_chunk_size * 4,
            int(np.ceil(
                configuration.sample_rate_hz * configuration.rolling_window_seconds,
            )),
        )
        self.capacity = max(1, minimum_capacity if capacity is None else int(capacity))
        self.channel_count = max(1, len(self.channel_names))
        self._raw_values = self._mp_ctx.RawArray(
            "f", self.channel_count * self.capacity,
        )
        self._write_sequence = self._mp_ctx.RawValue("Q", 0)
        self._epoch = self._mp_ctx.RawValue("Q", 0)
        self._generation = self._mp_ctx.RawValue("Q", 0)
        self._end_sample_index = self._mp_ctx.RawValue("q", 0)
        self._valid_sample_count = self._mp_ctx.RawValue("Q", 0)
        self._source_perf_time = self._mp_ctx.RawValue("d", 0.0)
        self._source_wall_time = self._mp_ctx.RawValue("d", 0.0)
        self._gap_count = self._mp_ctx.RawValue("Q", 0)
        self._values_view = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_values_view"] = None
        return state

    @property
    def values(self) -> np.ndarray:
        if self._values_view is None:
            self._values_view = np.frombuffer(self._raw_values, dtype=np.float32).reshape(
                self.channel_count, self.capacity,
            )
        return self._values_view

    @property
    def signature(self) -> Tuple[Tuple[str, ...], float, int]:
        return self.channel_names, self.sample_rate_hz, self.capacity

    @property
    def generation(self) -> int:
        return int(self._generation.value)

    @property
    def gap_count(self) -> int:
        return int(self._gap_count.value)

    def reset(self) -> None:
        self._write_sequence.value += 1
        self._end_sample_index.value = 0
        self._valid_sample_count.value = 0
        self._source_perf_time.value = 0.0
        self._source_wall_time.value = 0.0
        self._gap_count.value = 0
        self._epoch.value += 1
        self._generation.value = 0
        self._write_sequence.value += 1

    def write_block(self, block: NidaqSignalSampleBlock) -> None:
        if not math.isclose(float(block.sample_rate_hz), self.sample_rate_hz):
            raise ValueError(
                f"sample block rate {block.sample_rate_hz:g} does not match "
                f"shared ring rate {self.sample_rate_hz:g}"
            )
        count = block.sample_count
        if count <= 0:
            return
        source_offset = max(0, count - self.capacity)
        write_count = count - source_offset
        start_sample_index = int(block.sample_index) + source_offset
        sources = tuple(
            np.asarray(block.values.get(channel_name, ()), dtype=np.float32).reshape(-1)
            for channel_name in self.channel_names
        )

        self._write_sequence.value += 1
        try:
            previous_valid = int(self._valid_sample_count.value)
            previous_end = int(self._end_sample_index.value)
            if previous_valid and int(block.sample_index) != previous_end:
                self._gap_count.value += 1
                previous_valid = 0

            position = start_sample_index % self.capacity
            first_count = min(write_count, self.capacity - position)
            target = self.values
            for channel_index, source in enumerate(sources):
                available = max(0, min(write_count, source.size - source_offset))
                if available:
                    first_available = min(available, first_count)
                    target[channel_index, position:position + first_available] = source[
                        source_offset:source_offset + first_available
                    ]
                    if available > first_available:
                        target[channel_index, :available - first_available] = source[
                            source_offset + first_available:source_offset + available
                        ]
                if available < write_count:
                    missing_start = (position + available) % self.capacity
                    missing_count = write_count - available
                    missing_first = min(missing_count, self.capacity - missing_start)
                    target[
                        channel_index,
                        missing_start:missing_start + missing_first,
                    ] = np.nan
                    if missing_count > missing_first:
                        target[channel_index, :missing_count - missing_first] = np.nan

            self._end_sample_index.value = start_sample_index + write_count
            self._valid_sample_count.value = min(
                self.capacity,
                write_count if previous_valid == 0 else previous_valid + write_count,
            )
            self._source_perf_time.value = time.perf_counter()
            self._source_wall_time.value = time.time()
            self._generation.value += 1
        except BaseException:
            self._valid_sample_count.value = 0
            raise
        finally:
            self._write_sequence.value += 1

    def copy_since(
        self,
        last_sample_index: Optional[int],
        destination: np.ndarray,
    ) -> Optional[SharedSampleRead]:
        """Copy every unread sample, returning None during a concurrent write."""
        sequence_before = int(self._write_sequence.value)
        if sequence_before & 1:
            return None
        end_sample_index = int(self._end_sample_index.value)
        valid_count = int(self._valid_sample_count.value)
        oldest_sample_index = end_sample_index - valid_count
        requested = oldest_sample_index if last_sample_index is None else int(last_sample_index)
        overrun_samples = max(0, oldest_sample_index - requested)
        start_sample_index = max(oldest_sample_index, min(requested, end_sample_index))
        count = end_sample_index - start_sample_index
        if destination.ndim != 2 or destination.shape[0] < len(self.channel_names):
            raise ValueError("destination does not have enough NI-DAQ channel rows")
        if destination.shape[1] < count:
            raise ValueError(f"destination holds {destination.shape[1]} samples; {count} required")

        position = start_sample_index % self.capacity
        first_count = min(count, self.capacity - position)
        if count:
            destination[:len(self.channel_names), :first_count] = self.values[
                :len(self.channel_names), position:position + first_count
            ]
            if count > first_count:
                destination[:len(self.channel_names), first_count:count] = self.values[
                    :len(self.channel_names), :count - first_count
                ]
        generation = int(self._generation.value)
        epoch = int(self._epoch.value)
        source_perf_time = float(self._source_perf_time.value)
        source_wall_time = float(self._source_wall_time.value)
        gap_count = int(self._gap_count.value)
        sequence_after = int(self._write_sequence.value)
        if sequence_before != sequence_after or sequence_after & 1:
            return None
        return SharedSampleRead(
            start_sample_index=start_sample_index,
            end_sample_index=end_sample_index,
            sample_rate_hz=self.sample_rate_hz,
            epoch=epoch,
            generation=generation,
            source_perf_time=source_perf_time,
            source_wall_time=source_wall_time,
            overrun_samples=overrun_samples,
            gap_count=gap_count,
        )

    def current_end_sample_index(self) -> int:
        while True:
            sequence_before = int(self._write_sequence.value)
            if sequence_before & 1:
                continue
            end_sample_index = int(self._end_sample_index.value)
            if sequence_before == int(self._write_sequence.value):
                return end_sample_index
