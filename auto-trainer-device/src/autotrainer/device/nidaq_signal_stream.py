from __future__ import annotations

import dataclasses
import logging
import numbers
import time
from typing import Dict, List, Optional, Tuple

from autotrainer.core import NidaqSignalChannelConfiguration, NidaqSignalStreamConfiguration
from autotrainer.core.logging import log_hardware_initialization


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class NidaqSignalSampleBlock:
    """A chunk of scaled NI-DAQ input samples from one shared stream read."""

    wall_time: float
    perf_time: float
    sample_rate_hz: float
    sample_index: int
    channels: Tuple[NidaqSignalChannelConfiguration, ...]
    values: Dict[str, Tuple[float, ...]]

    @property
    def sample_count(self) -> int:
        if not self.values:
            return 0
        return max(len(channel_values) for channel_values in self.values.values())


class NidaqSignalStreamController:
    """Continuous NI-DAQmx input reader for acquisition signal verification."""

    def __init__(self, configuration: NidaqSignalStreamConfiguration):
        if not configuration.is_enabled:
            raise RuntimeError("NI-DAQ signal stream is disabled")
        if not configuration.channels:
            raise RuntimeError("NI-DAQ signal stream has no configured channels")
        runtime_started = time.perf_counter()
        log_hardware_initialization(logger, "START | NI-DAQmx runtime | consumer=signal-stream")
        self._nidaqmx = _load_nidaqmx()
        log_hardware_initialization(
            logger,
            "READY | NI-DAQmx runtime | consumer=signal-stream elapsed=%.3fs",
            time.perf_counter() - runtime_started,
        )
        self._configuration = configuration
        self._analog_task: Optional[object] = None
        self._digital_task: Optional[object] = None
        self._sample_index = 0
        self._is_started = False
        try:
            self._create_tasks()
        except Exception:
            self.close()
            raise

    @property
    def configuration(self) -> NidaqSignalStreamConfiguration:
        return self._configuration

    def start(self) -> None:
        if self._is_started:
            return
        started = time.perf_counter()
        log_hardware_initialization(logger, "START | NI-DAQ signal tasks")
        try:
            # Start DI before AI when DI is clocked from the AI sample clock.
            if self._digital_task is not None:
                self._digital_task.start()
            if self._analog_task is not None:
                self._analog_task.start()
            self._is_started = True
            log_hardware_initialization(
                logger,
                "READY | NI-DAQ signal tasks | analog=%s digital=%s elapsed=%.3fs",
                self._analog_task is not None,
                self._digital_task is not None,
                time.perf_counter() - started,
            )
        except Exception:
            self.close()
            raise

    def read_chunk(self) -> NidaqSignalSampleBlock:
        if not self._is_started:
            self.start()
        cfg = self._configuration
        chunk_size = cfg.read_chunk_size
        timeout = max(1.0, chunk_size / cfg.sample_rate_hz * 5.0)
        values: Dict[str, Tuple[float, ...]] = {}
        wall_time = time.time()
        perf_time = time.perf_counter()

        if self._analog_task is not None:
            raw = self._analog_task.read(number_of_samples_per_channel=chunk_size, timeout=timeout)
            for channel, samples in zip(cfg.analog_channels, _normalize_samples(raw, len(cfg.analog_channels))):
                values[channel.name] = tuple(_scale_sample(sample, channel) for sample in samples)

        if self._digital_task is not None:
            raw = self._digital_task.read(number_of_samples_per_channel=chunk_size, timeout=timeout)
            for channel, samples in zip(cfg.digital_channels, _normalize_samples(raw, len(cfg.digital_channels))):
                values[channel.name] = tuple(_scale_sample(1.0 if bool(sample) else 0.0, channel) for sample in samples)

        sample_index = self._sample_index
        if values:
            self._sample_index += max(len(channel_values) for channel_values in values.values())

        return NidaqSignalSampleBlock(
            wall_time=wall_time,
            perf_time=perf_time,
            sample_rate_hz=cfg.sample_rate_hz,
            sample_index=sample_index,
            channels=cfg.channels,
            values=values,
        )

    def close(self) -> None:
        errors = []
        for name, task in (
            ("digital input", self._digital_task),
            ("analog input", self._analog_task),
        ):
            if task is None:
                continue
            try:
                task.stop()
            except Exception as exc:
                errors.append((f"{name} stop", exc))
                logger.exception("Failed to stop NI-DAQ signal stream %s task", name)
            try:
                task.close()
            except Exception as exc:
                errors.append((f"{name} close", exc))
                logger.exception("Failed to close NI-DAQ signal stream %s task", name)
        self._digital_task = None
        self._analog_task = None
        self._is_started = False
        if errors:
            locations = ", ".join(location for location, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ signal stream task(s): {locations}") from errors[0][1]

    def _create_tasks(self) -> None:
        cfg = self._configuration
        buffer_size = max(cfg.read_chunk_size * 10, cfg.read_chunk_size)

        analog_channels = cfg.analog_channels
        if analog_channels:
            started = time.perf_counter()
            log_hardware_initialization(
                logger,
                "START | NI-DAQ analog input task | channels=%s sample_rate=%s buffer=%s",
                tuple(channel.physical_channel for channel in analog_channels),
                cfg.sample_rate_hz,
                buffer_size,
            )
            self._analog_task = self._nidaqmx.Task("reachaq_signal_stream_ai")
            for channel in analog_channels:
                kwargs = {}
                if channel.minimum is not None:
                    kwargs["min_val"] = channel.minimum
                if channel.maximum is not None:
                    kwargs["max_val"] = channel.maximum
                self._analog_task.ai_channels.add_ai_voltage_chan(channel.physical_channel, **kwargs)
            self._analog_task.timing.cfg_samp_clk_timing(
                rate=cfg.sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size,
            )
            log_hardware_initialization(
                logger,
                "READY | NI-DAQ analog input task | elapsed=%.3fs",
                time.perf_counter() - started,
            )

        digital_channels = cfg.digital_channels
        if digital_channels:
            started = time.perf_counter()
            log_hardware_initialization(
                logger,
                "START | NI-DAQ digital input task | channels=%s sample_rate=%s buffer=%s",
                tuple(channel.physical_channel for channel in digital_channels),
                cfg.sample_rate_hz,
                buffer_size,
            )
            self._digital_task = self._nidaqmx.Task("reachaq_signal_stream_di")
            line_grouping = self._nidaqmx.constants.LineGrouping.CHAN_PER_LINE
            for channel in digital_channels:
                self._digital_task.di_channels.add_di_chan(channel.physical_channel, line_grouping=line_grouping)
            kwargs = {}
            if self._analog_task is not None:
                kwargs["source"] = self._analog_input_sample_clock_source(analog_channels[0].physical_channel)
            self._digital_task.timing.cfg_samp_clk_timing(
                rate=cfg.sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size,
                **kwargs,
            )
            log_hardware_initialization(
                logger,
                "READY | NI-DAQ digital input task | elapsed=%.3fs",
                time.perf_counter() - started,
            )

    def _analog_input_sample_clock_source(self, physical_channel: str) -> str:
        parts = physical_channel.strip("/").split("/")
        if len(parts) < 2 or not parts[0]:
            raise RuntimeError(f"cannot infer NI-DAQ AI sample clock source from {physical_channel!r}")
        return f"/{parts[0]}/ai/SampleClock"


def _normalize_samples(raw_samples, channel_count: int) -> List[List[object]]:
    if channel_count <= 0:
        return []
    if channel_count == 1:
        if isinstance(raw_samples, (list, tuple)) and raw_samples and not isinstance(raw_samples[0], numbers.Number):
            return [list(raw_samples[0])]
        return [list(raw_samples)]
    raw_samples = list(raw_samples)
    if len(raw_samples) != channel_count:
        raise RuntimeError(f"expected {channel_count} NI-DAQ stream channels, received {len(raw_samples)}")
    return [list(channel_samples) for channel_samples in raw_samples]


def _scale_sample(sample: float, channel: NidaqSignalChannelConfiguration) -> float:
    return float(sample) * channel.scale + channel.offset


def _load_nidaqmx():
    try:
        import nidaqmx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NI-DAQ signal streaming. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    return nidaqmx
