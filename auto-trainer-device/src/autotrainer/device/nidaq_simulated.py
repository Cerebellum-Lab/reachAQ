"""A hardware-free NI input stream for simulation runs.

It never imports nidaqmx. Samples are generated on the perf clock, which every
process shares: `cam_frames` follows the simulated camera exposure grid
(autotrainer.core.simulation.sim_epoch plus k periods), and any other line
follows the windows other processes publish on the signal bus.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy

from autotrainer.core import NidaqSignalStreamConfiguration, NidaqTimingPlan
from autotrainer.core.latency.raw_clock import monotonic_raw_time
from autotrainer.core.simulation import (
    ENV_SIM_BUS,
    SimSignalBusReader,
    SimSignalEvent,
    sim_camera_exposure_s,
    sim_camera_fps,
    sim_epoch,
)
from autotrainer.device.nidaq_signal_stream import (
    NidaqSignalSampleBlock,
    NidaqSignalStreamController,
    _scale_samples,
)

logger = logging.getLogger(__name__)

SIM_NIDAQ_DEVICE = "SimDAQ1"

#: JSON object {nidaq channel name: bus line name}, for a channel that reads
#: a line published under another name (laser1_trigger reading board_stim2).
ENV_SIM_LINE_ALIASES = "REACHAQ_SIM_LINE_ALIASES"

CAM_FRAMES_CHANNEL = "cam_frames"
#: Standard deviation of the noise on every analog sample, in volts.
ANALOG_NOISE_V = 0.0005


def sim_camera_period() -> float:
    return 1.0 / sim_camera_fps()


def sim_camera_exposure() -> float:
    return sim_camera_exposure_s()


def sim_line_aliases() -> Dict[str, str]:
    raw = os.environ.get(ENV_SIM_LINE_ALIASES, "")
    if not raw:
        return {}
    aliases = json.loads(raw)
    if not isinstance(aliases, dict):
        raise ValueError(f"{ENV_SIM_LINE_ALIASES} must be a JSON object, not {raw!r}")
    return {str(channel): str(line) for channel, line in aliases.items()}


class SimulatedNidaqStream:
    """NidaqInputBackend generating samples instead of reading a board.

    Sample i is at epoch_perf + i / rate, epoch_perf being the perf time taken
    in start(), as the real stream's is. read_chunk stamps first and then sleeps
    until its last sample exists, the same order as the real blocking read, so
    the block stamp rule holds and the session's NI-to-host fit sees the same
    shape of delay.
    """

    def __init__(
        self,
        configuration: NidaqSignalStreamConfiguration,
        *,
        timing_plan: Optional[NidaqTimingPlan] = None,
    ):
        if not configuration.is_enabled:
            raise RuntimeError("NI-DAQ signal stream is disabled")
        if not configuration.channels:
            raise RuntimeError("NI-DAQ signal stream has no configured channels")
        if timing_plan is not None and not timing_plan.is_valid:
            raise RuntimeError(f"Invalid NI-DAQ timing plan: {timing_plan.reason}")
        NidaqSignalStreamController._require_matching_clock_rate(configuration, timing_plan)
        self._configuration = configuration
        self._timing_plan = timing_plan
        self._period = sim_camera_period()
        self._exposure = sim_camera_exposure()
        self._grid_epoch = sim_epoch()
        aliases = sim_line_aliases()
        self._bus_lines = {
            channel.name: aliases.get(channel.name, channel.name)
            for channel in configuration.channels
        }
        bus_path = os.environ.get(ENV_SIM_BUS)
        self._bus = SimSignalBusReader(bus_path) if bus_path else None
        self._events: List[SimSignalEvent] = []
        self._rng = numpy.random.default_rng()
        self._sample_index = 0
        self._read_telemetry = {
            "blocks": 0,
            "read_seconds": 0.0,
            "read_failures": 0,
            "short_reads": 0,
        }
        self._is_started = False
        self._epoch_perf_time: Optional[float] = None
        self._epoch_wall_time: Optional[float] = None

    @property
    def configuration(self) -> NidaqSignalStreamConfiguration:
        return self._configuration

    @property
    def read_telemetry(self):
        return dict(self._read_telemetry)

    def start(self) -> None:
        if self._is_started:
            return
        # Sample 0 is at the new epoch, also after a close() and restart.
        self._sample_index = 0
        self._epoch_perf_time = time.perf_counter()
        self._epoch_wall_time = time.time()
        self._is_started = True
        logger.info(
            "simulated NI-DAQ stream started: %g Hz, %d-sample chunks, camera grid "
            "%.6f s period %.6f s exposure, bus %s",
            self._configuration.sample_rate_hz, self._configuration.read_chunk_size,
            self._period, self._exposure, "on" if self._bus is not None else "off",
        )

    def read_chunk(self) -> NidaqSignalSampleBlock:
        if not self._is_started:
            self.start()
        cfg = self._configuration
        chunk_size = cfg.read_chunk_size
        rate = cfg.sample_rate_hz
        wall_time = time.time()
        perf_time = time.perf_counter()
        raw_time = monotonic_raw_time()
        read_started = time.perf_counter()

        first = self._sample_index
        # No buffer limit: a stalled consumer never overflows here as a board's buffer would.
        ready_at = self._epoch_perf_time + (first + chunk_size) / rate
        delay = ready_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

        times = self._epoch_perf_time + numpy.arange(first, first + chunk_size) / rate
        self._take_bus_events(times[0])
        values = {
            channel.name: _scale_samples(self._raw_samples(channel, times), channel)
            for channel in cfg.channels
        }
        self._sample_index += chunk_size
        self._read_telemetry["blocks"] += 1
        self._read_telemetry["read_seconds"] += time.perf_counter() - read_started
        return NidaqSignalSampleBlock(
            wall_time=wall_time,
            perf_time=perf_time,
            sample_rate_hz=rate,
            sample_index=first,
            channels=tuple(cfg.channels),
            values=values,
            epoch_perf_time=self._epoch_perf_time,
            epoch_wall_time=self._epoch_wall_time,
            raw_time=raw_time,
        )

    def _take_bus_events(self, chunk_start: float) -> None:
        """Add newly published events and drop those wholly before this chunk."""
        if self._bus is not None:
            self._events.extend(self._bus.poll())
        self._events = [event for event in self._events if event.end >= chunk_start]

    def _raw_samples(self, channel, times: numpy.ndarray) -> numpy.ndarray:
        digital = channel.kind == "digital"
        if channel.name == CAM_FRAMES_CHANNEL:
            # High from each exposure's start for the exposure, as the
            # camera's strobe is: the NI edge marks the start of exposure.
            phase = numpy.mod(times - self._grid_epoch, self._period)
            samples = (phase < self._exposure).astype(numpy.float64)
        else:
            samples = numpy.zeros(times.shape, dtype=numpy.float64)
            line = self._bus_lines[channel.name]
            for event in self._events:
                if event.line != line:
                    continue
                inside = (times >= event.start) & (times < event.end)
                if digital:
                    samples[inside] = 1.0 if event.value else 0.0
                else:
                    samples[inside] = event.value
        if not digital:
            samples = samples + self._rng.normal(0.0, ANALOG_NOISE_V, times.shape)
        return samples

    def verify_tasks(self, *, commit: bool = True) -> Tuple[str, ...]:
        del commit  # nothing to commit; there are no tasks
        devices = dict.fromkeys(
            str(channel.physical_channel).strip("/").split("/", 1)[0]
            for channel in self._configuration.channels
        )
        return tuple(f"simulated:{device}" for device in devices)

    def close(self) -> None:
        self._is_started = False
        self._epoch_perf_time = None
        self._epoch_wall_time = None
        self._events = []
