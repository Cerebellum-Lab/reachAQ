"""The NI-DAQ input stream contract and its one selection point."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

from autotrainer.core import NidaqSignalStreamConfiguration, NidaqTimingPlan
from autotrainer.core.simulation import is_simulation
from autotrainer.device.nidaq_signal_stream import (
    NidaqSignalSampleBlock,
    NidaqSignalStreamController,
)


@runtime_checkable
class NidaqInputBackend(Protocol):
    """A continuous, hardware-clocked NI input stream, owned by one thread.

    The NI worker process constructs it, calls `start` once, then `read_chunk`
    in a loop until it is stopped, then `close`. The preflight constructs one,
    calls `verify_tasks` and closes it without ever starting it.

    - `start` stamps the stream's epoch (`epoch_perf_time`, `epoch_wall_time`):
      sample 0 is acquired at about that instant, and sample i at
      epoch + i / sample_rate_hz on the board's own clock.
    - `read_chunk` blocks until `configuration.read_chunk_size` new samples
      exist and returns them. `sample_index` is the first sample's index since
      start and blocks are contiguous. `values` maps channel name to that
      channel's scaled samples (volts for analog, 0/1 for digital, after the
      channel's scale and offset).
    - Block stamp rule: `perf_time`, `wall_time` and `raw_time` are taken
      *before* the blocking read, so a block's stamp is roughly the end of the
      previous block, not of its own samples. The session's NI-to-host fit
      relies on this.
    - `verify_tasks` checks the task graph without starting it and returns the
      ids it verified. Failures raise.
    - `close` releases everything and may be called more than once.
    """

    @property
    def configuration(self) -> NidaqSignalStreamConfiguration: ...

    @property
    def read_telemetry(self) -> dict: ...

    def start(self) -> None: ...

    def read_chunk(self) -> NidaqSignalSampleBlock: ...

    def verify_tasks(self, *, commit: bool = True) -> Tuple[str, ...]: ...

    def close(self) -> None: ...


def make_nidaq_input_backend(
    configuration: NidaqSignalStreamConfiguration,
    *,
    timing_plan: Optional[NidaqTimingPlan] = None,
    hardware_class=None,
) -> NidaqInputBackend:
    """The simulated stream in a simulation run, else the NI-DAQmx one.

    `hardware_class` lets a call site pass the class it imported, so tests that
    patch that module attribute keep patching what is constructed.
    """
    if is_simulation():
        # Imported here so a hardware run never loads the simulated stream.
        from autotrainer.device.nidaq_simulated import SimulatedNidaqStream

        return SimulatedNidaqStream(configuration, timing_plan=timing_plan)
    cls = hardware_class if hardware_class is not None else NidaqSignalStreamController
    return cls(configuration, timing_plan=timing_plan)
