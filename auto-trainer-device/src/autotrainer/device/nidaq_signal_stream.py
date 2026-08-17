from __future__ import annotations

import dataclasses
import logging
import numbers
import time
from typing import Dict, List, Optional, Tuple

import numpy

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    NidaqTimingPlan,
)
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
    epoch_perf_time: Optional[float] = None
    epoch_wall_time: Optional[float] = None

    @property
    def sample_count(self) -> int:
        if not self.values:
            return 0
        return max(len(channel_values) for channel_values in self.values.values())


class NidaqSignalStreamController:
    """Continuous NI-DAQmx input reader for acquisition signal verification."""

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
        runtime_started = time.perf_counter()
        log_hardware_initialization(logger, "START | NI-DAQmx runtime | consumer=signal-stream")
        self._nidaqmx = _load_nidaqmx()
        log_hardware_initialization(
            logger,
            "READY | NI-DAQmx runtime | consumer=signal-stream elapsed=%.3fs",
            time.perf_counter() - runtime_started,
        )
        self._configuration = configuration
        self._timing_plan = timing_plan
        if timing_plan is not None and not timing_plan.is_valid:
            raise RuntimeError(f"Invalid NI-DAQ timing plan: {timing_plan.reason}")
        self._analog_tasks: Dict[str, object] = {}
        self._digital_tasks: Dict[str, object] = {}
        self._digital_clock_tasks: Dict[str, object] = {}
        # Compatibility aliases for existing diagnostics and focused tests.
        self._analog_task: Optional[object] = None
        self._digital_task: Optional[object] = None
        self._digital_clock_task: Optional[object] = None
        self._analog_channels_by_device: Dict[
            str, Tuple[NidaqSignalChannelConfiguration, ...]
        ] = {}
        self._digital_channels_by_device: Dict[
            str, Tuple[NidaqSignalChannelConfiguration, ...]
        ] = {}
        self._sample_index = 0
        self._analog_readers: Dict[str, object] = {}
        self._digital_readers: Dict[str, object] = {}
        self._analog_buffers: Dict[str, numpy.ndarray] = {}
        self._digital_buffers: Dict[str, numpy.ndarray] = {}
        self._read_telemetry = {
            "blocks": 0,
            "availability_wait_seconds": 0.0,
            "read_seconds": 0.0,
            "late_barriers": 0,
            "short_reads": 0,
        }
        self._is_started = False
        self._epoch_perf_time: Optional[float] = None
        self._epoch_wall_time: Optional[float] = None
        try:
            self._create_tasks()
        except Exception:
            self.close()
            raise

    @property
    def configuration(self) -> NidaqSignalStreamConfiguration:
        return self._configuration

    @property
    def read_telemetry(self):
        return dict(self._read_telemetry)

    def start(self) -> None:
        if self._is_started:
            return
        started = time.perf_counter()
        log_hardware_initialization(logger, "START | NI-DAQ signal tasks")
        try:
            device_order = self._task_start_order()
            master = (
                self._timing_plan.master_device
                if self._timing_plan is not None
                else (device_order[-1] if device_order else None)
            )
            for device_name in device_order:
                if device_name == master:
                    self._epoch_perf_time = time.perf_counter()
                    self._epoch_wall_time = time.time()
                # Arm consumers before the task that produces their sample clock.
                digital_task = self._digital_tasks.get(device_name)
                if digital_task is not None:
                    digital_task.start()
                analog_task = self._analog_tasks.get(device_name)
                if analog_task is not None:
                    analog_task.start()
                clock_task = self._digital_clock_tasks.get(device_name)
                if clock_task is not None:
                    clock_task.start()
            if self._epoch_perf_time is None:
                self._epoch_perf_time = time.perf_counter()
                self._epoch_wall_time = time.time()
            self._is_started = True
            log_hardware_initialization(
                logger,
                "READY | NI-DAQ signal tasks | analog=%s digital=%s digital_timing=hardware elapsed=%.3fs",
                bool(self._analog_tasks),
                bool(self._digital_tasks),
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
        wait_started = time.perf_counter()
        self._wait_all_available(chunk_size, timeout)
        self._read_telemetry["availability_wait_seconds"] += (
            time.perf_counter() - wait_started
        )
        read_started = time.perf_counter()

        for device_name in self._task_start_order():
            analog_task = self._analog_tasks.get(device_name)
            analog_channels = self._analog_channels_by_device.get(device_name, tuple())
            if analog_task is not None:
                raw = self._read_analog(device_name, analog_task, chunk_size, timeout)
                for channel, samples in zip(
                    analog_channels,
                    _normalize_samples(raw, len(analog_channels)),
                ):
                    values[channel.name] = tuple(
                        _scale_sample(sample, channel) for sample in samples
                    )

            digital_task = self._digital_tasks.get(device_name)
            digital_channels = self._digital_channels_by_device.get(
                device_name, tuple()
            )
            if digital_task is not None:
                raw = self._read_digital(device_name, digital_task, chunk_size, timeout)
                for channel, samples in zip(
                    digital_channels,
                    _normalize_samples(raw, len(digital_channels)),
                ):
                    values[channel.name] = tuple(
                        _scale_sample(1.0 if bool(sample) else 0.0, channel)
                        for sample in samples
                    )

        self._read_telemetry["blocks"] += 1
        self._read_telemetry["read_seconds"] += time.perf_counter() - read_started

        sample_index = self._sample_index
        if values:
            sample_counts = {len(channel_values) for channel_values in values.values()}
            if len(sample_counts) != 1:
                raise RuntimeError(
                    "Synchronized NI-DAQ tasks returned different sample counts: "
                    + ", ".join(map(str, sorted(sample_counts)))
                )
            self._sample_index += next(iter(sample_counts))

        return NidaqSignalSampleBlock(
            wall_time=wall_time,
            perf_time=perf_time,
            sample_rate_hz=cfg.sample_rate_hz,
            sample_index=sample_index,
            channels=cfg.channels,
            values=values,
            epoch_perf_time=self._epoch_perf_time,
            epoch_wall_time=self._epoch_wall_time,
        )

    def verify_tasks(self, *, commit: bool = True) -> Tuple[str, ...]:
        """Verify the exact disposable graph without starting any task."""
        verified = []
        modes = self._nidaqmx.constants.TaskMode
        for task_id, task in self._owned_task_records():
            task.control(modes.TASK_VERIFY)
            if commit:
                task.control(modes.TASK_COMMIT)
            verified.append(task_id)
        return tuple(verified)

    def close(self) -> None:
        errors = []
        tasks = tuple(
            (f"digital sample clock {device}", task)
            for device, task in self._digital_clock_tasks.items()
        ) + tuple(
            (f"digital input {device}", task)
            for device, task in self._digital_tasks.items()
        ) + tuple(
            (f"analog input {device}", task)
            for device, task in self._analog_tasks.items()
        )
        for name, task in tasks:
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
        self._digital_tasks.clear()
        self._digital_clock_tasks.clear()
        self._analog_tasks.clear()
        self._analog_channels_by_device.clear()
        self._digital_channels_by_device.clear()
        self._digital_task = None
        self._digital_clock_task = None
        self._analog_task = None
        self._analog_readers.clear()
        self._digital_readers.clear()
        self._analog_buffers.clear()
        self._digital_buffers.clear()
        self._is_started = False
        self._epoch_perf_time = None
        self._epoch_wall_time = None
        if errors:
            locations = ", ".join(location for location, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ signal stream task(s): {locations}") from errors[0][1]

    def _owned_task_records(self):
        return tuple(
            (f"{device}.counter-clock", task)
            for device, task in self._digital_clock_tasks.items()
        ) + tuple(
            (f"{device}.di", task)
            for device, task in self._digital_tasks.items()
        ) + tuple(
            (f"{device}.ai", task)
            for device, task in self._analog_tasks.items()
        )

    def _create_tasks(self) -> None:
        cfg = self._configuration
        buffer_size = max(cfg.read_chunk_size * 10, cfg.read_chunk_size)

        analog_by_device = self._channels_by_device(cfg.analog_channels)
        digital_by_device = self._channels_by_device(cfg.digital_channels)
        if self._uses_multidevice_tasks():
            master = self._timing_plan.master_device
            if len(analog_by_device) > 1:
                analog_by_device = {master: tuple(cfg.analog_channels)}
            if len(digital_by_device) > 1:
                digital_by_device = {master: tuple(cfg.digital_channels)}
        self._analog_channels_by_device = analog_by_device
        self._digital_channels_by_device = digital_by_device

        for device_name, analog_channels in analog_by_device.items():
            started = time.perf_counter()
            log_hardware_initialization(
                logger,
                "START | NI-DAQ analog input task | channels=%s sample_rate=%s buffer=%s",
                tuple(channel.physical_channel for channel in analog_channels),
                cfg.sample_rate_hz,
                buffer_size,
            )
            analog_task = self._nidaqmx.Task(
                f"reachaq_signal_stream_{device_name}_ai"
            )
            self._analog_tasks[device_name] = analog_task
            if self._analog_task is None:
                self._analog_task = analog_task
            for channel in analog_channels:
                kwargs = {}
                if channel.minimum is not None:
                    kwargs["min_val"] = channel.minimum
                if channel.maximum is not None:
                    kwargs["max_val"] = channel.maximum
                analog_task.ai_channels.add_ai_voltage_chan(
                    channel.physical_channel, **kwargs
                )
            timing_kwargs = self._sample_clock_kwargs(device_name)
            analog_task.timing.cfg_samp_clk_timing(
                rate=cfg.sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size,
                **timing_kwargs,
            )
            self._configure_reference_clock(analog_task)
            self._configure_start_trigger(analog_task, device_name)
            self._configure_exports(analog_task, device_name, "ai")
            self._make_stream_reader(device_name, analog_task, analog_channels, analog=True)
            log_hardware_initialization(
                logger,
                "READY | NI-DAQ analog input task | elapsed=%.3fs",
                time.perf_counter() - started,
            )

        for device_name, digital_channels in digital_by_device.items():
            started = time.perf_counter()
            log_hardware_initialization(
                logger,
                "START | NI-DAQ digital input task | channels=%s sample_rate=%s buffer=%s",
                tuple(channel.physical_channel for channel in digital_channels),
                cfg.sample_rate_hz,
                buffer_size,
            )
            digital_task = self._make_digital_task(
                digital_channels,
                f"reachaq_signal_stream_{device_name}_di",
            )
            self._digital_tasks[device_name] = digital_task
            if self._digital_task is None:
                self._digital_task = digital_task
            kwargs = {}
            if device_name in self._analog_tasks:
                kwargs["source"] = f"/{device_name}/ai/SampleClock"
            elif (
                self._timing_plan is not None
                and self._timing_plan.clock_producer == "external"
            ):
                kwargs["source"] = self._timing_plan.sample_clock_source
            elif self._is_master_device(device_name):
                kwargs["source"] = self._create_digital_sample_clock(
                    device_name,
                    cfg.sample_rate_hz,
                    buffer_size,
                )
            else:
                timing_source = (
                    None
                    if self._timing_plan is None
                    else self._timing_plan.sample_clock_source
                )
                if not timing_source:
                    raise RuntimeError(
                        f"NI-DAQ slave {device_name} has no routed sample clock"
                    )
                kwargs["source"] = timing_source
            digital_task.timing.cfg_samp_clk_timing(
                rate=cfg.sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size,
                **kwargs,
            )
            # The Linux NI-DAQmx default waits for the device FIFO to become
            # more than half full before servicing an interrupt.  On the
            # PXI-6221 that delivered roughly six 10 kHz chunks every 100 ms.
            # Requesting a transfer whenever the FIFO is non-empty preserves
            # correct interrupt-based digital values while delivering each
            # display-sized chunk at its acquisition cadence.
            digital_task.di_channels.all.di_data_xfer_req_cond = (
                self._nidaqmx.constants.InputDataTransferCondition.ON_BOARD_MEMORY_NOT_EMPTY
            )
            self._configure_reference_clock(digital_task)
            self._configure_start_trigger(digital_task, device_name)
            self._configure_exports(digital_task, device_name, "di")
            self._make_stream_reader(device_name, digital_task, digital_channels, analog=False)
            log_hardware_initialization(
                logger,
                "READY | NI-DAQ digital input task | timing=hardware source=%s elapsed=%.3fs",
                kwargs["source"],
                time.perf_counter() - started,
            )

    def _make_digital_task(self, digital_channels, task_name) -> object:
        task = self._nidaqmx.Task(task_name)
        line_grouping = self._nidaqmx.constants.LineGrouping.CHAN_PER_LINE
        for channel in digital_channels:
            task.di_channels.add_di_chan(channel.physical_channel, line_grouping=line_grouping)
        # NI-DAQmx's default DMA path returns zero-filled buffered DI samples on
        # the Linux PXI-6221 runtime, while interrupt transfer returns the
        # correct correlated digital states.
        task.di_channels.all.di_data_xfer_mech = (
            self._nidaqmx.constants.DataTransferActiveTransferMode.INTERRUPT
        )
        return task

    def _uses_multidevice_tasks(self) -> bool:
        plan = self._timing_plan
        return bool(
            plan is not None
            and plan.multidevice_probe_status == "verified"
            and plan.task_graph is not None
            and plan.task_graph.strategy
            in {"auto_multidevice", "forced_multidevice"}
        )

    def _make_stream_reader(self, device_name, task, channels, *, analog):
        in_stream = getattr(task, "in_stream", None)
        readers = getattr(self._nidaqmx, "stream_readers", None)
        if in_stream is None or readers is None:
            return
        chunk_size = self._configuration.read_chunk_size
        if analog:
            self._analog_readers[device_name] = readers.AnalogMultiChannelReader(
                in_stream
            )
            self._analog_buffers[device_name] = numpy.empty(
                (len(channels), chunk_size), dtype=numpy.float64
            )
        else:
            self._digital_readers[device_name] = readers.DigitalMultiChannelReader(
                in_stream
            )
            self._digital_buffers[device_name] = numpy.empty(
                (len(channels), chunk_size), dtype=numpy.bool_
            )

    def _wait_all_available(self, sample_count: int, timeout: float) -> None:
        streams = tuple(
            task.in_stream
            for task_id, task in self._owned_task_records()
            if not task_id.endswith("counter-clock")
            and hasattr(task, "in_stream")
            and hasattr(task.in_stream, "avail_samp_per_chan")
        )
        if not streams:
            return
        deadline = time.perf_counter() + timeout
        while True:
            if all(int(stream.avail_samp_per_chan) >= sample_count for stream in streams):
                return
            if time.perf_counter() >= deadline:
                self._read_telemetry["late_barriers"] += 1
                availability = tuple(
                    int(stream.avail_samp_per_chan) for stream in streams
                )
                raise TimeoutError(
                    "NI-DAQ synchronized availability barrier timed out: "
                    f"required={sample_count} available={availability}"
                )
            time.sleep(0.0005)

    def _read_analog(self, device_name, task, sample_count, timeout):
        reader = self._analog_readers.get(device_name)
        if reader is None:
            return task.read(
                number_of_samples_per_channel=sample_count,
                timeout=timeout,
            )
        buffer = self._analog_buffers[device_name]
        count = reader.read_many_sample(
            buffer,
            number_of_samples_per_channel=sample_count,
            timeout=timeout,
        )
        self._require_exact_reader_count(device_name, count, sample_count)
        return buffer[:, :sample_count]

    def _read_digital(self, device_name, task, sample_count, timeout):
        reader = self._digital_readers.get(device_name)
        if reader is None:
            return task.read(
                number_of_samples_per_channel=sample_count,
                timeout=timeout,
            )
        buffer = self._digital_buffers[device_name]
        count = reader.read_many_sample_multi_line(
            buffer,
            number_of_samples_per_channel=sample_count,
            timeout=timeout,
        )
        self._require_exact_reader_count(device_name, count, sample_count)
        return buffer[:, :sample_count]

    def _require_exact_reader_count(self, device_name, observed, expected):
        if int(observed) == int(expected):
            return
        self._read_telemetry["short_reads"] += 1
        raise RuntimeError(
            f"NI-DAQ task {device_name} returned {observed} samples; expected {expected}"
        )

    def _create_digital_sample_clock(
        self,
        device_name: str,
        sample_rate_hz: float,
        buffer_size: int,
    ) -> str:
        task = self._nidaqmx.Task(f"reachaq_signal_stream_{device_name}_clock")
        self._digital_clock_tasks[device_name] = task
        if self._digital_clock_task is None:
            self._digital_clock_task = task
        task.co_channels.add_co_pulse_chan_freq(f"{device_name}/ctr0", freq=sample_rate_hz)
        task.timing.cfg_implicit_timing(
            sample_mode=self._nidaqmx.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=buffer_size,
        )
        self._configure_reference_clock(task)
        self._configure_exports(task, device_name, "counter")
        return f"/{device_name}/Ctr0InternalOutput"

    def _configure_exports(self, task, device_name: str, subsystem: str) -> None:
        plan = self._timing_plan
        if plan is None or device_name != plan.master_device:
            return
        exporter = getattr(task, "export_signals", None)
        export = None if exporter is None else getattr(exporter, "export_signal", None)
        signals = getattr(self._nidaqmx.constants, "Signal", None)
        if not callable(export) or signals is None:
            if plan.sample_clock_export_terminal or plan.start_trigger_export_terminal:
                raise RuntimeError("NI-DAQ task does not expose signal routing APIs")
            return
        if plan.sample_clock_export_terminal and subsystem in {"ai", "counter"}:
            signal = (
                signals.SAMPLE_CLOCK
                if subsystem == "ai"
                else signals.COUNTER_OUTPUT_EVENT
            )
            export(signal, plan.sample_clock_export_terminal)
        if plan.start_trigger_export_terminal and subsystem == "ai":
            export(signals.START_TRIGGER, plan.start_trigger_export_terminal)

    def _task_start_order(self) -> Tuple[str, ...]:
        available = tuple(dict.fromkeys((
            *self._analog_tasks.keys(),
            *self._digital_tasks.keys(),
            *self._digital_clock_tasks.keys(),
        )))
        if self._timing_plan is None:
            return available
        return tuple(
            device
            for device in self._timing_plan.task_start_order
            if device in available
        ) + tuple(
            device
            for device in available
            if device not in self._timing_plan.task_start_order
        )

    def _is_master_device(self, device_name: str) -> bool:
        if self._timing_plan is None:
            devices = self._task_start_order()
            return not devices or device_name == devices[-1]
        return device_name == self._timing_plan.master_device

    def _sample_clock_kwargs(self, device_name: str) -> Dict[str, str]:
        if (
            self._timing_plan is not None
            and self._timing_plan.clock_producer == "external"
        ):
            if not self._timing_plan.sample_clock_source:
                raise RuntimeError("external NI-DAQ timing has no sample clock")
            return {"source": self._timing_plan.sample_clock_source}
        if self._is_master_device(device_name):
            return {}
        source = (
            None if self._timing_plan is None
            else self._timing_plan.sample_clock_source
        )
        if not source:
            raise RuntimeError(f"NI-DAQ slave {device_name} has no sample clock")
        return {"source": source}

    def _configure_reference_clock(self, task) -> None:
        plan = self._timing_plan
        if plan is None or not plan.reference_clock_source:
            return
        timing = getattr(task, "timing", None)
        if timing is None:
            return
        if hasattr(timing, "ref_clk_src"):
            timing.ref_clk_src = plan.reference_clock_source
        if (
            plan.reference_clock_rate_hz is not None
            and hasattr(timing, "ref_clk_rate")
        ):
            timing.ref_clk_rate = plan.reference_clock_rate_hz

    def _configure_start_trigger(self, task, device_name: str) -> None:
        plan = self._timing_plan
        if (
            plan is None
            or (
                self._is_master_device(device_name)
                and plan.clock_producer != "external"
            )
            or not plan.start_trigger_source
        ):
            return
        triggers = getattr(task, "triggers", None)
        start_trigger = None if triggers is None else getattr(
            triggers, "start_trigger", None
        )
        configure = None if start_trigger is None else getattr(
            start_trigger, "cfg_dig_edge_start_trig", None
        )
        if callable(configure):
            configure(plan.start_trigger_source)

    @staticmethod
    def _channels_by_device(channels) -> Dict[
        str, Tuple[NidaqSignalChannelConfiguration, ...]
    ]:
        grouped: Dict[str, List[NidaqSignalChannelConfiguration]] = {}
        for channel in channels:
            device_name = NidaqSignalStreamController._device_name(
                channel.physical_channel
            )
            grouped.setdefault(device_name, []).append(channel)
        return {
            device_name: tuple(device_channels)
            for device_name, device_channels in grouped.items()
        }

    @staticmethod
    def _device_name(physical_channel: str) -> str:
        parts = physical_channel.strip("/").split("/")
        if len(parts) < 2 or not parts[0]:
            raise RuntimeError(f"cannot infer NI-DAQ device from {physical_channel!r}")
        return parts[0]

    def _analog_input_sample_clock_source(self, physical_channel: str) -> str:
        parts = physical_channel.strip("/").split("/")
        if len(parts) < 2 or not parts[0]:
            raise RuntimeError(f"cannot infer NI-DAQ AI sample clock source from {physical_channel!r}")
        return f"/{parts[0]}/ai/SampleClock"


def _normalize_samples(raw_samples, channel_count: int) -> List[List[object]]:
    if channel_count <= 0:
        return []
    if isinstance(raw_samples, numpy.ndarray):
        array = raw_samples
        if array.ndim == 1:
            return [array.tolist()]
        if array.ndim == 2 and array.shape[0] == channel_count:
            return [row.tolist() for row in array]
        raise RuntimeError(
            f"expected {channel_count} NI-DAQ stream channels, received array "
            f"shape {array.shape}"
        )
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
        from nidaqmx import stream_readers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NI-DAQ signal streaming. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    nidaqmx.stream_readers = stream_readers
    return nidaqmx
