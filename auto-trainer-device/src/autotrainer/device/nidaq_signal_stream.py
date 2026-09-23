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


from autotrainer.device.nidaq_reference_clock import (
    apply_reference_clock,
    resolve_reference_clock,
)

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
        self._require_matching_clock_rate(configuration, timing_plan)
        self._analog_tasks: Dict[str, object] = {}
        #: Per device, where each configured digital channel lives in the
        #: port words that come back: (port index, bit within that port's
        #: channel). Buffered digital is read per port, not per line.
        self._digital_layouts: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        self._digital_port_counts: Dict[str, int] = {}
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
        #: Per physical channel, the referencing the driver says it takes.
        #: None means the question could not be asked.
        self._terminal_config_support = {}
        self._read_telemetry = {
            "blocks": 0,
            "read_seconds": 0.0,
            "read_failures": 0,
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
        read_started = time.perf_counter()

        try:
            self._read_all_devices(values, chunk_size, timeout)
        except Exception as error:
            # A blocking read that times out says only that it did. Which
            # task was short, and whether it is running at all, is what
            # separates "not clocked" from "faulted", and the two need
            # different fixes.
            self._read_telemetry["read_failures"] += 1
            raise RuntimeError(
                f"NI-DAQ read of {chunk_size} samples failed: {error}; "
                f"per task: {self._describe_task_state()}"
            ) from error

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
            channels=tuple(cfg.channels),
            values=values,
            epoch_perf_time=self._epoch_perf_time,
            epoch_wall_time=self._epoch_wall_time,
        )

    def _read_all_devices(self, values, chunk_size, timeout) -> None:
        """Every task's chunk, in start order, into `values`."""
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
                    self._unpack_digital(
                        device_name, raw, len(digital_channels)),
                ):
                    values[channel.name] = tuple(
                        _scale_sample(1.0 if bool(sample) else 0.0, channel)
                        for sample in samples
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
                terminal_config = self._analog_terminal_config()
                # Per channel, because not every analog channel accepts the
                # same referencing and forcing one is fatal rather than
                # approximate. A board's internal AO readback channel
                # (_ao0_vs_aognd) reports DIFF and only DIFF, so naming RSE
                # on it fails the whole task with "requested value is not a
                # supported value" - and takes the sixteen real inputs that
                # do accept RSE down with it.
                if terminal_config is not None and self._channel_accepts(
                        channel.physical_channel, terminal_config):
                    kwargs["terminal_config"] = terminal_config
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
            self._apply_analog_transfer(analog_task, device_name)
            self._configure_reference_clock(analog_task, device_name)
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
                device_name,
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
            self._configure_reference_clock(digital_task, device_name)
            self._configure_start_trigger(digital_task, device_name)
            self._configure_exports(digital_task, device_name, "di")
            self._make_stream_reader(device_name, digital_task, digital_channels, analog=False)
            log_hardware_initialization(
                logger,
                "READY | NI-DAQ digital input task | timing=hardware source=%s elapsed=%.3fs",
                kwargs["source"],
                time.perf_counter() - started,
            )

    def _channel_accepts(self, physical_channel: str, terminal_config) -> bool:
        """Whether this one channel supports the referencing being asked for.

        The driver is asked per channel and its answer is cached. A channel
        that cannot be asked - an older nidaqmx, or a name the system does
        not resolve - is assumed to accept it, so this can only ever remove a
        setting that would have failed, never add one that was not wanted.
        """
        if physical_channel in self._terminal_config_support:
            supported = self._terminal_config_support[physical_channel]
            return supported is None or terminal_config in supported
        try:
            channel = self._nidaqmx.system.PhysicalChannel(physical_channel)
            supported = frozenset(channel.ai_term_cfgs)
        except Exception:
            supported = None
        if not supported:
            self._terminal_config_support[physical_channel] = None
            return True
        self._terminal_config_support[physical_channel] = supported
        return terminal_config in supported

    def _analog_terminal_config(self):
        """How analog inputs should be referenced, or None to let DAQmx pick.

        Letting it pick is not neutral. On a PXI-6221 it gives ai0-ai7
        differential - pairing each with ai8-ai15 - and ai8 upwards
        single-ended, so a channel list spanning both halves reads some
        channels against pins it also reads directly. Measured on
        christielab10: one laser's command copy appeared on three inputs it is
        not wired to, and holding the scan rate down to 10 Hz did not shift
        it, which is how it was told apart from a settling artefact.
        """
        name = getattr(self._configuration, "analog_terminal_config", "") or ""
        if not name:
            return None
        configs = self._nidaqmx.constants.TerminalConfiguration
        resolved = getattr(configs, name.upper(), None)
        if resolved is None:
            logger.warning(
                "unknown analog terminal configuration %r; letting DAQmx "
                "choose per channel, which is not uniform across a 6221",
                name,
            )
        return resolved

    def _make_digital_task(self, device_name, digital_channels, task_name) -> object:
        """One channel per port, holding only the lines this stream reads.

        Buffered digital input has no line-based many-sample read - nidaqmx
        offers the port variants and nothing else - so a task built one
        channel per line can only be read by task.read(), which allocates a
        fresh list of lists every chunk. A channel spanning several lines is
        read by read_many_sample_port_uint32 into a preallocated array
        instead, which is the path NI documents for correlated DIO.

        Only the configured lines go in. Taking the whole port would be
        simpler and would collide with the laser shutter outputs, which sit
        on other lines of this same port.

        A line's bit in the returned word is its physical line number, not
        its position in the channel, and not an offset from the channel's
        lowest line. Measured, after assuming otherwise and getting it wrong:
        a channel declared line1:2 puts line1 at 0x02, and a channel declared
        in the order line2,line3,line0,line1 still puts line0 at 0x01 and
        line1 at 0x02.
        """
        task = self._nidaqmx.Task(task_name)
        line_grouping = self._nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES
        ports: List[str] = []
        lines_by_port: Dict[str, List[str]] = {}
        for channel in digital_channels:
            port = channel.physical_channel.rsplit("/", 1)[0]
            if port not in lines_by_port:
                lines_by_port[port] = []
                ports.append(port)
            lines_by_port[port].append(channel.physical_channel)
        for port in ports:
            task.di_channels.add_di_chan(
                ",".join(lines_by_port[port]), line_grouping=line_grouping)
        self._digital_layouts[device_name] = tuple(
            (
                ports.index(channel.physical_channel.rsplit("/", 1)[0]),
                _line_number(channel.physical_channel),
            )
            for channel in digital_channels
        )
        self._digital_port_counts[device_name] = len(ports)
        # NI-DAQmx's default DMA path returns zero-filled buffered DI
        # samples on the Linux PXI-6221 runtime, while interrupt transfer
        # returns the correct correlated digital states. This is the default
        # rather than the only option: transferMechanismOverrides names a
        # different one per device and subsystem, and until C4 that setting
        # reached the plan and was then ignored by everything.
        mechanism = self._transfer_mechanism(device_name, "di")
        task.di_channels.all.di_data_xfer_mech = (
            mechanism if mechanism is not None
            else self._nidaqmx.constants.DataTransferActiveTransferMode.INTERRUPT
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
            reader = readers.DigitalMultiChannelReader(in_stream)
            # Port words, not lines: read_many_sample_port_uint32 is the
            # widest of the three port variants nidaqmx offers and covers a
            # 32-line port. There is no line-based many-sample read on any
            # reader class, which is what sent an earlier version of this
            # down the allocating task.read() path.
            if not hasattr(reader, "read_many_sample_port_uint32"):
                logger.warning(
                    "nidaqmx %s offers no buffered port read; digital "
                    "channels fall back to task.read(), which allocates per "
                    "chunk",
                    getattr(self._nidaqmx, "__version__", "?"),
                )
                return
            self._digital_readers[device_name] = reader
            self._digital_buffers[device_name] = numpy.zeros(
                (self._digital_port_counts.get(device_name, 1), chunk_size),
                dtype=numpy.uint32,
            )

    @staticmethod
    def _acquires_samples(task) -> bool:
        """Whether this task will ever have input samples to wait for.

        Every nidaqmx task carries an in_stream, including a pure output
        task, whose avail_samp_per_chan is zero for as long as it exists.
        Enabling hardware-timed laser output put the PXI-6713 - an analog
        output board with no input channels - into the same task graph, and
        the barrier then waited on a number that could never arrive:
        "required=167 available=(513, 0)", and the stream never started.
        """
        for attribute in ("ai_channels", "di_channels"):
            channels = getattr(task, attribute, None)
            try:
                if channels is not None and len(channels):
                    return True
            except TypeError:
                continue
        return False

    def _describe_task_state(self) -> str:
        """Per task, how much is waiting and whether it is running at all.

        This was the body of an availability barrier that ran before every
        read, polling each task at half a millisecond until all had a
        chunk. It was removed for being a second wait in front of a
        blocking read that already waits, and for the failure mode it
        carried: it once waited on a pure output task whose available
        count could never rise, and the stream never started.

        It was not removed to save time, and it does not. Measured on
        christielab10 at 10 kHz, 160 chunks over eight seconds, three
        runs each alternating: with the barrier 11.2/11.7/12.1% of a
        core, without it 13.2/13.5/14.0%. The DAQmx blocking read does
        not sleep through the wait - it costs slightly more than the
        spin it replaces. Both deliver the same 160 chunks, because a
        chunk takes its own 50 ms to arrive whatever asks for it.

        What the barrier was genuinely good for was saying which task had
        stalled. That is kept, and moved to where it is needed: a read that
        fails is re-raised carrying this, so the diagnosis survives without
        the spin. It is also what found the analog input fault - "ai=0
        [not-running(DaqError: Onboard device memory overflow...)]" beside a
        digital task at 411 - which a bare DAQmx timeout would not have said.
        """
        def state(task):
            done = getattr(task, "is_task_done", None)
            try:
                return "done" if done() else "running"
            except Exception as error:
                # The message matters: DAQmx reports a task that started and
                # then faulted through the same query as one that never
                # started, and only the text tells them apart.
                text = str(error).replace(chr(10), " ")[:160]
                return f"not-running({type(error).__name__}: {text})"

        def available(task):
            try:
                return str(int(task.in_stream.avail_samp_per_chan))
            except Exception:
                # A polled task holds no host buffer, so it has no answer to
                # give rather than an answer of zero.
                return "unbuffered"

        return ", ".join(
            f"{task_id}={available(task)}[{state(task)}]"
            for task_id, task in self._owned_task_records()
            if not task_id.endswith("counter-clock")
            and self._acquires_samples(task)
        )

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
        """Port words for this device, one row per port."""
        reader = self._digital_readers.get(device_name)
        if reader is None:
            return numpy.atleast_2d(numpy.asarray(task.read(
                number_of_samples_per_channel=sample_count,
                timeout=timeout,
            ), dtype=numpy.uint32))
        buffer = self._digital_buffers[device_name]
        count = reader.read_many_sample_port_uint32(
            buffer,
            number_of_samples_per_channel=sample_count,
            timeout=timeout,
        )
        self._require_exact_reader_count(device_name, count, sample_count)
        return buffer[:, :sample_count]

    def _unpack_digital(self, device_name, ports, channel_count):
        """One row of 0/1 per configured line, taken out of the port words."""
        layout = self._digital_layouts.get(device_name)
        if not layout:
            return _normalize_samples(ports, channel_count)
        return [
            (ports[port_index] >> bit) & 1
            for port_index, bit in layout[:channel_count]
        ]

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
        self._configure_reference_clock(task, device_name)
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

    def _transfer_mechanism(self, device_name: str, subsystem: str):
        """The configured transfer mechanism for this task, or None.

        C4. `transferMechanismOverrides` was carried from the configuration
        into the task graph and read by nothing, while the stream applied a
        fixed interrupt mode to every digital task - a knob that did nothing
        beside a decision nobody could change. It is applied here now, and an
        unrecognised name is refused rather than silently ignored, because
        the whole point of the setting is the case where the default is
        wrong and somebody needs to know their override took effect.
        """
        plan = self._timing_plan
        graph = getattr(plan, "task_graph", None) if plan is not None else None
        for task in (getattr(graph, "tasks", ()) or ()):
            if task.device != device_name or task.subsystem != subsystem:
                continue
            name = (task.transfer_mechanism or "").strip().upper()
            if not name:
                return None
            modes = self._nidaqmx.constants.DataTransferActiveTransferMode
            resolved = getattr(modes, name, None)
            if resolved is None:
                available = ", ".join(
                    sorted(m for m in dir(modes) if m.isupper()))
                raise RuntimeError(
                    f"NI-DAQ transfer mechanism {task.transfer_mechanism!r} "
                    f"for {device_name}.{subsystem} is not one DAQmx has "
                    f"(available: {available})"
                )
            return resolved
        return None

    def _apply_analog_transfer(self, task, device_name: str) -> None:
        """An analog override, when one is configured. There is no default."""
        mechanism = self._transfer_mechanism(device_name, "ai")
        if mechanism is None:
            return
        task.ai_channels.all.ai_data_xfer_mech = mechanism

    @staticmethod
    def _require_matching_clock_rate(configuration, plan) -> None:
        """Refuse a stream whose declared rate is not the clock it will get.

        C7. The rate handed to cfg_samp_clk_timing beside an external source
        does not set the rate - the clock does - it sizes the buffer and is
        what every sample count downstream is computed from. When the two
        disagree, nothing fails: every timestamp is wrong by the ratio, and
        on this rig that shape of defect once stretched a two-second pulse
        train to twenty seconds and nobody saw an error.

        Only the laser derived its rate from the plan. The stream took its
        own configured value and never compared them, which is the same
        latent bug one layer up.
        """
        if plan is None:
            return
        planned = getattr(plan, "sample_clock_rate_hz", None)
        declared = getattr(configuration, "sample_rate_hz", None)
        if not planned or not declared:
            return
        # A ratio rather than equality: these are floats that have been
        # through YAML and a plan, and a part in a million is not a
        # disagreement about the clock.
        if abs(planned - declared) / max(planned, declared) > 1e-6:
            raise RuntimeError(
                f"NI-DAQ signal stream is configured for {declared:g} Hz and "
                f"the timing plan will clock it at {planned:g} Hz; a chunk "
                f"the stream treats as one second would really take "
                f"{declared / planned:g} seconds, and every duration derived "
                "from the configured rate is wrong by that factor"
            )

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

    def _reference_clock_for(self, device_name: str):
        """The device's own reference-clock terminal, or None if it has none."""
        plan = self._timing_plan
        return resolve_reference_clock(
            self._nidaqmx,
            device_name,
            None if plan is None else plan.reference_clock_source,
        )

    def _configure_reference_clock(self, task, device_name: str) -> None:
        apply_reference_clock(
            self._nidaqmx, task, self._timing_plan, device_name=device_name)

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


def _line_number(physical_channel: str) -> int:
    """The line's own number, which is its bit in the port word.

    The last segment has to be a line: harvesting digits from whatever is
    there accepts "port0" and calls it line zero, which is a whole port
    silently mistaken for one line.
    """
    tail = physical_channel.rsplit("/", 1)[-1].strip().lower()
    if not tail.startswith("line") or not tail[4:].isdigit():
        raise ValueError(
            f"NI-DAQ digital channel {physical_channel!r} does not name a line"
        )
    return int(tail[4:])


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
