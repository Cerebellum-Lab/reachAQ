from __future__ import annotations

import dataclasses
import logging
import numbers
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from autotrainer.core import NidaqTimingPlan
from autotrainer.core.logging import log_hardware_initialization

from .laser import (
    LaserCalibrationPoint,
    LaserCalibrationRamp,
    LaserChannelConfiguration,
    LaserChannelId,
    LaserFeedbackSample,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    LaserSystemConfiguration,
    normalize_laser_channel_id,
)


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _NidaqLaserTasks:
    analog_output: Optional[object]
    diode_input: Optional[object]
    command_copy_input: Optional[object]
    shutter_output: object
    auxiliary_output: Optional[object]

    def close(self) -> None:
        errors = []
        for name, task in (
            ("shutter_output", self.shutter_output),
            ("auxiliary_output", self.auxiliary_output),
            ("analog_output", self.analog_output),
            ("command_copy_input", self.command_copy_input),
            ("diode_input", self.diode_input),
        ):
            if task is not None:
                try:
                    task.close()
                except Exception as exc:
                    errors.append((name, exc))
        if errors:
            names = ", ".join(name for name, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ laser task(s): {names}") from errors[0][1]


class NidaqLaserController:
    """NI-DAQmx-backed laser controller.

    Manual command/shutter paths use on-demand NI-DAQmx writes. When
    ``LaserSystemConfiguration.hardware_timed`` is enabled, finite pulse trains
    use transient sample-clocked AO tasks so manual command tasks do not reserve
    the same physical output channel.
    """

    def __init__(
        self,
        configuration: LaserSystemConfiguration,
        *,
        feedback_reader: Optional[Callable[[str], float]] = None,
        timing_plan: Optional[NidaqTimingPlan] = None,
    ):
        if configuration.backend != "nidaq":
            raise ValueError("NidaqLaserController requires laser backend 'nidaq'")
        runtime_started = time.perf_counter()
        log_hardware_initialization(logger, "START | NI-DAQmx runtime | consumer=laser")
        self._nidaqmx = _load_nidaqmx()
        log_hardware_initialization(
            logger,
            "READY | NI-DAQmx runtime | consumer=laser elapsed=%.3fs",
            time.perf_counter() - runtime_started,
        )
        self._configuration = configuration
        self._feedback_reader = feedback_reader
        self._timing_plan = timing_plan
        self._last_timing_status = {
            "status": "independent",
            "reason": "No finite laser waveform has been executed",
        }
        self._tasks: Dict[LaserChannelId, _NidaqLaserTasks] = {}
        self._command_volts: Dict[LaserChannelId, float] = {}
        try:
            for channel in configuration.channels:
                channel_started = time.perf_counter()
                log_hardware_initialization(
                    logger,
                    "START | NI-DAQ laser channel | id=%s AO=%s diode_AI=%s shutter_DO=%s auxiliary_DO=%s",
                    channel.channel_id.value,
                    channel.analog_output,
                    channel.diode_input,
                    channel.shutter_output,
                    channel.auxiliary_output,
                )
                self._tasks[channel.channel_id] = self._create_channel_tasks(channel)
                self._command_volts[channel.channel_id] = channel.minimum_command_volts
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
                self.set_shutter_open(channel.channel_id, False)
                if channel.auxiliary_output is not None:
                    self.set_auxiliary_output(channel.channel_id, False)
                log_hardware_initialization(
                    logger,
                    "READY | NI-DAQ laser channel | id=%s elapsed=%.3fs",
                    channel.channel_id.value,
                    time.perf_counter() - channel_started,
                )
        except Exception:
            try:
                self.close()
            except Exception:
                logger.exception("Failed to close partially initialized NI-DAQ laser controller")
            raise

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    @property
    def timing_status(self) -> dict:
        return dict(self._last_timing_status)

    def set_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> float:
        channel = self._configuration.get_channel(channel_id)
        applied = channel.clamp_command_voltage(volts)
        tasks = self._tasks[channel.channel_id]
        if tasks.analog_output is None:
            self._write_transient_analog_sample(channel, applied)
        else:
            tasks.analog_output.write(applied, auto_start=True)
        self._command_volts[channel.channel_id] = applied
        return applied

    def set_shutter_open(self, channel_id: Union[LaserChannelId, int], is_open: bool) -> None:
        channel = self._configuration.get_channel(channel_id)
        self._tasks[channel.channel_id].shutter_output.write(bool(is_open), auto_start=True)

    def set_auxiliary_output(self, channel_id: Union[LaserChannelId, int], enabled: bool) -> None:
        channel = self._configuration.get_channel(channel_id)
        auxiliary_output = self._tasks[channel.channel_id].auxiliary_output
        if auxiliary_output is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no auxiliary_output configured"
            )
        auxiliary_output.write(bool(enabled), auto_start=True)

    def read_diode_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        channel = self._configuration.get_channel(channel_id)
        if self._feedback_reader is not None:
            return float(self._feedback_reader(channel.diode_input))
        raw = self._tasks[channel.channel_id].diode_input.read()
        return float(raw) * channel.feedback_scale

    def read_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        channel = self._configuration.get_channel(channel_id)
        command_copy_volts = self._read_optional_command_copy_voltage(channel.channel_id)
        if command_copy_volts is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no AI command-copy input configured"
            )
        return command_copy_volts

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        normalized = normalize_laser_channel_id(channel_id)
        return LaserFeedbackSample(
            channel_id=normalized,
            command_volts=self._command_volts[normalized],
            diode_volts=self.read_diode_voltage(normalized),
            command_copy_volts=self._read_optional_command_copy_voltage(normalized),
        )

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        self.run_synchronized_pulse_train(
            LaserSynchronizedPulseTrain(
                pulse_trains=(pulse_train,),
                trigger_source=pulse_train.trigger_source,
                trigger_edge=pulse_train.trigger_edge,
                enable_pmt_shutter=pulse_train.enable_pmt_shutter,
                wait=pulse_train.wait,
                timeout_seconds=pulse_train.timeout_seconds,
            )
        )

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain) -> None:
        if not self._configuration.hardware_timed:
            raise RuntimeError("Hardware-timed laser pulse trains require laser configuration hardware_timed=True")
        if not pulse_train.wait:
            raise NotImplementedError("Asynchronous hardware-timed laser output is not implemented yet")
        channels = [
            self._configuration.get_channel(channel_pulse.channel_id)
            for channel_pulse in pulse_train.pulse_trains
        ]
        for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
            self._validate_command_voltage(channel, channel_pulse.amplitude_volts)
        sample_rate_hz = self._require_sample_rate()
        waveforms = [
            self._build_pulse_train_waveform(channel, channel_pulse)
            for channel, channel_pulse in zip(channels, pulse_train.pulse_trains)
        ]
        pmt_enabled = pulse_train.enable_pmt_shutter or any(
            channel_pulse.enable_pmt_shutter for channel_pulse in pulse_train.pulse_trains
        )
        pmt_open_delay_ms = max(
            (channel_pulse.pmt_shutter_open_delay_ms for channel_pulse in pulse_train.pulse_trains),
            default=0.0,
        ) if pmt_enabled else 0.0
        pmt_close_delay_ms = max(
            (channel_pulse.pmt_shutter_close_delay_ms for channel_pulse in pulse_train.pulse_trains),
            default=0.0,
        ) if pmt_enabled else 0.0
        pre_samples = _samples_from_ms(pmt_open_delay_ms, sample_rate_hz)
        post_samples = _samples_from_ms(pmt_close_delay_ms, sample_rate_hz)
        max_waveform_samples = max(len(waveform) for waveform in waveforms)
        timed_waveforms = []
        for channel, waveform in zip(channels, waveforms):
            minimum = channel.minimum_command_volts
            timed_waveforms.append(
                [minimum] * pre_samples
                + waveform
                + [minimum] * (max_waveform_samples - len(waveform) + post_samples)
            )
        total_samples = len(timed_waveforms[0])
        timeout_seconds = pulse_train.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = total_samples / sample_rate_hz + 5.0
        ao_task = self._create_synchronized_analog_output_task(channels, "laser_sync_pulse_ao")
        digital_tasks = []
        run_error = None
        try:
            timing_kwargs, timing_status = self._resolve_pulse_timing(
                channels, pulse_train,
            )
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=total_samples,
                **timing_kwargs,
            )
            self._configure_timing_reference(ao_task, timing_status)
            if pulse_train.trigger_source:
                ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    pulse_train.trigger_source,
                    trigger_edge=self._get_trigger_edge(pulse_train.trigger_edge),
                )
            ao_task.write(timed_waveforms[0] if len(timed_waveforms) == 1 else timed_waveforms, auto_start=False)
            sample_clock_source = self._analog_output_sample_clock_source(channels[0].analog_output)
            if pmt_enabled:
                pmt_line = self._require_pmt_shutter_output()
                digital_tasks.append(
                    self._create_finite_digital_output_task(
                        pmt_line,
                        "laser_pmt_shutter_do",
                        [True] * total_samples,
                        sample_rate_hz,
                        total_samples,
                        sample_clock_source,
                        pulse_train.trigger_source,
                        pulse_train.trigger_edge,
                    )
                )
            for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
                if channel_pulse.emit_trigger_output:
                    if channel.trigger_output is None:
                        raise RuntimeError(
                            f"laser channel {channel.channel_id.value} requested trigger output, "
                            "but trigger_output is not configured"
                        )
                    digital_tasks.append(
                        self._create_finite_digital_output_task(
                            channel.trigger_output,
                            f"laser_{channel.channel_id.value}_trigger_do",
                            self._build_digital_pulse_waveform(
                                total_samples,
                                channel_pulse.trigger_output_pulse_ms,
                                sample_rate_hz,
                            ),
                            sample_rate_hz,
                            total_samples,
                            sample_clock_source,
                            pulse_train.trigger_source,
                            pulse_train.trigger_edge,
                        )
                    )
                if channel_pulse.emit_timing_trigger_output:
                    if channel.timing_trigger_output is None:
                        raise RuntimeError(
                            f"laser channel {channel.channel_id.value} requested timing trigger output, "
                            "but timing_trigger_output is not configured"
                        )
                    digital_tasks.append(
                        self._create_finite_digital_output_task(
                            channel.timing_trigger_output,
                            f"laser_{channel.channel_id.value}_timing_trigger_do",
                            self._build_digital_pulse_waveform(
                                total_samples,
                                channel_pulse.timing_trigger_output_pulse_ms,
                                sample_rate_hz,
                            ),
                            sample_rate_hz,
                            total_samples,
                            sample_clock_source,
                            pulse_train.trigger_source,
                            pulse_train.trigger_edge,
                        )
                    )
            for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
                if channel_pulse.open_shutter:
                    self.set_shutter_open(channel.channel_id, True)
            for task in digital_tasks:
                task.start()
            ao_task.start()
            self._last_timing_status = timing_status
            ao_task.wait_until_done(timeout=timeout_seconds)
            for task in digital_tasks:
                task.wait_until_done(timeout=timeout_seconds)
        except Exception as exc:
            run_error = exc
            raise
        finally:
            self._cleanup_pulse_train(
                ao_task=ao_task,
                digital_tasks=digital_tasks,
                channels=channels,
                channel_pulses=pulse_train.pulse_trains,
                close_pmt=pmt_enabled,
                run_error=run_error,
            )

    def _resolve_pulse_timing(self, channels, pulse_train):
        plan = self._timing_plan
        output_devices = tuple(dict.fromkeys(
            channel.analog_output.strip("/").split("/", 1)[0]
            for channel in channels
        ))
        base = {
            "devices": output_devices,
            "sampleClockSource": None,
            "startTriggerSource": pulse_train.trigger_source,
            "referenceClockSource": None,
        }
        if plan is None or not plan.is_valid:
            return {}, {
                **base,
                "status": "independent",
                "reason": "No valid shared NI timing plan was applied",
            }
        undeclared = tuple(
            device for device in output_devices
            if device not in plan.hardware_output_devices
        )
        if undeclared:
            return {}, {
                **base,
                "status": "unsupported",
                "reason": "Laser output device was not in the resolved timing topology",
            }
        if not pulse_train.trigger_source:
            return {}, {
                **base,
                "status": "declared_not_armed",
                "reason": (
                    "The acquisition clock is already running and this on-demand "
                    "waveform has no future hardware start trigger"
                ),
            }
        if not plan.sample_clock_source:
            return {}, {
                **base,
                "status": "unsupported",
                "reason": "Resolved timing topology has no shared sample clock",
            }
        return {"source": plan.sample_clock_source}, {
            **base,
            "status": "hardware_synchronized",
            "reason": "Finite output armed for a future trigger on the shared sample clock",
            "sampleClockSource": plan.sample_clock_source,
            "referenceClockSource": plan.reference_clock_source,
        }

    def _configure_timing_reference(self, task, timing_status) -> None:
        if timing_status.get("status") != "hardware_synchronized":
            return
        plan = self._timing_plan
        timing = getattr(task, "timing", None)
        if plan is None or timing is None or not plan.reference_clock_source:
            return
        if hasattr(timing, "ref_clk_src"):
            timing.ref_clk_src = plan.reference_clock_source
        if (
            plan.reference_clock_rate_hz is not None
            and hasattr(timing, "ref_clk_rate")
        ):
            timing.ref_clk_rate = plan.reference_clock_rate_hz

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp) -> Tuple[LaserCalibrationPoint, ...]:
        if self._feedback_reader is not None:
            raise RuntimeError(
                "Laser calibration is unavailable while the shared NI-DAQ input "
                "stream owns the feedback channels"
            )
        if not self._configuration.hardware_timed:
            raise RuntimeError("Hardware-timed laser calibration ramps require laser configuration hardware_timed=True")
        channel = self._configuration.get_channel(ramp.channel_id)
        self._validate_command_voltage(channel, ramp.start_volts)
        self._validate_command_voltage(channel, ramp.stop_volts)
        if ramp.enable_pmt_shutter and (
            ramp.pmt_shutter_open_delay_ms > 0 or ramp.pmt_shutter_close_delay_ms > 0
        ):
            raise NotImplementedError("PMT shutter delays for calibration ramps are not implemented yet")
        sample_rate_hz = self._require_sample_rate()
        waveform = self._build_calibration_ramp_waveform(ramp)
        timeout_seconds = ramp.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = len(waveform) / sample_rate_hz + 5.0
        ao_task = self._create_analog_output_task(channel, f"laser_{channel.channel_id.value}_calibration_ao")
        ai_task = self._create_calibration_input_task(channel)
        digital_tasks = []
        run_error = None
        points = ()
        try:
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(waveform),
            )
            sample_clock_source = self._analog_output_sample_clock_source(channel.analog_output)
            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                source=sample_clock_source,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(waveform),
            )
            if ramp.enable_pmt_shutter:
                digital_tasks.append(
                    self._create_finite_digital_output_task(
                        self._require_pmt_shutter_output(),
                        "laser_pmt_shutter_calibration_do",
                        [True] * len(waveform),
                        sample_rate_hz,
                        len(waveform),
                        sample_clock_source,
                        None,
                        "rising",
                    )
                )
            ao_task.write(waveform, auto_start=False)
            if ramp.open_shutter:
                self.set_shutter_open(channel.channel_id, True)
            for task in digital_tasks:
                task.start()
            ai_task.start()
            ao_task.start()
            ao_task.wait_until_done(timeout=timeout_seconds)
            ai_task.wait_until_done(timeout=timeout_seconds)
            raw_samples = ai_task.read(number_of_samples_per_channel=len(waveform), timeout=timeout_seconds)
            points = self._build_calibration_points(channel, ramp, raw_samples)
        except Exception as exc:
            run_error = exc
            raise
        finally:
            self._cleanup_calibration_ramp(
                ao_task=ao_task,
                ai_task=ai_task,
                digital_tasks=digital_tasks,
                channel=channel,
                ramp=ramp,
                run_error=run_error,
            )
        return points

    def _validate_command_voltage(self, channel: LaserChannelConfiguration, volts: float) -> None:
        if not channel.minimum_command_volts <= volts <= channel.maximum_command_volts:
            raise ValueError(
                f"laser channel {channel.channel_id.value} command {volts} V is outside "
                f"the configured range {channel.minimum_command_volts}..{channel.maximum_command_volts} V"
            )

    def _cleanup_pulse_train(
        self,
        ao_task: object,
        digital_tasks: List[object],
        channels: List[LaserChannelConfiguration],
        channel_pulses: Tuple[LaserPulseTrain, ...],
        close_pmt: bool,
        run_error: Optional[BaseException],
    ) -> None:
        errors = []
        self._stop_and_close_task("pulse analog output task", ao_task, errors)
        for index, task in enumerate(digital_tasks):
            self._stop_and_close_task(f"pulse digital output task {index}", task, errors)
        for channel in channels:
            try:
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} command reset", exc))
                logger.exception("Failed to reset NI-DAQ laser command for channel %s", channel.channel_id.value)
        for channel, channel_pulse in zip(channels, channel_pulses):
            if channel_pulse.close_shutter:
                try:
                    self.set_shutter_open(channel.channel_id, False)
                except Exception as exc:
                    errors.append((f"channel {channel.channel_id.value} shutter close", exc))
                    logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
        if close_pmt:
            try:
                self._write_transient_digital_line(
                    self._require_pmt_shutter_output(),
                    False,
                    "laser_pmt_shutter_reset",
                )
            except Exception as exc:
                errors.append(("PMT shutter close", exc))
                logger.exception("Failed to close NI-DAQ PMT shutter output")
        self._raise_or_log_cleanup_errors("NI-DAQ laser pulse train", errors, run_error)

    def _cleanup_calibration_ramp(
        self,
        ao_task: object,
        ai_task: object,
        digital_tasks: List[object],
        channel: LaserChannelConfiguration,
        ramp: LaserCalibrationRamp,
        run_error: Optional[BaseException],
    ) -> None:
        errors = []
        self._stop_and_close_task("calibration analog output task", ao_task, errors)
        self._stop_and_close_task("calibration analog input task", ai_task, errors)
        for index, task in enumerate(digital_tasks):
            self._stop_and_close_task(f"calibration digital output task {index}", task, errors)
        try:
            self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
        except Exception as exc:
            errors.append((f"channel {channel.channel_id.value} command reset", exc))
            logger.exception("Failed to reset NI-DAQ laser command for channel %s", channel.channel_id.value)
        if ramp.close_shutter:
            try:
                self.set_shutter_open(channel.channel_id, False)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} shutter close", exc))
                logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
        if ramp.enable_pmt_shutter:
            try:
                self._write_transient_digital_line(
                    self._require_pmt_shutter_output(),
                    False,
                    "laser_pmt_shutter_calibration_reset",
                )
            except Exception as exc:
                errors.append(("PMT shutter close", exc))
                logger.exception("Failed to close NI-DAQ PMT shutter output after calibration")
        self._raise_or_log_cleanup_errors("NI-DAQ laser calibration ramp", errors, run_error)

    def _stop_and_close_task(self, name: str, task: object, errors: list) -> None:
        try:
            task.stop()
        except Exception as exc:
            errors.append((f"{name} stop", exc))
            logger.exception("Failed to stop %s", name)
        try:
            task.close()
        except Exception as exc:
            errors.append((f"{name} close", exc))
            logger.exception("Failed to close %s", name)

    def _raise_or_log_cleanup_errors(
        self,
        context: str,
        errors: list,
        run_error: Optional[BaseException],
    ) -> None:
        if not errors:
            return
        locations = ", ".join(location for location, _ in errors)
        cleanup_error = RuntimeError(f"Failed to clean up {context}: {locations}")
        if run_error is None:
            raise cleanup_error from errors[0][1]
        logger.error("Failed to clean up %s after output error: %s", context, locations)

    def close_all_shutters(self) -> None:
        errors = []
        for channel in self._configuration.channels:
            try:
                self.set_shutter_open(channel.channel_id, False)
            except Exception as exc:
                errors.append((channel.channel_id, exc))
                logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
        if errors:
            channels = ", ".join(str(channel_id.value) for channel_id, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ laser shutter(s) for channel(s): {channels}") from errors[0][1]

    def close(self) -> None:
        errors = []
        for channel in self._configuration.channels:
            if channel.channel_id not in self._tasks:
                continue
            try:
                self.set_shutter_open(channel.channel_id, False)
                if channel.auxiliary_output is not None:
                    self.set_auxiliary_output(channel.channel_id, False)
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} reset", exc))
                logger.exception("Failed to reset NI-DAQ laser channel %s during close", channel.channel_id.value)
        for channel_id, tasks in self._tasks.items():
            try:
                tasks.close()
            except Exception as exc:
                errors.append((f"channel {channel_id.value} task close", exc))
                logger.exception("Failed to close NI-DAQ laser tasks for channel %s", channel_id.value)
        self._tasks.clear()
        if errors:
            locations = ", ".join(location for location, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ laser controller cleanly: {locations}") from errors[0][1]

    def _create_channel_tasks(self, channel: LaserChannelConfiguration) -> _NidaqLaserTasks:
        analog_output = None
        if not self._configuration.hardware_timed:
            analog_output = self._create_analog_output_task(channel, f"laser_{channel.channel_id.value}_ao")
        diode_input = None
        if self._feedback_reader is None:
            diode_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_ai")
            diode_input.ai_channels.add_ai_voltage_chan(channel.diode_input)

        command_copy_input = None
        if channel.command_copy_input and self._feedback_reader is None:
            command_copy_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_command_copy_ai")
            command_copy_input.ai_channels.add_ai_voltage_chan(channel.command_copy_input)

        shutter_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_shutter")
        shutter_output.do_channels.add_do_chan(channel.shutter_output)

        auxiliary_output = None
        if channel.auxiliary_output is not None:
            auxiliary_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_aux")
            auxiliary_output.do_channels.add_do_chan(channel.auxiliary_output)

        return _NidaqLaserTasks(
            analog_output=analog_output,
            diode_input=diode_input,
            command_copy_input=command_copy_input,
            shutter_output=shutter_output,
            auxiliary_output=auxiliary_output,
        )

    def _create_analog_output_task(self, channel: LaserChannelConfiguration, name: str):
        analog_output = self._nidaqmx.Task(name)
        analog_output.ao_channels.add_ao_voltage_chan(
            channel.analog_output,
            min_val=channel.minimum_command_volts,
            max_val=channel.maximum_command_volts,
        )
        return analog_output

    def _create_synchronized_analog_output_task(self, channels: List[LaserChannelConfiguration], name: str):
        analog_output = self._nidaqmx.Task(name)
        for channel in channels:
            analog_output.ao_channels.add_ao_voltage_chan(
                channel.analog_output,
                min_val=channel.minimum_command_volts,
                max_val=channel.maximum_command_volts,
            )
        return analog_output

    def _create_calibration_input_task(self, channel: LaserChannelConfiguration):
        analog_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_calibration_ai")
        analog_input.ai_channels.add_ai_voltage_chan(channel.diode_input)
        if channel.command_copy_input is not None:
            analog_input.ai_channels.add_ai_voltage_chan(channel.command_copy_input)
        return analog_input

    def _create_finite_digital_output_task(
        self,
        physical_line: str,
        name: str,
        waveform: List[bool],
        sample_rate_hz: float,
        total_samples: int,
        sample_clock_source: str,
        trigger_source: Optional[str],
        trigger_edge: str,
    ):
        if len(waveform) != total_samples:
            raise RuntimeError(
                f"digital output waveform for {physical_line} has {len(waveform)} samples, "
                f"expected {total_samples}"
            )
        digital_output = self._nidaqmx.Task(name)
        try:
            digital_output.do_channels.add_do_chan(physical_line)
            digital_output.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                source=sample_clock_source,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=total_samples,
            )
            if trigger_source:
                digital_output.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source,
                    trigger_edge=self._get_trigger_edge(trigger_edge),
                )
            digital_output.write(waveform, auto_start=False)
        except Exception:
            digital_output.close()
            raise
        return digital_output

    def _write_transient_analog_sample(self, channel: LaserChannelConfiguration, volts: float) -> None:
        task = self._create_analog_output_task(channel, f"laser_{channel.channel_id.value}_manual_ao")
        try:
            task.write(volts, auto_start=True)
        finally:
            task.close()

    def _write_transient_digital_line(self, physical_line: str, enabled: bool, name: str) -> None:
        task = self._nidaqmx.Task(name)
        try:
            task.do_channels.add_do_chan(physical_line)
            task.write(bool(enabled), auto_start=True)
        finally:
            task.close()

    def _read_optional_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> Optional[float]:
        channel = self._configuration.get_channel(channel_id)
        if channel.command_copy_input is None:
            return None
        if self._feedback_reader is not None:
            return float(self._feedback_reader(channel.command_copy_input))
        task = self._tasks[channel.channel_id].command_copy_input
        if task is None:
            return None
        return float(task.read()) * channel.command_copy_scale

    def _build_pulse_train_waveform(
        self,
        channel: LaserChannelConfiguration,
        pulse_train: LaserPulseTrain,
    ) -> list:
        sample_rate_hz = self._require_sample_rate()
        baseline_samples = _samples_from_ms(pulse_train.baseline_ms, sample_rate_hz)
        high_samples = max(1, _samples_from_ms(pulse_train.duration_ms, sample_rate_hz))
        post_stim_samples = _samples_from_ms(pulse_train.post_stim_ms, sample_rate_hz)
        minimum = channel.minimum_command_volts
        amplitude = pulse_train.amplitude_volts
        waveform = [minimum] * baseline_samples
        if pulse_train.pulse_count == 1:
            waveform.extend([amplitude] * high_samples)
        else:
            period_samples = max(1, int(round(sample_rate_hz / pulse_train.frequency_hz)))
            if high_samples > period_samples:
                raise ValueError(
                    f"laser pulse duration {pulse_train.duration_ms} ms exceeds pulse period "
                    f"at {pulse_train.frequency_hz} Hz"
                )
            low_samples = period_samples - high_samples
            for pulse_index in range(pulse_train.pulse_count):
                waveform.extend([amplitude] * high_samples)
                if pulse_index < pulse_train.pulse_count - 1:
                    waveform.extend([minimum] * low_samples)
        waveform.extend([minimum] * post_stim_samples)
        if not waveform:
            raise ValueError("laser pulse train waveform is empty")
        return waveform

    def _build_digital_pulse_waveform(
        self,
        total_samples: int,
        pulse_ms: float,
        sample_rate_hz: float,
    ) -> List[bool]:
        pulse_samples = min(total_samples, max(1, _samples_from_ms(pulse_ms, sample_rate_hz)))
        return [True] * pulse_samples + [False] * (total_samples - pulse_samples)

    def _build_calibration_ramp_waveform(self, ramp: LaserCalibrationRamp) -> List[float]:
        waveform = []
        for index in range(ramp.steps):
            fraction = index / (ramp.steps - 1)
            command_volts = ramp.start_volts + fraction * (ramp.stop_volts - ramp.start_volts)
            waveform.extend([command_volts] * ramp.samples_per_step)
        if not waveform:
            raise ValueError("laser calibration ramp waveform is empty")
        return waveform

    def _build_calibration_points(
        self,
        channel: LaserChannelConfiguration,
        ramp: LaserCalibrationRamp,
        raw_samples,
    ) -> Tuple[LaserCalibrationPoint, ...]:
        channel_count = 2 if channel.command_copy_input is not None else 1
        samples = self._normalize_ai_samples(raw_samples, channel_count)
        points = []
        for index in range(ramp.steps):
            start = index * ramp.samples_per_step
            stop = start + ramp.samples_per_step
            fraction = index / (ramp.steps - 1)
            command_volts = ramp.start_volts + fraction * (ramp.stop_volts - ramp.start_volts)
            diode_volts = _mean(samples[0][start:stop]) * channel.feedback_scale
            command_copy_volts = None
            if channel.command_copy_input is not None:
                command_copy_volts = _mean(samples[1][start:stop]) * channel.command_copy_scale
            points.append(
                LaserCalibrationPoint(
                    channel_id=channel.channel_id,
                    command_volts=command_volts,
                    diode_volts=diode_volts,
                    command_copy_volts=command_copy_volts,
                )
            )
        return tuple(points)

    def _normalize_ai_samples(self, raw_samples, channel_count: int) -> List[List[float]]:
        raw_samples = list(raw_samples)
        if channel_count == 1:
            if raw_samples and not isinstance(raw_samples[0], numbers.Number):
                return [list(raw_samples[0])]
            return [list(raw_samples)]
        if len(raw_samples) != channel_count:
            raise RuntimeError(f"expected {channel_count} analog input channels, received {len(raw_samples)}")
        return [list(channel_samples) for channel_samples in raw_samples]

    def _require_pmt_shutter_output(self) -> str:
        pmt_shutter_output = self._configuration.pmt_shutter_output
        if pmt_shutter_output is None:
            raise RuntimeError("PMT shutter output requested, but laser pmt_shutter_output is not configured")
        return pmt_shutter_output

    def _analog_output_sample_clock_source(self, physical_channel: str) -> str:
        parts = physical_channel.strip("/").split("/")
        if len(parts) < 2 or not parts[0]:
            raise RuntimeError(f"cannot infer NI-DAQ AO sample clock source from physical channel {physical_channel}")
        return f"/{parts[0]}/ao/SampleClock"

    def _require_sample_rate(self) -> float:
        sample_rate_hz = self._configuration.sample_rate_hz
        if sample_rate_hz is None:
            raise RuntimeError("hardware-timed laser output requires sample_rate_hz")
        return sample_rate_hz

    def _get_trigger_edge(self, trigger_edge: str):
        if trigger_edge == "rising":
            return self._nidaqmx.constants.Edge.RISING
        if trigger_edge == "falling":
            return self._nidaqmx.constants.Edge.FALLING
        raise ValueError("trigger_edge must be 'rising' or 'falling'")


def _samples_from_ms(value_ms: float, sample_rate_hz: float) -> int:
    if value_ms <= 0:
        return 0
    return max(1, int(round(value_ms * sample_rate_hz / 1000.0)))


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise RuntimeError("cannot average an empty calibration sample segment")
    return float(sum(values)) / len(values)


def _load_nidaqmx():
    try:
        import nidaqmx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NidaqLaserController. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    return nidaqmx
