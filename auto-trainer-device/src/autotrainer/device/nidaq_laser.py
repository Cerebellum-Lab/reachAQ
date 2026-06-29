from __future__ import annotations

import dataclasses
import logging
from typing import Dict, Optional, Union

from .laser import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserFeedbackSample,
    LaserPulseTrain,
    LaserSystemConfiguration,
    normalize_laser_channel_id,
)


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _NidaqLaserTasks:
    analog_output: Optional[object]
    diode_input: object
    command_copy_input: Optional[object]
    shutter_output: object
    auxiliary_output: object

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

    def __init__(self, configuration: LaserSystemConfiguration):
        if configuration.backend != "nidaq":
            raise ValueError("NidaqLaserController requires laser backend 'nidaq'")
        self._nidaqmx = _load_nidaqmx()
        self._configuration = configuration
        self._tasks: Dict[LaserChannelId, _NidaqLaserTasks] = {}
        self._command_volts: Dict[LaserChannelId, float] = {}
        try:
            for channel in configuration.channels:
                self._tasks[channel.channel_id] = self._create_channel_tasks(channel)
                self._command_volts[channel.channel_id] = channel.minimum_command_volts
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
                self.set_shutter_open(channel.channel_id, False)
                self.set_auxiliary_output(channel.channel_id, False)
        except Exception:
            try:
                self.close()
            except Exception:
                logger.exception("Failed to close partially initialized NI-DAQ laser controller")
            raise

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

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
        self._tasks[channel.channel_id].auxiliary_output.write(bool(enabled), auto_start=True)

    def read_diode_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        channel = self._configuration.get_channel(channel_id)
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
        if not self._configuration.hardware_timed:
            raise RuntimeError("Hardware-timed laser pulse trains require laser configuration hardware_timed=True")
        if not pulse_train.wait:
            raise NotImplementedError("Asynchronous hardware-timed laser output is not implemented yet")
        if pulse_train.enable_pmt_shutter:
            raise NotImplementedError("Hardware-timed PMT shutter sequencing is not implemented yet")
        channel = self._configuration.get_channel(pulse_train.channel_id)
        if not channel.minimum_command_volts <= pulse_train.amplitude_volts <= channel.maximum_command_volts:
            raise ValueError(
                f"laser channel {channel.channel_id.value} amplitude {pulse_train.amplitude_volts} V is outside "
                f"the configured range {channel.minimum_command_volts}..{channel.maximum_command_volts} V"
            )
        waveform = self._build_pulse_train_waveform(channel, pulse_train)
        sample_rate_hz = self._require_sample_rate()
        timeout_seconds = pulse_train.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = len(waveform) / sample_rate_hz + 5.0
        task = self._create_analog_output_task(channel, f"laser_{channel.channel_id.value}_pulse_ao")
        run_error = None
        try:
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(waveform),
            )
            if pulse_train.trigger_source:
                task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    pulse_train.trigger_source,
                    trigger_edge=self._get_trigger_edge(pulse_train.trigger_edge),
                )
            task.write(waveform, auto_start=False)
            if pulse_train.open_shutter:
                self.set_shutter_open(channel.channel_id, True)
            task.start()
            task.wait_until_done(timeout=timeout_seconds)
        except Exception as exc:
            run_error = exc
            raise
        finally:
            errors = []
            try:
                task.stop()
            except Exception as exc:
                errors.append(("pulse task stop", exc))
                logger.exception("Failed to stop NI-DAQ laser pulse task for channel %s", channel.channel_id.value)
            try:
                task.close()
            except Exception as exc:
                errors.append(("pulse task close", exc))
                logger.exception("Failed to close NI-DAQ laser pulse task for channel %s", channel.channel_id.value)
            try:
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
            except Exception as exc:
                errors.append(("command reset", exc))
                logger.exception("Failed to reset NI-DAQ laser command for channel %s", channel.channel_id.value)
            if pulse_train.close_shutter:
                try:
                    self.set_shutter_open(channel.channel_id, False)
                except Exception as exc:
                    errors.append(("shutter close", exc))
                    logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
            if errors:
                locations = ", ".join(location for location, _ in errors)
                cleanup_error = RuntimeError(
                    f"Failed to clean up NI-DAQ laser pulse train for channel {channel.channel_id.value}: {locations}"
                )
                if run_error is None:
                    raise cleanup_error from errors[0][1]
                logger.error(
                    "Failed to clean up NI-DAQ laser pulse train for channel %s after output error: %s",
                    channel.channel_id.value,
                    locations,
                )

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
        diode_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_ai")
        diode_input.ai_channels.add_ai_voltage_chan(channel.diode_input)

        command_copy_input = None
        if channel.command_copy_input:
            command_copy_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_command_copy_ai")
            command_copy_input.ai_channels.add_ai_voltage_chan(channel.command_copy_input)

        shutter_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_shutter")
        shutter_output.do_channels.add_do_chan(channel.shutter_output)

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

    def _write_transient_analog_sample(self, channel: LaserChannelConfiguration, volts: float) -> None:
        task = self._create_analog_output_task(channel, f"laser_{channel.channel_id.value}_manual_ao")
        try:
            task.write(volts, auto_start=True)
        finally:
            task.close()

    def _read_optional_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> Optional[float]:
        channel = self._configuration.get_channel(channel_id)
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


def _load_nidaqmx():
    try:
        import nidaqmx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NidaqLaserController. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    return nidaqmx
