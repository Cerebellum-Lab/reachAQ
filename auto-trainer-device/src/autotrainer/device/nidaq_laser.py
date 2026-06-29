from __future__ import annotations

import dataclasses
import logging
from typing import Dict, Optional, Union

from .laser import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserFeedbackSample,
    LaserSystemConfiguration,
    normalize_laser_channel_id,
)


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _NidaqLaserTasks:
    analog_output: object
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

    The first implementation uses on-demand NI-DAQmx tasks. The channel model is
    intentionally compatible with hardware-timed task construction later: one
    analog output task per laser writes the command voltage, one analog input
    task reads diode feedback, an optional second analog input task reads a
    measured AI command copy, and each digital output line is controlled
    independently.
    """

    def __init__(self, configuration: LaserSystemConfiguration):
        if configuration.backend != "nidaq":
            raise ValueError("NidaqLaserController requires laser backend 'nidaq'")
        if configuration.hardware_timed:
            raise NotImplementedError(
                "Hardware-timed laser output is not implemented yet. Use hardware_timed=False for "
                "manual/on-demand voltage and shutter control."
            )
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
        analog_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_ao")
        analog_output.ao_channels.add_ao_voltage_chan(
            channel.analog_output,
            min_val=channel.minimum_command_volts,
            max_val=channel.maximum_command_volts,
        )
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

    def _read_optional_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> Optional[float]:
        channel = self._configuration.get_channel(channel_id)
        task = self._tasks[channel.channel_id].command_copy_input
        if task is None:
            return None
        return float(task.read()) * channel.command_copy_scale


def _load_nidaqmx():
    try:
        import nidaqmx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NidaqLaserController. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    return nidaqmx
