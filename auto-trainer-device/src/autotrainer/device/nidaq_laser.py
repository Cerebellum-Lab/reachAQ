from __future__ import annotations

import dataclasses
from typing import Dict, Union

from .laser import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserFeedbackSample,
    LaserSystemConfiguration,
    normalize_laser_channel_id,
)


@dataclasses.dataclass
class _NidaqLaserTasks:
    analog_output: object
    diode_input: object
    shutter_output: object
    auxiliary_output: object

    def close(self) -> None:
        for task in (
            self.shutter_output,
            self.auxiliary_output,
            self.analog_output,
            self.diode_input,
        ):
            task.close()


class NidaqLaserController:
    """NI-DAQmx-backed laser controller.

    The first implementation uses on-demand NI-DAQmx tasks. The channel model is
    intentionally compatible with hardware-timed task construction later: one
    analog output task per laser writes the command voltage and optional command
    copy together, one analog input task reads diode feedback, and each digital
    output line is controlled independently.
    """

    def __init__(self, configuration: LaserSystemConfiguration):
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
            self.close()
            raise

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    def set_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> float:
        channel = self._configuration.get_channel(channel_id)
        applied = channel.clamp_command_voltage(volts)
        tasks = self._tasks[channel.channel_id]
        values = [applied]
        if channel.command_copy_output:
            values.append(applied)
        tasks.analog_output.write(values[0] if len(values) == 1 else values, auto_start=True)
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

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        normalized = normalize_laser_channel_id(channel_id)
        return LaserFeedbackSample(
            channel_id=normalized,
            command_volts=self._command_volts[normalized],
            diode_volts=self.read_diode_voltage(normalized),
        )

    def close_all_shutters(self) -> None:
        for channel in self._configuration.channels:
            self.set_shutter_open(channel.channel_id, False)

    def close(self) -> None:
        for channel in self._configuration.channels:
            if channel.channel_id not in self._tasks:
                continue
            try:
                self.set_shutter_open(channel.channel_id, False)
                self.set_auxiliary_output(channel.channel_id, False)
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
            except Exception:
                pass
        for tasks in self._tasks.values():
            tasks.close()
        self._tasks.clear()

    def _create_channel_tasks(self, channel: LaserChannelConfiguration) -> _NidaqLaserTasks:
        analog_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_ao")
        analog_output.ao_channels.add_ao_voltage_chan(
            channel.analog_output,
            min_val=channel.minimum_command_volts,
            max_val=channel.maximum_command_volts,
        )
        if channel.command_copy_output:
            analog_output.ao_channels.add_ao_voltage_chan(
                channel.command_copy_output,
                min_val=channel.minimum_command_volts,
                max_val=channel.maximum_command_volts,
            )

        diode_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_ai")
        diode_input.ai_channels.add_ai_voltage_chan(channel.diode_input)

        shutter_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_shutter")
        shutter_output.do_channels.add_do_chan(channel.shutter_output)

        auxiliary_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_aux")
        auxiliary_output.do_channels.add_do_chan(channel.auxiliary_output)

        return _NidaqLaserTasks(
            analog_output=analog_output,
            diode_input=diode_input,
            shutter_output=shutter_output,
            auxiliary_output=auxiliary_output,
        )


def _load_nidaqmx():
    try:
        import nidaqmx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NidaqLaserController. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    return nidaqmx
