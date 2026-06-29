from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Protocol, Union

from autotrainer.core import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    normalize_laser_channel_id,
)


@dataclasses.dataclass(frozen=True)
class LaserFeedbackSample:
    channel_id: LaserChannelId
    command_volts: float
    diode_volts: float
    command_monitor_volts: Optional[float] = None


class LaserControllerProtocol(Protocol):
    """Application-facing contract for laser control implementations."""

    @property
    def configuration(self) -> LaserSystemConfiguration:
        """Active laser system configuration."""

    def set_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> float:
        """Set laser/AOM command voltage and return the applied voltage."""

    def set_shutter_open(self, channel_id: Union[LaserChannelId, int], is_open: bool) -> None:
        """Set the laser shutter digital output."""

    def set_auxiliary_output(self, channel_id: Union[LaserChannelId, int], enabled: bool) -> None:
        """Set the second per-laser digital output."""

    def read_diode_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        """Read diode feedback voltage for the requested laser."""

    def read_command_monitor_voltage(self, channel_id: Union[LaserChannelId, int]) -> Optional[float]:
        """Read measured command-monitor voltage when the channel has a monitor input."""

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        """Read diode feedback and command-monitor feedback with the current command voltage."""

    def close_all_shutters(self) -> None:
        """Force all configured shutters closed."""

    def close(self) -> None:
        """Release controller resources."""


class NullLaserController:
    """In-memory laser controller for UI development and tests."""

    def __init__(self, configuration: LaserSystemConfiguration):
        self._configuration = configuration
        self._command_volts: Dict[LaserChannelId, float] = {
            channel.channel_id: channel.minimum_command_volts
            for channel in configuration.channels
        }
        self._shutter_open: Dict[LaserChannelId, bool] = {
            channel.channel_id: False for channel in configuration.channels
        }
        self._aux_enabled: Dict[LaserChannelId, bool] = {
            channel.channel_id: False for channel in configuration.channels
        }

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    def set_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> float:
        channel = self._configuration.get_channel(channel_id)
        applied = channel.clamp_command_voltage(volts)
        self._command_volts[channel.channel_id] = applied
        return applied

    def set_shutter_open(self, channel_id: Union[LaserChannelId, int], is_open: bool) -> None:
        channel = self._configuration.get_channel(channel_id)
        self._shutter_open[channel.channel_id] = bool(is_open)

    def set_auxiliary_output(self, channel_id: Union[LaserChannelId, int], enabled: bool) -> None:
        channel = self._configuration.get_channel(channel_id)
        self._aux_enabled[channel.channel_id] = bool(enabled)

    def read_diode_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        channel = self._configuration.get_channel(channel_id)
        return self._command_volts[channel.channel_id] * channel.feedback_scale

    def read_command_monitor_voltage(self, channel_id: Union[LaserChannelId, int]) -> Optional[float]:
        channel = self._configuration.get_channel(channel_id)
        if channel.command_monitor_input is None:
            return None
        return self._command_volts[channel.channel_id] * channel.command_monitor_scale

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        channel = self._configuration.get_channel(channel_id)
        return LaserFeedbackSample(
            channel_id=channel.channel_id,
            command_volts=self._command_volts[channel.channel_id],
            diode_volts=self.read_diode_voltage(channel.channel_id),
            command_monitor_volts=self.read_command_monitor_voltage(channel.channel_id),
        )

    def close_all_shutters(self) -> None:
        for channel_id in self._shutter_open:
            self._shutter_open[channel_id] = False

    def close(self) -> None:
        self.close_all_shutters()

    def is_shutter_open(self, channel_id: Union[LaserChannelId, int]) -> bool:
        channel = self._configuration.get_channel(channel_id)
        return self._shutter_open[channel.channel_id]

    def is_auxiliary_output_enabled(self, channel_id: Union[LaserChannelId, int]) -> bool:
        channel = self._configuration.get_channel(channel_id)
        return self._aux_enabled[channel.channel_id]
