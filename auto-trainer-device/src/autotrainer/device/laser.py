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
    command_copy_volts: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class LaserPulseTrain:
    """Finite pulse-train command for hardware-timed laser output."""

    channel_id: LaserChannelId
    amplitude_volts: float
    duration_ms: float
    baseline_ms: float = 0.0
    post_stim_ms: float = 0.0
    pulse_count: int = 1
    frequency_hz: Optional[float] = None
    trigger_source: Optional[str] = None
    trigger_edge: str = "rising"
    open_shutter: bool = True
    close_shutter: bool = True
    enable_pmt_shutter: bool = False
    wait: bool = True
    timeout_seconds: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))
        object.__setattr__(self, "trigger_edge", self.trigger_edge.lower())
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.baseline_ms < 0:
            raise ValueError("baseline_ms cannot be negative")
        if self.post_stim_ms < 0:
            raise ValueError("post_stim_ms cannot be negative")
        if self.pulse_count <= 0:
            raise ValueError("pulse_count must be positive")
        if self.frequency_hz is not None and self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive when provided")
        if self.pulse_count > 1 and self.frequency_hz is None:
            raise ValueError("frequency_hz is required when pulse_count is greater than one")
        if self.trigger_edge not in ("rising", "falling"):
            raise ValueError("trigger_edge must be 'rising' or 'falling'")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")


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

    def read_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        """Read measured AI command-copy voltage for the requested laser."""

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        """Read diode feedback and optional AI command-copy feedback with the current command voltage."""

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        """Run a finite pulse train on a configured laser channel."""

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

    def read_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        channel = self._configuration.get_channel(channel_id)
        command_copy_volts = self._read_optional_command_copy_voltage(channel.channel_id)
        if command_copy_volts is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no AI command-copy input configured"
            )
        return command_copy_volts

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        channel = self._configuration.get_channel(channel_id)
        return LaserFeedbackSample(
            channel_id=channel.channel_id,
            command_volts=self._command_volts[channel.channel_id],
            diode_volts=self.read_diode_voltage(channel.channel_id),
            command_copy_volts=self._read_optional_command_copy_voltage(channel.channel_id),
        )

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        channel = self._configuration.get_channel(pulse_train.channel_id)
        if pulse_train.enable_pmt_shutter:
            raise NotImplementedError("PMT shutter sequencing is not implemented for null laser control")
        if pulse_train.trigger_source is not None:
            raise NotImplementedError("External trigger waiting is not implemented for null laser control")
        if not channel.minimum_command_volts <= pulse_train.amplitude_volts <= channel.maximum_command_volts:
            raise ValueError(
                f"laser channel {channel.channel_id.value} amplitude {pulse_train.amplitude_volts} V is outside "
                f"the configured range {channel.minimum_command_volts}..{channel.maximum_command_volts} V"
            )
        if pulse_train.open_shutter:
            self.set_shutter_open(channel.channel_id, True)
        self.set_command_voltage(channel.channel_id, pulse_train.amplitude_volts)
        self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
        if pulse_train.close_shutter:
            self.set_shutter_open(channel.channel_id, False)

    def _read_optional_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> Optional[float]:
        channel = self._configuration.get_channel(channel_id)
        if channel.command_copy_input is None:
            return None
        return self._command_volts[channel.channel_id] * channel.command_copy_scale

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
