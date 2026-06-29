from __future__ import annotations

import dataclasses
import enum
from typing import Dict, Iterable, Optional, Protocol, Tuple, Union


class LaserChannelId(enum.IntEnum):
    LASER_1 = 1
    LASER_2 = 2
    LASER_3 = 3
    LASER_4 = 4


def normalize_laser_channel_id(value: Union[LaserChannelId, int]) -> LaserChannelId:
    return value if isinstance(value, LaserChannelId) else LaserChannelId(int(value))


@dataclasses.dataclass(frozen=True)
class LaserChannelConfiguration:
    """NI-DAQ channel assignment for one independently controlled laser."""

    channel_id: LaserChannelId
    analog_output: str
    diode_input: str
    shutter_output: str
    auxiliary_output: str
    command_copy_output: Optional[str] = None
    minimum_command_volts: float = 0.0
    maximum_command_volts: float = 5.0
    feedback_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))
        for name in ("analog_output", "diode_input", "shutter_output", "auxiliary_output"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be provided")
        if self.maximum_command_volts <= self.minimum_command_volts:
            raise ValueError("maximum_command_volts must be greater than minimum_command_volts")
        if self.feedback_scale <= 0:
            raise ValueError("feedback_scale must be positive")

    def clamp_command_voltage(self, volts: float) -> float:
        return min(max(volts, self.minimum_command_volts), self.maximum_command_volts)


@dataclasses.dataclass(frozen=True)
class LaserSystemConfiguration:
    """Configuration for up to four laser channels."""

    channels: Tuple[LaserChannelConfiguration, ...] = tuple()
    hardware_timed: bool = True
    sample_rate_hz: Optional[float] = None

    def __post_init__(self):
        channel_ids = tuple(channel.channel_id for channel in self.channels)
        if len(channel_ids) > 4:
            raise ValueError("at most four laser channels are supported")
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("laser channel IDs must be unique")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")

    @classmethod
    def from_channels(
        cls,
        channels: Iterable[LaserChannelConfiguration],
        *,
        hardware_timed: bool = True,
        sample_rate_hz: Optional[float] = None,
    ) -> "LaserSystemConfiguration":
        return cls(tuple(channels), hardware_timed=hardware_timed, sample_rate_hz=sample_rate_hz)

    def get_channel(self, channel_id: Union[LaserChannelId, int]) -> LaserChannelConfiguration:
        normalized = normalize_laser_channel_id(channel_id)
        for channel in self.channels:
            if channel.channel_id == normalized:
                return channel
        raise KeyError(f"laser channel {normalized.value} is not configured")


@dataclasses.dataclass(frozen=True)
class LaserFeedbackSample:
    channel_id: LaserChannelId
    command_volts: float
    diode_volts: float


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

    def close_all_shutters(self) -> None:
        """Force all configured shutters closed."""


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

    def close_all_shutters(self) -> None:
        for channel_id in self._shutter_open:
            self._shutter_open[channel_id] = False

    def is_shutter_open(self, channel_id: Union[LaserChannelId, int]) -> bool:
        channel = self._configuration.get_channel(channel_id)
        return self._shutter_open[channel.channel_id]

    def is_auxiliary_output_enabled(self, channel_id: Union[LaserChannelId, int]) -> bool:
        channel = self._configuration.get_channel(channel_id)
        return self._aux_enabled[channel.channel_id]
