from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar, Iterable, Optional, Tuple, Union

import yaml

from autotrainer.core import make_camelize_representer, make_decamelize_constructor
from autotrainer.core.configuration import SystemConfigurationDumper, SystemConfigurationLoader


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
    command_monitor_input: Optional[str] = None
    command_copy_output: Optional[str] = None
    trigger_source: Optional[str] = None
    trigger_output: Optional[str] = None
    timing_trigger_output: Optional[str] = None
    minimum_command_volts: float = 0.0
    maximum_command_volts: float = 5.0
    feedback_scale: float = 1.0
    command_monitor_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))
        for name in ("analog_output", "diode_input", "shutter_output", "auxiliary_output"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be provided")
        if self.maximum_command_volts <= self.minimum_command_volts:
            raise ValueError("maximum_command_volts must be greater than minimum_command_volts")
        if self.feedback_scale <= 0:
            raise ValueError("feedback_scale must be positive")
        if self.command_monitor_scale <= 0:
            raise ValueError("command_monitor_scale must be positive")

    def clamp_command_voltage(self, volts: float) -> float:
        return min(max(volts, self.minimum_command_volts), self.maximum_command_volts)


@dataclasses.dataclass(frozen=True)
class LaserSystemConfiguration:
    """Configuration for up to four laser channels."""

    VALID_BACKENDS: ClassVar[Tuple[str, ...]] = ("disabled", "null", "nidaq")

    channels: Tuple[LaserChannelConfiguration, ...] = tuple()
    hardware_timed: bool = True
    sample_rate_hz: Optional[float] = None
    backend: str = "disabled"
    pmt_shutter_output: Optional[str] = None
    trigger_listener_inputs: Tuple[str, ...] = tuple()

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "trigger_listener_inputs", tuple(self.trigger_listener_inputs))
        channel_ids = tuple(channel.channel_id for channel in self.channels)
        if len(channel_ids) > 4:
            raise ValueError("at most four laser channels are supported")
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("laser channel IDs must be unique")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError(f"laser backend must be one of: {', '.join(self.VALID_BACKENDS)}")
        if any(not value for value in self.trigger_listener_inputs):
            raise ValueError("trigger_listener_inputs cannot contain empty channel names")

    @classmethod
    def from_channels(
        cls,
        channels: Iterable[LaserChannelConfiguration],
        *,
        hardware_timed: bool = True,
        sample_rate_hz: Optional[float] = None,
        backend: str = "disabled",
        pmt_shutter_output: Optional[str] = None,
        trigger_listener_inputs: Iterable[str] = tuple(),
    ) -> "LaserSystemConfiguration":
        return cls(
            tuple(channels),
            hardware_timed=hardware_timed,
            sample_rate_hz=sample_rate_hz,
            backend=backend,
            pmt_shutter_output=pmt_shutter_output,
            trigger_listener_inputs=tuple(trigger_listener_inputs),
        )

    def get_channel(self, channel_id: Union[LaserChannelId, int]) -> LaserChannelConfiguration:
        normalized = normalize_laser_channel_id(channel_id)
        for channel in self.channels:
            if channel.channel_id == normalized:
                return channel
        raise KeyError(f"laser channel {normalized.value} is not configured")


def laser_channel_id_representer(dumper: yaml.SafeDumper, obj: LaserChannelId):
    return dumper.represent_data(int(obj))


SystemConfigurationDumper.add_representer(LaserChannelId, laser_channel_id_representer)

for _tag, _cls in (
    ("LaserChannelConfiguration", LaserChannelConfiguration),
    ("LaserSystemConfiguration", LaserSystemConfiguration),
):
    SystemConfigurationDumper.add_representer(_cls, make_camelize_representer(f"!{_tag}"))
    SystemConfigurationLoader.add_constructor(f"!{_tag}", make_decamelize_constructor(_cls))
