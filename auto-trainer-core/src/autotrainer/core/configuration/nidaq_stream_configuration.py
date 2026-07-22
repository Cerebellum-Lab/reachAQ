from __future__ import annotations

import dataclasses
from typing import ClassVar, Iterable, Optional, Tuple

from autotrainer.core import make_camelize_representer, make_decamelize_constructor
from autotrainer.core.configuration import SystemConfigurationDumper, SystemConfigurationLoader


@dataclasses.dataclass(frozen=True)
class NidaqSignalChannelConfiguration:
    """One visualization-only NI-DAQ input channel."""

    name: str
    physical_channel: str
    kind: str = "analog"
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    VALID_KINDS: ClassVar[Tuple[str, ...]] = ("analog", "digital")

    def __post_init__(self):
        object.__setattr__(self, "kind", self.kind.lower())
        if not self.name:
            raise ValueError("NI-DAQ stream channel name must be provided")
        if not self.physical_channel:
            raise ValueError("NI-DAQ stream physical_channel must be provided")
        if self.kind not in self.VALID_KINDS:
            raise ValueError(f"NI-DAQ stream channel kind must be one of: {', '.join(self.VALID_KINDS)}")
        if self.scale == 0:
            raise ValueError("NI-DAQ stream channel scale cannot be zero")
        if self.minimum is not None and self.maximum is not None and self.maximum <= self.minimum:
            raise ValueError("NI-DAQ stream channel maximum must be greater than minimum")
        if not self.unit:
            object.__setattr__(self, "unit", "logic" if self.kind == "digital" else "V")


@dataclasses.dataclass(frozen=True)
class NidaqSignalStreamConfiguration:
    """Continuous NI-DAQ input stream shown in the acquisition Analysis panel."""

    channels: Tuple[NidaqSignalChannelConfiguration, ...] = tuple()
    is_enabled: bool = False
    sample_rate_hz: float = 10000.0
    read_chunk_size: int = 500
    rolling_window_seconds: float = 10.0
    # Both fields are retained only so older YAML configurations continue to
    # load. The analysis stream is visualization-only and they are ignored.
    record_to_acquisition: bool = False
    output_name: str = "nidaq_signals"

    def __post_init__(self):
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        if self.sample_rate_hz <= 0:
            raise ValueError("NI-DAQ stream sample_rate_hz must be positive")
        if self.read_chunk_size <= 0:
            raise ValueError("NI-DAQ stream read_chunk_size must be positive")
        if self.rolling_window_seconds <= 0:
            raise ValueError("NI-DAQ stream rolling_window_seconds must be positive")
        if self.is_enabled and not channels:
            raise ValueError("NI-DAQ signal stream is enabled but no channels are configured")
        names = tuple(channel.name for channel in channels)
        if len(set(names)) != len(names):
            raise ValueError("NI-DAQ stream channel names must be unique")
        if not self.output_name:
            raise ValueError("NI-DAQ stream output_name must be provided")

    @classmethod
    def from_channels(
        cls,
        channels: Iterable[NidaqSignalChannelConfiguration],
        *,
        is_enabled: bool = False,
        sample_rate_hz: float = 10000.0,
        read_chunk_size: int = 500,
        rolling_window_seconds: float = 10.0,
        record_to_acquisition: bool = False,
        output_name: str = "nidaq_signals",
    ) -> "NidaqSignalStreamConfiguration":
        return cls(
            tuple(channels),
            is_enabled=is_enabled,
            sample_rate_hz=sample_rate_hz,
            read_chunk_size=read_chunk_size,
            rolling_window_seconds=rolling_window_seconds,
            record_to_acquisition=record_to_acquisition,
            output_name=output_name,
        )

    @property
    def analog_channels(self) -> Tuple[NidaqSignalChannelConfiguration, ...]:
        return tuple(channel for channel in self.channels if channel.kind == "analog")

    @property
    def digital_channels(self) -> Tuple[NidaqSignalChannelConfiguration, ...]:
        return tuple(channel for channel in self.channels if channel.kind == "digital")


for _tag, _cls in (
    ("NidaqSignalChannelConfiguration", NidaqSignalChannelConfiguration),
    ("NidaqSignalStreamConfiguration", NidaqSignalStreamConfiguration),
):
    SystemConfigurationDumper.add_representer(_cls, make_camelize_representer(f"!{_tag}"))
    SystemConfigurationLoader.add_constructor(f"!{_tag}", make_decamelize_constructor(_cls))
