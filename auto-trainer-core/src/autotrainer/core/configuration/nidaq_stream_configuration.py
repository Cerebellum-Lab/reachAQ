from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, Iterable, Optional, Tuple

from autotrainer.core import make_camelize_representer, make_decamelize_constructor
from autotrainer.core.configuration import SystemConfigurationDumper, SystemConfigurationLoader


@dataclasses.dataclass(frozen=True)
class NidaqSignalChannelConfiguration:
    """One continuously acquired NI-DAQ input channel."""

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
class NidaqDeliveryCalibration:
    """Where a correct camera-latch pairing's residual sits on this rig, measured by loopback.

    The pairing's median residual is the NI-to-host fit's delivery floor, plus the
    camera's own stamp offset, minus the exposure (the camera stamps a frame at the
    end of its exposure, the NI edge marks its start). offset_seconds is the mean of
    residual + exposure over loopback-verified sessions and spread_seconds its
    standard deviation between them: the residual gate is centred on offset minus
    the session's exposure. tools/hardware/calibrate_ni_delivery.py measures both
    and records what the sessions covered; the gate reads only the two numbers.
    """

    offset_seconds: float
    spread_seconds: float
    sessions: int
    measured: str = ""
    """When the calibration ran (ISO 8601)."""
    read_chunk_samples: Tuple[int, ...] = tuple()
    frame_rates_hz: Tuple[float, ...] = tuple()
    exposures_seconds: Tuple[float, ...] = tuple()
    camera_serials: Tuple[str, ...] = tuple()
    delivery_floor_min_seconds: Optional[float] = None
    delivery_floor_max_seconds: Optional[float] = None

    def __post_init__(self):
        # Every number is checked here: alignment.json, which records the calibration,
        # is written without NaN, and a hand-edited value (YAML reads 35e-6 as text) must
        # fail when the configuration loads, not when a session is finalized.
        def number(name, value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"NI delivery calibration {name} must be a number, "
                                 f"not {value!r}") from None
            if not math.isfinite(value):
                raise ValueError(f"NI delivery calibration {name} must be finite")
            return value

        object.__setattr__(self, "offset_seconds", number("offset_seconds", self.offset_seconds))
        object.__setattr__(self, "spread_seconds", number("spread_seconds", self.spread_seconds))
        if not self.spread_seconds > 0:
            raise ValueError("NI delivery calibration spread_seconds must be positive")
        if isinstance(self.sessions, bool) or int(self.sessions) != self.sessions:
            raise ValueError("NI delivery calibration sessions must be a whole number")
        object.__setattr__(self, "sessions", int(self.sessions))
        if self.sessions < 2:
            raise ValueError("NI delivery calibration needs at least two sessions")
        object.__setattr__(self, "measured", str(self.measured))
        object.__setattr__(self, "read_chunk_samples",
                           tuple(int(value) for value in self.read_chunk_samples))
        for name in ("frame_rates_hz", "exposures_seconds"):
            object.__setattr__(self, name, tuple(number(name, value)
                                                 for value in getattr(self, name)))
        object.__setattr__(self, "camera_serials", tuple(str(value) for value in self.camera_serials))
        for name in ("delivery_floor_min_seconds", "delivery_floor_max_seconds"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, number(name, value))


@dataclasses.dataclass(frozen=True)
class NidaqSignalStreamConfiguration:
    """Continuous NI-DAQ acquisition stream and independent display selection."""

    channels: Tuple[NidaqSignalChannelConfiguration, ...] = tuple()
    is_enabled: bool = False
    sample_rate_hz: float = 10000.0
    read_chunk_size: int = 500
    rolling_window_seconds: float = 10.0
    # Plot selection is independent of acquisition and persistence. ``None``
    # selects every configured channel by default; an empty tuple intentionally
    # plots nothing. All configured channels are still read and written.
    display_channels: Optional[Tuple[str, ...]] = None
    #: How analog inputs are referenced. Left unset, DAQmx chooses per channel
    #: and the choice is not uniform: on a PXI-6221 ai0-ai7 default to
    #: differential, pairing each with ai8-ai15, while ai8 and above can only
    #: be single-ended. A configuration naming channels from both halves then
    #: reads some of them against pins it also names as channels in their own
    #: right, which was measured coupling one laser's command into three
    #: unrelated inputs. "rse" matches a BNC breakout's shared ground and is
    #: the default; "diff" and "nrse" are available for other wiring.
    analog_terminal_config: str = "rse"
    #: This rig's NI delivery calibration, or None. Without one the camera-latch
    #: pairing is refused, so no session is rated hardware_edge by camera clock.
    delivery_calibration: Optional[NidaqDeliveryCalibration] = None

    def __post_init__(self):
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        display_channels = self.display_channels
        if display_channels is None:
            display_channels = tuple(channel.name for channel in channels)
        else:
            display_channels = tuple(display_channels)
        object.__setattr__(self, "display_channels", display_channels)
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
        unknown_display_channels = sorted(set(display_channels) - set(names))
        if unknown_display_channels:
            raise ValueError(
                "NI-DAQ display channels are not present in the acquisition plan: "
                + ", ".join(unknown_display_channels)
            )

    @classmethod
    def from_channels(
        cls,
        channels: Iterable[NidaqSignalChannelConfiguration],
        *,
        is_enabled: bool = False,
        sample_rate_hz: float = 10000.0,
        read_chunk_size: int = 500,
        rolling_window_seconds: float = 10.0,
        display_channels: Optional[Iterable[str]] = None,
    ) -> "NidaqSignalStreamConfiguration":
        return cls(
            tuple(channels),
            is_enabled=is_enabled,
            sample_rate_hz=sample_rate_hz,
            read_chunk_size=read_chunk_size,
            rolling_window_seconds=rolling_window_seconds,
            display_channels=(
                None if display_channels is None else tuple(display_channels)
            ),
        )

    @property
    def analog_channels(self) -> Tuple[NidaqSignalChannelConfiguration, ...]:
        return tuple(channel for channel in self.channels if channel.kind == "analog")

    @property
    def digital_channels(self) -> Tuple[NidaqSignalChannelConfiguration, ...]:
        return tuple(channel for channel in self.channels if channel.kind == "digital")


for _tag, _cls in (
    ("NidaqSignalChannelConfiguration", NidaqSignalChannelConfiguration),
    ("NidaqDeliveryCalibration", NidaqDeliveryCalibration),
    ("NidaqSignalStreamConfiguration", NidaqSignalStreamConfiguration),
):
    SystemConfigurationDumper.add_representer(_cls, make_camelize_representer(f"!{_tag}"))
    SystemConfigurationLoader.add_constructor(f"!{_tag}", make_decamelize_constructor(_cls))
