from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Protocol, Tuple, Union

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
    pmt_shutter_open_delay_ms: float = 0.0
    pmt_shutter_close_delay_ms: float = 0.0
    emit_trigger_output: bool = False
    emit_timing_trigger_output: bool = False
    trigger_output_pulse_ms: float = 1.0
    timing_trigger_output_pulse_ms: float = 1.0
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
        if self.pmt_shutter_open_delay_ms < 0:
            raise ValueError("pmt_shutter_open_delay_ms cannot be negative")
        if self.pmt_shutter_close_delay_ms < 0:
            raise ValueError("pmt_shutter_close_delay_ms cannot be negative")
        if self.trigger_output_pulse_ms <= 0:
            raise ValueError("trigger_output_pulse_ms must be positive")
        if self.timing_trigger_output_pulse_ms <= 0:
            raise ValueError("timing_trigger_output_pulse_ms must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")


@dataclasses.dataclass(frozen=True)
class LaserSynchronizedPulseTrain:
    """Finite, sample-clocked pulse train spanning one or more laser channels."""

    pulse_trains: Tuple[LaserPulseTrain, ...]
    trigger_source: Optional[str] = None
    trigger_edge: str = "rising"
    enable_pmt_shutter: bool = False
    wait: bool = True
    timeout_seconds: Optional[float] = None

    def __post_init__(self):
        pulse_trains = tuple(self.pulse_trains)
        if not pulse_trains:
            raise ValueError("pulse_trains cannot be empty")
        object.__setattr__(self, "pulse_trains", pulse_trains)
        object.__setattr__(self, "trigger_edge", self.trigger_edge.lower())
        if self.trigger_edge not in ("rising", "falling"):
            raise ValueError("trigger_edge must be 'rising' or 'falling'")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        channel_ids = [pulse_train.channel_id for pulse_train in pulse_trains]
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("synchronized laser pulse trains cannot repeat a channel_id")
        trigger_sources = {
            pulse_train.trigger_source
            for pulse_train in pulse_trains
            if pulse_train.trigger_source is not None
        }
        if len(trigger_sources) > 1:
            raise ValueError("synchronized laser pulse trains require a single shared trigger_source")
        if self.trigger_source is not None and trigger_sources and self.trigger_source not in trigger_sources:
            raise ValueError("synchronized trigger_source conflicts with a per-laser trigger_source")
        if self.trigger_source is None and trigger_sources:
            object.__setattr__(self, "trigger_source", next(iter(trigger_sources)))
        if any(not pulse_train.wait for pulse_train in pulse_trains):
            raise NotImplementedError("per-laser asynchronous output is not supported in synchronized pulse trains")


@dataclasses.dataclass(frozen=True)
class LaserCalibrationRamp:
    """Sample-clocked command ramp used to acquire diode and command-copy feedback."""

    channel_id: LaserChannelId
    start_volts: float
    stop_volts: float
    steps: int
    samples_per_step: int
    open_shutter: bool = True
    close_shutter: bool = True
    enable_pmt_shutter: bool = False
    pmt_shutter_open_delay_ms: float = 0.0
    pmt_shutter_close_delay_ms: float = 0.0
    timeout_seconds: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if self.samples_per_step <= 0:
            raise ValueError("samples_per_step must be positive")
        if self.pmt_shutter_open_delay_ms < 0:
            raise ValueError("pmt_shutter_open_delay_ms cannot be negative")
        if self.pmt_shutter_close_delay_ms < 0:
            raise ValueError("pmt_shutter_close_delay_ms cannot be negative")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")


@dataclasses.dataclass(frozen=True)
class LaserCalibrationPoint:
    channel_id: LaserChannelId
    command_volts: float
    diode_volts: float
    command_copy_volts: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))


@dataclasses.dataclass(frozen=True)
class LaserDiodePowerCurve:
    """Monotonic diode-voltage curve derived from laser calibration samples."""

    channel_id: LaserChannelId
    points: Tuple[LaserCalibrationPoint, ...]

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))
        points = tuple(self.points)
        if len(points) < 2:
            raise ValueError("laser diode power curves require at least two calibration points")
        for point in points:
            if point.channel_id != self.channel_id:
                raise ValueError("all calibration points must belong to the curve channel_id")
        points = tuple(sorted(points, key=lambda point: point.diode_volts))
        diode_values = [point.diode_volts for point in points]
        if any(b <= a for a, b in zip(diode_values, diode_values[1:])):
            raise ValueError("calibration diode_volts must be strictly increasing")
        object.__setattr__(self, "points", points)

    def command_for_diode_voltage(self, target_diode_volts: float) -> float:
        if target_diode_volts < self.points[0].diode_volts or target_diode_volts > self.points[-1].diode_volts:
            raise ValueError(
                f"target diode voltage {target_diode_volts} V is outside calibration range "
                f"{self.points[0].diode_volts}..{self.points[-1].diode_volts} V"
            )
        for low, high in zip(self.points, self.points[1:]):
            if target_diode_volts <= high.diode_volts:
                span = high.diode_volts - low.diode_volts
                fraction = (target_diode_volts - low.diode_volts) / span
                return low.command_volts + fraction * (high.command_volts - low.command_volts)
        return self.points[-1].command_volts


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

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain) -> None:
        """Run a finite pulse train across one or more laser channels."""

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp) -> Tuple[LaserCalibrationPoint, ...]:
        """Run a finite command ramp and return acquired diode feedback samples."""

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
        if not pulse_train.wait:
            raise NotImplementedError("Asynchronous laser output is not implemented for null laser control")
        if pulse_train.enable_pmt_shutter or any(train.enable_pmt_shutter for train in pulse_train.pulse_trains):
            raise NotImplementedError("PMT shutter sequencing is not implemented for null laser control")
        if any(train.emit_trigger_output or train.emit_timing_trigger_output for train in pulse_train.pulse_trains):
            raise NotImplementedError("Digital trigger output is not implemented for null laser control")
        if pulse_train.trigger_source is not None:
            raise NotImplementedError("External trigger waiting is not implemented for null laser control")
        for train in pulse_train.pulse_trains:
            self._validate_command_voltage(train.channel_id, train.amplitude_volts)
        for train in pulse_train.pulse_trains:
            channel = self._configuration.get_channel(train.channel_id)
            if train.open_shutter:
                self.set_shutter_open(channel.channel_id, True)
            self.set_command_voltage(channel.channel_id, train.amplitude_volts)
        for train in pulse_train.pulse_trains:
            channel = self._configuration.get_channel(train.channel_id)
            self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
            if train.close_shutter:
                self.set_shutter_open(channel.channel_id, False)

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp) -> Tuple[LaserCalibrationPoint, ...]:
        if ramp.enable_pmt_shutter:
            raise NotImplementedError("PMT shutter sequencing is not implemented for null laser control")
        self._validate_command_voltage(ramp.channel_id, ramp.start_volts)
        self._validate_command_voltage(ramp.channel_id, ramp.stop_volts)
        channel = self._configuration.get_channel(ramp.channel_id)
        points = []
        if ramp.open_shutter:
            self.set_shutter_open(channel.channel_id, True)
        try:
            for index in range(ramp.steps):
                fraction = index / (ramp.steps - 1)
                command_volts = ramp.start_volts + fraction * (ramp.stop_volts - ramp.start_volts)
                self.set_command_voltage(channel.channel_id, command_volts)
                points.append(
                    LaserCalibrationPoint(
                        channel_id=channel.channel_id,
                        command_volts=command_volts,
                        diode_volts=self.read_diode_voltage(channel.channel_id),
                        command_copy_volts=self._read_optional_command_copy_voltage(channel.channel_id),
                    )
                )
        finally:
            self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
            if ramp.close_shutter:
                self.set_shutter_open(channel.channel_id, False)
        return tuple(points)

    def _validate_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> None:
        channel = self._configuration.get_channel(channel_id)
        if not channel.minimum_command_volts <= volts <= channel.maximum_command_volts:
            raise ValueError(
                f"laser channel {channel.channel_id.value} command {volts} V is outside "
                f"the configured range {channel.minimum_command_volts}..{channel.maximum_command_volts} V"
            )

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
