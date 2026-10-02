from __future__ import annotations

import dataclasses
import math
import numbers
import threading
import time
from typing import Dict, Mapping, Optional, Protocol, Tuple, Union

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
    defer_start: bool = False
    timeout_seconds: Optional[float] = None
    operation_context: Mapping[str, object] = dataclasses.field(default_factory=dict)
    #: How long a deferred start waits for trigger() before the operation
    #: fails, when given; None leaves it to timeout_seconds, which also bounds
    #: the wait for the output to end. An arm that must not stay armed for
    #: long, such as one held for a button press, sets its own.
    start_wait_seconds: Optional[float] = None

    def __post_init__(self):
        pulse_trains = tuple(self.pulse_trains)
        if not pulse_trains:
            raise ValueError("pulse_trains cannot be empty")
        object.__setattr__(self, "pulse_trains", pulse_trains)
        object.__setattr__(self, "operation_context", dict(self.operation_context))
        object.__setattr__(self, "trigger_edge", self.trigger_edge.lower())
        if self.trigger_edge not in ("rising", "falling"):
            raise ValueError("trigger_edge must be 'rising' or 'falling'")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        if self.start_wait_seconds is not None:
            # A NaN passed "<= 0", and an infinite wait is no bound at all.
            if not (math.isfinite(self.start_wait_seconds) and self.start_wait_seconds > 0):
                raise ValueError(
                    "start_wait_seconds must be a finite time above 0 s when provided")
            if not self.defer_start:
                raise ValueError("start_wait_seconds applies only to a deferred start (defer_start)")
        if self.defer_start and self.trigger_source is not None:
            raise ValueError("defer_start cannot be combined with a hardware trigger")
        if self.defer_start and self.wait:
            raise ValueError("deferred start requires nonblocking operation ownership")
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


#: How long the start of each calibration step is left out of its point, by
#: default: a time, whatever the step's length. Confirmed on christielab10
#: with 5 ms steps at 100 kHz (H2b, c12189cd): laser 2's diode came within 2%
#: in 270-610 us, laser 1's in 420-430 us, the command copies at once, and
#: every point lands within 0.5% of its step. Laser 2 needs 55-61 samples
#: against the 60 dropped, so there is little margin. What settles is the
#: laser and the input path together, which H2b cannot separate; at the
#: ramp's start, with the command steady at 0 V, both inputs also decay from
#: a false level with a time constant of 93-120 us, which is the input path.
CALIBRATION_SETTLE_SECONDS = 600e-6


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
    #: How long the start of each step is left out of its point. The input
    #: converts on the edge the output updates on, and the laser driver, the
    #: diode and the input path take time to follow, so the first samples of
    #: a step still read the step before: averaged in, they pulled every point
    #: of a rising ramp low. Taken as round(settle_seconds x rate) samples at
    #: the ramp's rate (settle_sample_count).
    settle_seconds: float = CALIBRATION_SETTLE_SECONDS

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if self.samples_per_step <= 0:
            raise ValueError("samples_per_step must be positive")
        if (isinstance(self.settle_seconds, bool)
                or not isinstance(self.settle_seconds, numbers.Real)
                or not math.isfinite(self.settle_seconds)
                or self.settle_seconds < 0):
            # float() took "600e-6" silently; a bool is a number to Python.
            raise ValueError(
                f"settle_seconds must be a time of 0 s or more, not "
                f"{self.settle_seconds!r}")
        object.__setattr__(self, "settle_seconds", float(self.settle_seconds))
        if self.pmt_shutter_open_delay_ms < 0:
            raise ValueError("pmt_shutter_open_delay_ms cannot be negative")
        if self.pmt_shutter_close_delay_ms < 0:
            raise ValueError("pmt_shutter_close_delay_ms cannot be negative")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")

    def settle_sample_count(self, sample_rate_hz: float) -> int:
        """The samples at the start of each step left out of its point.

        At least one sample of each step must be left for the point.
        """
        samples = int(round(self.settle_seconds * sample_rate_hz))
        if samples >= self.samples_per_step:
            raise ValueError(
                f"a settle of {self.settle_seconds * 1e6:g} µs is {samples} samples "
                f"at {sample_rate_hz:g} Hz, which leaves none of the "
                f"{self.samples_per_step} samples of each step: shorten Settle or "
                "lengthen Samples/step")
        return samples


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
        """Close the shutters and release controller resources.

        The laser model leaves the shutters to this: it calls nothing else
        on the controller first (LaserModel.close).
        """


class LaserPulseRefused(RuntimeError):
    """A pulse train refused before anything was driven: its outputs untouched.

    The board rule, a closed controller, a configuration that cannot run
    it, or an amplitude outside its laser's range (NidaqLaserController). A
    caller that recorded the pulse's amplitude as what its output may hold
    takes that back (LaserModel).
    """


class LaserPulseCancelled(RuntimeError):
    """A waited-for pulse train cancelled while it ran, by a cancel or a close.

    Not a failure of the train (NidaqLaserController.run_synchronized_pulse_train),
    and told apart from one by its class: the laser model records a manual
    Run Pulse stopped by System Mode's Stop as cancelled. A RuntimeError, as
    the error a cancel raised before it had a class of its own.
    """


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
        self._operation_lock = threading.RLock()
        self._live_operations = {}

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
        if channel.auxiliary_output is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no auxiliary_output configured"
            )
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

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain):
        if not pulse_train.wait:
            return self._run_emulated_async_pulse_train(pulse_train)
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

    def _run_emulated_async_pulse_train(self, pulse_train):
        """Exercise software-start ownership without claiming physical timing."""
        from .nidaq_laser import NidaqLaserOperation

        if pulse_train.trigger_source is not None:
            raise RuntimeError(
                "Null laser control cannot emulate a hardware-synchronized trigger"
            )
        if not pulse_train.defer_start:
            raise RuntimeError(
                "Null asynchronous laser control supports Direct NI software start only"
            )
        resources = tuple(
            self._configuration.get_channel(item.channel_id).analog_output
            for item in pulse_train.pulse_trains
        )
        for train in pulse_train.pulse_trains:
            self._validate_command_voltage(train.channel_id, train.amplitude_volts)
        with self._operation_lock:
            conflicts = [
                item.operation_id
                for item in self._live_operations.values()
                if set(item.resources) & set(resources)
                and item.state.value not in {"completed", "failed", "cancelled"}
            ]
            if conflicts:
                raise LaserPulseRefused(
                    "Emulated laser output resource is already owned by operation(s): "
                    + ", ".join(conflicts)
                )
            operation = NidaqLaserOperation(
                resources=resources,
                context=pulse_train.operation_context,
                terminal_callback=self._release_emulated_operation,
            )
            operation._set_timing_status({
                "status": "emulated_software_start",
                "reason": "Null laser backend; no physical AO timing is claimed",
                "emulated": True,
            })
            self._live_operations[operation.operation_id] = operation

        def execute():
            terminal_error = None
            try:
                operation._mark_armed()
                timeout = (
                    pulse_train.timeout_seconds or 30.0
                    if pulse_train.start_wait_seconds is None
                    else pulse_train.start_wait_seconds
                )
                if not operation._start_requested.wait(timeout):
                    raise TimeoutError(
                        "Emulated deferred laser operation did not receive a start request"
                    )
                operation._require_not_cancelled()
                operation._mark_triggered("emulated software start accepted")
                for train in pulse_train.pulse_trains:
                    channel = self._configuration.get_channel(train.channel_id)
                    if train.open_shutter:
                        self.set_shutter_open(channel.channel_id, True)
                    self.set_command_voltage(channel.channel_id, train.amplitude_volts)
                deadline = (
                    time.perf_counter()
                    + self._emulated_pulse_duration_seconds(pulse_train)
                )
                while time.perf_counter() < deadline:
                    operation._require_not_cancelled()
                    time.sleep(min(0.01, max(0.0, deadline - time.perf_counter())))
                operation._require_not_cancelled()
            except Exception as error:
                terminal_error = error
            finally:
                for train in pulse_train.pulse_trains:
                    channel = self._configuration.get_channel(train.channel_id)
                    self.set_command_voltage(
                        channel.channel_id, channel.minimum_command_volts
                    )
                    if train.close_shutter:
                        self.set_shutter_open(channel.channel_id, False)
            if terminal_error is None:
                operation._complete()
            elif operation.state.value == "cancelled":
                operation._finish_terminal()
            else:
                operation._fail(terminal_error)

        operation._thread = threading.Thread(
            target=execute,
            name=f"NullLaser-{operation.operation_id[:8]}",
            daemon=True,
        )
        operation._thread.start()
        try:
            operation.wait_until_armed(timeout=1.0)
        except Exception:
            operation.cancel()
            raise
        return operation

    @staticmethod
    def _emulated_pulse_duration_seconds(pulse_train):
        durations = []
        for train in pulse_train.pulse_trains:
            duration = (
                train.baseline_ms
                + train.duration_ms
                + train.post_stim_ms
                + train.pmt_shutter_open_delay_ms
                + train.pmt_shutter_close_delay_ms
            ) / 1000.0
            if train.pulse_count > 1:
                duration += (train.pulse_count - 1) / train.frequency_hz
            durations.append(duration)
        return max(durations, default=0.0)

    def _release_emulated_operation(self, operation):
        with self._operation_lock:
            self._live_operations.pop(operation.operation_id, None)

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
        for operation in tuple(self._live_operations.values()):
            operation.cancel()
            try:
                operation.wait(timeout=1.0)
            except Exception:
                pass
        self.close_all_shutters()

    def is_shutter_open(self, channel_id: Union[LaserChannelId, int]) -> bool:
        channel = self._configuration.get_channel(channel_id)
        return self._shutter_open[channel.channel_id]

    def is_auxiliary_output_enabled(self, channel_id: Union[LaserChannelId, int]) -> bool:
        channel = self._configuration.get_channel(channel_id)
        if channel.auxiliary_output is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no auxiliary_output configured"
            )
        return self._aux_enabled[channel.channel_id]
