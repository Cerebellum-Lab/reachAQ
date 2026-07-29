from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Callable, Optional, Tuple, Union

from autotrainer.core import ObservableObject
from autotrainer.core.logging import get_verbose_logger, log_hardware_initialization
from autotrainer.device import (
    LaserControllerProtocol,
    LaserCalibrationPoint,
    LaserCalibrationRamp,
    LaserChannelId,
    LaserDiodePowerCurve,
    LaserFeedbackSample,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    LaserSystemConfiguration,
    NidaqLaserController,
    NullLaserController,
)


logger = get_verbose_logger(__name__)


@dataclasses.dataclass(frozen=True)
class LaserTraceBlock:
    channel_id: LaserChannelId
    source: str
    x_values: Tuple[float, ...] = tuple()
    command_volts: Tuple[float, ...] = tuple()
    diode_volts: Tuple[float, ...] = tuple()
    command_copy_volts: Tuple[float, ...] = tuple()
    output_name: str = ""
    output_value: Optional[float] = None
    replace: bool = False


class LaserModelEvents:
    trace_received = Callable[[LaserTraceBlock], None]


class LaserModel(ObservableObject):
    CONFIGURATION = "configuration"
    IS_CONNECTED = "is_connected"
    LAST_FEEDBACK_SAMPLE = "last_feedback_sample"
    trace_received: LaserModelEvents.trace_received

    def __init__(self, controller: Optional[LaserControllerProtocol] = None):
        super().__init__(("trace_received",))
        self._controller: Optional[LaserControllerProtocol] = None
        self._configuration = LaserSystemConfiguration()
        self._last_feedback_sample: Optional[LaserFeedbackSample] = None
        if controller is not None:
            self.set_controller(controller)

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    @property
    def is_connected(self) -> bool:
        return self._controller is not None

    @property
    def last_feedback_sample(self) -> Optional[LaserFeedbackSample]:
        return self._last_feedback_sample

    def configure_null(self, configuration: LaserSystemConfiguration) -> None:
        self.set_controller(NullLaserController(configuration))

    def configure_nidaq(
        self,
        configuration: LaserSystemConfiguration,
        *,
        feedback_reader: Optional[Callable[[str], float]] = None,
    ) -> None:
        self.set_controller(
            NidaqLaserController(
                configuration,
                feedback_reader=feedback_reader,
            )
        )

    def load_configuration(
        self,
        configuration: LaserSystemConfiguration,
        *,
        feedback_reader: Optional[Callable[[str], float]] = None,
    ) -> None:
        backend = configuration.backend
        if backend == "disabled":
            prev_config = self._configuration
            self.close()
            self._configuration = configuration
            self._on_property_changed(self.CONFIGURATION, configuration, prev_config)
            log_hardware_initialization(logger, "SKIP | laser controller | backend=disabled")
            return

        started = time.perf_counter()
        log_hardware_initialization(
            logger,
            "START | laser controller | backend=%s hardware_timed=%s channels=%s",
            backend,
            configuration.hardware_timed,
            tuple(
                (
                    channel.channel_id.value,
                    channel.analog_output,
                    channel.diode_input,
                    channel.shutter_output,
                )
                for channel in configuration.channels
            ),
        )
        try:
            if backend == "null":
                self.configure_null(configuration)
            elif backend == "nidaq":
                self.configure_nidaq(
                    configuration,
                    feedback_reader=feedback_reader,
                )
            else:
                raise ValueError(f"Unsupported laser backend: {backend}")
        except Exception as exc:
            log_hardware_initialization(
                logger,
                "FAILED | laser controller | backend=%s elapsed=%.3fs error=%s",
                backend,
                time.perf_counter() - started,
                str(exc) or exc.__class__.__name__,
                level=logging.ERROR,
            )
            raise
        log_hardware_initialization(
            logger,
            "READY | laser controller | backend=%s elapsed=%.3fs",
            backend,
            time.perf_counter() - started,
        )

    def set_configuration_offline(self, configuration: LaserSystemConfiguration) -> None:
        prev_config = self._configuration
        self.close()
        self._configuration = configuration
        self._on_property_changed(self.CONFIGURATION, configuration, prev_config)

    def save_configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    def set_controller(self, controller: LaserControllerProtocol) -> None:
        prev_config = self._configuration
        self.close()
        self._controller = controller
        self._configuration = controller.configuration
        self._on_property_changed(self.CONFIGURATION, self._configuration, prev_config)
        self._on_property_changed(self.IS_CONNECTED, True, False)

    def set_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> float:
        applied = self._require_controller().set_command_voltage(channel_id, volts)
        self.trace_received(
            LaserTraceBlock(
                channel_id=LaserChannelId(int(channel_id)),
                source="manual command",
                x_values=(0.0,),
                command_volts=(applied,),
            )
        )
        return applied

    def set_shutter_open(self, channel_id: Union[LaserChannelId, int], is_open: bool) -> None:
        self._require_controller().set_shutter_open(channel_id, is_open)
        self.trace_received(
            LaserTraceBlock(
                channel_id=LaserChannelId(int(channel_id)),
                source="output state",
                output_name="shutter_open",
                output_value=float(bool(is_open)),
            )
        )

    def set_auxiliary_output(self, channel_id: Union[LaserChannelId, int], enabled: bool) -> None:
        self._require_controller().set_auxiliary_output(channel_id, enabled)
        self.trace_received(
            LaserTraceBlock(
                channel_id=LaserChannelId(int(channel_id)),
                source="output state",
                output_name="auxiliary_enabled",
                output_value=float(bool(enabled)),
            )
        )

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        sample = self._require_controller().read_feedback_sample(channel_id)
        prev, self._last_feedback_sample = self._last_feedback_sample, sample
        self._on_property_changed(self.LAST_FEEDBACK_SAMPLE, sample, prev)
        self._emit_feedback_trace(sample)
        return sample

    def read_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        return self._require_controller().read_command_copy_voltage(channel_id)

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        self._require_controller().run_pulse_train(pulse_train)
        self.trace_received(self._make_pulse_trace(pulse_train))

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain) -> None:
        self._require_controller().run_synchronized_pulse_train(pulse_train)
        for channel_pulse in pulse_train.pulse_trains:
            self.trace_received(self._make_pulse_trace(channel_pulse))

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp) -> Tuple[LaserCalibrationPoint, ...]:
        self.trace_received(
            LaserTraceBlock(
                channel_id=ramp.channel_id,
                source="calibration",
                replace=True,
            )
        )
        points = self._require_controller().run_calibration_ramp(ramp)
        if points:
            prev, self._last_feedback_sample = self._last_feedback_sample, LaserFeedbackSample(
                channel_id=points[-1].channel_id,
                command_volts=points[-1].command_volts,
                diode_volts=points[-1].diode_volts,
                command_copy_volts=points[-1].command_copy_volts,
            )
            self._on_property_changed(self.LAST_FEEDBACK_SAMPLE, self._last_feedback_sample, prev)
            sample_rate_hz = self._configuration.sample_rate_hz or 1.0
            seconds_per_step = ramp.samples_per_step / sample_rate_hz
            self.trace_received(
                LaserTraceBlock(
                    channel_id=ramp.channel_id,
                    source="calibration",
                    x_values=tuple(index * seconds_per_step for index in range(len(points))),
                    command_volts=tuple(point.command_volts for point in points),
                    diode_volts=tuple(point.diode_volts for point in points),
                    command_copy_volts=tuple(
                        math.nan if point.command_copy_volts is None else point.command_copy_volts
                        for point in points
                    ),
                    replace=True,
                )
            )
        return points

    def make_diode_power_curve(self, points: Tuple[LaserCalibrationPoint, ...]) -> LaserDiodePowerCurve:
        if not points:
            raise ValueError("calibration points cannot be empty")
        return LaserDiodePowerCurve(points[0].channel_id, points)

    def close_all_shutters(self) -> None:
        controller = self._controller
        if controller is not None:
            controller.close_all_shutters()

    def close(self) -> None:
        controller = self._controller
        if controller is None:
            return
        was_connected = self.is_connected
        try:
            controller.close_all_shutters()
        finally:
            try:
                controller.close()
            finally:
                self._controller = None
                self._on_property_changed(self.IS_CONNECTED, False, was_connected)

    def _require_controller(self) -> LaserControllerProtocol:
        controller = self._controller
        if controller is None:
            raise RuntimeError("Laser controller is not configured")
        return controller

    def _emit_feedback_trace(self, sample: LaserFeedbackSample) -> None:
        self.trace_received(
            LaserTraceBlock(
                channel_id=sample.channel_id,
                source="feedback",
                x_values=(0.0,),
                command_volts=(sample.command_volts,),
                diode_volts=(sample.diode_volts,),
                command_copy_volts=(
                    math.nan if sample.command_copy_volts is None else sample.command_copy_volts,
                ),
            )
        )

    def _make_pulse_trace(self, pulse_train: LaserPulseTrain) -> LaserTraceBlock:
        channel = self._configuration.get_channel(pulse_train.channel_id)
        minimum = channel.minimum_command_volts
        amplitude = pulse_train.amplitude_volts
        duration_s = pulse_train.duration_ms / 1000.0
        baseline_s = pulse_train.baseline_ms / 1000.0
        post_stim_s = pulse_train.post_stim_ms / 1000.0
        period_s = (
            1.0 / pulse_train.frequency_hz
            if pulse_train.frequency_hz is not None
            else duration_s
        )
        x_values = [0.0]
        y_values = [minimum]
        current_t = 0.0

        def horizontal(to_t: float) -> None:
            nonlocal current_t
            if to_t <= current_t:
                return
            x_values.append(to_t)
            y_values.append(y_values[-1])
            current_t = to_t

        def transition(value: float) -> None:
            x_values.extend((current_t, current_t))
            y_values.extend((y_values[-1], value))

        horizontal(baseline_s)
        for pulse_index in range(pulse_train.pulse_count):
            pulse_start = baseline_s + pulse_index * period_s
            horizontal(pulse_start)
            transition(amplitude)
            horizontal(pulse_start + duration_s)
            transition(minimum)
        horizontal(current_t + post_stim_s)
        trigger_kind = "external" if pulse_train.trigger_source else "internal"
        return LaserTraceBlock(
            channel_id=pulse_train.channel_id,
            source=f"{trigger_kind} pulse",
            x_values=tuple(x_values),
            command_volts=tuple(y_values),
        )
