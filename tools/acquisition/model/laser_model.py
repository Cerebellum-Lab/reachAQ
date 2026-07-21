from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, Union

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


class LaserModel(ObservableObject):
    CONFIGURATION = "configuration"
    IS_CONNECTED = "is_connected"
    LAST_FEEDBACK_SAMPLE = "last_feedback_sample"

    def __init__(self, controller: Optional[LaserControllerProtocol] = None):
        super().__init__()
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

    def configure_nidaq(self, configuration: LaserSystemConfiguration) -> None:
        self.set_controller(NidaqLaserController(configuration))

    def load_configuration(self, configuration: LaserSystemConfiguration) -> None:
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
                self.configure_nidaq(configuration)
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
        return self._require_controller().set_command_voltage(channel_id, volts)

    def set_shutter_open(self, channel_id: Union[LaserChannelId, int], is_open: bool) -> None:
        self._require_controller().set_shutter_open(channel_id, is_open)

    def set_auxiliary_output(self, channel_id: Union[LaserChannelId, int], enabled: bool) -> None:
        self._require_controller().set_auxiliary_output(channel_id, enabled)

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        sample = self._require_controller().read_feedback_sample(channel_id)
        prev, self._last_feedback_sample = self._last_feedback_sample, sample
        self._on_property_changed(self.LAST_FEEDBACK_SAMPLE, sample, prev)
        return sample

    def read_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        return self._require_controller().read_command_copy_voltage(channel_id)

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        self._require_controller().run_pulse_train(pulse_train)

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain) -> None:
        self._require_controller().run_synchronized_pulse_train(pulse_train)

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp) -> Tuple[LaserCalibrationPoint, ...]:
        points = self._require_controller().run_calibration_ramp(ramp)
        if points:
            prev, self._last_feedback_sample = self._last_feedback_sample, LaserFeedbackSample(
                channel_id=points[-1].channel_id,
                command_volts=points[-1].command_volts,
                diode_volts=points[-1].diode_volts,
                command_copy_volts=points[-1].command_copy_volts,
            )
            self._on_property_changed(self.LAST_FEEDBACK_SAMPLE, self._last_feedback_sample, prev)
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
