from __future__ import annotations

from typing import Optional, Union

from autotrainer.core import ObservableObject
from autotrainer.device import (
    LaserControllerProtocol,
    LaserChannelId,
    LaserFeedbackSample,
    LaserSystemConfiguration,
    NidaqLaserController,
    NullLaserController,
)


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
        if configuration.backend == "null":
            self.configure_null(configuration)
        elif configuration.backend == "nidaq":
            self.configure_nidaq(configuration)
        elif configuration.backend == "disabled":
            prev_config = self._configuration
            self.close()
            self._configuration = configuration
            self._on_property_changed(self.CONFIGURATION, configuration, prev_config)
        else:
            raise ValueError(f"Unsupported laser backend: {configuration.backend}")

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
