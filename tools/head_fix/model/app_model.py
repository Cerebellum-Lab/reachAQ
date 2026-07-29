import logging
import queue
from typing import Optional

from autotrainer.core import (ObservableObject, ProjectInterval, SystemMessageHandler,
                              SystemCommandKind, SensorAnalysis, Motor, EventManager, MessageHandler)
from autotrainer.device import DeviceConnection, CanDevice, HAVE_CAN_DEVICE

from tools.head_fix.model.user_settings import UserSettings

logger = logging.getLogger(__name__)


class AppModel(ObservableObject):
    def __init__(self):
        super().__init__()

        self._user_settings = UserSettings()

        self._device_connection = None

        self._message_handler = SystemMessageHandler(queue.Queue())
        self._message_handler.property_changed += self.message_handler_property_changed
        self._message_handler.ack_received += self.reader_ack_received

        self._analysis = self._message_handler.analysis
        self._analysis.interval = ProjectInterval.HOUR

        self._is_connected = False

        self._firmware_version = ""

        self._magnet_intensity = -1.0
        self._gate_position = 0.0
        self._config = None

    @property
    def user_settings(self) -> UserSettings:
        return self._user_settings

    @property
    def allow_can_emulation(self) -> bool:
        return self._allow_can_emulation

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def firmware_version(self) -> str:
        return self._firmware_version

    @firmware_version.setter
    def firmware_version(self, value):
        if value is None or value.find("Magnet") == -1:
            # Might not be magnet module response.
            return
        self._firmware_version = self._on_property_changed(MessageHandler.FIRMWARE_VERSION_PROPERTY,
                                                           value,
                                                           self._firmware_version)

    @property
    def magnet_intensity(self) -> float:
        return self._magnet_intensity

    @magnet_intensity.setter
    def magnet_intensity(self, value: float):
        self._magnet_intensity = self._on_property_changed(
            MessageHandler.HEAD_MAGNET_INTENSITY_PROPERTY, value,
            self._magnet_intensity)

    @property
    def gate_position(self) -> float:
        return self._gate_position

    @gate_position.setter
    def gate_position(self, value: float):
        self._gate_position = self._on_property_changed(MessageHandler.HEAD_GATE_PROPERTY, value,
                                                        self._gate_position)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = self._on_property_changed(MessageHandler.CONFIG_PROPERTY, value,
                                                 self._config)

    @property
    def message_handler(self) -> SystemMessageHandler:
        return self._message_handler

    @property
    def analysis(self) -> SensorAnalysis:
        return self._analysis

    def set_magnet_position(self, value: float):
        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.MOVE_MAGNET_SERVO, value,
                                                 context="set magnet")

    def set_gate_position(self, value: float):
        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.MOVE_GATE_SERVO, value,
                                                 context="set gate")

    def set_tone(self, freq: float, duration: float):
        """
        Args:
            freq: Hz
            duration: sec
        """

        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.PLAY_TONE, (int(freq),
                                                                               int(duration *
                                                                                   1000)),
                                                 context="tone")

    def open_tunnel_gate(self):
        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.OPEN_TUNNEL_GATE,
                                                 context="open gate")

    def close_tunnel_gate(self):
        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.CLOSE_TUNNEL_GATE,
                                                 context="close gate")

    def set_config(self, config):
        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.WRITE_MOTOR_CONFIGURATION,
                                                 config, context="set motor cfg")

    def get_config(self, motor: Motor):
        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.READ_MOTOR_CONFIGURATION, motor,
                                                 context="get motor cfg")

    def set_stream_enabled(self, enable: bool):
        if enable:
            self._enable_data_stream()
        else:
            if self._device_connection is not None:
                self._device_connection.send_message(SystemCommandKind.STREAM_STOP)

        self._user_settings.stream_enabled = enable

    def connect_to_device(self):
        # This is specific to wanting to be able to test UI changes w/the emulation interface, which is not
        # configured to generate messages as frequently as the real device.
        buffer_size = 10 if HAVE_CAN_DEVICE else 1
        device_connection = self._device_connection = DeviceConnection(CanDevice(buffer_size=buffer_size),
                                                                       self._message_handler.input_queue)

        device_connection.name = "head-fix"

        device_connection.request_connect()

        device_connection.load_default_motor_config()
        device_connection.load_default_move_config()

        device_connection.send_message(SystemCommandKind.REQUEST_VERSION)

        if self._user_settings.stream_enabled:
            self._enable_data_stream()

        self._is_connected = True

    def disconnect_from_device(self):
        if self._is_connected:
            # End DeviceConnection for this connection.  Do not kill the message handler which is connection agnostic.
            if self._device_connection is not None:
                self._device_connection.request_disconnect()
                self._device_connection = None

            self._is_connected = False

            self._firmware_version = ""

    def on_activated(self):
        self._message_handler.start()

    def on_close(self):
        self.disconnect_from_device()

        # End all threads so application exits cleanly.
        if self._device_connection is not None:
            self._device_connection.request_disconnect()
        if self._message_handler is not None:
            self._message_handler.request_terminate()

    def message_handler_property_changed(self, name: str, value, _old_value):
        if name == SystemMessageHandler.FIRMWARE_VERSION_PROPERTY:
            self.firmware_version = value
        elif name == SystemMessageHandler.HEAD_MAGNET_INTENSITY_PROPERTY:
            self.magnet_intensity = value
        elif name == SystemMessageHandler.HEAD_GATE_PROPERTY:
            self.gate_position = value
        elif name == "config":
            self.config = value

    @staticmethod
    def reader_ack_received(ack, *, perf_c: Optional[float]=None):
        logger.info(f"ack context received: {ack}")

    def _enable_data_stream(self):
        if self._device_connection is not None:
            self._device_connection.send_message(SystemCommandKind.STREAM_START)
        if self._message_handler is not None and self._message_handler.input_queue is not None:
            self._message_handler.input_queue.put((SystemCommandKind.STREAM_START, None))
