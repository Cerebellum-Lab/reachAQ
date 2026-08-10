import math
import queue
import time
import uuid
from functools import partial, partialmethod
from pathlib import Path
from typing import Optional

from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core import (ObservableObject, SystemMessageHandler, SystemCommandKind, MessageHandler, Motor,
                              EventManager, Offset3DTuple, MotorConfigurations, SystemStatusMessageKind)
from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import (CanDevice, MotorConfigurationFile, DeviceConnection, CompoundMovements, Target)

from tools.pellet_delivery.model.user_settings import UserSettings

logger = get_verbose_logger(__name__)

# TODO: This is just to see if the behavior is correct.  They should end up somewhere that any application or script can
#  access.
_alogus_travel_limits = {
    "x": (0, 35),
    "y": (0, 35),
    "z": (0, 35),
}


class AppModel(ObservableObject):
    def __init__(
        self,
        *,
        diamond_triangle_config_path: Path = DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH,
    ):
        super().__init__()

        self._user_settings = UserSettings()

        self._hardware_configuration = None

        self._device_connection: Optional[DeviceConnection] = None

        msg_handler = self._message_handler = SystemMessageHandler(queue.Queue())
        msg_handler.start()
        #
        msg_handler.property_changed += self._message_handler_property_changed
        msg_handler.ack_received += self.reader_ack_received

        self._is_connected = False

        self._firmware_version = ""

        # The values the device is reporting - not a requested value.
        self._x = None
        self._y = None
        self._z = None
        self._send_x = None
        self._send_y = None
        self._send_z = None
        self._load_arm = None
        self._cover_arm = None
        self._stimuli = None
        self._color_led = None
        self._config = None

        self._command_pending = False
        self._last_command = None

        self._travel_limits = None  # _alogus_travel_limits

        self._diamond_triangle_config = DiamondTriangleOffsetConfig.load_config(
            diamond_triangle_config_path)

    @property
    def user_settings(self) -> UserSettings:
        return self._user_settings

    @property
    def hardware_configuration(self):
        return self._hardware_configuration

    @hardware_configuration.setter
    def hardware_configuration(self, value):
        self._hardware_configuration = self._on_property_changed("hardware_configuration", value,
                                                                 self._hardware_configuration)

    def to_diamond_coordinates(self, motor_xyz: Offset3DTuple) -> Offset3DTuple:
        diam_triangle_cfg = self._diamond_triangle_config
        if diam_triangle_cfg is None:
            return Offset3DTuple(math.nan, math.nan, math.nan)
        return diam_triangle_cfg.motor_to_diamond(motor_xyz)

    def to_motor_coordinates(self, diamond_xyz: Offset3DTuple) -> Offset3DTuple:
        diam_triangle_cfg = self._diamond_triangle_config
        if diam_triangle_cfg is None:
            return Offset3DTuple(math.nan, math.nan, math.nan)
        return diam_triangle_cfg.diamond_to_motor(diamond_xyz)

    @property
    def is_connected(self):
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value):
        prev, self._is_connected = self._is_connected, value  # is important to set before sending the event:
        self._on_property_changed("is_connected", value, prev)

    @property
    def firmware_version(self) -> str:
        return self._firmware_version

    @firmware_version.setter
    def firmware_version(self, value):
        if value is None or value.find("Pellet") == -1:
            # Might not be pellet module response.
            return

        self._firmware_version = self._on_property_changed("firmware_version", value,
                                                           self._firmware_version)

    @property
    def xyz(self) -> Offset3DTuple:
        x, y, z = map(lambda v: math.nan if v is None else v, (self._x, self._y, self._z))
        return Offset3DTuple(x, y, z)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        prev, self._x = self._x, value
        self._on_property_changed("x", value, prev)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        prev, self._y = self._y, value
        self._on_property_changed("y", value, prev)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        prev, self._z = self._z, value
        self._on_property_changed("z", value, prev)

    #

    @property
    def send_xyz(self) -> Offset3DTuple:
        x, y, z = map(lambda v: math.nan if v is None else v, (self._send_x, self._send_y, self._send_z))
        return Offset3DTuple(x, y, z)

    @property
    def send_x(self):
        return self._send_x

    @send_x.setter
    def send_x(self, value):
        prev, self._send_x = self._send_x, value
        self._on_property_changed("send_x", value, prev)

    @property
    def send_y(self):
        return self._send_y

    @send_y.setter
    def send_y(self, value):
        prev, self._send_y = self._send_y, value
        self._on_property_changed("send_y", value, prev)

    @property
    def send_z(self):
        return self._send_z

    @send_z.setter
    def send_z(self, value):
        prev, self._send_z = self._send_z, value
        self._on_property_changed("send_z", value, prev)

    @property
    def load_arm(self):
        return self._load_arm

    @load_arm.setter
    def load_arm(self, value):
        self._load_arm = self._on_property_changed("load_arm", value, self._load_arm)

    @property
    def cover_arm(self):
        return self._cover_arm

    @cover_arm.setter
    def cover_arm(self, value):
        self._cover_arm = self._on_property_changed("cover_arm", value, self._cover_arm)

    @property
    def travel_limits(self):
        return self._travel_limits

    @travel_limits.setter
    def travel_limits(self, value):
        prev, self._travel_limits = self._travel_limits, value
        self._on_property_changed("travel_limits", value, prev)

    @property
    def command_pending(self):
        return self._command_pending

    @command_pending.setter
    def command_pending(self, value):
        self._command_pending = self._on_property_changed("command_pending", value,
                                                          self._command_pending)

    @property
    def stimuli(self):
        return self._stimuli

    @stimuli.setter
    def stimuli(self, value):
        prev, self._stimuli = self._stimuli, value
        self._on_property_changed(MessageHandler.STIMULI_PROPERTY, value, prev)

    @property
    def color_led(self):
        return self._color_led

    @color_led.setter
    def color_led(self, value):
        prev, self._color_led = self._color_led, value
        self._on_property_changed(MessageHandler.COLOR_LED, value, prev)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = self._on_property_changed("config", value, self._config)

    def send_home(self):
        self._send_command(SystemCommandKind.SEND_HOME, context=uuid.uuid4())

    def load_pellet(self):
        self._send_command(SystemCommandKind.LOAD_PELLET, context=uuid.uuid4())

    def send_pellet(self):
        self._send_command(SystemCommandKind.SEND_PELLET, context=uuid.uuid4())

    def move_retract(self, *, force: bool=False):
        self._send_command(SystemCommandKind.SEND_RETRACT, context=uuid.uuid4(), force=force)

    def release_pellet(self):
        self._send_command(SystemCommandKind.RELEASE_PELLET, context=uuid.uuid4())

    def cover_pellet(self):
        self._send_command(SystemCommandKind.COVER_PELLET, context=uuid.uuid4())

    def set_color_led(self, r, g, b):
        """0 -> 100"""
        self._send_command(SystemCommandKind.SET_RGB_LED, (r, g, b),
                           context=uuid.uuid4())

    def _exec_xyz(self, value, *, system_cmd):
        return self._send_command(system_cmd, value, context=uuid.uuid4())

    def _get_send_xyz(self):
        return self.send_xyz

    set_x = partialmethod(_exec_xyz, system_cmd=SystemCommandKind.SET_X)
    set_y = partialmethod(_exec_xyz, system_cmd=SystemCommandKind.SET_Y)
    set_z = partialmethod(_exec_xyz, system_cmd=SystemCommandKind.SET_Z)

    def _get_xyz(self):
        return self.xyz

    move_x = partialmethod(_exec_xyz, system_cmd=SystemCommandKind.MOVE_X)
    move_y = partialmethod(_exec_xyz, system_cmd=SystemCommandKind.MOVE_Y)
    move_z = partialmethod(_exec_xyz, system_cmd=SystemCommandKind.MOVE_Z)

    def set_config(self, config):
        self._send_command(SystemCommandKind.WRITE_MOTOR_CONFIGURATION, config)

    def get_config(self, motor: Motor):
        self._send_command(SystemCommandKind.READ_MOTOR_CONFIGURATION, motor)

    def load_move_file(self, filename: str):
        if self._device_connection is not None:
            movements = CompoundMovements.from_file(filename)
            self._device_connection.use_compound_movements(movements)

    def connect_to_device(self):
        self._device_connection = DeviceConnection(
            CanDevice(required_targets=(Target.PELLET_DEVICE,)),
            self._message_handler.input_queue,
            name="pellet-can",
        )
        self._device_connection.request_connect()
        self._send_command(SystemCommandKind.REQUEST_VERSION)
        #
        if self._hardware_configuration is None:
            for attempt in (
                    MotorConfigurationFile.DEFAULT_LOCATION.expanduser(),
                    Path.home().joinpath(".alogus_config.yaml"),
                    Path.home().joinpath("alogus_config.yaml"),
            ):
                if attempt.exists():
                    logger.notice("Will load motor config %s", attempt)
                    self.hardware_configuration = attempt.as_posix()
                    break
            else:
                logger.warning("No motor config file found, motors are possibly unconfigured ; this might be critical")

        if self._hardware_configuration is None:
            motors_cfg = MotorConfigurationFile()
        else:
            logger.info("Reading motor config file %s", self._hardware_configuration)
            try:
                motors_cfg = MotorConfigurationFile.from_file(self._hardware_configuration)
            except Exception as err:
                logger.error(
                    "failed to read motor configuration file %s: %s", self._hardware_configuration,
                    err)
                self.hardware_configuration = None
                raise  # do not take any risk

        #
        self._diamond_triangle_config = DiamondTriangleOffsetConfig.load_config(
            DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH)

        # only set it after having loaded motor config
        self.travel_limits = _alogus_travel_limits

        self._device_connection.use_motor_configurations(motors_cfg)
        self._device_connection.load_default_move_config()

        self.is_connected = True

    def disconnect_from_device(self):
        if self._is_connected:
            # End DeviceConnection for this connection.  Do not kill the message handler which is connection agnostic.
            if self._device_connection is not None:
                self._device_connection.request_disconnect()
                self._device_connection = None

        # to have new events emitted on next connect, we better set these to None,
        # so that event listener(s) will get an on_property_changed callback triggered
        self.config = None
        self._x = self._y = self._z = None
        self._send_x = self._send_y = self._send_z = None
        self.travel_limits = None
        self.firmware_version = ""
        self.is_connected = False  # last

    def on_activated(self):
        pass

    def on_close(self):
        self.disconnect_from_device()
        if self._message_handler is not None:
            self._message_handler.request_terminate()
            self._message_handler.wait_terminated()

    def _message_handler_property_changed(self, name: str, value, _old_value):
        if name == SystemMessageHandler.FIRMWARE_VERSION_PROPERTY:
            self.firmware_version = value
        elif name == MessageHandler.STEPPER_X_PROPERTY:
            self.x = value.position
            self.send_x = value.send_position
        elif name == MessageHandler.STEPPER_Y_PROPERTY:
            self.y = value.position
            self.send_y = value.send_position
        elif name == MessageHandler.STEPPER_Z_PROPERTY:
            self.z = value.position
            self.send_z = value.send_position
        elif name == MessageHandler.LOAD_ARM_ANGLE_PROPERTY:
            self.load_arm = value
        elif name == MessageHandler.COVER_ARM_ANGLE_PROPERTY:
            self.cover_arm = value
        elif name == MessageHandler.STIMULI_PROPERTY:
            self.stimuli = value
        elif name == MessageHandler.CONFIG_PROPERTY:
            self.config = value
        elif name == MessageHandler.COLOR_LED:
            self.color_led = value

    def reader_ack_received(self, ack, *, perf_c: Optional[float]=None):
        logger.info(f"ack context received: {ack}")
        if self._last_command is not None and ack == self._last_command:
            self._last_command = None
            self.command_pending = False

    def _send_command(self, message, data=None, *, context=None, force: bool=False):
        if self._last_command is not None and not force:
            logger.verbose("ignoring command %s while existing command is in process with context=%s",
                           message, self._last_command)
            return

        if context is not None:
            # If not planning to confirm the response token, don't block the UI.
            self.command_pending = True
            self._last_command = context

        # if context is not None:
        logger.debug("sending message %s with context: %s", message, context)
        if self._device_connection is not None:
            self._device_connection.send_message(message, data, context)

    def set_tunnel_fan_on(self):
        return self._send_command(SystemCommandKind.TUNNEL_FAN_ON, context=uuid.uuid4())

    def set_tunnel_fan_off(self):
        return self._send_command(SystemCommandKind.TUNNEL_FAN_OFF, context=uuid.uuid4())
