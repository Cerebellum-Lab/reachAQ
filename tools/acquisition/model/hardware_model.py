import math
import re
import threading
import time
import uuid
from functools import partial
from queue import Queue
from uuid import UUID, uuid4
from typing import Optional, Tuple, Dict, Union, List

from autotrainer.api import ApiEventKind, ApiDetectorKind
from autotrainer.core import (ObservableObject, SystemCommandKind, MessageHandler, AnimalSubject, Offset3DTuple,
                              get_verbose_logger, Motor, SensorAnalysis, EventManager, HardwareConfiguration,
                              get_perf_now, SystemStatusMessageKind)
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core.event import post_api_detector_event_content
from autotrainer.core.message import SystemDataArgsKwargs
from autotrainer.device import (DeviceConnectionProtocol, HAVE_CAN_DEVICE, DeviceConnection, CanDevice,
                                StepperConfig, ServoConfig, Device, ColorLed, Target)
from autotrainer.behavior import TunnelDeviceProtocol, PelletDeviceProtocol

logger = get_verbose_logger(__name__)

_nans_offset3dTuple = Offset3DTuple.get_nan()


_reg_pellet_version_clean = re.compile("pellet ?:? *")
_reg_tunnel_version_clean = re.compile("tunnel ?:? *")
_reg_magnet_version_clean = re.compile("magnet ?:? *")


class HardwareModel(ObservableObject, TunnelDeviceProtocol, PelletDeviceProtocol):

    TUNNEL_VERSION_PROPERTY = "tunnel_version"
    PELLET_VERSION_PROPERTY = "pellet_version"

    TUNNEL_IDENTIFIER_PROPERTY = "tunnel_identifier"
    PELLET_IDENTIFIER_PROPERTY = "pellet_identifier"

    PENDING_COMMAND_PROPERTY = "pending_command"

    FRONT_DOOR_PROPERTY = "front_door"
    SLIDE_DOOR_PROPERTY = "slide_door"

    DEVICE_ACK_TIMEOUT_ENGAGED = "device_ack_timeout_engaged"
    DEVICE_PELLET_STATUS_TIMEOUT_ENGAGED = "device_pellet_status_timeout_engaged"
    DEVICE_TUNNEL_STATUS_TIMEOUT_ENGAGED = "device_tunnel_status_timeout_engaged"

    # POS_X = "pos_x"
    # POS_Y = "pos_y"
    # POS_Z = "pos_z"
    POS_XYZ = "pos_xyz"

    SEND_X = "send_x"
    SEND_Y = "send_y"
    SEND_Z = "send_z"
    SEND_XYZ = "send_xyz"

    SET_X = "set_x"
    SET_Y = "set_y"
    SET_Z = "set_z"

    HEAD_MAGNET_INTENSITY = "head_magnet_intensity"
    TUNNEL_GATE_POSITION = "tunnel_gate_position"
    TUNNEL_GATE_OPEN_STATUS = "tunnel_gate_open_status"

    def __init__(
        self,
        message_handler: MessageHandler,
        sensor_analysis: SensorAnalysis,
    ):
        super().__init__()

        self._lock = threading.RLock()  # **required** re-entrant lock !!
        self._connect_count = 0

        self._event_manager = EventManager.default()
        self._board_status_timeout: Optional[float] = None
        self._device_ack_timeout_delay: Optional[float] = None
        self._device_conn: Optional[DeviceConnectionProtocol] = None
        self._can_device: Optional[CanDevice] = None
        self._sensor_analysis = sensor_analysis
        self._device_uuid_ack_timeout_engaged = False
        self._device_pellet_status_timeout_engaged = False
        self._device_tunnel_status_timeout_engaged = False
        self._device_stream_started = False
        self._tunnel_headfix_enabled = False

        self._pending_tokens: Dict[UUID, Tuple[SystemCommandKind, float]] = {}

        message_handler.property_changed += self._message_handler_property_changed
        message_handler.ack_received += self._ack_received

        self._head_magnet_position: Optional[float] = None
        self._tunnel_gate_position: float = math.nan
        self._tunnel_gate_open_status: bool = False

        self._dcs_config: Optional[DiamondTriangleOffsetConfig] = None
        # Support for relative x, y, z movements and whether they are persistent as the Send position various between
        # hardware implementations. One the Alogus hardware is used exclusively, it should be possible to remove these
        # and rely on SET_X/Y/Z commands with the extra arguments that support relative and/or movements that should
        # not affect the Send position.
        self._last_motor_coordinates = _nans_offset3dTuple
        # what the motors report they've been SET (with possible drift corrected):
        self._last_motor_send_coordinates = _nans_offset3dTuple
        # What we've SET as coordinates:
        self._last_requested_set_coordinates: Offset3DTuple = _nans_offset3dTuple

        self._front_door_open: bool = False
        self._slide_door_open: bool = False

        self._cover_arm_position: float = math.nan
        self._load_arm_position: float = math.nan

        self._color_led: Optional[ColorLed] = None

        self._device_ack_timeout_engaged = False
        self._disconnect_event = threading.Event()
        self._check_timedout_commands_thread: Optional[threading.Thread] = None

    @property
    def watchdog_reader_perf_c(self):
        dev = self._device_conn
        if dev is None:
            return math.nan
        return dev.watchdog_reader_perf_c

    @property
    def watchdog_writer_perf_c(self) -> float:
        dev = self._device_conn
        if dev is None:
            return math.nan
        return dev.watchdog_writer_perf_c

    def _check_timedout_commands_handler(self):
        while True:
            if self._disconnect_event.wait(1):
                break
            dev = self._device_conn
            if dev is None or not dev.device.connected:
                logger.verbose("device not connected, exiting check loop")
                break
            perf_now = get_perf_now()
            expired_tokens = set()
            with self._lock:
                for pending_token, (
                    pending_cmd,
                    pending_t_perf,
                ) in self._pending_tokens.items():
                    age_second = perf_now - pending_t_perf
                    if age_second > 30:
                        logger.warning(
                            "Giving up on pending cmd %s for too long ; token=%s age=%s seconds",
                            pending_cmd,
                            pending_token,
                            age_second,
                        )
                        expired_tokens.add(pending_token)
                for expired in expired_tokens:
                    self._pending_tokens.pop(expired, None)
                after_commands = list(self._pending_tokens.values())
            if len(expired_tokens) > 0:
                self._refresh_cmd_in_progress(after_commands)

    @property
    def pending_tokens(self) -> List[str]:
        return list(self._pending_tokens)

    def _check_dcs_cfg(self, *, return_none: bool=False):
        cfg = self._dcs_config
        if cfg is None or not cfg.fully_valid:
            if return_none:
                return None
            raise RuntimeError(f"DCS config not defined or not fully valid: {None if cfg is None else cfg.__dict__}")
        return cfg

    def set_diamond_triangle_config(self, config: Optional[DiamondTriangleOffsetConfig]):
        self._dcs_config = config

    @property
    def load_arm_position(self) -> float:
        return self._load_arm_position

    @property
    def cover_arm_position(self) -> float:
        return self._cover_arm_position

    @property
    def last_position(self) -> Optional[Offset3DTuple]:
        return self._last_motor_coordinates

    @property
    def last_dcs_position(self) -> Optional[Offset3DTuple]:
        cfg = self._check_dcs_cfg(return_none=True)
        if cfg is None:
            return None
        return cfg.motor_to_diamond(self._last_motor_coordinates)

    @property
    def last_set_position(self) -> Optional[Offset3DTuple]:
        value = self._last_requested_set_coordinates
        if any(map(math.isnan, value)):
            return None
        return value

    @property
    def last_dcs_set_position(self) -> Optional[Offset3DTuple]:
        value = self._last_requested_set_coordinates
        if any(map(math.isnan, value)):
            return None
        cfg = self._check_dcs_cfg(return_none=True)
        if cfg is None:
            return None
        return cfg.motor_to_diamond(value)

    @property
    def device_ack_timeout_engaged(self):
        return self._device_ack_timeout_engaged

    @device_ack_timeout_engaged.setter
    def device_ack_timeout_engaged(self, value):
        prev, self._device_ack_timeout_engaged = self._device_ack_timeout_engaged, value
        if prev == value:
            return
        self.property_changed(self.DEVICE_ACK_TIMEOUT_ENGAGED, value, prev)
        enabled = False
        dev_conn = self._device_conn
        if dev_conn is not None:
            dev = dev_conn.device
            if dev is not None:
                iface = dev.device_interface
                enabled = iface is not None and iface.is_open
        # could be in app_model or system_machine, in react property changed, but ok here too:
        post_api_detector_event_content(self._event_manager, ApiDetectorKind.deviceAckTimeOut, value, enabled)

    @property
    def send_x(self):
        return self._last_motor_send_coordinates.x

    @send_x.setter
    def send_x(self, value):
        prev, self._last_motor_send_coordinates = self._last_motor_send_coordinates, self._last_motor_send_coordinates.replace(x=value)
        self._on_property_changed(self.SEND_X, value, prev.x)

    @property
    def send_y(self):
        return self._last_motor_send_coordinates.y

    @send_y.setter
    def send_y(self, value):
        prev, self._last_motor_send_coordinates = self._last_motor_send_coordinates, self._last_motor_send_coordinates.replace(y=value)
        self._on_property_changed(self.SEND_Y, value, prev.y)

    @property
    def send_z(self):
        return self._last_motor_send_coordinates.z

    @send_z.setter
    def send_z(self, value):
        prev, self._last_motor_send_coordinates = self._last_motor_send_coordinates, self._last_motor_send_coordinates.replace(z=value)
        self._on_property_changed(self.SEND_Z, value, prev.z)

    @property
    def motor_send_coordinates(self) -> Offset3DTuple:
        return self._last_motor_send_coordinates

    @property
    def color_led(self) -> Optional[ColorLed]:
        return self._color_led

    @property
    def front_door_open(self):
        return self._front_door_open

    @front_door_open.setter
    def front_door_open(self, value: bool):
        prev, self._front_door_open = self._front_door_open, value
        self._on_property_changed(HardwareModel.FRONT_DOOR_PROPERTY, value, prev)

    @property
    def slide_door_open(self):
        return self._slide_door_open

    @slide_door_open.setter
    def slide_door_open(self, value: bool):
        prev, self._slide_door_open = self._slide_door_open, value
        self._on_property_changed(HardwareModel.SLIDE_DOOR_PROPERTY, value, prev)

    @property
    def head_magnet_intensity(self) -> Optional[float]:
        """
        This value is stored because there are other operations that depend on the current magnet intensity.  In most
        other cases, the only action is to update the UI with the current value.
        :return intensity in percent:
        """
        return self._head_magnet_position

    def update_head_magnet_intensity(self, value: Optional[float]) -> Optional[UUID]:
        if not self._tunnel_headfix_enabled:
            logger.debug("Skipping head magnet command because tunnel/headfix hardware is disabled")
            return None
        if value is None:  # caller should not call instead eventually
            return
        if isinstance(value, str):
            value = float(value)
        if value != self._head_magnet_position:
            logger.verbose("sending move magnet to %.3f", value)
            # self._head_magnet_position = value  # this is set from reading the hardware status
            return self._send_with_token(self._device_conn, SystemCommandKind.MOVE_MAGNET_SERVO,
                                         value)
        logger.debug("head magnet currently already at pos %.3f", value)
        return None

    @property
    def tunnel_gate_open_status(self) -> bool:
        return self._tunnel_gate_open_status

    def open_tunnel_gate(self) -> Optional[UUID]:
        if not self._tunnel_headfix_enabled:
            logger.debug("Skipping open tunnel gate command because tunnel/headfix hardware is disabled")
            return None
        return self._send_with_token(self._device_conn, SystemCommandKind.OPEN_TUNNEL_GATE)

    def close_tunnel_gate(self) -> Optional[UUID]:
        if not self._tunnel_headfix_enabled:
            logger.debug("Skipping close tunnel gate command because tunnel/headfix hardware is disabled")
            return None
        return self._send_with_token(self._device_conn, SystemCommandKind.CLOSE_TUNNEL_GATE)

    def tare_load_cell(self) -> Optional[UUID]:
        if not self._tunnel_headfix_enabled:
            logger.debug("Skipping load-cell tare command because tunnel/headfix hardware is disabled")
            return None
        return self._send_with_token(self._device_conn, SystemCommandKind.UPDATE_SCALE_TARE)

    def _set_axis(self, value: float, *, absolute: bool = True,
                  system_set_cmd: SystemCommandKind, coord_idx: int, sender: str="NA") -> Optional[UUID]:
        coord = "xyz"[coord_idx]
        logger.verbose("Sender=%s : SET_%s value=%.1f absolute=%s", sender, coord.upper(), value, absolute)
        prev_value = self._last_requested_set_coordinates[coord_idx]
        if absolute:
            new_value = value
        else:
            if math.isnan(prev_value):
                err_msg = (
                    "Cannot SET axis with relative value"
                    " when SET not already called/initialized with absolute value"
                )
                raise ValueError(err_msg)
            new_value = prev_value + value
        if new_value < 0:
            value = 0 if absolute else -prev_value
            new_value = 0
            logger.verbose("Axis-%s: limited move to 0 ; value=%.3f absolute=%s",
                         coord.upper(), value, absolute)
        res = self._send_with_token(self._device_conn, system_set_cmd,
                                    SystemDataArgsKwargs(value, relative=not absolute))
        if res is not None:
            self._last_requested_set_coordinates = self._last_requested_set_coordinates.replace(**{coord: new_value})
            self._on_property_changed(f"set_{coord}", new_value, prev_value)
        return res

    def set_x(self, value: float, *, absolute: bool = True, sender: str="NA") -> Optional[UUID]:
        return self._set_axis(value, absolute=absolute, system_set_cmd=SystemCommandKind.SET_X, coord_idx=0, sender=sender)

    def set_y(self, value: float, *, absolute: bool = True, sender: str="NA") -> Optional[UUID]:
        return self._set_axis(value, absolute=absolute, system_set_cmd=SystemCommandKind.SET_Y, coord_idx=1, sender=sender)

    def set_z(self, value: float, *, absolute: bool = True, sender: str="NA") -> Optional[UUID]:
        return self._set_axis(value, absolute=absolute, system_set_cmd=SystemCommandKind.SET_Z, coord_idx=2, sender=sender)

    def set_dcs_x(self, value: float, *, absolute: bool = True, sender: str="NA") -> Optional[UUID]:
        cfg = self._check_dcs_cfg()
        value = cfg.diamond_to_motor(Offset3DTuple(value, 0, 0)).x
        return self._set_axis(value, absolute=absolute, system_set_cmd=SystemCommandKind.SET_X, coord_idx=0, sender=sender)

    def set_dcs_y(self, value: float, *, absolute: bool = True, sender: str="NA") -> Optional[UUID]:
        cfg = self._check_dcs_cfg()
        value = cfg.diamond_to_motor(Offset3DTuple(0, value, 0)).y
        return self._set_axis(value, absolute=absolute, system_set_cmd=SystemCommandKind.SET_Y, coord_idx=1, sender=sender)

    def set_dcs_z(self, value: float, *, absolute: bool = True, sender: str="NA") -> Optional[UUID]:
        cfg = self._check_dcs_cfg()
        value = cfg.diamond_to_motor(Offset3DTuple(0, 0, value)).z
        return self._set_axis(value, absolute=absolute, system_set_cmd=SystemCommandKind.SET_Z, coord_idx=2, sender=sender)

    #

    def _move_axis(self, value: float, *, absolute: bool = True,
                   system_move_cmd: SystemCommandKind, coord_idx: int) -> Optional[UUID]:
        if not absolute:
            prev_value = self._last_motor_coordinates[coord_idx]
            if math.isnan(prev_value):
                logger.warning("Axis-%s: relative movement requested, but no previous position is set",
                               "XYZ"[coord_idx])
                return None
            value += prev_value
        res = self._send_with_token(self._device_conn, system_move_cmd, value)
        return res

    def move_x(self, value: float, *, absolute: bool = True) -> Optional[UUID]:
        return self._move_axis(value, absolute=absolute, system_move_cmd=SystemCommandKind.MOVE_X, coord_idx=0)

    def move_y(self, value: float, *, absolute: bool = True) -> Optional[UUID]:
        return self._move_axis(value, absolute=absolute, system_move_cmd=SystemCommandKind.MOVE_Y, coord_idx=1)

    def move_z(self, value: float, *, absolute: bool = True) -> Optional[UUID]:
        return self._move_axis(value, absolute=absolute, system_move_cmd=SystemCommandKind.MOVE_Z, coord_idx=2)

    def send_to_limits(self) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.SEND_TO_LIMITS,
                                     [Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR, Motor.PELLET_X_MOTOR])

    def send_retract(self) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.SEND_RETRACT)

    def send_home(self) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.SEND_HOME)

    def load_pellet(self) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.LOAD_PELLET)

    def send_pellet(self) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.SEND_PELLET)

    def release_pellet(self) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.RELEASE_PELLET)

    def cover_pellet(self) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.COVER_PELLET)

    def attach_servo(self, servo: Motor) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.SERVO_ATTACH, servo)

    def detach_servo(self, servo: Motor) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.SERVO_DETACH, servo)

    def play_tone(self, frequency: int, duration: float) -> Optional[UUID]:
        """Play a tone
        :param frequency: in Hz (integer)
        :param duration: in seconds (float)
        """
        duration_ms = int(duration * 1000)
        return self._send_with_token(self._device_conn, SystemCommandKind.PLAY_TONE, (frequency, duration_ms))

    def delay(self, amount: float) -> Optional[UUID]:
        return self._send_with_token(self._device_conn, SystemCommandKind.DELAY, amount)

    def set_tunnel_fan_on(self) -> Optional[UUID]:
        if not self._tunnel_headfix_enabled:
            logger.debug("Skipping tunnel fan command because tunnel/headfix hardware is disabled")
            return None
        return self._send_with_token(self._device_conn, SystemCommandKind.TUNNEL_FAN_ON)

    def set_tunnel_fan_off(self) -> Optional[UUID]:
        if not self._tunnel_headfix_enabled:
            logger.debug("Skipping tunnel fan command because tunnel/headfix hardware is disabled")
            return None
        return self._send_with_token(self._device_conn, SystemCommandKind.TUNNEL_FAN_OFF)

    def set_color_led(self, r: int, g: int, b: int):
        """0 -> 100"""
        return self._send_with_token(self._device_conn, SystemCommandKind.SET_RGB_LED, (r, g, b))

    def load_config(self, config: HardwareConfiguration):
        self._tunnel_headfix_enabled = config.tunnel_headfix_enabled
        self.set_device_ack_timeout(config.min_ack_timeout)
        self.set_board_status_timeout(config.board_status_timeout)

    def set_board_status_timeout(self, timeout: Optional[float]):
        self._board_status_timeout = timeout
        can_dev = self._can_device
        if can_dev is None:
            return
        if timeout is None:
            timeout = CanDevice.default_board_status_timeout_delay
        can_dev.default_board_status_timeout_delay = timeout

    def set_device_ack_timeout(self, delay: Optional[float]):
        self._device_ack_timeout_delay = delay
        can_dev = self._can_device
        if can_dev is not None:
            if delay is None:
                delay = CanDevice.default_command_ack_timeout_duration
            else:
                delay = max(CanDevice.default_command_ack_timeout_duration, delay)
            can_dev.default_command_ack_timeout_duration = delay
            logger.notice("Using %s for default device ack timeout delay", delay)

    @property
    def connected(self) -> bool:
        dev = self._device_conn
        dev_dev = None if dev is None else dev.device
        return dev_dev is not None and dev_dev.connected

    def connect(self, cmd_queue: Queue):
        logger.notice("%s: connect with %s", self, cmd_queue)
        self._disconnect_event.clear()

        prev_device = self._device_conn
        if prev_device is not None:
            logger.warning("auto-disconnecting from device before (re-)connect")
            self.disconnect()

        self._connect_count += 1
        self._last_motor_coordinates = \
        self._last_requested_set_coordinates = \
        self._last_motor_send_coordinates = _nans_offset3dTuple

        # This is specific to wanting to be able to test UI changes w/the emulation interface, which is not
        # configured to generate messages as frequently as the real device.
        buffer_size = 10 if HAVE_CAN_DEVICE else 1
        #
        can_device = self._can_device = CanDevice(
            buffer_size=buffer_size,
            required_targets=(Target.PELLET_DEVICE,),
        )
        self.set_device_ack_timeout(self._device_ack_timeout_delay)  # ensure it's used
        self.set_board_status_timeout(self._board_status_timeout)

        can_device.property_changed += self._can_device_property_changed

        device_conn = self._device_conn = DeviceConnection(can_device, cmd_queue, name="can-device")
        device_conn.request_connect()

        send_dev_cmd = partial(self._send_command, device_conn)
        def send_dev_ack_cmd(kind, data=None):
            tok = str(uuid.uuid4())
            with device_conn.await_acknowledge({tok}):
                send_dev_cmd(kind, data, context=tok)

        send_dev_ack_cmd(SystemCommandKind.REQUEST_VERSION)

        # load and set motors and move configs
        # 1)
        motors_config = device_conn.load_default_motor_config()
        # 2)
        device_conn.load_default_move_config()
        # 3)
        if self._connect_count == 1:
            logger.notice("Doing cover attach-release-detach on first connect")
            send_dev_ack_cmd(SystemCommandKind.SERVO_ATTACH, Motor.PELLET_COVER_SERVO)
            send_dev_ack_cmd(SystemCommandKind.RELEASE_PELLET)
            send_dev_ack_cmd(SystemCommandKind.SERVO_DETACH, Motor.PELLET_COVER_SERVO)
            send_dev_ack_cmd(SystemCommandKind.WRITE_MOTOR_CONFIGURATION, motors_config.cover_config)
            # also need to re-apply the config

        send_dev_ack_cmd(SystemCommandKind.STREAM_START)
        logger.success("STREAM_START acknowledged")
        self._device_stream_started = True

        if self._tunnel_headfix_enabled:
            send_dev_cmd(SystemCommandKind.UPDATE_SCALE_TARE)
        else:
            logger.debug("Skipping startup scale tare because tunnel/headfix hardware is disabled")

        prev_thread = self._check_timedout_commands_thread
        if prev_thread is None or not prev_thread.is_alive():
            thread = self._check_timedout_commands_thread = threading.Thread(
                target=self._check_timedout_commands_handler,
                daemon=True,
                name="check-timedout-commands",
            )
            thread.start()

    def disconnect(self):
        logger.verbose("disconnecting ..")
        self._disconnect_event.set()
        can_dev = self._can_device
        dev = self._device_conn
        if dev is not None:
            dev.request_disconnect()
            dev.join()
            self._device_conn = None
        if can_dev is not None:
            can_dev.property_changed -= self._can_device_property_changed
            self._can_device = None
        self._on_property_changed(self.TUNNEL_VERSION_PROPERTY, "", None)
        self._on_property_changed(self.PELLET_VERSION_PROPERTY, "", None)
        prev_thread = self._check_timedout_commands_thread
        if prev_thread is not None:
            logger.debug("joining checktimedout commands thread")
            prev_thread.join()
        self._device_stream_started = False

    def _can_device_property_changed(self, name: str, value, prev_value):
        logger.debug("_device_property_changed: %s : %s -> %s", name, prev_value, value)
        props = Device
        # only translate/relay what we want:
        is_dev_comm_err_possible = False
        if name == props.UUID_ACK_TIMEOUT_ENGAGED:
            self.device_ack_timeout_engaged = value
            is_dev_comm_err_possible = True
        elif name == props.PELLET_STATUS_TIMEOUT_ENGAGED:
            post_api_detector_event_content(
                self._event_manager,
                ApiDetectorKind.pelletStatusMessageInterruption,
                value,
                True,
            )
            self._device_pellet_status_timeout_engaged = value
            self.property_changed(self.DEVICE_PELLET_STATUS_TIMEOUT_ENGAGED, value, prev_value)
            is_dev_comm_err_possible = True
        elif name == props.TUNNEL_STATUS_TIMEOUT_ENGAGED:
            if not self._tunnel_headfix_enabled:
                logger.debug("Ignoring tunnel status timeout because tunnel/headfix hardware is disabled")
                return
            post_api_detector_event_content(
                self._event_manager,
                ApiDetectorKind.tunnelStatusMessageInterruption,
                value,
                True,
            )
            self._device_tunnel_status_timeout_engaged = value
            self.property_changed(self.DEVICE_TUNNEL_STATUS_TIMEOUT_ENGAGED, value, prev_value)
            is_dev_comm_err_possible = True
        #
        if is_dev_comm_err_possible:
            engaged = any((
                self._device_uuid_ack_timeout_engaged,
                self._device_tunnel_status_timeout_engaged,
                self._device_pellet_status_timeout_engaged,
            ))
            self._sensor_analysis.device_comm_alarm.is_engaged = engaged

    def _message_handler_property_changed(self, name: str, value, old_value):
        props = MessageHandler
        if not self._tunnel_headfix_enabled and name in {
            props.HEAD_MAGNET_INTENSITY_PROPERTY,
            props.HEAD_GATE_PROPERTY,
            props.TUNNEL_GATE_OPEN_STATUS,
        }:
            return
        if name == props.HEAD_MAGNET_INTENSITY_PROPERTY:
            prev, self._head_magnet_position = self._head_magnet_position, value
            self._on_property_changed(self.HEAD_MAGNET_INTENSITY, value, prev)

        elif name == props.HEAD_GATE_PROPERTY:
            # logger.verbose("HEAD_GATE_PROPERTY: %s ; old=%s", value, old_value)
            prev, self._tunnel_gate_position = self._tunnel_gate_position, value
            self._on_property_changed(self.TUNNEL_GATE_POSITION, value, prev)

        elif name == props.TUNNEL_GATE_OPEN_STATUS:
            prev, self._tunnel_gate_open_status = self._tunnel_gate_open_status, value

        elif name == props.STEPPER_X_PROPERTY:
            prev = self._last_motor_coordinates
            new = prev.replace(x=value.position)
            self._last_motor_coordinates = new
            self.send_x = value.send_position
            self._on_property_changed(self.POS_XYZ, new, prev)

        elif name == props.STEPPER_Y_PROPERTY:
            prev = self._last_motor_coordinates
            new = prev.replace(y=value.position)
            self._last_motor_coordinates = new
            self.send_y = value.send_position
            self._on_property_changed(self.POS_XYZ, new, prev)

        elif name == props.STEPPER_Z_PROPERTY:
            prev = self._last_motor_coordinates
            new = prev.replace(z=value.position)
            self._last_motor_coordinates = new
            self.send_z = value.send_position
            self._on_property_changed(self.POS_XYZ, new, prev)

        elif name == props.FRONT_DOOR_PROPERTY:
            self.front_door_open = value

        elif name == props.DRAWER_DOOR_PROPERTY:
            self.slide_door_open = value

        elif name == props.COVER_ARM_ANGLE_PROPERTY:
            self._cover_arm_position = value

        elif name == props.LOAD_ARM_ANGLE_PROPERTY:
            self._load_arm_position = value

        elif name == props.FIRMWARE_VERSION_PROPERTY and value is not None:
            version = str(value).lower()
            if version.find("module") != -1:
                version = version.replace("module", "").strip()
            if version.find("pellet") != -1:
                clean_v = _reg_pellet_version_clean.sub("", version).strip()
                self._on_property_changed(self.PELLET_VERSION_PROPERTY, clean_v, old_value)
            elif version.find("magnet") != -1:
                clean_v = _reg_magnet_version_clean.sub("", version).strip()
                self._on_property_changed(self.TUNNEL_VERSION_PROPERTY, clean_v, old_value)
            elif version.find("tunnel") != -1:
                clean_v = _reg_tunnel_version_clean.sub("", version).strip()
                self._on_property_changed(self.TUNNEL_VERSION_PROPERTY, clean_v, old_value)

        elif name == props.COLOR_LED:
            self._color_led = value

    def _send_with_token(self, device: Optional[DeviceConnectionProtocol], cmd: SystemCommandKind, data=None) -> Optional[UUID]:
        with self._lock:
            # ensure only 1 command can be sent at the same time
            # NB: there are multiple threads which can act on this instance,
            # and we don't want to try to send 2 messages at the same time,
            # or the pending command token might be overwritten.
            tok = self.__send_with_token(device, cmd, data)
            commands_tuple = list(self._pending_tokens.values())
        if tok is not None:
            self._refresh_cmd_in_progress(commands_tuple)
        return tok

    def __send_with_token(self, device: DeviceConnectionProtocol, cmd: SystemCommandKind, data=None) -> Optional[UUID]:
        token = uuid4()
        perf_now = get_perf_now()
        logger.debug("send_command cmd=%s token=%s nbr=%s", cmd, token, len(self._pending_tokens))
        if self._send_command(device, cmd, data, token):
            self._pending_tokens[token] = (cmd, perf_now)
            return token
        else:
            logger.verbose("send_command failed, device not setup yet: cmd=%s token=%s", cmd, token)
            return None

    # noinspection PyMethodMayBeStatic
    def _send_command(self,
                      device: DeviceConnectionProtocol,
                      cmd: SystemCommandKind,
                      data=None,
                      context=None) -> bool:
        if device is not None:
            device.send_message(cmd, data, context)
            return True
        return False

    def set_motors_drift(self, drift: Offset3DTuple):
        """Apply the pellet motor drift"""
        dev = self._device_conn
        if dev is None:
            return
        dev.send_message(SystemCommandKind.SET_MOTOR_DRIFT, drift)
        # this ensure the next send_to_fixed_pos command will get the corrected position:
        for cmd_kind in (SystemCommandKind.SET_X, SystemCommandKind.SET_Y, SystemCommandKind.SET_Z):
            dev.send_message(cmd_kind, SystemDataArgsKwargs(0, relative=True))

    def set_auto_correct_motor_drift(self, enabled: bool):
        dev = self._device_conn
        if dev is None:
            return
        dev.send_message(SystemCommandKind.SET_AUTO_CORRECT_DRIFT, enabled)
        if not enabled:
            logger.verbose("Doing SET_X/Y/Z relative=0 to clear possible motors drift")
            for cmd_kind in (SystemCommandKind.SET_X, SystemCommandKind.SET_Y, SystemCommandKind.SET_Z):
                dev.send_message(cmd_kind, SystemDataArgsKwargs(0, relative=True))

    def _refresh_cmd_in_progress(self, commands_tuple: List[Tuple[SystemCommandKind, float]]):
        if len(commands_tuple) == 0:
            self.property_changed(HardwareModel.PENDING_COMMAND_PROPERTY, None, True)
        else:
            # NB: sort on cmd perf_counter:
            list_after = (cmd.name for cmd, _ in sorted(commands_tuple, key=lambda t: t[1]))
            self.property_changed(
                HardwareModel.PENDING_COMMAND_PROPERTY, " - ".join(list_after), None
            )

    def _ack_received(self, token: UUID, *, perf_c: Optional[float]=None):
        with self._lock:
            popped = self._pending_tokens.pop(token, None)
            commands_in_prog = list(self._pending_tokens.values())
        if popped is None:
            # this can happen at device connection
            (logger.debug if not self._device_stream_started else logger.warning)(
                "Received unexpected ack token: %s", token)
        else:
            self._refresh_cmd_in_progress(commands_in_prog)

    def wait_pending_command_acked(
        self,
        token,
        *,
        timeout: float = 3,
        raise_on_timeout: bool = True,
    ):
        p_start = time.perf_counter()
        p_timeout = p_start + timeout
        logger.verbose("Waiting ack pending command %s", token)
        while True:
            with self._lock:
                if token not in self._pending_tokens:
                    logger.debug("Got ack for token=%s ; delay=%.6f",
                                 token, time.perf_counter() - p_start)
                    return
            p_now = time.perf_counter()
            if p_now > p_timeout:
                break
            time.sleep(0.0025)  # 2.5 ms
        if raise_on_timeout:
            raise RuntimeError(f"timeout waiting ack of pending token={token}")
        logger.warning("timeout waiting ack token %s, but continuing", token)
