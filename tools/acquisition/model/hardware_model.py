import logging
import math
import os
import re
import subprocess
import threading
import time
import uuid
from functools import partial
from queue import Queue
from uuid import UUID, uuid4
from typing import Optional, Tuple, Dict, Union, List

from autotrainer.api import ApiEventKind, ApiDetectorKind
from autotrainer.core import (ObservableObject, SystemCommandKind, MessageHandler, AnimalSubject, Offset3DTuple,
                              get_verbose_logger, Motor, EventManager, HardwareConfiguration,
                              get_perf_now, SystemStatusMessageKind)
from autotrainer.core.logging import log_hardware_initialization
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core.event import post_api_detector_event_content
from autotrainer.core.message import SystemDataArgsKwargs
from autotrainer.device import (CanTransportConfiguration, CanTransportKind, DeviceConnectionProtocol, HAVE_CAN_DEVICE,
                                DeviceConnection, CanDevice, StepperConfig, ServoConfig, Device, ColorLed, Target)
from autotrainer.behavior import PelletDeviceProtocol

logger = get_verbose_logger(__name__)

_nans_offset3dTuple = Offset3DTuple.get_nan()


_reg_pellet_version_clean = re.compile("pellet ?:? *")
class HardwareModel(ObservableObject, PelletDeviceProtocol):

    PELLET_VERSION_PROPERTY = "pellet_version"

    PELLET_IDENTIFIER_PROPERTY = "pellet_identifier"
    CAN_ENABLED = "can_enabled"
    PELLET_CONTROLLER_ENABLED = "pellet_controller_enabled"
    NIDAQ_ENABLED = "nidaq_enabled"

    PENDING_COMMAND_PROPERTY = "pending_command"

    DEVICE_ACK_TIMEOUT_ENGAGED = "device_ack_timeout_engaged"
    DEVICE_PELLET_STATUS_TIMEOUT_ENGAGED = "device_pellet_status_timeout_engaged"

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

    def __init__(
        self,
        message_handler: MessageHandler,
    ):
        super().__init__(event_names=("device_event",))

        self._lock = threading.RLock()  # **required** re-entrant lock !!
        self._safety_shutdown_lock = threading.Lock()
        self._safety_shutdown_started = False
        self._safety_shutdown_thread: Optional[threading.Thread] = None
        self._connect_count = 0

        self._event_manager = EventManager.default()
        self._board_status_timeout: Optional[float] = None
        self._device_ack_timeout_delay: Optional[float] = None
        self._device_conn: Optional[DeviceConnectionProtocol] = None
        self._can_device: Optional[CanDevice] = None
        self._device_uuid_ack_timeout_engaged = False
        self._device_pellet_status_timeout_engaged = False
        self._device_stream_started = False
        self._can_enabled = True
        self._pellet_controller_enabled = True
        self._nidaq_enabled = False

        self._pending_tokens: Dict[UUID, Tuple[SystemCommandKind, float]] = {}

        message_handler.property_changed += self._message_handler_property_changed
        message_handler.ack_received += self._ack_received

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

        self._cover_arm_position: float = math.nan
        self._load_arm_position: float = math.nan

        self._pellet_version = ""
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

    @property
    def pellet_status_timeout_engaged(self) -> bool:
        return self._device_pellet_status_timeout_engaged

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
    def pellet_version(self) -> str:
        return self._pellet_version

    @property
    def can_enabled(self) -> bool:
        return self._can_enabled

    @property
    def pellet_controller_enabled(self) -> bool:
        return self._pellet_controller_enabled

    @property
    def nidaq_enabled(self) -> bool:
        return self._nidaq_enabled

    @property
    def requires_connection(self) -> bool:
        return self._can_enabled and self._pellet_controller_enabled

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

    def set_color_led(self, r: int, g: int, b: int):
        """0 -> 100"""
        return self._send_with_token(self._device_conn, SystemCommandKind.SET_RGB_LED, (r, g, b))

    def load_config(self, config: HardwareConfiguration):
        self._set_boolean_config(self.CAN_ENABLED, "_can_enabled", config.can_enabled)
        self._set_boolean_config(
            self.PELLET_CONTROLLER_ENABLED,
            "_pellet_controller_enabled",
            config.pellet_controller_enabled,
        )
        self._set_boolean_config(self.NIDAQ_ENABLED, "_nidaq_enabled", config.nidaq_enabled)
        self.set_device_ack_timeout(config.min_ack_timeout)
        self.set_board_status_timeout(config.board_status_timeout)

    def _set_boolean_config(self, property_name: str, attr_name: str, value: bool) -> None:
        prev = getattr(self, attr_name)
        value = bool(value)
        setattr(self, attr_name, value)
        self._on_property_changed(property_name, value, prev)

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
        if not self._can_enabled:
            logger.notice("Skipping hardware connection because CAN bus is disabled in configuration")
            log_hardware_initialization(logger, "SKIP | CAN/pellet controller | CAN disabled")
            return
        if not self._pellet_controller_enabled:
            logger.notice("Skipping hardware connection because pellet controller is disabled in configuration")
            log_hardware_initialization(logger, "SKIP | CAN/pellet controller | pellet controller disabled")
            return
        connect_started = time.perf_counter()
        self._emit_device_event("state", "connect_start")
        transport = CanTransportConfiguration.from_environment()
        log_hardware_initialization(
            logger,
            "START | CAN/pellet controller | transport=%s channel=%s required_target=%s",
            transport.kind.value,
            transport.channel,
            Target.PELLET_DEVICE.name,
        )
        logger.notice("%s: connect with %s", self, cmd_queue)
        self._disconnect_event.clear()

        prev_device = self._device_conn
        if prev_device is not None:
            logger.warning("auto-disconnecting from device before (re-)connect")
            self._disconnect_transport()
        with self._safety_shutdown_lock:
            self._safety_shutdown_started = False
            self._safety_shutdown_thread = None

        self._connect_count += 1
        self._last_motor_coordinates = \
        self._last_requested_set_coordinates = \
        self._last_motor_send_coordinates = _nans_offset3dTuple
        self._device_pellet_status_timeout_engaged = False

        # This is specific to wanting to be able to test UI changes w/the emulation interface, which is not
        # configured to generate messages as frequently as the real device.
        buffer_size = 10 if HAVE_CAN_DEVICE else 1
        #
        can_device = self._can_device = CanDevice(
            buffer_size=buffer_size,
            required_targets=(Target.PELLET_DEVICE,),
            can_transport=transport,
            shutdown_callback=lambda reason: self.safety_shutdown(reason, wait=False),
        )
        log_hardware_initialization(
            logger,
            "READY | CAN adapter created | transport=%s channel=%s elapsed=%.3fs",
            transport.kind.value,
            transport.channel,
            time.perf_counter() - connect_started,
        )
        self._emit_device_event("state", "connected")
        self.set_device_ack_timeout(self._device_ack_timeout_delay)  # ensure it's used
        self.set_board_status_timeout(self._board_status_timeout)

        can_device.property_changed += self._can_device_property_changed

        device_conn = self._device_conn = DeviceConnection(can_device, cmd_queue, name="can-device")
        log_hardware_initialization(logger, "START | CAN connection worker | name=can-device")
        device_conn.request_connect()
        connection_started = time.perf_counter()
        connection_timeout = 3.0
        log_hardware_initialization(
            logger,
            "START | CAN connection readiness | timeout=%.1fs",
            connection_timeout,
        )
        try:
            device_conn.wait_connected(timeout=connection_timeout)
        except Exception as exc:
            log_hardware_initialization(
                logger,
                "FAILED | CAN connection readiness | elapsed=%.3fs error=%s",
                time.perf_counter() - connection_started,
                str(exc) or exc.__class__.__name__,
                level=logging.ERROR,
            )
            raise
        log_hardware_initialization(
            logger,
            "READY | CAN connection readiness | elapsed=%.3fs",
            time.perf_counter() - connection_started,
        )

        send_dev_cmd = partial(self._send_command, device_conn)
        def send_dev_ack_cmd(kind, data=None):
            command_started = time.perf_counter()
            log_hardware_initialization(logger, "START | pellet command | command=%s", kind.name)
            tok = str(uuid.uuid4())
            try:
                with device_conn.await_acknowledge(
                    {tok},
                    timeout=can_device.default_command_ack_timeout_duration,
                ):
                    send_dev_cmd(kind, data, context=tok)
            except Exception as exc:
                log_hardware_initialization(
                    logger,
                    "FAILED | pellet command | command=%s elapsed=%.3fs error=%s",
                    kind.name,
                    time.perf_counter() - command_started,
                    str(exc) or exc.__class__.__name__,
                    level=logging.ERROR,
                )
                raise
            log_hardware_initialization(
                logger,
                "READY | pellet command acknowledged | command=%s elapsed=%.3fs",
                kind.name,
                time.perf_counter() - command_started,
            )

        send_dev_ack_cmd(SystemCommandKind.REQUEST_VERSION)

        # load and set motors and move configs
        # 1)
        config_started = time.perf_counter()
        log_hardware_initialization(logger, "START | pellet motor configuration")
        motors_config = device_conn.load_default_motor_config()
        log_hardware_initialization(
            logger,
            "READY | pellet motor configuration | elapsed=%.3fs",
            time.perf_counter() - config_started,
        )
        # 2)
        config_started = time.perf_counter()
        log_hardware_initialization(logger, "START | pellet move configuration")
        device_conn.load_default_move_config()
        log_hardware_initialization(
            logger,
            "READY | pellet move configuration | elapsed=%.3fs",
            time.perf_counter() - config_started,
        )
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

        prev_thread = self._check_timedout_commands_thread
        if prev_thread is None or not prev_thread.is_alive():
            thread = self._check_timedout_commands_thread = threading.Thread(
                target=self._check_timedout_commands_handler,
                daemon=True,
                name="check-timedout-commands",
            )
            thread.start()
        log_hardware_initialization(
            logger,
            "READY | CAN/pellet controller | elapsed=%.3fs",
            time.perf_counter() - connect_started,
        )

    def _disconnect_transport(self):
        logger.verbose("disconnecting ..")
        self._emit_device_event("state", "disconnect_start")
        self._disconnect_event.set()
        with self._lock:
            self._pending_tokens.clear()
        can_dev = self._can_device
        dev = self._device_conn
        if dev is not None:
            dev.request_disconnect()
            dev.join()
            self._device_conn = None
        if can_dev is not None:
            can_dev.property_changed -= self._can_device_property_changed
            self._can_device = None
        prev, self._pellet_version = self._pellet_version, ""
        self._on_property_changed(self.PELLET_VERSION_PROPERTY, "", prev)
        prev_thread = self._check_timedout_commands_thread
        if prev_thread is not None:
            logger.debug("joining checktimedout commands thread")
            prev_thread.join()
        with self._lock:
            self._pending_tokens.clear()
        self._refresh_cmd_in_progress([])
        self._device_stream_started = False
        self._emit_device_event("state", "disconnected")

    def _reset_socketcan(self, transport: Optional[CanTransportConfiguration]) -> None:
        if transport is None:
            return
        if transport.kind != CanTransportKind.SOCKETCAN:
            return

        helper = "/usr/local/sbin/reachaq-reset-can"
        command = [helper] if os.geteuid() == 0 else ["sudo", "-n", helper]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.error("Could not reset SocketCAN after safety shutdown: %s", exc)
            return
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
            logger.error("SocketCAN safety reset failed: %s", detail)
        else:
            logger.notice("SocketCAN safety reset completed")

    def _run_safety_shutdown(self, reason: str) -> None:
        logger.critical("Starting CAN safety shutdown: %s", reason)
        self._emit_device_event(
            "state",
            "safety_shutdown_start",
            data={"reason": reason},
        )
        can_dev = self._can_device
        if can_dev is not None:
            transport = can_dev.can_transport_configuration
        else:
            try:
                transport = CanTransportConfiguration.from_environment()
            except Exception:
                logger.exception("Could not determine CAN transport for safety reset")
                transport = None
        try:
            self._disconnect_transport()
        except Exception:
            logger.exception("Failed to close CAN transport during safety shutdown")
        finally:
            self._reset_socketcan(transport)
        logger.notice("CAN safety shutdown finished")
        self._emit_device_event(
            "state",
            "safety_shutdown_finished",
            data={"reason": reason},
        )

    def safety_shutdown(self, reason: str, *, wait: bool = True) -> None:
        """Stop hardware once, discard commands, and flush the SocketCAN link."""
        with self._safety_shutdown_lock:
            thread = self._safety_shutdown_thread
            if not self._safety_shutdown_started:
                self._safety_shutdown_started = True
                thread = threading.Thread(
                    target=self._run_safety_shutdown,
                    args=(reason,),
                    name="CanSafetyShutdown",
                    daemon=False,
                )
                self._safety_shutdown_thread = thread
                thread.start()

        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(20)
            if thread.is_alive():
                logger.error("CAN safety shutdown did not finish within 20 seconds")

    def disconnect(self):
        self.safety_shutdown("hardware disconnect")

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
        #
        if is_dev_comm_err_possible:
            engaged = any((
                self._device_uuid_ack_timeout_engaged,
                self._device_pellet_status_timeout_engaged,
            ))

    def _message_handler_property_changed(self, name: str, value, old_value):
        props = MessageHandler
        if name == props.STEPPER_X_PROPERTY:
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
                prev, self._pellet_version = self._pellet_version, clean_v
                self._on_property_changed(self.PELLET_VERSION_PROPERTY, clean_v, prev)
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
        with self._safety_shutdown_lock:
            if self._safety_shutdown_started:
                logger.warning("Ignoring %s because CAN safety shutdown is active", cmd)
                return False
            if device is not None:
                self._emit_device_event(
                    "outbound",
                    cmd,
                    data=data,
                    context=context,
                )
                device.send_message(cmd, data, context)
                return True
        return False

    def _emit_device_event(
        self,
        direction,
        kind,
        *,
        data=None,
        context=None,
        target=None,
    ) -> None:
        try:
            self.device_event(
                direction,
                kind,
                data,
                context,
                target,
                time.perf_counter(),
                time.time(),
            )
        except Exception:
            logger.exception("Unable to publish structured device event")

    def set_motors_drift(self, drift: Offset3DTuple):
        """Apply the pellet motor drift"""
        dev = self._device_conn
        if dev is None:
            return
        self._send_command(dev, SystemCommandKind.SET_MOTOR_DRIFT, drift)
        # this ensure the next send_to_fixed_pos command will get the corrected position:
        for cmd_kind in (SystemCommandKind.SET_X, SystemCommandKind.SET_Y, SystemCommandKind.SET_Z):
            self._send_command(
                dev,
                cmd_kind,
                SystemDataArgsKwargs(0, relative=True),
            )

    def set_auto_correct_motor_drift(self, enabled: bool):
        dev = self._device_conn
        if dev is None:
            return
        self._send_command(dev, SystemCommandKind.SET_AUTO_CORRECT_DRIFT, enabled)
        if not enabled:
            logger.verbose("Doing SET_X/Y/Z relative=0 to clear possible motors drift")
            for cmd_kind in (SystemCommandKind.SET_X, SystemCommandKind.SET_Y, SystemCommandKind.SET_Z):
                self._send_command(
                    dev,
                    cmd_kind,
                    SystemDataArgsKwargs(0, relative=True),
                )

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
