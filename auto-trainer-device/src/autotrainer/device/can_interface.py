"""
Interface to the CANbus protocol to the device.

The interface to the CANbus is via a C++ library which provides high-level
functionality to the device (e.g. set a configuration, move a motor).

Data read from the CANbus is in the form of a JerryCANMsg class. That class is
translated to a more generic data class (see device_interface.py) for the rest
of the application to consume.

Writes to the device can be from almost any threaded context. Note that the low-level
CAN driver handles thread safety. Reads occur in the device_thread context, returning
a list of data sets that are then propagated to the rest of the application.
"""

import logging
import inspect
import math
import time
import warnings
import dataclasses
from enum import Enum, IntEnum
from operator import attrgetter
from pathlib import Path
from typing import Iterable, Type, Optional, Dict, Union, Any, Tuple


try:
    import pyjerrycan as jerry
    from pyjerrycan import JerryCAN, JerryCANMsg, JerryCANCmdType, JerryCANCfgMsg, AbsOrRel, \
        JerryCANBootloaderCmd
except ModuleNotFoundError:
    jerry = JerryCAN = None
    from .socketcan_jerrycan import JerryCANMsg, JerryCANCmdType, JerryCANCfgMsg, AbsOrRel, \
        JerryCANBootloaderCmd
else:
    from importlib.metadata import version
    jerry_v = tuple(
        map(lambda s: int(s) if s.isdigit() else s, version("pyjerrycan").split("."))
    )
    if jerry_v < (1, 2, 0):
        warnings.warn(f"expected pyjerrycan >= 1.2.0 ; got {jerry_v}", UserWarning)


from autotrainer.core import get_perf_now, Offset3DTuple
from autotrainer.core.logging import get_verbose_logger
from .can_transport import CanTransportConfiguration
from .can_ownership import CanChannelOwnership, CanChannelInUseError
from .device_interface import (
    DeviceInterface,
    Acknowledge,
    AnalogOutput,
    AnalogOutputs,
    Heartbeat,
    Motor,
    DigitalOutputs,
    Source,
    Status,
    Target,
    Tone,
    ColorLed,
    MotorSource,
    PelletDigitalInputs,
    ServoConfig,
    StepperConfig,
    ServoStatus,
    StepperStatus,
    Version,
    BoardTimeSync,
    BoardCapabilities,
    DigitalPulseStatus,
)
from .board_clock import BoardClockModel, BoardSequenceTracker
from .stepper_motor import mm_to_turns, turns_to_mm
from .socketcan_jerrycan import SocketCanJerryCAN
from . import socketcan_jerrycan


logger = get_verbose_logger(__name__)

_debug_path_fake_status_timeout = Path("/tmp/autotrainer_fake_device_timeout")


class MissingDeviceAddressError(RuntimeError):
    """Dedicated for when device address could not be read"""


_STEPPER_MAX_TURNS = 15  # absolute max nbr of turns for each stepper, hardcoded for now.
_STEPPER_MAX_POS = turns_to_mm(_STEPPER_MAX_TURNS)


def _is_pellet_by_addr(addr: int) -> bool:
    """
    Pellet device CAN address board type is 0 (bits 2 and 3)

    Args:
        addr: Physical CAN address

    Returns:
        bool: True if the address is associated with a pellet device
    """
    return addr & 0xC == 0


def _addr2tgt(addr: int) -> Target:
    """
    Convert a CANbus address to a target

    Args:
        addr: Physical CAN address

    Returns:
        Target: PELLET_DEVICE
    """
    if _is_pellet_by_addr(addr):
        return Target.PELLET_DEVICE
    raise RuntimeError(f"unknown addr for _addr2tgt: {addr}")


def target_to_str(target: Target) -> str:
    """
    Args:
        target: type of physical remote HW target

    Returns:
        str: Human-readable string identifier for the target
    """
    if target == Target.PELLET_DEVICE:
        return "Pellet"
    return "Unknown"


_MOTOR_TO_STR_MAP = {
    Motor.PELLET_X_MOTOR: "X",
    Motor.PELLET_Y_MOTOR: "Y",
    Motor.PELLET_Z_MOTOR: "Z",
    Motor.PELLET_LOAD_SERVO: "Load",
    Motor.PELLET_COVER_SERVO: "Cover",

}


def motor_to_str(motor: Motor) -> str:
    """
    Args:
        motor: motor (servo or stepper) identifier

    Returns:
        str: Human-readable string identifier for the motor
    """
    return _MOTOR_TO_STR_MAP.get(motor, "Unknown")


class MotorInstance(IntEnum):
    PELLET_X_MOTOR_ID = 0
    PELLET_Y_MOTOR_ID = 1
    PELLET_Z_MOTOR_ID = 2

    PELLET_COVER_SERVO_ID = 0
    PELLET_LOAD_SERVO_ID = 1


_MOTOR_TO_ID_MAP = {
    Motor.PELLET_X_MOTOR: MotorInstance.PELLET_X_MOTOR_ID,
    Motor.PELLET_Y_MOTOR: MotorInstance.PELLET_Y_MOTOR_ID,
    Motor.PELLET_Z_MOTOR: MotorInstance.PELLET_Z_MOTOR_ID,
    Motor.PELLET_LOAD_SERVO: MotorInstance.PELLET_LOAD_SERVO_ID,
    Motor.PELLET_COVER_SERVO: MotorInstance.PELLET_COVER_SERVO_ID,
}

def _motor_to_id(motor: Motor) -> int:
    """
    Args:
        motor: Motor identifier

    Returns:
        int: Physical identifier for the motor
    """

    motor_id = _MOTOR_TO_ID_MAP[motor]
    return motor_id.value


_motor_to_axis_idx_map = {
    Motor.PELLET_X_MOTOR: 0,
    Motor.PELLET_Y_MOTOR: 1,
    Motor.PELLET_Z_MOTOR: 2,
}

def _motor_to_axis_idx(
    motor: Motor,
    *,
    _map=_motor_to_axis_idx_map,  # noqa
) -> int:
    value = _map.get(motor, None)
    if value is None:
        raise ValueError(f"Invalid motor for offset idx map: {motor}")
    return value


_servo_motors = {
    Motor.PELLET_LOAD_SERVO,
    Motor.PELLET_COVER_SERVO,
}

def is_servo(motor: Motor,
             *,
             _servo_motors=_servo_motors,  # noqa
             ) -> bool:
    """
    Args:
        motor: motor identifier

    Returns:
        bool: True if the motor is a servo motor, False otherwise
    """
    return motor in _servo_motors


def is_stepper(motor: Motor) -> bool:
    """
    Args:
        motor: motor identifier

    Returns:
        bool: True if the motor is a stepper motor, False otherwise
    """
    return not is_servo(motor)


_pellet_board_motors = {
    Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR,
    Motor.PELLET_LOAD_SERVO, Motor.PELLET_COVER_SERVO,
}

_pellet_servo_2_motor = {
    MotorInstance.PELLET_COVER_SERVO_ID: Motor.PELLET_COVER_SERVO,
    MotorInstance.PELLET_LOAD_SERVO_ID: Motor.PELLET_LOAD_SERVO,
}

# Current pellet-board firmware still broadcasts status for servo channel 2,
# which formerly drove the tunnel gate.  The pellet-only runtime deliberately
# does not expose that retired motor, so discard its status without treating it
# as an unexpected protocol value.
_retired_pellet_servo_ids = frozenset({2})

_pellet_stepper_2_motor = {
    MotorInstance.PELLET_X_MOTOR_ID: Motor.PELLET_X_MOTOR,
    MotorInstance.PELLET_Y_MOTOR_ID: Motor.PELLET_Y_MOTOR,
    MotorInstance.PELLET_Z_MOTOR_ID: Motor.PELLET_Z_MOTOR,
}


def target_of_motor(motor: Motor) -> Target:
    """
    Args:
        motor: motor identifier

    Returns:
        Target: the hardware target that the motor resides on
    """
    if motor in _pellet_board_motors or motor in {Motor.DELAY, Motor.TONE}:
        return Target.PELLET_DEVICE
    raise ValueError(f"Unhandled motor for target_of_motor: {motor!r}")


def _id_to_motor(target: Target, isa_servo: bool, motor_id: int) -> Motor:
    """
    Convert a motor identifier from a CAN message to a Motor identifier

    Args:
        target: target from whence the id came from
        isa_servo: True if the motor is a servo
        motor_id: CAN message motor id

    Returns:
        Motor: associated Motor identifier
    """
    motor = Motor.NONE
    assert target == Target.PELLET_DEVICE
    if isa_servo:
        motor = _pellet_servo_2_motor.get(motor_id, motor)
    else:
        motor = _pellet_stepper_2_motor.get(motor_id, motor)

    is_retired_pellet_servo = (
        target == Target.PELLET_DEVICE
        and isa_servo
        and motor_id in _retired_pellet_servo_ids
    )
    if motor == Motor.NONE and not is_retired_pellet_servo:
        logger.warning("Unknown motor id for target: target=%s isa_servo=%s motor_id=%s",
                       target, isa_servo, motor_id)
    return motor


class CanInterface(DeviceInterface):
    """
    CanInterface implements the details of
        * communication (read and write) with Alogus hardware interface (pyjerrcan)

    Applications and scripts would generally not interact with this class directly,
    but with the more generalized behavior in the CanDevice class.
    """

    # UUIDs are used in a command/acknowledge protocol to know when a command is complete
    # A UUID of 0 is an invalid UUID.
    # UUIDs in the CAN message are 8 bits, so UUIDs here are maintained to 8 bits
    _uuid: int = 1

    @classmethod
    def next_uuid(cls) -> int:
        """
        Returns:
            int: Next UUID to use
        """
        # TODO: we should randomize the uuid we use to pass to our CAN messages,
        #  this would prevent possible conflict with a second client connected on the bus.
        cls._uuid = cls._uuid + 1 & 0xFF  # maintain 8 bits
        if cls._uuid == 0:  # don't allow 0's
            cls._uuid = 1
        if __debug__:
            # Get the current stack frame
            current_frame = inspect.currentframe()
            # Get the frame of the caller (one level up)
            caller_frame = current_frame.f_back
            # Extract the code object from the caller's frame
            caller_code = caller_frame.f_code
            # Get the name of the function from the code object
            caller_name = caller_code.co_name
            logger.debug("next_uuid: caller=%s uuid=%s", caller_name, cls._uuid)
        return cls._uuid

    @classmethod
    def uuid(cls) -> int:
        """
        Returns:
            int: UUID of active command
        """
        return cls._uuid

    def __init__(self, required_targets: Optional[Iterable[Target]] = None,
                 can_transport: Optional[CanTransportConfiguration] = None):
        """
        Initialize the CanInterface Class.

        Creates default Configurations for motors. Expected to be updated during
        the connection protocol.

        Sets the known pellet address to None. Expected to be updated during
        the connection protocol.
        """
        super().__init__()
        self._required_targets = tuple(required_targets or (Target.PELLET_DEVICE,))
        self._can_transport = can_transport or CanTransportConfiguration()
        self._channel_ownership = CanChannelOwnership(self._can_transport.channel)
        if self._can_transport.uses_linux_can_stack:
            self._jerrycan_msg_cls = socketcan_jerrycan.JerryCANMsg
            self._jerrycan_cmd_type = socketcan_jerrycan.JerryCANCmdType
            self._jerrycan_cfg_msg_cls = socketcan_jerrycan.JerryCANCfgMsg
            self._abs_or_rel = socketcan_jerrycan.AbsOrRel
            self._bootloader_cmd_cls = socketcan_jerrycan.JerryCANBootloaderCmd
        else:
            self._jerrycan_msg_cls = JerryCANMsg
            self._jerrycan_cmd_type = JerryCANCmdType
            self._jerrycan_cfg_msg_cls = JerryCANCfgMsg
            self._abs_or_rel = AbsOrRel
            self._bootloader_cmd_cls = JerryCANBootloaderCmd

        if self._can_transport.uses_linux_can_stack:
            self._jc = SocketCanJerryCAN(self._can_transport)
        elif JerryCAN is None:
            self._jc = None
        else:
            self._jc = JerryCAN()

        # by default, will be set in open() too
        self._read_msgs = self._read_by_one_msg
        self._get_timestamp_ns = self._assign_timestamp_ns
        self._get_index = lambda m: time.perf_counter_ns()

        self._cnt_none = 0
        self._is_open = False

        self._next_status_log_perf_c = -math.inf
        self._pellet_board_last_status_perf_c = {
            motor: -math.inf
            for motor in Motor
            if motor in _pellet_board_motors
        }

        self._pellet_addr: Optional[int] = None
        self._board_clock_model = BoardClockModel()
        self._board_sequence_tracker = BoardSequenceTracker()
        self._time_sync_requests = {}
        self._next_time_sync_request_id = 1

        self._servo_configs = {}
        self.load_config = ServoConfig()
        self.cover_config = ServoConfig()

        self._motor_configs = {}
        self.x_config = StepperConfig()
        self.y_config = StepperConfig()
        self.z_config = StepperConfig()

        no_op = lambda msg: None

        self._last_positions = Offset3DTuple.get_nan()
        self._prev_send_pos = Offset3DTuple.get_nan()

        self._last_gpio_status_perf: Dict[int, Tuple[
            PelletDigitalInputs, int]] = {}

        # Simple handlers implemented as lambdas
        cmd_type = self._jerrycan_cmd_type
        self._handlers = {
            cmd_type.HEARTBEAT: lambda msg: Heartbeat(target=_addr2tgt(msg.dst_id)),
            cmd_type.BOOTLOADER_RESPONSE: self._translate_bootloader,
            cmd_type.CFG_RESPONSE: self._translate_config,
            cmd_type.GPIO_READ: self._translate_gpio,
            cmd_type.TONE: lambda msg: Tone(
                target=_addr2tgt(msg.dst_id),
                time_remaining_ms=msg.tone.duration_ms,
                frequency_hz=msg.tone.frequency_hz
            ),
            cmd_type.ANALOG_OUT: self._translate_analog_out,
            cmd_type.RESERVED_0E: no_op,
            cmd_type.PRESSURE_READ: no_op,
            cmd_type.RGB_LED: lambda msg: ColorLed(
                target=_addr2tgt(msg.dst_id),
                red=msg.rgb_led.red,
                green=msg.rgb_led.green,
                blue=msg.rgb_led.blue
            ),
            cmd_type.AUDIO_MAGNITUDE_DATA_BEGIN: no_op,
            cmd_type.AUDIO_MAGNITUDE_DATA_CONT: no_op,
            cmd_type.AUDIO_MAGNITUDE_DATA_END: no_op,
            cmd_type.DOOR_SENSOR: no_op,
            cmd_type.SERVO_STATUS: self._translate_servo_status,
            cmd_type.STEPPER_STATUS: self._handle_stepper_status,
            cmd_type.TEMP_HUM_READ: no_op,
            cmd_type.ACKNOWLEDGE: lambda msg: Acknowledge(uuid=msg.uuid),
            # no-op handlers, to silence the warning if unknown message type
            cmd_type.STEPPER_HOME: no_op,
            cmd_type.STEPPER_MOVE: no_op,
            cmd_type.CFG_WRITE: no_op,
            cmd_type.SERVO_MOVE: no_op,
            cmd_type.GPIO_WRITE: no_op,
            cmd_type.DELAY: no_op,
            cmd_type.BOOTLOADER_DATA: no_op,
        }
        optional_handlers = {
            "TIME_SYNC_RESPONSE": self._translate_time_sync,
            "CAPABILITIES_RESPONSE": self._translate_capabilities,
            "GPIO_PULSE_STATUS": self._translate_gpio_pulse_status,
            "TIME_SYNC_REQUEST": no_op,
            "CAPABILITIES_REQUEST": no_op,
            "GPIO_PULSE": no_op,
        }
        for name, handler in optional_handlers.items():
            kind = getattr(cmd_type, name, None)
            if kind is not None:
                self._handlers[kind] = handler

    def __allow_fake_status_time(self, motor):
        if _debug_path_fake_status_timeout.exists():
            for line in _debug_path_fake_status_timeout.read_text().split("\n"):
                m_idx, age = line.split("=")
                age = float(age)
                m = Motor(int(m_idx))
                if m == motor:
                    return age

    def _handle_motor_status_age(self, motor: Motor):
        p_now = motor_p_now = get_perf_now()
        if p_now > self._next_status_log_perf_c:
            self._next_status_log_perf_c = p_now + 15
            vals = self._pellet_board_last_status_perf_c
            logger.verbose("motor status age: %s",
                ' '.join(f"{k.name}={p_now - v:.6f}s"
                for k, v in sorted(vals.items(), key=lambda i: i[1])))
        if __debug__:
            try:
                fake_age = self.__allow_fake_status_time(motor)
                if fake_age is not None:
                    motor_p_now -= fake_age
            except Exception:
                logger.exception("__allow_fake_status_time")
        if motor in _pellet_board_motors:
            vals = self._pellet_board_last_status_perf_c
            vals[motor] = motor_p_now
            # use the oldest for the "global" pellet status perf_c
            oldest = min(vals.values())
            self.pellet_status_perf_c = oldest
        else:
            return
        for m, p in vals.items():
            prev_warn, prev_err = self._motors_prev_warn_error[m]
            age = p_now - p
            if age > 1.5:
                if not prev_err:
                    prev_err = True
                    logger.error("%s status age = %.3fs", m, age)
            elif age > 0.5:
                prev_err = False
                if not prev_warn:
                    prev_warn = True
                    logger.warning("%s status age = %.3fs", m, age)
            else:
                prev_warn = prev_err = False
            self._motors_prev_warn_error[m] = (prev_warn, prev_err)

    def _set_servo_config(self, motor: Motor, config: ServoConfig):
        config.motor = motor
        config.target = target_of_motor(motor)
        self._servo_configs[motor] = config

    @property
    def load_config(self):
        """
        Returns:
            Handle to the load arm servo configuration
        """
        return self._load_config

    @load_config.setter
    def load_config(self, config: ServoConfig):
        """
        Updates the load arm servo configuration (local copy)

        Args:
            config: new configuration
        """
        self._load_config = config
        self._set_servo_config(Motor.PELLET_LOAD_SERVO, config)

    @property
    def cover_config(self):
        """
        Returns:
            Handle to the cover servo configuration
        """
        return self._cover_config

    @cover_config.setter
    def cover_config(self, config: ServoConfig):
        """
        Updates the cover servo configuration (local copy)

        Args:
            config: new configuration
        """
        self._cover_config = config
        self._set_servo_config(Motor.PELLET_COVER_SERVO, config)

    def _set_motor_config(self, motor: Motor, config):
        config.motor = motor
        config.target = target_of_motor(motor)
        self._motor_configs[motor] = config

    @property
    def x_config(self):
        """
        Returns:
            Handle to the X stepper motor configuration
        """
        return self._motor_configs[Motor.PELLET_X_MOTOR]

    @x_config.setter
    def x_config(self, config: StepperConfig):
        """
        Updates the X stepper motor configuration (local copy)

        Args:
            config: new configuration
        """
        self._set_motor_config(Motor.PELLET_X_MOTOR, config)

    @property
    def y_config(self):
        """
        Returns:
            Handle to the Y stepper motor configuration
        """
        return self._motor_configs[Motor.PELLET_Y_MOTOR]

    @y_config.setter
    def y_config(self, config: StepperConfig):
        """
        Updates the Y stepper motor configuration (local copy)

        Args:
            config: new configuration
        """
        self._set_motor_config(Motor.PELLET_Y_MOTOR, config)

    @property
    def z_config(self):
        """
        Returns:
            Handle to the Z stepper motor configuration
        """
        return self._motor_configs[Motor.PELLET_Z_MOTOR]

    @z_config.setter
    def z_config(self, config: StepperConfig):
        """
        Updates the Z stepper motor configuration (local copy)

        Args:
            config: new configuration
        """
        self._set_motor_config(Motor.PELLET_Z_MOTOR, config)

    @property
    def pellet_address(self):
        return self._pellet_addr

    @pellet_address.setter
    def pellet_address(self, addr: int):
        """
        Set the pellet CAN address. Used primarily for testing, as after data is received
        from the device(s), the address for each target will be updated automatically.

        Args:
            addr: Pellet CAN address
        """
        self._pellet_addr = addr
        logger.info(f"pellet module located at {self._pellet_addr}")

    def are_addresses_valid(self) -> bool:
        """
        Returns:
             bool: True if all required CANbus target addresses are valid
        """
        return all(self._has_address(target) for target in self._required_targets)

    @property
    def required_targets(self) -> Tuple[Target, ...]:
        return self._required_targets

    def is_target_required(self, target: Target) -> bool:
        return target in self._required_targets

    def _has_address(self, target: Target) -> bool:
        if target == Target.PELLET_DEVICE:
            return self.pellet_address is not None
        raise ValueError(f"Unhandled target: {target}")

    def _missing_required_targets(self) -> Tuple[Target, ...]:
        return tuple(target for target in self._required_targets if not self._has_address(target))

    def _tgt2addr(self, target: Target) -> int:
        """
        Args:
            target

        Returns:
             int: CANbus address of the given target
        """
        if target == Target.PELLET_DEVICE:
            dst = self.pellet_address
        else:
            raise ValueError(f"Unhandled target: {target}")
        if dst is None:
            raise MissingDeviceAddressError(f"{target} missing address")
        return dst

    def _assign_address(self, message):
        """
        Assign the pellet CANbus address based on an incoming message.

        Args:
            message: Jerrycan message
        """
        if self.pellet_address is None and _is_pellet_by_addr(message.dst_id):
            self.pellet_address = message.dst_id

    @property
    def is_open(self) -> bool:
        """
        Returns:
            bool: Connection to hardware is open (True) or closed (False)
        """
        return self._is_open

    @property
    def motors_position(self) -> Offset3DTuple:
        return self._last_positions

    def open(self) -> bool:
        """
        Open the interface (CANbus) connection

        Returns:
            bool: True if success False otherwise
        """
        if self._jc is None:
            return False

        try:
            self._channel_ownership.acquire()
        except CanChannelInUseError as exc:
            logger.error("%s", exc)
            return False

        try:
            self._is_open = self._jc.Open() == 0
        except BaseException:
            self._channel_ownership.release()
            raise

        self._read_msgs = self._jc.ReceiveMessages if hasattr(self._jc, "ReceiveMessages") else self._read_by_one_msg
        msg_cls = self._jerrycan_msg_cls
        self._get_timestamp_ns = attrgetter("timestamp_ns") if hasattr(msg_cls(), "timestamp_ns") else self._assign_timestamp_ns
        self._get_index = attrgetter("index") if hasattr(msg_cls(), "index") else (lambda _: time.perf_counter_ns())
        logger.debug("Using %s and %s and %s", self._read_msgs, self._get_timestamp_ns, self._get_index)

        self._cnt_none = 0

        p_now = get_perf_now()
        for m in self._pellet_board_last_status_perf_c:
            self._pellet_board_last_status_perf_c[m] = p_now

        if self._is_open:
            tot_flushed = 0
            t_end = time.perf_counter() + 1.5
            while True:
                flushed = self._read_msgs(100, collect_ms=5)  # purge whatever is available
                tot_flushed += len(flushed)
                for msg in flushed:
                    self._assign_address(msg)
                    if self.are_addresses_valid():
                        break
                if self.are_addresses_valid():
                    break
                if time.perf_counter() > t_end:
                    logger.critical("Could not obtain required CAN bus addresses in time, "
                                    "missing targets: %s. Either the required board is shutdown, "
                                    "there is a CAN bus or CAN system related issue, or required_targets "
                                    "needs to match the attached hardware.",
                                    self._missing_required_targets())
                    break
            logger.notice("pellet_address=%s ; flushed %s",
                        self.pellet_address, tot_flushed)
            if not self.are_addresses_valid():
                try:
                    self._jc.Close()
                finally:
                    self._is_open = False
                    self._channel_ownership.release()
                return False
            try:
                self._query_configuration()
                if self.request_capabilities():
                    capabilities = self.get_response(
                        BoardCapabilities,
                        Target.PELLET_DEVICE,
                        timeout=0.25,
                    )
                    if capabilities is not None and capabilities.capabilities & 0x2:
                        for _ in range(4):
                            if self.request_clock_sync():
                                self.get_response(
                                    BoardTimeSync,
                                    Target.PELLET_DEVICE,
                                    timeout=0.25,
                                )
            except BaseException:
                try:
                    self._jc.Close()
                finally:
                    self._is_open = False
                    self._channel_ownership.release()
                raise
        else:
            self._channel_ownership.release()
        return self._is_open

    def close(self):
        """
        Close the interface (CANbus) connection
        """
        try:
            if self._is_open:
                self._jc.Close()
        finally:
            self._is_open = False
            self._channel_ownership.release()

    def can_read(self) -> bool:
        """
        Returns:
            bool: True if Data can be read from the connection else False
        """
        return self._is_open

    def _read_by_one_msg(self, max_count: int, collect_ms: int):
        t_end = time.perf_counter() + collect_ms / 1000
        messages = []
        while True:
            msg = self._jc.ReceiveMessage()
            if msg is None:
                if collect_ms == 0 or time.perf_counter() > t_end:
                    return messages
            else:
                messages.append(msg)
            if 0 < max_count <= len(messages):
                return messages

    def read(self, max_count: int = 1, *, collect_ms: int = 0) -> Any:
        """
        Read a set of packets from the CANbus.

        Args:
            max_count: Maximum number of messages to return
            collect_ms: Maximum duration to read messages ; if <= 0 only read while message are read,
              on first non-available message: return what was already obtained.

        Returns:
            a list of data classes (see device_interface.py for list of classes)
        """
        if self._is_open:
            messages = self._read_msgs(max_count, collect_ms)
            if len(messages) == 0:
                self._cnt_none += 1
        else:
            messages = []
        # some handlers can return None, so we have to filter:
        return list(filter(lambda v: v is not None, map(self._translate, messages)))

    def write(self, value: Any) -> int:
        """
        Do not allow the application to write unknown messages to the CANbus

        Args:
            value

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    def write_str(self, value: str) -> int:
        """
        Do not allow the application to write unknown messages to the CANbus

        Args:
            value

        Raises:
            NotImplementedErrord
        """
        raise NotImplementedError()

    def get_response(
        self,
        typeof: Type[MotorSource],
        target: Target,
        *,
        motor: Optional[Motor],
        timeout: float = 2.0):
        """
        Read data until a specific response is detected

        Args:
            typeof: Class name to look for
            target: Source board target
            motor: Optional motor to check against too
            timeout: Maximum time to wait (sec). Default=2.0
        """
        perf_timeout = time.perf_counter() + timeout
        final_res = None
        dropped = set()
        tot_dropped = 0
        logger.debug("get_response: typeof=%s target=%s motor=%s", typeof, target, motor)
        while time.perf_counter() < perf_timeout:
            messages = self.read(15, collect_ms=5)
            if len(messages) == 0:
                self._cnt_none += 1
                continue
            # loop reversed, given we break and so that we return the most recent one:
            for msg in reversed(messages):
                if isinstance(msg, typeof):
                    # logger.debug("got msg of typeof ; msg.target=%s msg.motor=%s", msg.target, msg.motor)
                    if (
                        msg.target == target
                        and (motor is None or msg.motor == motor)
                    ):
                        final_res = msg
                        break
                dropped.add(type(msg))
                tot_dropped += 1
            if final_res is not None:
                break

        logger.debug("get_response(%s): res=%s ; dropped %s msgs, types=%s",
                     typeof.__qualname__, final_res, tot_dropped, dropped)

        return final_res

    def get_motor_configuration(self, motor: Motor) -> Union[ServoConfig, StepperConfig]:
        """
        Args:
            motor

        Returns:
             ServoConfig or StepperConfig: The configuration for the given motor
        """
        if is_servo(motor):

            # Not all configuration items get pushed/pulled to the target
            # Reminder: config points to same object after assignment
            if motor == Motor.PELLET_COVER_SERVO:
                config = self.cover_config
            elif motor == Motor.PELLET_LOAD_SERVO:
                config = self.load_config
            else:
                logger.warning("Unknown motor servo config requested: motor=%s", motor)
                config = ServoConfig()
        else:
            assert is_stepper(motor)
            if motor == Motor.PELLET_X_MOTOR:
                config = self.x_config
            elif motor == Motor.PELLET_Y_MOTOR:
                config = self.y_config
            elif motor == Motor.PELLET_Z_MOTOR:
                config = self.z_config
            else:
                logger.warning("Unknown motor stepper config requested: motor=%s", motor)
                config = StepperConfig()

        return config

    def set_motor_configuration(self, motor: Motor, config, write_to_remote: bool = True) -> bool:
        """
        Set a motor's configuration.

        Args:
            motor - Motor associated with the configuration
            config - Configuration data
            write - Indication to push data to target (True), or locally store new configuration (False)

        Returns:
            bool: True if successful else False
        """
        if config is None:
            return False

        rc = True

        config.motor = motor

        if motor == Motor.PELLET_X_MOTOR:
            self.x_config = config
            if write_to_remote:
                rc = self._write_stepper_config(self.x_config)

        elif motor == Motor.PELLET_Y_MOTOR:
            self.y_config = config
            if write_to_remote:
                rc = self._write_stepper_config(self.y_config)

        elif motor == Motor.PELLET_Z_MOTOR:
            self.z_config = config
            if write_to_remote:
                rc = self._write_stepper_config(self.z_config)

        elif motor == Motor.PELLET_COVER_SERVO:
            self.cover_config = config
            if write_to_remote:
                rc = self._write_servo_config(self.cover_config)

        elif motor == Motor.PELLET_LOAD_SERVO:
            self.load_config = config
            if write_to_remote:
                rc = self._write_servo_config(self.load_config)

        else:
            logger.error("Unhandled motor for set config: %s", motor)
            rc = False

        return rc

    def _query_motor_configuration(
        self,
        motor: Motor,
        config_type: Type[MotorSource],
        *,
        timeout: float = 0.5,
    ):
        """
         Read the configurations from the remote device and print it out.

         Args:
             motor:
             config_type: Either a ServoConfig or StepperConfig class reference
             timeout: Duration before failure
         """
        t_perf_end = time.perf_counter() + timeout
        while True:
            if self.request_motor_config(motor):
                config = self.get_response(config_type, target_of_motor(motor),
                                           motor=motor, timeout=0.1)
                if config is not None:
                    self.set_motor_configuration(motor, config, write_to_remote=False)
                    logger.info("Pulled configuration for %s", motor)
                    break
                logger.error("Failed to get configuration for %s", motor)
            else:
                logger.error("Failed to request motor configuration for %s", motor)
                time.sleep(0.01)
            if time.perf_counter() > t_perf_end:
                raise RuntimeError(f"Could not get config for motor {motor} in time")

    def _query_configuration(self):
        """
        Query all motor configurations from the remote devices
        """
        query = self._query_motor_configuration
        query(Motor.PELLET_X_MOTOR, StepperConfig)
        query(Motor.PELLET_Y_MOTOR, StepperConfig)
        query(Motor.PELLET_Z_MOTOR, StepperConfig)
        query(Motor.PELLET_LOAD_SERVO, ServoConfig)
        query(Motor.PELLET_COVER_SERVO, ServoConfig)

    def delay(self, delay_sec) -> bool:
        """
        Issue a commanded delay at the target

        Args:
            delay_sec: Delay (seconds)

        Returns:
            bool: True if successful else False
        """
        addr = self._tgt2addr(Target.PELLET_DEVICE)
        uuid = CanInterface.next_uuid()
        res = self._jc.Delay(addr, int(delay_sec * 1000), uuid)
        logger.debug("Delay addr=%s res=%s uuid=%s", addr, res, uuid)
        return res == 0

    def move_servo_motor(self, motor: Motor, position: Union[float, Tuple[float, float]]):
        config = self._servo_configs.get(motor)
        if config is None:
            raise RuntimeError(f"Unhandled servo: {motor}")
        return self._move_servo_motor(motor, position, config)

    def _move_servo_motor(self, motor: Motor, position: Union[float, Tuple[float, float]], config: ServoConfig):
        """
        Move a servo motor.

        Args:
            motor
            position: Either a position (float) or a (position, rate (%)) pair
            config: associated motor configuration

        Returns:
            bool: True if successful else False
        """

        if isinstance(position, float) or isinstance(position, int):
            velocity = config.maximum_velocity
        elif isinstance(position, tuple):
            velocity = float(position[1]) / 100.0 * config.maximum_velocity
            position = float(position[0])
        else:
            raise RuntimeError("Unhandled position: %s", position)

        prev_pos = position
        if position < 0:
            position = 0
        elif position > 180:
            position = 180
        if prev_pos != position:
            logger.verbose("Limiting servo %s move from %.1f to %.1f", motor, prev_pos, position)

        acceleration = config.maximum_acceleration

        addr = self._tgt2addr(target_of_motor(motor))

        uuid = CanInterface.next_uuid()
        res = self._jc.ServoMove(addr, _motor_to_id(motor),
                                                       position,
                                                       velocity,
                                                       acceleration,
                                                       self._abs_or_rel.ABSOLUTE,
                                                       uuid)
        logger.debug("%s: servo move %.3f mm with v=%.3f mm/s**2 ; res=%s uuid=%s ; config=%s",
                     motor, position, velocity, res, uuid, config)
        return res == 0

    def _move_stepper_motor(
        self,
        motor: Motor,
        position: Union[float, Tuple[float, float]],
        config: StepperConfig,
        save_as_fixed: bool,
        relative: bool = False,
    ):
        """
        Move a stepper motor.
        
        Args:
            motor
            position: Either a position (float) or a (position, rate (%)) pair
            config: associated motor configuration
            save_as_fixed: To save the passed position as fixed.
            relative: Relative movement or absolute, default absolute.

        Returns:
            bool: True if successful else False
        """
        addr = self._tgt2addr(target_of_motor(motor))

        if isinstance(position, (float, int)):
            velocity = config.maximum_velocity
        elif isinstance(position, tuple) and len(position) == 2:
            velocity = float(position[1]) / 100.0 * config.maximum_velocity
            position = float(position[0])
        else:
            logger.error("Unhandled position type: %s ; value=%r", type(position), position)
            return False

        motor_axis_idx = _motor_to_axis_idx(motor)
        axis_prev_send_pos = self._prev_send_pos[motor_axis_idx]
        char_coord = "xyz"[motor_axis_idx]

        if save_as_fixed:
            if relative:
                # force absolute for save_as_fixed position,
                # this allows to only account 1 time for the possible auto-corrected drift
                relative = False
                position += axis_prev_send_pos

        corrected_position = position
        if save_as_fixed and self._auto_correct_motor_drift:
            axis_drift = self._motors_drift[motor_axis_idx]
            self._active_motors_drift = self._active_motors_drift.replace(**{char_coord: axis_drift})
            corrected_position = position - axis_drift
            # logger.debug("%s: corrected %.3f -> %.3f", motor, position, corrected_position)

        logger.verbose("%s: %s %s %.3f mm (corrected %.3f) with v=%.3f mm/s**2",
                     motor,
                       ("move", "save_as_fixed")[save_as_fixed],
                       ("absolute", "relative")[relative],
                       position, corrected_position, velocity)

        turns_position = mm_to_turns(corrected_position)
        turns_velocity = mm_to_turns(velocity)
        turns_acceleration = mm_to_turns(config.maximum_acceleration)

        if relative:
            assert not save_as_fixed
            motor_idx = _motor_to_axis_idx_map.get(motor, None)
            if motor_idx is not None:
                last_pos = self._last_positions[motor_idx]
                if not math.isfinite(last_pos):
                    logger.error("Motor %s: refusing relative movement with no last_pos known: %s",
                                 motor, last_pos)
                    return False
                tentative = last_pos + position
                if tentative < 0:
                    position -= tentative
                    turns_position = mm_to_turns(position)
                    logger.verbose("Limiting relative move to %s", position)
                elif tentative > _STEPPER_MAX_POS:
                    position = _STEPPER_MAX_POS - last_pos
                    turns_position = mm_to_turns(position)
                    logger.verbose("Limiting relative move to %s", position)
        else:
            # absolute move
            if turns_position < 0:
                logger.debug("limited turns_position to 0 ; was %.1f", turns_position)
                turns_position = 0
            elif turns_position > _STEPPER_MAX_TURNS:
                logger.debug("limited turns_position to max ; was %.1f", turns_position)
                turns_position = _STEPPER_MAX_TURNS

        uuid = CanInterface.next_uuid()
        res = self._jc.StepperMove(
            addr,
            _motor_to_id(motor),
            turns_position,
            turns_velocity,
            turns_acceleration,
            self._abs_or_rel.RELATIVE if relative else self._abs_or_rel.ABSOLUTE,
            save_as_fixed,
            uuid,
        )
        logger.debug("%s: StepperMove res=%s uuid=%s", motor, res, uuid)
        return res == 0

    def set_motor_x(self, position: float, *, relative: bool = False) -> bool:
        # NB: SET == saved-as-fixed:
        return self.move_motor_x(position, save_as_fixed=True, relative=relative)

    def move_motor_x(
        self,
        position: Union[float, Tuple[float, float]],
        save_as_fixed: bool = False,
        *,
        relative: bool = False,
    ) -> bool:
        """
         Move the X-direction motor

        Args:
            position: Either a position (float) or a (position, rate (%)) pair
            save_as_fixed: Save the position as a new fixed location for this motor
                If True then the position is only saved-as-fixed, the motor is not moved.
            relative: Relative movement or absolute, default absolute.

         Returns:
             bool: True if successful else False
         """
        return self._move_stepper_motor(Motor.PELLET_X_MOTOR, position, self.x_config,
                                        save_as_fixed=save_as_fixed, relative=relative)

    def set_motor_y(self, position, *, relative: bool = False) -> bool:
        return self.move_motor_y(position, save_as_fixed=True, relative=relative)

    def move_motor_y(self, position, save_as_fixed: bool = False, *, relative: bool = False) -> bool:
        """
         Move the Y-direction motor

         Args:
             position: Either a position (float) or a (position, rate (%)) pair
             save_as_fixed: Save the position as a new fixed location for this motor
                If True then the position is only saved-as-fixed, the motor is not moved.
             relative: Relative movement or absolute, default absolute.

         Returns:
             bool: True if successful else False
         """
        return self._move_stepper_motor(Motor.PELLET_Y_MOTOR, position, self.y_config,
                                        save_as_fixed=save_as_fixed, relative=relative)

    def set_motor_z(self, position, *, relative: bool = False) -> bool:
        return self.move_motor_z(position, save_as_fixed=True, relative=relative)

    def move_motor_z(self, position, save_as_fixed: bool = False, *, relative: bool = False) -> bool:
        """
         Move the Z-direction motor

         Args:
             position: Either a position (float) or a (position, rate (%)) pair
             save_as_fixed: Save the position as a new fixed location for this motor
                If True then the position is only saved-as-fixed, the motor is not moved.
             relative: Relative movement or absolute, default absolute.

         Returns:
             bool: True if successful else False
         """
        return self._move_stepper_motor(Motor.PELLET_Z_MOTOR, position, self.z_config,
                                        save_as_fixed=save_as_fixed, relative=relative)

    def fixed_position(self) -> bool:
        """
        Move the X, Y, Z motor to a fixed location, known by the device

        Returns:
            bool: True if successful else False
        """
        addr = self._tgt2addr(Target.PELLET_DEVICE)
        uuid = CanInterface.next_uuid()
        res = self._jc.SendToFixedXYZ(addr, uuid)
        logger.debug("SendToFixedXYZ res=%s uuid=%s", res, uuid)
        return res == 0

    def move_load_servo(self, position):
        """
        Move the load arm

        Args:
            position: Either a position (float) or a (position, rate (%)) pair

        Returns:
            bool: True if successful else False
        """
        return self._move_servo_motor(Motor.PELLET_LOAD_SERVO, position, self.load_config)

    def scoop_pellet(self) -> bool:
        """
        Move to scoop a pellet

        Returns:
            bool: True if successful else False
        """
        return self.move_load_servo(self.load_config.minimum_position)

    def retrieve_pellet(self) -> bool:
        """
        Move to retrieve a pellet

            Returns:
                bool: True if successful else False
        """
        return self.move_load_servo(self.load_config.maximum_position)

    def move_cover_servo(self, position):
        """
        Move the cover servo

        Args:
            position: Either a position (float) or a (position, rate (%)) pair

        Returns:
            bool: True if successful else False
        """
        return self._move_servo_motor(Motor.PELLET_COVER_SERVO, position, self.cover_config)

    def release_pellet(self) -> bool:
        """
        Open the cover so the pellet is visible to the animal

        Returns:
            bool: True if successful else False
        """
        return self.move_cover_servo(self.cover_config.minimum_position)

    def cover_pellet(self) -> bool:
        """
        Close the cover so the pellet is not visible to the animal

        Returns:
            bool: True if successful else False
        """
        return self.move_cover_servo(self.cover_config.maximum_position)

    def stepper_home(self, motor: Motor) -> bool:
        """
        Send the given motor (X, Y, or Z) to the 0 position

        Args:
            motor:

        Returns:
            bool: True if successful else False
        """
        logger.info(f"Homing Stepper Motor {motor_to_str(motor)}")
        if is_servo(motor):
            logger.warning("invalid stepper home motor: %s", motor)
            return False

        addr = self._tgt2addr(Target.PELLET_DEVICE)
        # Third arg - forward/rev. Go in forward direction if the non-zero locations are negative
        uuid = CanInterface.next_uuid()
        res = self._jc.StepperHome(addr, _motor_to_id(motor), uuid)
        logger.debug("%s StepperHome res=%s uuid=%s", motor, res, uuid)
        return res == 0

    def _write_stepper_config(self, config: StepperConfig) -> bool:
        """
        Update a stepper motor configuration on the target

`       Args:
            config: Configuration to update

        Returns:
            bool: True if successful else False
        """
        motor_id = _motor_to_id(config.motor)
        addr = self._tgt2addr(Target.PELLET_DEVICE)

        max_vel = mm_to_turns(config.maximum_velocity)
        max_acc = mm_to_turns(config.maximum_acceleration)
        home_vel = mm_to_turns(config.homing_velocity)

        uuid = CanInterface.next_uuid()
        res = self._jc.StepperCfgWrite(addr, motor_id,
                                    config.microsteps,
                                    config.steps_per_revolution,
                                    max_vel,
                                    max_acc,
                                    home_vel,
                                    config.flip_limit_orientation,
                                    uuid)
        logger.debug("StepperCfgWrite addr=%s config=%s: res=%s uuid=%s",
                     addr, config, res, uuid)
        if res != 0:
            logger.error(
                "stepper %s addr=%s config write failed", config.motor, addr)
            return False
        return True

    def _write_servo_config(self, config: ServoConfig) -> bool:
        """
        Update a servo motor configuration on the target

        Args:
            config: Configuration to update

        Returns:
            bool: True if successful else False
        """
        motor_id = _motor_to_id(config.motor)
        target = target_of_motor(config.motor)

        addr = self._tgt2addr(target)

        uuid = CanInterface.next_uuid()
        res = self._jc.ServoCfgWrite(addr, motor_id,
                                  config.minimum_position,
                                  config.maximum_position,
                                  config.minimum_pwm_duration,
                                  config.maximum_pwm_duration,
                                  config.maximum_velocity,
                                  config.maximum_acceleration,
                                  uuid)
        logger.debug("ServoCfgWrite addr=%s config=%s: res=%s uuid=%s", addr, config, res, uuid)
        if res != 0:
            logger.error("servo %s %s config write failed", addr, motor_id)
            return False
        return True

    def request_motor_config(self, motor: Motor) -> bool:
        """
        Request a stepper motor configuration from a target

        Args:
            motor:

        Returns:
            bool: True if successful else False
        """
        target = target_of_motor(motor)
        msg = self._jerrycan_cfg_msg_cls()
        if is_servo(motor):
            msg.type = self._jerrycan_cfg_msg_cls.Type.SERVO
            msg.servo.motor_id = _motor_to_id(motor)
        else:
            msg.type = self._jerrycan_cfg_msg_cls.Type.STEPPER
            msg.stepper.motor_id = _motor_to_id(motor)

        addr = self._tgt2addr(target)
        res = self._jc.CfgRead(addr, msg)
        logger.debug("motor=%s: tentative request addr=%s tgt=%s => res=%s", motor, addr, target, res)
        return res == 0

    def send_heartbeat(self) -> bool:
        """
        Send a heartbeat message to the target; causes an LED to blink briefly.

        Returns:
            bool: True if successful else False
        """
        return self._jc.Heartbeat() == 0

    def set_digital_output(self, gpio: DigitalOutputs, state: bool) -> bool:
        """
        Set a digital output value on the pellet device

        Args:
            gpio: Digital output to change
            state: On (True) or Off (False) state

         Returns:
             bool: True if successful else False
         """

        # These values are based on the order and listing in the DTS files for
        # each board.
        if gpio == DigitalOutputs.STIMULUS_1:
            gpio_id = 4
        elif gpio == DigitalOutputs.STIMULUS_2:
            gpio_id = 5
        elif gpio == DigitalOutputs.STIMULUS_3:
            gpio_id = 6
        elif gpio == DigitalOutputs.STIMULUS_4:
            gpio_id = 7
        else:
            raise ValueError(f"Unhandled digital output {gpio}")

        addr = self._tgt2addr(Target.PELLET_DEVICE)
        uuid = CanInterface.next_uuid()
        res = self._jc.GPIOWrite(addr, 0, gpio_id, state, uuid)
        logger.debug("set_digital_output addr=%s gpio_id=%s state=%s res=%s uuid=%s",
                     addr, gpio_id, state, res, uuid)
        return res == 0

    def emit_tone(self, frequency: int, duration_ms: int) -> bool:
        """
        Emit a tone for the animal to hear

        Args:
            frequency: Frequency of tone (Hz)
            duration_ms: Duration of tone (milliseconds)

        Returns:
            bool: True if successful else False
        """
        addr = self._tgt2addr(Target.PELLET_DEVICE)
        uuid = CanInterface.next_uuid()
        res = self._jc.ToneWrite(addr, 0, frequency, duration_ms, uuid)
        logger.debug("emit_tone addr=%s freq=%s duration_ms=%s res=%s uuid=%s",
                     addr, frequency, duration_ms, res, uuid)
        return res == 0

    def set_analog_output(self, channel: AnalogOutputs, millivolts: int) -> bool:
        """
        Set an analog output on the pellet device

        Args:
            channel: Channel #
            millivolts: desired voltage output (millivolts)

        Returns:
            bool: True if successful else False
        """
        if channel == AnalogOutputs.STATUS_OUT:
            channel = 0
        else:
            raise ValueError(f"Unhandled channel {channel}")

        addr = self._tgt2addr(Target.PELLET_DEVICE)
        return self._jc.AnalogOutWrite(addr, channel, millivolts, CanInterface.next_uuid()) == 0

    def set_color_led(self, red_percent: int, green_percent: int, blue_percent: int) -> bool:
        """
        Set the colors of a 3-color LED.

        Args:
            red_percent: % in red
            green_percent: % in green
            blue_percent: % in blue

        Returns:
            bool: True if successful else False
        """
        addr = self._tgt2addr(Target.PELLET_DEVICE)
        rc = self._jc.RGBLEDWrite(addr,
                                    red_percent,
                                    green_percent,
                                    blue_percent,
                                    CanInterface.next_uuid())
        return rc == 0

    def request_version(self) -> bool:
        """
        Request the versions of the required board firmware

        Returns:
            bool: True if successful else False
        """
        rc = 0
        for target in self.required_targets:
            addr = self._tgt2addr(target)
            rc = self._jc.BootloaderCommand(addr, self._bootloader_cmd_cls.SubCommand.VERSION)
            if rc != 0:
                break
        return rc == 0

    def pulse_digital_output(
        self,
        gpio: DigitalOutputs,
        duration_us: int,
    ) -> bool:
        """Request the firmware-owned, guaranteed-return-low STIM3 pulse."""
        if DigitalOutputs(gpio) is not DigitalOutputs.STIMULUS_4:
            raise ValueError("Finite pulse output currently supports STIM3 only")
        duration_us = int(duration_us)
        if not 100 <= duration_us <= 5_000_000:
            raise ValueError("STIM3 pulse duration must be within 100 us..5 s")
        pulse = getattr(self._jc, "GPIOPulse", None)
        if pulse is None:
            raise RuntimeError("Pellet firmware/transport does not support finite GPIO pulse")
        addr = self._tgt2addr(Target.PELLET_DEVICE)
        uuid = CanInterface.next_uuid()
        # Physical STIM3 = the fourth logical stimulus output = GPIO 0:7.
        return pulse(addr, 0, 7, duration_us, uuid) == 0

    def request_capabilities(self) -> bool:
        request = getattr(self._jc, "RequestCapabilities", None)
        if request is None or self.pellet_address is None:
            return False
        return request(self.pellet_address) == 0

    def request_clock_sync(self) -> bool:
        request = getattr(self._jc, "TimeSync", None)
        if request is None or self.pellet_address is None:
            return False
        request_id = self._next_time_sync_request_id
        self._next_time_sync_request_id = (request_id + 1) & 0xFFFFFFFF
        host_send_ns = time.perf_counter_ns()
        self._time_sync_requests[request_id] = host_send_ns
        if request(self.pellet_address, request_id, host_send_ns) != 0:
            self._time_sync_requests.pop(request_id, None)
            return False
        while len(self._time_sync_requests) > 128:
            self._time_sync_requests.pop(next(iter(self._time_sync_requests)))
        return True

    @staticmethod
    def _assign_timestamp_ns(message):
        return time.time_ns()

    def _translate(self, message) -> Optional[Any]:
        """
        Translate from a JerryCANCmd class to a class specific to the data type received

        Args:
            message: jerrycan_msg_t type; interpret using message.type

        Returns:
            Populated class type (see device_interface.py) or None
        """
        handler = self._handlers.get(message.type, None)
        if handler is None:
            logger.warning("Unhandled message type: %s", message.type)
            return None
        res = handler(message)
        if res is not None:
            if not isinstance(res, Source):
                raise TypeError(f"CAN decoder returned unsupported value: {type(res)}")
            res.timestamp_ns = self._get_timestamp_ns(message)
            res.index = self._get_index(message)
            self._apply_board_timing(res, message)
        return res

    def _apply_board_timing(self, source: Source, message) -> None:
        source.event_perf_time = source.index / 1e9
        if not bool(getattr(message, "board_timing_valid", False)):
            return
        timing = message.timing
        source.board_boot_id = int(timing.boot_id)
        source.board_sequence = int(timing.sequence)
        source.board_time_us = int(timing.board_time_us)
        kind = getattr(timing.kind, "name", str(timing.kind))
        source.board_timestamp_kind = str(kind).lower()
        source.board_sequence_status = self._board_sequence_tracker.observe(
            source.board_boot_id, source.board_sequence,
        )
        estimate = self._board_clock_model.estimate
        if (
            source.board_sequence_status.get("reboot")
            and (estimate is None or estimate.boot_id != source.board_boot_id)
        ):
            self._board_clock_model.reset(source.board_boot_id)
        aligned = self._board_clock_model.align(source.board_time_us)
        if aligned is None:
            return
        source.board_aligned_perf_time = aligned["perf_time"]
        source.board_clock_model_id = aligned["model_id"]
        source.board_clock_uncertainty_seconds = aligned["uncertainty_seconds"]
        source.event_perf_time = aligned["perf_time"]
        source.timestamp_method = "board_clock_affine"
        source.timing_confidence = "board_timestamp"

    def _translate_time_sync(self, message) -> BoardTimeSync:
        response = message.time_sync_response
        request_id = int(response.request_id)
        host_send_ns = self._time_sync_requests.pop(request_id, None)
        model = {}
        timing = getattr(message, "timing", None)
        if (
            host_send_ns is not None
            and bool(getattr(message, "board_timing_valid", False))
            and timing is not None
        ):
            estimate = self._board_clock_model.observe(
                request_id=request_id,
                host_send_perf_ns=host_send_ns,
                board_receive_time_us=int(response.request_receive_time_us),
                board_send_time_us=int(response.response_queue_time_us),
                host_receive_perf_ns=self._get_index(message),
                boot_id=int(timing.boot_id),
            )
            model = dataclasses.asdict(estimate)
        return BoardTimeSync(
            target=_addr2tgt(message.dst_id),
            request_id=request_id,
            request_receive_time_us=int(response.request_receive_time_us),
            response_queue_time_us=int(response.response_queue_time_us),
            clock_model=model,
        )

    @staticmethod
    def _translate_capabilities(message) -> BoardCapabilities:
        response = message.capabilities_response
        return BoardCapabilities(
            target=_addr2tgt(message.dst_id),
            wire_schema_version=int(response.wire_schema_version),
            capabilities=int(response.capabilities),
            boot_id=int(response.boot_id),
        )

    @staticmethod
    def _translate_gpio_pulse_status(message) -> DigitalPulseStatus:
        status = message.gpio_pulse_status
        phase = getattr(status.phase, "name", str(status.phase)).lower()
        return DigitalPulseStatus(
            target=_addr2tgt(message.dst_id),
            channel=DigitalOutputs.STIMULUS_4,
            duration_us=int(status.duration_us),
            phase=phase,
            error=int(status.error),
        )

    def _translate_bootloader(self, message) -> Optional[Version]:
        """
        Translate bootloader response messages.

        Args:
            message: JerryCANMsg with bootloader response data

        Returns:
            Version object if the bootloader response is a version request, None otherwise
        """
        if message.bootloader_response.type == self._bootloader_cmd_cls.SubCommand.VERSION:
            target = _addr2tgt(message.dst_id)
            if hasattr(message.bootloader_response.version, "running_major"):
                # pyjerrycan < 1.2.0
                version_str = f"{target_to_str(target)}: {message.bootloader_response.version.running_major}." \
                              f"{message.bootloader_response.version.running_minor}." \
                              f"{message.bootloader_response.version.running_patch}"
            else:
                # pyjerrycan >= 1.2.0
                version_str = f"{target_to_str(target)}: {message.bootloader_response.version.running_version_major}." \
                              f"{message.bootloader_response.version.running_version_minor}." \
                              f"{message.bootloader_response.version.running_version_patch}"
            return Version(target, version=version_str)
        return None

    def _translate_config(self, message) -> Optional[Union[ServoConfig, StepperConfig]]:
        """
        Translate configuration response messages for servo or stepper motors.

        Args:
            message: JerryCANMsg with configuration data

        Returns:
            ServoConfig or StepperConfig object depending on the message type
        """
        if message.cfg_response.type == self._jerrycan_cfg_msg_cls.Type.SERVO:
            return self._translate_servo_config(message)
        elif message.cfg_response.type == self._jerrycan_cfg_msg_cls.Type.STEPPER:
            return self._translate_stepper_config(message)
        logger.warning("Unknown config type: %s", message.cfg_response.type)
        return None

    def _translate_servo_config(self, message) -> ServoConfig:
        """
        Translate servo configuration response messages.

        Args:
            message: JerryCANMsg with servo configuration data

        Returns:
            ServoConfig object with updated settings
        """
        target = _addr2tgt(message.dst_id)
        r_cfg = message.cfg_response.servo
        motor = _id_to_motor(target, True, r_cfg.motor_id)

        config = self.get_motor_configuration(motor)

        # Update configuration with values from the message
        config.minimum_position = r_cfg.min_position
        config.maximum_position = r_cfg.max_position
        config.minimum_pwm_duration = r_cfg.min_pwm_duration_us
        config.maximum_pwm_duration = r_cfg.max_pwm_duration_us
        config.maximum_velocity = r_cfg.motor_max_velocity
        config.maximum_acceleration = r_cfg.motor_max_acceleration

        return config

    def _translate_stepper_config(self, message) -> StepperConfig:
        """
        Translate stepper configuration response messages.

        Args:
            message: JerryCANMsg with stepper configuration data

        Returns:
            StepperConfig object with updated settings
        """
        target = _addr2tgt(message.dst_id)
        r_cfg = message.cfg_response.stepper
        motor = _id_to_motor(target, False, r_cfg.motor_id)

        config = self.get_motor_configuration(motor)

        # Update configuration with values from the message
        config.microsteps = r_cfg.microsteps
        config.steps_per_revolution = r_cfg.steps_per_revolution
        config.flip_limit_orientation = r_cfg.flip_limit_orientation
        config.maximum_velocity = turns_to_mm(r_cfg.motor_max_velocity)
        config.maximum_acceleration = turns_to_mm(r_cfg.motor_max_acceleration)
        config.homing_velocity = turns_to_mm(r_cfg.homing_velocity)

        return config

    def _translate_gpio(self, message) -> Optional[PelletDigitalInputs]:
        """
        Translate GPIO read response messages.

        Args:
            message: JerryCANMsg with GPIO state data

        Returns:
            PelletDigitalInputs for the pellet board, otherwise None
        """
        prev = self._last_gpio_status_perf.get(message.dst_id, None)
        state = message.gpio_read.state
        if _is_pellet_by_addr(message.dst_id):
            tgt = Target.PELLET_DEVICE
            digital_inputs = PelletDigitalInputs(
                target=_addr2tgt(message.dst_id),
                stimulus_1=state & 0x010 == 0x010,
                stimulus_2=state & 0x020 == 0x020,
                stimulus_3=state & 0x040 == 0x040,
                stimulus_4=state & 0x080 == 0x080,
            )
        else:
            logger.warning("unknown gpio msg.dst_id: %s", message.dst_id)
            return None

        if state != prev:
            logger.verbose(
                "%s: digital_inputs: dst_id=%s state=%s (prev=%s)",
                tgt, message.dst_id, message.gpio_read.state, prev
            )
        self._last_gpio_status_perf[message.dst_id] = state
        return digital_inputs

    @staticmethod
    def _translate_analog_out(message) -> Optional[AnalogOutput]:
        """
        Translate analog output response messages.

        Args:
            message: JerryCANMsg with analog output data

        Returns:
            AnalogOutput object if it's from the pellet device, None otherwise
        """
        if message.analog_out.instance == 0 and _is_pellet_by_addr(message.dst_id):
            return AnalogOutput(
                target=_addr2tgt(message.dst_id),
                status_out_mv=message.analog_out.value_mv,
            )
        return None

    def _translate_servo_status(self, message) -> Optional[ServoStatus]:
        """
        Translate servo status response messages.

        Args:
            message: JerryCANMsg with servo status data

        Returns:
            ServoStatus object if the motor is recognized, None otherwise
        """
        target = _addr2tgt(message.dst_id)
        motor = _id_to_motor(target, True, message.servo_status.motor_id)
        if motor == Motor.NONE:
            return None
        self._handle_motor_status_age(motor)
        return ServoStatus(target, motor, self.round_float(message.servo_status.position))

    def _handle_stepper_status(self, message):
        status = self._translate_stepper_status(message)
        if status is None:
            return None
        motor_idx = _motor_to_axis_idx_map.get(status.motor, None)
        if motor_idx is not None:
            new_pos = list(self._last_positions)
            new_pos[motor_idx] = status.position
            new_send_pos = list(self._prev_send_pos)
            self._last_positions = new_pos
            new_send_pos[motor_idx] = status.send_position
            self._prev_send_pos = Offset3DTuple(*new_send_pos)
        return status

    def _translate_stepper_status(self, message) -> Optional[StepperStatus]:
        """
        Translate stepper status response messages.

        Args:
            message: JerryCANMsg with stepper status data

        Returns:
            StepperStatus object if the motor is recognized, None otherwise
        """
        target = _addr2tgt(message.dst_id)
        motor = _id_to_motor(target, False, message.stepper_status.motor_id)
        if motor == Motor.NONE:
            logger.warning("_translate_stepper_status: target=%s motor=%s dst_id=%s motor_id=%s",
                           target, motor, message.dst_id, message.stepper_status.motor_id)
            return None
        self._handle_motor_status_age(motor)
        motor_axis_idx = _motor_to_axis_idx(motor)
        motor_send_pos = turns_to_mm(message.stepper_status.send_position)
        if self._auto_correct_motor_drift:
            motor_send_pos += self._active_motors_drift[motor_axis_idx]
        round_val = self.round_float
        return StepperStatus(
            target=target,
            motor=motor,
            position=round_val(turns_to_mm(message.stepper_status.position)),
            send_position=round_val(motor_send_pos),
            limit_switch=message.stepper_status.limit_switch == 1,
            position_error=self._motors_drift_error[motor_axis_idx],
        )

    def servo_attach(self, motor: Motor):
        addr = self._tgt2addr(target_of_motor(motor))
        motor_id = _motor_to_id(motor)
        res = self._jc.ServoAttach(addr, motor_id)
        if res != 0:
            logger.error("%s: ServoAttach failed: %s", motor, res)
        return res == 0

    def servo_detach(self, motor: Motor):
        addr = self._tgt2addr(target_of_motor(motor))
        motor_id = _motor_to_id(motor)
        res = self._jc.ServoDetach(addr, motor_id)
        if res != 0:
            logger.error("%s: ServoDetach failed: %s", motor, res)
        return res == 0

    def board_reboot(self, target: Target):
        addr = self._tgt2addr(target)
        rc = self._jc.BootloaderCommand(addr, self._bootloader_cmd_cls.SubCommand.REBOOT)
        if rc != 0:
            logger.error("board_reboot failed: SendMessage(dst_id=%s): rc=%s", addr, rc)
        return rc == 0
