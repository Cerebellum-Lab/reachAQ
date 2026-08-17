from __future__ import annotations

import dataclasses
import enum
import errno
import logging
import socket
import struct
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .can_transport import CanTransportConfiguration, CanTransportKind


JERRYCAN_ACTUAL_PAYLOAD_SIZE = 64
CAN_FD_MTU = 72
JERRYCAN_TIMING_TRAILER_TAG = 0xA7
JERRYCAN_TIMING_TRAILER_VERSION = 1
_TIMING_TRAILER = struct.Struct("<BBBBIIQ")

logger = logging.getLogger(__name__)


class CanTransportReadError(RuntimeError):
    """A classified failure raised by the physical CAN receive path."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        error_code: Optional[int] = None,
        diagnostics=None,
    ):
        super().__init__(message)
        self.category = category
        self.error_code = error_code
        self.diagnostics = dict(diagnostics or {})


def _exception_errno(exc: BaseException) -> Optional[int]:
    current: Optional[BaseException] = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("errno", "error_code"):
            value = getattr(current, attribute, None)
            if isinstance(value, int):
                return value
        current = current.__cause__ or current.__context__
    return None


def _classify_receive_error(exc: BaseException) -> Tuple[str, Optional[int]]:
    error_code = _exception_errno(exc)
    message = str(exc).lower()
    if error_code == errno.ENETDOWN:
        return "network_down", error_code
    if error_code in {errno.ENODEV, errno.ENXIO, errno.ENOTCONN}:
        return "device_removed", error_code
    if "bus-off" in message or "bus off" in message:
        return "bus_off", error_code
    if exc.__class__.__name__ == "CanOperationError":
        return "can_operation_error", error_code
    return "receive_error", error_code


class JerryCANCmdType(enum.IntEnum):
    ESTOP = 0x00
    HEARTBEAT = 0x3F
    STATUS = 0x01
    STEPPER_MOVE = 0x02
    SERVO_MOVE = 0x03
    STEPPER_HOME = 0x04
    CFG_WRITE = 0x05
    CFG_READ = 0x06
    CFG_RESPONSE = 0x07
    PRESSURE_READ = 0x08
    TEMP_HUM_READ = 0x09
    GPIO_READ = 0x0A
    GPIO_WRITE = 0x0B
    TONE = 0x0C
    ANALOG_OUT = 0x0D
    RESERVED_0E = 0x0E
    DOOR_SENSOR = 0x0F
    AUDIO_MAGNITUDE_DATA_BEGIN = 0x10
    AUDIO_MAGNITUDE_DATA_CONT = 0x11
    AUDIO_MAGNITUDE_DATA_END = 0x12
    RGB_LED = 0x13
    RESERVED_14 = 0x14
    STEPPER_STATUS = 0x16
    SERVO_STATUS = 0x17
    BOOTLOADER_COMMAND = 0x18
    BOOTLOADER_RESPONSE = 0x19
    BOOTLOADER_DATA = 0x1A
    DELAY = 0x1B
    FIXED_XYZ = 0x1C
    SERVO_ATTACH = 0x1D
    SERVO_DETACH = 0x1E
    TIME_SYNC_REQUEST = 0x1F
    TIME_SYNC_RESPONSE = 0x20
    GPIO_PULSE = 0x21
    GPIO_PULSE_STATUS = 0x22
    CAPABILITIES_REQUEST = 0x23
    CAPABILITIES_RESPONSE = 0x24
    ACKNOWLEDGE = 0x30


class AbsOrRel(enum.IntEnum):
    ABSOLUTE = 0
    RELATIVE = 1


class JerryCANBootloaderCmd:
    class SubCommand(enum.IntEnum):
        VERSION = 0
        START = 1
        END = 2
        REBOOT = 3
        FINALIZE = 4
        ACK = 5
        NACK = 6


@dataclasses.dataclass
class _EmptyCommand:
    rsvd: int = 0


@dataclasses.dataclass
class StepperMoveCommand:
    motor_id: int = 0
    save: bool = False
    abs_or_rel: AbsOrRel = AbsOrRel.ABSOLUTE
    position: float = 0.0
    max_velocity: float = 0.0
    max_acceleration: float = 0.0


ServoMoveCommand = StepperMoveCommand


@dataclasses.dataclass
class StepperHomeCommand:
    motor_id: int = 0


@dataclasses.dataclass
class ServoAttachCommand:
    motor_id: int = 0


ServoDetachCommand = ServoAttachCommand


@dataclasses.dataclass
class ServoCfg:
    motor_id: int = 0
    error: int = 0
    min_position: float = 0.0
    max_position: float = 0.0
    min_pwm_duration_us: float = 0.0
    max_pwm_duration_us: float = 0.0
    motor_max_velocity: float = 0.0
    motor_max_acceleration: float = 0.0


@dataclasses.dataclass
class StepperCfg:
    motor_id: int = 0
    error: int = 0
    flip_limit_orientation: bool = False
    microsteps: int = 0
    steps_per_revolution: float = 0.0
    motor_max_velocity: float = 0.0
    motor_max_acceleration: float = 0.0
    homing_velocity: float = 0.0


class JerryCANCfgMsg:
    class Type(enum.IntEnum):
        STEPPER = 0
        SERVO = 1

    ServoCfg = ServoCfg
    StepperCfg = StepperCfg

    def __init__(self):
        self.type = self.Type.STEPPER
        self.servo = ServoCfg()
        self.stepper = StepperCfg()


@dataclasses.dataclass
class PressureRead:
    instance: int = 0
    pressure: int = 0


@dataclasses.dataclass
class TempHumRead:
    instance: int = 0
    temperature: int = 0
    humidity: int = 0


@dataclasses.dataclass
class GPIORead:
    instance: int = 0
    state: int = 0


@dataclasses.dataclass
class GPIOWrite:
    instance: int = 0
    gpio_idx: int = 0
    state: bool = False


@dataclasses.dataclass
class Tone:
    instance: int = 0
    frequency_hz: int = 0
    duration_ms: int = 0


@dataclasses.dataclass
class AnalogOut:
    instance: int = 0
    value_mv: int = 0


@dataclasses.dataclass
class Doors:
    door1: int = 0
    door2: int = 0
    door3: int = 0
    ext_button: int = 0


@dataclasses.dataclass
class RGBLED:
    red: int = 0
    green: int = 0
    blue: int = 0


@dataclasses.dataclass
class AudioDataCmd:
    stream_id: int = 0


@dataclasses.dataclass
class AudioData:
    magnitudes: List[float] = dataclasses.field(default_factory=lambda: [0.0] * 16)


@dataclasses.dataclass
class StepperStatus:
    motor_id: int = 0
    status: int = 0
    homing_status: int = 0
    limit_switch: int = 0
    position: float = 0.0
    send_position: float = 0.0


@dataclasses.dataclass
class ServoStatus:
    motor_id: int = 0
    status: int = 0
    position: float = 0.0


@dataclasses.dataclass
class BootloaderCommand:
    type: JerryCANBootloaderCmd.SubCommand = JerryCANBootloaderCmd.SubCommand.VERSION


@dataclasses.dataclass
class BootloaderVersion:
    running_version_major: int = 0
    running_version_minor: int = 0
    running_version_patch: int = 0
    slot1_version_major: int = 0
    slot1_version_minor: int = 0
    slot1_version_patch: int = 0


@dataclasses.dataclass
class BootloaderStatus:
    active: int = 0
    bytes_written: int = 0


@dataclasses.dataclass
class BootloaderResponse:
    type: JerryCANBootloaderCmd.SubCommand = JerryCANBootloaderCmd.SubCommand.VERSION
    version: BootloaderVersion = dataclasses.field(default_factory=BootloaderVersion)
    status: BootloaderStatus = dataclasses.field(default_factory=BootloaderStatus)


@dataclasses.dataclass
class BootloaderData:
    data: bytes = bytes(JERRYCAN_ACTUAL_PAYLOAD_SIZE)


@dataclasses.dataclass
class DelayCommand:
    delay: int = 0


@dataclasses.dataclass
class FixedXyzCommand:
    rsvd: int = 0


@dataclasses.dataclass
class Acknowledge:
    error: int = 0


class JerryCANTimestampKind(enum.IntEnum):
    TX_QUEUE = 0
    OBSERVED = 1
    COMMAND_RECEIVED = 2
    PHYSICAL_START = 3
    COMPLETED = 4
    SAMPLED = 5


@dataclasses.dataclass
class JerryCANTiming:
    version: int = 0
    kind: JerryCANTimestampKind = JerryCANTimestampKind.TX_QUEUE
    flags: int = 0
    boot_id: int = 0
    sequence: int = 0
    board_time_us: int = 0


@dataclasses.dataclass
class TimeSyncRequest:
    request_id: int = 0
    host_send_perf_ns: int = 0


@dataclasses.dataclass
class TimeSyncResponse:
    request_id: int = 0
    request_receive_time_us: int = 0
    response_queue_time_us: int = 0


@dataclasses.dataclass
class CapabilitiesResponse:
    wire_schema_version: int = 0
    capabilities: int = 0
    boot_id: int = 0


@dataclasses.dataclass
class GPIOPulse:
    instance: int = 0
    gpio_idx: int = 0
    duration_us: int = 0


@dataclasses.dataclass
class GPIOPulseStatus(GPIOPulse):
    phase: int = 0
    error: int = 0


class JerryCANMsg:
    def __init__(self):
        self.type = JerryCANCmdType.HEARTBEAT
        self.dst_id = 0
        self.timestamp_ns = 0
        self.index = 0
        self.wire_length = 0
        self.board_timing_valid = False
        self.timing = JerryCANTiming()
        self.uuid = 0
        self.estop = _EmptyCommand()
        self.status = _EmptyCommand()
        self.heartbeat = _EmptyCommand()
        self.stepper_move = StepperMoveCommand()
        self.servo_move = ServoMoveCommand()
        self.stepper_home = StepperHomeCommand()
        self.servo_attach = ServoAttachCommand()
        self.servo_detach = ServoDetachCommand()
        self.cfg_response = JerryCANCfgMsg()
        self.cfg_read = JerryCANCfgMsg()
        self.cfg_write = JerryCANCfgMsg()
        self.stepper_status = StepperStatus()
        self.servo_status = ServoStatus()
        self.pressure_read = PressureRead()
        self.temp_hum_read = TempHumRead()
        self.gpio_read = GPIORead()
        self.gpio_write = GPIOWrite()
        self.tone = Tone()
        self.analog_out = AnalogOut()
        self.rgb_led = RGBLED()
        self.doors = Doors()
        self.audio_data_cmd = AudioDataCmd()
        self.audio_data = AudioData()
        self.bootloader_command = BootloaderCommand()
        self.bootloader_response = BootloaderResponse()
        self.bootloader_data = BootloaderData()
        self.delay = DelayCommand()
        self.fixed_xyz = FixedXyzCommand()
        self.ack = Acknowledge()
        self.time_sync_request = TimeSyncRequest()
        self.time_sync_response = TimeSyncResponse()
        self.capabilities_response = CapabilitiesResponse()
        self.gpio_pulse = GPIOPulse()
        self.gpio_pulse_status = GPIOPulseStatus()


_PAYLOAD_SIZES = {
    JerryCANCmdType.ESTOP: 1,
    JerryCANCmdType.HEARTBEAT: 1,
    JerryCANCmdType.STATUS: 8,
    JerryCANCmdType.STEPPER_MOVE: 13,
    JerryCANCmdType.SERVO_MOVE: 13,
    JerryCANCmdType.SERVO_ATTACH: 1,
    JerryCANCmdType.SERVO_DETACH: 1,
    JerryCANCmdType.STEPPER_HOME: 1,
    JerryCANCmdType.CFG_WRITE: 28,
    JerryCANCmdType.CFG_RESPONSE: 28,
    JerryCANCmdType.CFG_READ: 28,
    JerryCANCmdType.STEPPER_STATUS: 12,
    JerryCANCmdType.SERVO_STATUS: 6,
    JerryCANCmdType.PRESSURE_READ: 5,
    JerryCANCmdType.TEMP_HUM_READ: 5,
    JerryCANCmdType.GPIO_READ: 5,
    JerryCANCmdType.GPIO_WRITE: 4,
    JerryCANCmdType.TONE: 5,
    JerryCANCmdType.ANALOG_OUT: 3,
    JerryCANCmdType.RESERVED_0E: 5,
    JerryCANCmdType.AUDIO_MAGNITUDE_DATA_BEGIN: 4,
    JerryCANCmdType.AUDIO_MAGNITUDE_DATA_CONT: 64,
    JerryCANCmdType.AUDIO_MAGNITUDE_DATA_END: 4,
    JerryCANCmdType.RESERVED_14: 1,
    JerryCANCmdType.RGB_LED: 3,
    JerryCANCmdType.DOOR_SENSOR: 1,
    JerryCANCmdType.BOOTLOADER_COMMAND: 1,
    JerryCANCmdType.BOOTLOADER_RESPONSE: 7,
    JerryCANCmdType.BOOTLOADER_DATA: 64,
    JerryCANCmdType.DELAY: 2,
    JerryCANCmdType.FIXED_XYZ: 1,
    JerryCANCmdType.TIME_SYNC_REQUEST: 12,
    JerryCANCmdType.TIME_SYNC_RESPONSE: 20,
    JerryCANCmdType.GPIO_PULSE: 7,
    JerryCANCmdType.GPIO_PULSE_STATUS: 12,
    JerryCANCmdType.CAPABILITIES_REQUEST: 1,
    JerryCANCmdType.CAPABILITIES_RESPONSE: 9,
    JerryCANCmdType.ACKNOWLEDGE: 4,
}


def _payload_size(message_type: JerryCANCmdType) -> int:
    return _PAYLOAD_SIZES.get(JerryCANCmdType(message_type), 0)


def encode_can_id(message_type: JerryCANCmdType, dst_id: int) -> int:
    return ((int(message_type) & 0x3F) << 5) | (dst_id & 0x1F)


def _jerrycan_filters():
    """Kernel-friendly filters for standard JerryCAN command identifiers."""
    return [
        {
            "can_id": int(message_type) << 5,
            "can_mask": 0x7E0,
            "extended": False,
        }
        for message_type in JerryCANCmdType
    ]


def _socketcan_mtu(channel: str) -> Optional[int]:
    try:
        return int(Path("/sys/class/net", channel, "mtu").read_text().strip())
    except (OSError, ValueError):
        return None


def _pack_stepper_move(command: StepperMoveCommand) -> bytes:
    flags = (command.motor_id & 0x03)
    flags |= (1 if command.save else 0) << 2
    flags |= (int(command.abs_or_rel) & 0x01) << 7
    return struct.pack(
        "<Bfff",
        flags,
        float(command.position),
        float(command.max_velocity),
        float(command.max_acceleration),
    )


def _pack_cfg(cfg: JerryCANCfgMsg) -> bytes:
    cfg_type = JerryCANCfgMsg.Type(cfg.type)
    if cfg_type == JerryCANCfgMsg.Type.SERVO:
        first = (cfg.servo.motor_id & 0x03) | ((cfg.servo.error & 0x01) << 2)
        return struct.pack(
            "<BBhffffff",
            int(cfg_type),
            first,
            0,
            float(cfg.servo.min_position),
            float(cfg.servo.max_position),
            float(cfg.servo.min_pwm_duration_us),
            float(cfg.servo.max_pwm_duration_us),
            float(cfg.servo.motor_max_velocity),
            float(cfg.servo.motor_max_acceleration),
        )
    first = (cfg.stepper.motor_id & 0x03) | ((cfg.stepper.error & 0x01) << 2)
    first |= (1 if cfg.stepper.flip_limit_orientation else 0) << 3
    content = struct.pack(
        "<BBHffff",
        int(cfg_type),
        first,
        int(cfg.stepper.microsteps),
        float(cfg.stepper.steps_per_revolution),
        float(cfg.stepper.motor_max_velocity),
        float(cfg.stepper.motor_max_acceleration),
        float(cfg.stepper.homing_velocity),
    )
    return content + bytes(28 - len(content))


def _pack_payload(message: JerryCANMsg) -> bytes:
    message_type = JerryCANCmdType(message.type)
    if message_type == JerryCANCmdType.ESTOP:
        return struct.pack("<B", int(message.estop.rsvd))
    if message_type == JerryCANCmdType.HEARTBEAT:
        return struct.pack("<B", int(message.heartbeat.rsvd))
    if message_type == JerryCANCmdType.STEPPER_MOVE:
        return _pack_stepper_move(message.stepper_move)
    if message_type == JerryCANCmdType.SERVO_MOVE:
        return _pack_stepper_move(message.servo_move)
    if message_type == JerryCANCmdType.SERVO_ATTACH:
        return struct.pack("<B", int(message.servo_attach.motor_id))
    if message_type == JerryCANCmdType.SERVO_DETACH:
        return struct.pack("<B", int(message.servo_detach.motor_id))
    if message_type == JerryCANCmdType.STEPPER_HOME:
        return struct.pack("<B", int(message.stepper_home.motor_id) & 0x7F)
    if message_type == JerryCANCmdType.CFG_WRITE:
        return _pack_cfg(message.cfg_write)
    if message_type == JerryCANCmdType.CFG_READ:
        return _pack_cfg(message.cfg_read)
    if message_type == JerryCANCmdType.GPIO_WRITE:
        return struct.pack(
            "<BHB",
            int(message.gpio_write.instance),
            int(message.gpio_write.gpio_idx),
            1 if message.gpio_write.state else 0,
        )
    if message_type == JerryCANCmdType.TONE:
        return struct.pack("<BHH", int(message.tone.instance), int(message.tone.frequency_hz),
                           int(message.tone.duration_ms))
    if message_type == JerryCANCmdType.ANALOG_OUT:
        return struct.pack("<BH", int(message.analog_out.instance), int(message.analog_out.value_mv))
    if message_type == JerryCANCmdType.RGB_LED:
        return struct.pack("<BBB", int(message.rgb_led.red), int(message.rgb_led.green),
                           int(message.rgb_led.blue))
    if message_type == JerryCANCmdType.BOOTLOADER_COMMAND:
        return struct.pack("<B", int(message.bootloader_command.type))
    if message_type == JerryCANCmdType.BOOTLOADER_DATA:
        return bytes(message.bootloader_data.data).ljust(JERRYCAN_ACTUAL_PAYLOAD_SIZE, b"\0")[:64]
    if message_type == JerryCANCmdType.DELAY:
        return struct.pack("<H", int(message.delay.delay))
    if message_type == JerryCANCmdType.FIXED_XYZ:
        return struct.pack("<B", int(message.fixed_xyz.rsvd))
    if message_type == JerryCANCmdType.TIME_SYNC_REQUEST:
        return struct.pack(
            "<IQ", message.time_sync_request.request_id,
            message.time_sync_request.host_send_perf_ns,
        )
    if message_type == JerryCANCmdType.CAPABILITIES_REQUEST:
        return b"\0"
    if message_type == JerryCANCmdType.GPIO_PULSE:
        return struct.pack(
            "<BHI", message.gpio_pulse.instance, message.gpio_pulse.gpio_idx,
            message.gpio_pulse.duration_us,
        )
    raise NotImplementedError(f"packing {message_type.name} is not implemented")


def encode_frame(message: JerryCANMsg, dst_id: int) -> Tuple[int, bytes]:
    message_type = JerryCANCmdType(message.type)
    payload = _pack_payload(message)
    payload_size = _payload_size(message_type)
    payload = payload[:payload_size]
    if payload_size < JERRYCAN_ACTUAL_PAYLOAD_SIZE:
        payload += struct.pack("<B", int(message.uuid) & 0xFF)
    return encode_can_id(message_type, dst_id), payload


def _unpack_cfg(data: bytes) -> JerryCANCfgMsg:
    cfg = JerryCANCfgMsg()
    cfg.type = JerryCANCfgMsg.Type(data[0])
    if cfg.type == JerryCANCfgMsg.Type.SERVO:
        first, _, min_pos, max_pos, min_pwm, max_pwm, max_vel, max_acc = struct.unpack("<Bhffffff", data[1:28])
        cfg.servo.motor_id = first & 0x03
        cfg.servo.error = (first >> 2) & 0x01
        cfg.servo.min_position = min_pos
        cfg.servo.max_position = max_pos
        cfg.servo.min_pwm_duration_us = min_pwm
        cfg.servo.max_pwm_duration_us = max_pwm
        cfg.servo.motor_max_velocity = max_vel
        cfg.servo.motor_max_acceleration = max_acc
    else:
        first, microsteps, steps_per_rev, max_vel, max_acc, homing_vel = struct.unpack("<BHffff", data[1:20])
        cfg.stepper.motor_id = first & 0x03
        cfg.stepper.error = (first >> 2) & 0x01
        cfg.stepper.flip_limit_orientation = bool((first >> 3) & 0x01)
        cfg.stepper.microsteps = microsteps
        cfg.stepper.steps_per_revolution = steps_per_rev
        cfg.stepper.motor_max_velocity = max_vel
        cfg.stepper.motor_max_acceleration = max_acc
        cfg.stepper.homing_velocity = homing_vel
    return cfg


def decode_frame(arbitration_id: int, data: bytes, *, timestamp_ns: Optional[int] = None,
                 index: Optional[int] = None) -> JerryCANMsg:
    message = JerryCANMsg()
    message.type = JerryCANCmdType((arbitration_id >> 5) & 0x3F)
    message.dst_id = arbitration_id & 0x1F
    message.timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
    message.index = time.perf_counter_ns() if index is None else index
    message.wire_length = len(data)
    payload_size = _payload_size(message.type)
    payload = bytes(data[:payload_size])
    if payload_size < JERRYCAN_ACTUAL_PAYLOAD_SIZE and len(data) > payload_size:
        message.uuid = data[payload_size]
        trailer_offset = payload_size + 1
        if len(data) >= trailer_offset + _TIMING_TRAILER.size:
            tag, version, kind, flags, boot_id, sequence, board_time_us = (
                _TIMING_TRAILER.unpack_from(data, trailer_offset)
            )
            if tag == JERRYCAN_TIMING_TRAILER_TAG and version == 1:
                message.board_timing_valid = True
                message.timing = JerryCANTiming(
                    version=version,
                    kind=JerryCANTimestampKind(kind),
                    flags=flags,
                    boot_id=boot_id,
                    sequence=sequence,
                    board_time_us=board_time_us,
                )
    _unpack_payload(message, payload)
    return message


def _unpack_payload(message: JerryCANMsg, payload: bytes) -> None:
    message_type = JerryCANCmdType(message.type)
    if message_type == JerryCANCmdType.BOOTLOADER_RESPONSE:
        response_type = JerryCANBootloaderCmd.SubCommand(payload[0])
        message.bootloader_response.type = response_type
        if response_type == JerryCANBootloaderCmd.SubCommand.VERSION:
            fields = payload[1:7].ljust(6, b"\0")
            version = BootloaderVersion(*struct.unpack("<BBBBBB", fields))
            message.bootloader_response.version = version
        else:
            fields = payload[1:6].ljust(5, b"\0")
            active, bytes_written = struct.unpack("<BI", fields)
            message.bootloader_response.status = BootloaderStatus(active, bytes_written)
    elif message_type == JerryCANCmdType.CFG_RESPONSE:
        message.cfg_response = _unpack_cfg(payload.ljust(28, b"\0"))
    elif message_type == JerryCANCmdType.GPIO_READ:
        message.gpio_read.instance, message.gpio_read.state = struct.unpack("<BI", payload)
    elif message_type == JerryCANCmdType.TONE:
        message.tone.instance, message.tone.frequency_hz, message.tone.duration_ms = struct.unpack("<BHH", payload)
    elif message_type == JerryCANCmdType.ANALOG_OUT:
        message.analog_out.instance, message.analog_out.value_mv = struct.unpack("<BH", payload)
    elif message_type == JerryCANCmdType.PRESSURE_READ:
        message.pressure_read.instance, message.pressure_read.pressure = struct.unpack("<BI", payload)
    elif message_type == JerryCANCmdType.RGB_LED:
        message.rgb_led.red, message.rgb_led.green, message.rgb_led.blue = struct.unpack("<BBB", payload)
    elif message_type == JerryCANCmdType.AUDIO_MAGNITUDE_DATA_BEGIN:
        message.audio_data_cmd.stream_id = struct.unpack("<I", payload)[0]
    elif message_type == JerryCANCmdType.AUDIO_MAGNITUDE_DATA_CONT:
        message.audio_data.magnitudes = list(struct.unpack("<16f", payload[:64]))
    elif message_type == JerryCANCmdType.AUDIO_MAGNITUDE_DATA_END:
        message.audio_data_cmd.stream_id = struct.unpack("<I", payload)[0]
    elif message_type == JerryCANCmdType.DOOR_SENSOR:
        state = payload[0]
        message.doors.door1 = state & 0x01
        message.doors.door2 = (state >> 1) & 0x01
        message.doors.door3 = (state >> 2) & 0x01
        message.doors.ext_button = (state >> 3) & 0x01
    elif message_type == JerryCANCmdType.SERVO_STATUS:
        message.servo_status.motor_id, message.servo_status.status, message.servo_status.position = struct.unpack(
            "<BBf", payload)
    elif message_type == JerryCANCmdType.STEPPER_STATUS:
        fields = struct.unpack("<BBBBff", payload)
        message.stepper_status = StepperStatus(*fields)
    elif message_type == JerryCANCmdType.TEMP_HUM_READ:
        fields = struct.unpack("<BHH", payload)
        message.temp_hum_read = TempHumRead(*fields)
    elif message_type == JerryCANCmdType.ACKNOWLEDGE:
        message.ack.error = struct.unpack("<i", payload)[0]
    elif message_type == JerryCANCmdType.TIME_SYNC_RESPONSE:
        message.time_sync_response = TimeSyncResponse(*struct.unpack("<IQQ", payload))
    elif message_type == JerryCANCmdType.CAPABILITIES_RESPONSE:
        message.capabilities_response = CapabilitiesResponse(*struct.unpack("<BII", payload))
    elif message_type == JerryCANCmdType.GPIO_PULSE_STATUS:
        message.gpio_pulse_status = GPIOPulseStatus(*struct.unpack("<BHIBi", payload))


class SocketCanJerryCAN:
    """pyjerrycan-compatible JerryCAN backend implemented with python-can."""

    def __init__(self, configuration: CanTransportConfiguration):
        self.configuration = configuration
        self._bus = None
        self._receive_buffer_bytes: Optional[int] = None
        self._received_frames = 0
        self._received_since_log = 0
        self._receive_log_started = time.perf_counter()
        self._next_receive_log = self._receive_log_started + 60

    @property
    def receive_statistics(self):
        return {
            "received_frames": self._received_frames,
            "receive_buffer_bytes": self._receive_buffer_bytes,
        }

    def _configure_receive_buffer(self) -> None:
        raw_socket = getattr(self._bus, "socket", None) or getattr(self._bus, "_socket", None)
        if raw_socket is None or not hasattr(raw_socket, "setsockopt"):
            logger.debug("CAN backend does not expose its raw socket; SO_RCVBUF unchanged")
            return
        requested = self.configuration.receive_buffer_bytes
        try:
            raw_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested)
            self._receive_buffer_bytes = raw_socket.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_RCVBUF,
            )
        except OSError as exc:
            logger.warning("Could not set CAN socket receive buffer to %s bytes: %s", requested, exc)
            return
        logger.info(
            "CAN socket receive buffer configured: requested=%s effective=%s",
            requested,
            self._receive_buffer_bytes,
        )

    def Open(self) -> int:
        try:
            import can
        except ModuleNotFoundError:
            return -errno.ENODEV
        transport_kind = self.configuration.kind
        interface = {
            CanTransportKind.SOCKETCAN: "socketcan",
            CanTransportKind.PCAN_BASIC: "pcan",
        }[transport_kind]
        kwargs = {
            "interface": interface,
            "channel": self.configuration.channel,
            "can_filters": _jerrycan_filters(),
        }
        if self.configuration.fd or self.configuration.data_bitrate is not None:
            kwargs["fd"] = True
        if transport_kind == CanTransportKind.SOCKETCAN:
            # SocketCAN bit timing belongs to the network device and must be
            # configured with ip/netlink before the application opens it.
            # Keep error frames visible so bus-off and controller failures can
            # be classified instead of surfacing later as a generic timeout.
            kwargs["ignore_rx_error_frames"] = False
            if self.configuration.fd:
                mtu = _socketcan_mtu(self.configuration.channel)
                if mtu is not None and mtu < CAN_FD_MTU:
                    logger.error(
                        "SocketCAN channel %s has MTU %s; JerryCAN requires CAN FD MTU %s. "
                        "Configure the interface with 'fd on' before launching reachAQ.",
                        self.configuration.channel,
                        mtu,
                        CAN_FD_MTU,
                    )
                    return -errno.EPROTONOSUPPORT
        else:
            # PCAN-Basic owns adapter initialization, unlike SocketCAN.
            if self.configuration.bitrate is not None:
                kwargs["bitrate"] = self.configuration.bitrate
            if self.configuration.data_bitrate is not None:
                kwargs["data_bitrate"] = self.configuration.data_bitrate
        try:
            self._bus = can.Bus(**kwargs)
        except OSError as exc:
            logger.error(
                "Could not open CAN transport=%s channel=%s: %s",
                transport_kind.value,
                self.configuration.channel,
                exc,
            )
            return -getattr(exc, "errno", errno.EIO)
        except Exception:
            logger.exception(
                "Could not open CAN transport=%s channel=%s",
                transport_kind.value,
                self.configuration.channel,
            )
            return -errno.EIO
        self._configure_receive_buffer()
        return 0

    def Close(self) -> int:
        if self._bus is None:
            return 0
        self._bus.shutdown()
        self._bus = None
        return 0

    def SendMessage(self, msg: JerryCANMsg, dst_id: int) -> int:
        if self._bus is None:
            return -errno.ENOTCONN
        can_id, payload = encode_frame(msg, dst_id)
        if len(payload) > 8 and not self.configuration.fd:
            logger.error(
                "Refusing %s-byte JerryCAN payload because CAN FD is disabled",
                len(payload),
            )
            return -errno.EMSGSIZE
        try:
            import can
            message = can.Message(
                arbitration_id=can_id,
                data=payload,
                is_extended_id=False,
                is_fd=self.configuration.fd or len(payload) > 8,
                bitrate_switch=self.configuration.fd,
            )
            self._bus.send(message)
        except OSError as exc:
            return -getattr(exc, "errno", errno.EIO)
        except Exception:
            return -errno.EIO
        return 0

    def ReceiveMessage(self) -> Optional[JerryCANMsg]:
        messages = self.ReceiveMessages(1, 0)
        return messages[0] if messages else None

    def ReceiveMessages(self, max_count: int = 1, collect_ms: int = 0) -> List[JerryCANMsg]:
        if self._bus is None:
            return []
        messages = []
        end = time.perf_counter() + collect_ms / 1000
        timeout = self.configuration.receive_timeout_seconds or 0.001
        while True:
            try:
                raw_message = self._bus.recv(timeout=timeout)
            except BaseException as exc:
                category, error_code = _classify_receive_error(exc)
                from .can_diagnostics import capture_can_diagnostics
                raise CanTransportReadError(
                    f"CAN receive failed on {self.configuration.channel}: {exc}",
                    category=category,
                    error_code=error_code,
                    diagnostics=capture_can_diagnostics(self.configuration.channel),
                ) from exc
            if raw_message is not None:
                if raw_message.is_error_frame:
                    category = "bus_off" if raw_message.arbitration_id & 0x40 else "error_frame"
                    from .can_diagnostics import capture_can_diagnostics
                    raise CanTransportReadError(
                        "Unsupported CAN frame "
                        f"id={raw_message.arbitration_id:#x} error={raw_message.is_error_frame} "
                        f"remote={raw_message.is_remote_frame} extended={raw_message.is_extended_id}",
                        category=category,
                        diagnostics=capture_can_diagnostics(self.configuration.channel),
                    )
                if raw_message.is_remote_frame or raw_message.is_extended_id:
                    logger.warning(
                        "Ignoring unsupported CAN frame id=%#x remote=%s extended=%s",
                        raw_message.arbitration_id,
                        raw_message.is_remote_frame,
                        raw_message.is_extended_id,
                    )
                    if collect_ms == 0:
                        break
                    continue
                else:
                    timestamp_ns = (
                        int(raw_message.timestamp * 1e9)
                        if raw_message.timestamp is not None
                        else None
                    )
                    try:
                        decoded = decode_frame(
                            raw_message.arbitration_id,
                            bytes(raw_message.data),
                            timestamp_ns=timestamp_ns,
                        )
                    except (IndexError, struct.error, ValueError) as exc:
                        from .can_diagnostics import capture_can_diagnostics
                        raise CanTransportReadError(
                            "Malformed JerryCAN frame "
                            f"id={raw_message.arbitration_id:#x} length={len(raw_message.data)}: {exc}",
                            category="malformed_frame",
                            diagnostics=capture_can_diagnostics(self.configuration.channel),
                        )
                    else:
                        messages.append(decoded)
                        self._received_frames += 1
                        self._received_since_log += 1
                        if 0 < max_count <= len(messages):
                            break
            if collect_ms == 0 or time.perf_counter() > end:
                break
        now = time.perf_counter()
        if now >= self._next_receive_log:
            elapsed = max(now - self._receive_log_started, 1e-9)
            logger.info(
                "CAN reader throughput: frames=%s rate=%.1f frames/s receive_buffer=%s",
                self._received_since_log,
                self._received_since_log / elapsed,
                self._receive_buffer_bytes,
            )
            self._received_since_log = 0
            self._receive_log_started = now
            self._next_receive_log = now + 60
        return messages

    def Heartbeat(self) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.HEARTBEAT
        msg.heartbeat.rsvd = 0xFF
        return self.SendMessage(msg, 0x1F)

    def EStop(self, enable: bool) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.ESTOP
        msg.estop.rsvd = int(enable)
        return self.SendMessage(msg, 0x1F)

    def StepperMove(self, dst_id: int, motor_id: int, position: float, max_velocity: float,
                    max_acceleration: float, abs_or_rel: AbsOrRel, save: bool, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.STEPPER_MOVE
        msg.uuid = uuid
        msg.stepper_move = StepperMoveCommand(motor_id, save, AbsOrRel(abs_or_rel), position, max_velocity,
                                              max_acceleration)
        return self.SendMessage(msg, dst_id)

    def ServoMove(self, dst_id: int, motor_id: int, position: float, max_velocity: float,
                  max_acceleration: float, abs_or_rel: AbsOrRel, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.SERVO_MOVE
        msg.uuid = uuid
        msg.servo_move = ServoMoveCommand(motor_id, False, AbsOrRel(abs_or_rel), position, max_velocity,
                                          max_acceleration)
        return self.SendMessage(msg, dst_id)

    def ServoAttach(self, dst_id: int, motor_id: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.SERVO_ATTACH
        msg.servo_attach.motor_id = motor_id
        return self.SendMessage(msg, dst_id)

    def ServoDetach(self, dst_id: int, motor_id: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.SERVO_DETACH
        msg.servo_detach.motor_id = motor_id
        return self.SendMessage(msg, dst_id)

    def StepperHome(self, dst_id: int, motor_id: int, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.STEPPER_HOME
        msg.uuid = uuid
        msg.stepper_home.motor_id = motor_id
        return self.SendMessage(msg, dst_id)

    def CfgRead(self, dst_id: int, cfg: JerryCANCfgMsg) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.CFG_READ
        msg.cfg_read = cfg
        return self.SendMessage(msg, dst_id)

    def _cfg_write(self, dst_id: int, cfg: JerryCANCfgMsg, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.CFG_WRITE
        msg.uuid = uuid
        msg.cfg_write = cfg
        return self.SendMessage(msg, dst_id)

    def StepperCfgWrite(self, dst_id: int, motor_id: int, microsteps: int, steps_per_revolution: float,
                        motor_max_velocity: float, motor_max_acceleration: float, homing_velocity: float,
                        flip_limit_orientation: bool, uuid: int) -> int:
        cfg = JerryCANCfgMsg()
        cfg.type = JerryCANCfgMsg.Type.STEPPER
        cfg.stepper.motor_id = motor_id
        cfg.stepper.flip_limit_orientation = flip_limit_orientation
        cfg.stepper.microsteps = microsteps
        cfg.stepper.steps_per_revolution = steps_per_revolution
        cfg.stepper.motor_max_velocity = motor_max_velocity
        cfg.stepper.motor_max_acceleration = motor_max_acceleration
        cfg.stepper.homing_velocity = homing_velocity
        return self._cfg_write(dst_id, cfg, uuid)

    def ServoCfgWrite(self, dst_id: int, motor_id: int, min_position: float, max_position: float,
                      min_pwm_duration_us: float, max_pwm_duration_us: float, motor_max_velocity: float,
                      motor_max_acceleration: float, uuid: int) -> int:
        cfg = JerryCANCfgMsg()
        cfg.type = JerryCANCfgMsg.Type.SERVO
        cfg.servo.motor_id = motor_id
        cfg.servo.min_position = min_position
        cfg.servo.max_position = max_position
        cfg.servo.min_pwm_duration_us = min_pwm_duration_us
        cfg.servo.max_pwm_duration_us = max_pwm_duration_us
        cfg.servo.motor_max_velocity = motor_max_velocity
        cfg.servo.motor_max_acceleration = motor_max_acceleration
        return self._cfg_write(dst_id, cfg, uuid)

    def StepperCfgRead(self, dst_id: int, motor_id: int) -> int:
        cfg = JerryCANCfgMsg()
        cfg.type = JerryCANCfgMsg.Type.STEPPER
        cfg.stepper.motor_id = motor_id
        return self.CfgRead(dst_id, cfg)

    def ServoCfgRead(self, dst_id: int, motor_id: int) -> int:
        cfg = JerryCANCfgMsg()
        cfg.type = JerryCANCfgMsg.Type.SERVO
        cfg.servo.motor_id = motor_id
        return self.CfgRead(dst_id, cfg)

    def GPIOWrite(self, dst_id: int, instance: int, gpio_idx: int, state: bool, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.GPIO_WRITE
        msg.uuid = uuid
        msg.gpio_write = GPIOWrite(instance, gpio_idx, state)
        return self.SendMessage(msg, dst_id)

    def ToneWrite(self, dst_id: int, instance: int, frequency: int, duration: int, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.TONE
        msg.uuid = uuid
        msg.tone = Tone(instance, frequency, duration)
        return self.SendMessage(msg, dst_id)

    def AnalogOutWrite(self, dst_id: int, instance: int, value_mv: int, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.ANALOG_OUT
        msg.uuid = uuid
        msg.analog_out = AnalogOut(instance, value_mv)
        return self.SendMessage(msg, dst_id)

    def RGBLEDWrite(self, dst_id: int, red: int, green: int, blue: int, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.RGB_LED
        msg.uuid = uuid
        msg.rgb_led = RGBLED(red, green, blue)
        return self.SendMessage(msg, dst_id)

    def BootloaderCommand(self, dst_id: int, subcmd: JerryCANBootloaderCmd.SubCommand) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.BOOTLOADER_COMMAND
        msg.bootloader_command.type = JerryCANBootloaderCmd.SubCommand(subcmd)
        return self.SendMessage(msg, dst_id)

    def BootloaderData(self, dst_id: int, data: BootloaderData) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.BOOTLOADER_DATA
        msg.bootloader_data = data
        return self.SendMessage(msg, dst_id)

    def Delay(self, dst_id: int, delay: int, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.DELAY
        msg.uuid = uuid
        msg.delay.delay = delay
        return self.SendMessage(msg, dst_id)

    def SendToFixedXYZ(self, dst_id: int, uuid: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.FIXED_XYZ
        msg.uuid = uuid
        return self.SendMessage(msg, dst_id)

    def TimeSync(self, dst_id: int, request_id: int, host_send_perf_ns: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.TIME_SYNC_REQUEST
        msg.time_sync_request = TimeSyncRequest(request_id, host_send_perf_ns)
        return self.SendMessage(msg, dst_id)

    def RequestCapabilities(self, dst_id: int) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.CAPABILITIES_REQUEST
        return self.SendMessage(msg, dst_id)

    def GPIOPulse(
        self,
        dst_id: int,
        instance: int,
        gpio_idx: int,
        duration_us: int,
        uuid: int,
    ) -> int:
        msg = JerryCANMsg()
        msg.type = JerryCANCmdType.GPIO_PULSE
        msg.uuid = uuid
        msg.gpio_pulse = GPIOPulse(instance, gpio_idx, duration_us)
        return self.SendMessage(msg, dst_id)
