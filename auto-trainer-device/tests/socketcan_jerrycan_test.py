import struct

from autotrainer.device.socketcan_jerrycan import (
    AbsOrRel,
    BootloaderVersion,
    JerryCANBootloaderCmd,
    JerryCANCfgMsg,
    JerryCANCmdType,
    JerryCANMsg,
    decode_frame,
    encode_frame,
)


def test_encode_stepper_move_matches_jerrycan_frame_layout():
    message = JerryCANMsg()
    message.type = JerryCANCmdType.STEPPER_MOVE
    message.uuid = 7
    message.stepper_move.motor_id = 2
    message.stepper_move.save = True
    message.stepper_move.abs_or_rel = AbsOrRel.RELATIVE
    message.stepper_move.position = 1.5
    message.stepper_move.max_velocity = 2.5
    message.stepper_move.max_acceleration = 3.5

    can_id, payload = encode_frame(message, 0x0A)

    assert can_id == 0x4A
    assert len(payload) == 14
    flags, position, velocity, acceleration, uuid = struct.unpack("<BfffB", payload)
    assert flags == 0x86
    assert position == 1.5
    assert velocity == 2.5
    assert acceleration == 3.5
    assert uuid == 7


def test_encode_cfg_read_pads_stepper_union_to_jerrycan_size():
    message = JerryCANMsg()
    message.type = JerryCANCmdType.CFG_READ
    message.cfg_read.type = JerryCANCfgMsg.Type.STEPPER
    message.cfg_read.stepper.motor_id = 1

    can_id, payload = encode_frame(message, 0x00)

    assert can_id == 0xC0
    assert len(payload) == 29
    assert payload[0] == JerryCANCfgMsg.Type.STEPPER
    assert payload[1] == 1
    assert payload[-1] == 0


def test_decode_bootloader_version_response_matches_pyjerrycan_shape():
    can_id = (JerryCANCmdType.BOOTLOADER_RESPONSE << 5) | 0x01
    payload = bytes([
        JerryCANBootloaderCmd.SubCommand.VERSION,
        1,
        2,
        3,
        4,
        5,
        6,
        0,
    ])

    message = decode_frame(can_id, payload, timestamp_ns=123, index=456)

    assert message.type == JerryCANCmdType.BOOTLOADER_RESPONSE
    assert message.dst_id == 0x01
    assert message.timestamp_ns == 123
    assert message.index == 456
    assert message.bootloader_response.type == JerryCANBootloaderCmd.SubCommand.VERSION
    assert message.bootloader_response.version == BootloaderVersion(1, 2, 3, 4, 5, 6)
