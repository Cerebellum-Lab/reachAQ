"""Wire codec for the CAN sync clap, the NI-DAQ barcode report, IO-module
analog blocks and PWM commands (spec section 4.2)."""

import struct

import pytest

from autotrainer.device import socketcan_jerrycan
from autotrainer.device.socketcan_jerrycan import (
    JerryCANCmdType,
    JerryCANMsg,
    JerryCANTimestampKind,
    PwmSet,
)

IO_MODULE_ADDRESS = 0x08


def _trailer(kind, *, boot_id=0xB007, sequence=5, board_time_us=1_234_567):
    return struct.pack("<BBBBIIQ", 0xA7, 1, int(kind), 0, boot_id, sequence, board_time_us)


def _can_id(message_type, address):
    return (int(message_type) << 5) | address


def test_new_types_have_the_specified_ids():
    assert JerryCANCmdType.SYNC_CLAP == 0x25
    assert JerryCANCmdType.SYNC_CLAP_OBSERVED == 0x26
    assert JerryCANCmdType.SYNC_BARCODE == 0x27
    assert JerryCANCmdType.ANALOG_BLOCK == 0x28
    assert JerryCANCmdType.PWM_SET == 0x29


def test_receive_filters_admit_the_new_types():
    admitted = {f["can_id"] >> 5 for f in socketcan_jerrycan._jerrycan_filters()}

    assert {0x25, 0x26, 0x27, 0x28, 0x29} <= admitted


def test_decode_sync_clap():
    data = struct.pack("<I", 41) + b"\x00" + _trailer(JerryCANTimestampKind.TX_QUEUE)

    message = socketcan_jerrycan.decode_frame(
        _can_id(JerryCANCmdType.SYNC_CLAP, IO_MODULE_ADDRESS), data,
    )

    assert message.sync_clap.tick == 41
    assert message.dst_id == IO_MODULE_ADDRESS


def test_decode_sync_clap_observed_with_its_observation_time():
    data = struct.pack("<IB", 41, IO_MODULE_ADDRESS) + b"\x00" + _trailer(
        JerryCANTimestampKind.OBSERVED, board_time_us=9_000_123,
    )

    message = socketcan_jerrycan.decode_frame(_can_id(JerryCANCmdType.SYNC_CLAP_OBSERVED, 0), data)

    assert message.sync_clap_observed.tick == 41
    assert message.sync_clap_observed.clapper_node == IO_MODULE_ADDRESS
    assert message.board_timing_valid
    assert message.timing.kind == JerryCANTimestampKind.OBSERVED
    assert message.timing.board_time_us == 9_000_123


def test_decode_sync_barcode_with_its_start_edge_time():
    data = struct.pack("<I", 41) + b"\x00" + _trailer(JerryCANTimestampKind.PHYSICAL_START)

    message = socketcan_jerrycan.decode_frame(
        _can_id(JerryCANCmdType.SYNC_BARCODE, IO_MODULE_ADDRESS), data,
    )

    assert message.sync_barcode.tick == 41
    assert message.timing.kind == JerryCANTimestampKind.PHYSICAL_START


def test_decode_analog_block_keeps_scan_major_order_through_dlc_padding():
    samples = list(range(100, 116))  # scan 0 ch0..7, then scan 1 ch0..7
    payload = struct.pack("<BBBBH16H", 0, 8, 2, 0, 1000, *samples)
    # 38 B payload + uuid + 20 B trailer = 59 B, padded by the board to DLC 64.
    data = payload + b"\x00" + _trailer(JerryCANTimestampKind.SAMPLED) + bytes(5)

    message = socketcan_jerrycan.decode_frame(
        _can_id(JerryCANCmdType.ANALOG_BLOCK, IO_MODULE_ADDRESS), data,
    )

    block = message.analog_block
    assert (block.instance, block.channel_count, block.scan_count, block.scan_period_us) == (0, 8, 2, 1000)
    assert block.samples == tuple(samples)
    assert message.timing.kind == JerryCANTimestampKind.SAMPLED


def test_decode_analog_block_trims_unused_sample_slots():
    payload = struct.pack("<BBBBH16H", 1, 4, 2, 0, 500, *range(16))

    message = socketcan_jerrycan.decode_frame(
        _can_id(JerryCANCmdType.ANALOG_BLOCK, IO_MODULE_ADDRESS), payload + b"\x00",
    )

    assert message.analog_block.samples == tuple(range(8))


def test_decode_analog_block_rejects_more_samples_than_slots():
    payload = struct.pack("<BBBBH16H", 0, 8, 3, 0, 1000, *range(16))

    with pytest.raises(ValueError):
        socketcan_jerrycan.decode_frame(
            _can_id(JerryCANCmdType.ANALOG_BLOCK, IO_MODULE_ADDRESS), payload + b"\x00",
        )


def test_encode_pwm_set():
    message = JerryCANMsg()
    message.type = JerryCANCmdType.PWM_SET
    message.uuid = 0x5A
    message.pwm_set = PwmSet(channel=2, duty_permyriad=2500, period_us=1000)

    can_id, data = socketcan_jerrycan.encode_frame(message, IO_MODULE_ADDRESS)

    assert can_id == _can_id(JerryCANCmdType.PWM_SET, IO_MODULE_ADDRESS)
    assert data == struct.pack("<BHI", 2, 2500, 1000) + b"\x5a"
