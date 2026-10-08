"""reachAQ must share the bus with boards it has no driver for.

A board's frames carry its own 5-bit address; bits 2-3 are the board type and
the pellet board is type 0. Before this guard, every handler resolved its
target through _addr2tgt, which raises for any other type, and read()
translated a whole batch at once - so one frame from another board threw away
the pellet frames read alongside it.
"""

import logging

import pytest

from autotrainer.device import CanTransportConfiguration, Heartbeat, Target
from autotrainer.device.can_interface import CanInterface
from autotrainer.device.socketcan_jerrycan import JerryCANCmdType, JerryCANMsg

IO_MODULE_ADDRESS = 0x08  # board type 2, node 0
PELLET_ADDRESS = 0x00


def _interface():
    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(kind="socketcan", fd=True),
    )
    interface._get_index = lambda item: item.index
    interface._get_timestamp_ns = lambda item: item.timestamp_ns
    return interface


def _message(message_type, address):
    message = JerryCANMsg()
    message.type = message_type
    message.dst_id = address
    return message


@pytest.mark.parametrize("message_type", [
    JerryCANCmdType.HEARTBEAT,
    JerryCANCmdType.GPIO_READ,
    JerryCANCmdType.TONE,
    JerryCANCmdType.PRESSURE_READ,
    JerryCANCmdType.RGB_LED,
    JerryCANCmdType.ANALOG_OUT,
    JerryCANCmdType.TIME_SYNC_RESPONSE,
    JerryCANCmdType.CAPABILITIES_RESPONSE,
    JerryCANCmdType.GPIO_PULSE_STATUS,
    JerryCANCmdType.ACKNOWLEDGE,
])
def test_foreign_board_frame_translates_to_nothing(message_type):
    assert _interface()._translate(_message(message_type, IO_MODULE_ADDRESS)) is None


def test_lowest_foreign_board_address_is_dropped():
    # 0x04 is board type 1 (bit 2 only): a board-type shift of 3 reads it as
    # the pellet's type 0.
    assert _interface()._translate(_message(JerryCANCmdType.HEARTBEAT, 0x04)) is None


def test_read_keeps_pellet_frames_when_a_foreign_board_shares_the_batch():
    interface = _interface()
    interface._is_open = True
    batch = [
        _message(JerryCANCmdType.HEARTBEAT, IO_MODULE_ADDRESS),
        _message(JerryCANCmdType.HEARTBEAT, PELLET_ADDRESS),
    ]
    interface._read_msgs = lambda max_count, collect_ms: list(batch)

    result = interface.read(max_count=2)

    assert len(result) == 1
    assert isinstance(result[0], Heartbeat)
    assert result[0].target == Target.PELLET_DEVICE


def test_foreign_board_is_logged_once_per_address(caplog):
    interface = _interface()

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            interface._translate(_message(JerryCANCmdType.HEARTBEAT, IO_MODULE_ADDRESS))
        interface._translate(_message(JerryCANCmdType.HEARTBEAT, IO_MODULE_ADDRESS + 1))

    notices = [r for r in caplog.records if "no driver for that board" in r.getMessage()]
    assert len(notices) == 2


@pytest.mark.parametrize("address", [PELLET_ADDRESS, 0x03])  # board type 0: nodes 0 and 3
def test_pellet_frames_are_unaffected(address):
    translated = _interface()._translate(_message(JerryCANCmdType.HEARTBEAT, address))

    assert isinstance(translated, Heartbeat)
    assert translated.target == Target.PELLET_DEVICE


@pytest.mark.parametrize("message_type", [
    JerryCANCmdType.SYNC_CLAP,
    JerryCANCmdType.SYNC_CLAP_OBSERVED,
    JerryCANCmdType.SYNC_BARCODE,
    JerryCANCmdType.ANALOG_BLOCK,
    JerryCANCmdType.PWM_SET,
])
def test_sync_and_io_types_from_the_pellet_arrive_silently(message_type, caplog):
    with caplog.at_level(logging.WARNING):
        translated = _interface()._translate(_message(message_type, PELLET_ADDRESS))

    assert translated is None
    assert "Unhandled message type" not in caplog.text
