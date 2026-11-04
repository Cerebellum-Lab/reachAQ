"""reachAQ must share the bus with boards it has no driver for.

A board's frames carry its own 5-bit address; bits 2-3 are the board type and
the pellet board is type 0. Before this guard, every handler resolved its
target through _addr2tgt, which raises for any other type, and read()
translated a whole batch at once - so one frame from another board threw away
the pellet frames read alongside it.
"""

import logging

import pytest

from autotrainer.device import (CanTransportConfiguration, Heartbeat, SyncBarcodeReport,
                                SyncClapObservation, Target, socketcan_jerrycan)
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
    JerryCANCmdType.ANALOG_BLOCK,
    JerryCANCmdType.PWM_SET,
])
def test_sync_and_io_types_from_the_pellet_arrive_silently(message_type, caplog):
    with caplog.at_level(logging.WARNING):
        translated = _interface()._translate(_message(message_type, PELLET_ADDRESS))

    assert translated is None
    assert "Unhandled message type" not in caplog.text


def test_a_rejected_time_sync_exchange_does_not_escape_translate():
    interface = _interface()
    message = _message(JerryCANCmdType.TIME_SYNC_RESPONSE, PELLET_ADDRESS)
    message.index = 1_000_000_000
    message.board_timing_valid = True
    message.timing = socketcan_jerrycan.JerryCANTiming(
        version=1, kind=socketcan_jerrycan.JerryCANTimestampKind.TX_QUEUE,
        boot_id=9, sequence=1, board_time_us=2_000,
    )
    # Board send before board receive: observe() rejects it as out of order.
    message.time_sync_response = socketcan_jerrycan.TimeSyncResponse(
        request_id=7, request_receive_time_us=2_000, response_queue_time_us=1_000,
    )
    interface._time_sync_requests[7] = 999_000_000

    translated = interface._translate(message)

    assert translated.clock_model == {}


@pytest.mark.parametrize("address", [PELLET_ADDRESS, IO_MODULE_ADDRESS])
def test_clap_observations_are_recorded_from_any_board(address):
    message = _message(JerryCANCmdType.SYNC_CLAP_OBSERVED, address)
    message.sync_clap_observed = socketcan_jerrycan.SyncClapObserved(tick=41, clapper_node=IO_MODULE_ADDRESS)

    translated = _interface()._translate(message)

    assert isinstance(translated, SyncClapObservation)
    assert (translated.board_address, translated.tick, translated.clapper_node) == (address, 41, IO_MODULE_ADDRESS)


def test_barcode_reports_are_recorded_from_the_timekeeper():
    message = _message(JerryCANCmdType.SYNC_BARCODE, IO_MODULE_ADDRESS)
    message.sync_barcode = socketcan_jerrycan.SyncBarcode(tick=41)

    translated = _interface()._translate(message)

    assert isinstance(translated, SyncBarcodeReport)
    assert (translated.board_address, translated.tick) == (IO_MODULE_ADDRESS, 41)


@pytest.mark.parametrize("message_type", [
    JerryCANCmdType.HEARTBEAT, JerryCANCmdType.ANALOG_BLOCK, JerryCANCmdType.GPIO_READ,
])
def test_other_io_module_traffic_is_still_ignored(message_type):
    assert _interface()._translate(_message(message_type, IO_MODULE_ADDRESS)) is None


def test_the_message_handler_passes_sync_reports_through_silently(caplog):
    from queue import Queue
    from autotrainer.core import SystemStatusMessageKind
    from autotrainer.core.message.system_message_handler import SystemMessageHandler

    handler = SystemMessageHandler(Queue())
    with caplog.at_level(logging.WARNING):
        handler.message_received(SystemStatusMessageKind.SYNC_CLAP_OBSERVATION, object())
        handler.message_received(SystemStatusMessageKind.SYNC_BARCODE_REPORT, object())

    assert "unhandled msg" not in caplog.text
