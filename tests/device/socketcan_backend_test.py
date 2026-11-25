import errno
import logging
import socket
import struct
import sys
import types
from unittest import mock

import pytest

from reachaq.device import (
    AnalogOutput,
    CanTransportConfiguration,
    CanTransportKind,
    ColorLed,
    Motor,
    ServoStatus,
    Target,
    BoardTimeSync,
)
from reachaq.device import can_interface
from reachaq.device import socketcan_jerrycan
from reachaq.device.can_interface import CanInterface
from reachaq.device.socketcan_jerrycan import (
    CanTransportReadError,
    JerryCANCfgMsg,
    JerryCANCmdType,
    JerryCANMsg,
    SocketCanJerryCAN,
)


class FakeBus:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.sent = []
        self.shutdown_called = False

    def recv(self, timeout):
        del timeout
        return self.messages.pop(0) if self.messages else None

    def send(self, message):
        self.sent.append(message)

    def shutdown(self):
        self.shutdown_called = True


class FakeSocket:
    def __init__(self):
        self.options = {}

    def setsockopt(self, level, option, value):
        self.options[(level, option)] = value

    def getsockopt(self, level, option):
        # Linux normally reports a doubled value for SO_RCVBUF.
        return self.options[(level, option)] * 2


def test_close_marks_interface_closed_even_if_backend_close_fails():
    interface = object.__new__(CanInterface)
    interface._is_open = True
    interface._jc = mock.Mock()
    interface._channel_ownership = mock.Mock()
    interface._jc.Close.side_effect = RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        interface.close()

    assert interface.is_open is False
    interface._channel_ownership.release.assert_called_once_with()


def test_interface_open_releases_channel_when_backend_open_raises():
    interface = object.__new__(CanInterface)
    interface._jc = mock.Mock()
    interface._jc.Open.side_effect = RuntimeError("open failed")
    interface._channel_ownership = mock.Mock()

    with pytest.raises(RuntimeError, match="open failed"):
        interface.open()

    interface._channel_ownership.acquire.assert_called_once_with()
    interface._channel_ownership.release.assert_called_once_with()


def _raw_message(
    arbitration_id,
    data=b"",
    *,
    error=False,
    remote=False,
    extended=False,
    is_fd=True,
):
    return types.SimpleNamespace(
        arbitration_id=arbitration_id,
        data=data,
        timestamp=1.25,
        is_error_frame=error,
        is_remote_frame=remote,
        is_extended_id=extended,
        is_fd=is_fd,
    )


def test_socketcan_open_uses_fd_filters_and_netdev_owned_bit_timing(monkeypatch):
    captured = {}

    def bus_factory(**kwargs):
        captured.update(kwargs)
        return FakeBus()

    fake_can = types.SimpleNamespace(Bus=bus_factory)
    monkeypatch.setitem(sys.modules, "can", fake_can)
    monkeypatch.setattr(socketcan_jerrycan, "_socketcan_mtu", lambda channel: 72)
    backend = SocketCanJerryCAN(CanTransportConfiguration(
        kind="socketcan",
        channel="can0",
        bitrate=1000000,
        data_bitrate=1000000,
        fd=True,
    ))

    assert backend.Open() == 0
    assert captured["interface"] == "socketcan"
    assert captured["channel"] == "can0"
    assert captured["fd"] is True
    assert captured["ignore_rx_error_frames"] is False
    assert captured["can_filters"]
    assert "bitrate" not in captured
    assert "data_bitrate" not in captured


def test_socketcan_open_increases_exposed_socket_receive_buffer(monkeypatch):
    raw_socket = FakeSocket()
    bus = FakeBus()
    bus._socket = raw_socket
    fake_can = types.SimpleNamespace(Bus=lambda **kwargs: bus)
    monkeypatch.setitem(sys.modules, "can", fake_can)
    monkeypatch.setattr(socketcan_jerrycan, "_socketcan_mtu", lambda channel: 72)
    backend = SocketCanJerryCAN(CanTransportConfiguration(
        kind="socketcan",
        fd=True,
        receive_buffer_bytes=4 * 1024 * 1024,
    ))

    assert backend.Open() == 0
    assert raw_socket.options[(socket.SOL_SOCKET, socket.SO_RCVBUF)] == 4 * 1024 * 1024
    assert backend.receive_statistics["receive_buffer_bytes"] == 8 * 1024 * 1024


def test_socketcan_open_rejects_classical_can_mtu(monkeypatch):
    fake_can = types.SimpleNamespace(Bus=lambda **kwargs: FakeBus())
    monkeypatch.setitem(sys.modules, "can", fake_can)
    monkeypatch.setattr(socketcan_jerrycan, "_socketcan_mtu", lambda channel: 16)
    backend = SocketCanJerryCAN(CanTransportConfiguration(
        kind="socketcan",
        fd=True,
    ))

    assert backend.Open() == -errno.EPROTONOSUPPORT
    assert backend._bus is None


def test_receive_ignores_remote_and_extended_frames():
    valid_id = (JerryCANCmdType.HEARTBEAT << 5) | 0x01
    backend = SocketCanJerryCAN(CanTransportConfiguration(
        kind="socketcan",
        fd=True,
    ))
    backend._bus = FakeBus([
        _raw_message(valid_id, remote=True),
        _raw_message(valid_id, extended=True),
        _raw_message(valid_id, b"\xff\0"),
    ])

    messages = backend.ReceiveMessages(max_count=1, collect_ms=5)

    assert len(messages) == 1
    assert messages[0].type == JerryCANCmdType.HEARTBEAT
    assert messages[0].dst_id == 0x01


def test_timestamped_frame_preserves_legacy_uuid_and_decodes_trailer():
    can_id = (JerryCANCmdType.TONE << 5) | 0x01
    legacy = struct.pack("<BHHB", 0, 7000, 100, 23)
    timing = struct.pack(
        "<BBBBIIQ", 0xA7, 1, 3, 0, 0x12345678, 99, 7_654_321,
    )

    message = socketcan_jerrycan.decode_frame(can_id, legacy + timing)

    assert message.uuid == 23
    assert message.tone.frequency_hz == 7000
    assert message.board_timing_valid
    assert message.timing.boot_id == 0x12345678
    assert message.timing.sequence == 99
    assert message.timing.board_time_us == 7_654_321
    assert message.timing.kind.name == "PHYSICAL_START"


def test_legacy_frame_has_explicit_host_timing_fallback():
    can_id = (JerryCANCmdType.TONE << 5) | 0x01
    message = socketcan_jerrycan.decode_frame(
        can_id, struct.pack("<BHHB", 0, 7000, 100, 23),
    )

    assert not message.board_timing_valid
    assert message.timing.board_time_us == 0


def test_can_interface_fits_and_applies_board_clock_response():
    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(kind="socketcan", fd=True),
    )
    interface._get_index = lambda item: item.index
    interface._get_timestamp_ns = lambda item: item.timestamp_ns
    message = JerryCANMsg()
    message.type = JerryCANCmdType.TIME_SYNC_RESPONSE
    message.dst_id = 0
    message.index = 101_002_000_000
    message.timestamp_ns = 1_000_000_000
    message.board_timing_valid = True
    message.timing = socketcan_jerrycan.JerryCANTiming(
        version=1, kind=socketcan_jerrycan.JerryCANTimestampKind.OBSERVED,
        boot_id=9, sequence=1, board_time_us=1_000_100,
    )
    message.time_sync_response = socketcan_jerrycan.TimeSyncResponse(
        request_id=7,
        request_receive_time_us=1_000_000,
        response_queue_time_us=1_000_200,
    )
    interface._time_sync_requests[7] = 101_000_000_000

    translated = interface._translate(message)

    assert isinstance(translated, BoardTimeSync)
    assert translated.clock_model["boot_id"] == 9
    assert translated.board_boot_id == 9
    assert translated.board_aligned_perf_time == pytest.approx(101.001)
    assert translated.timestamp_method == "board_clock_affine"


def test_receive_classifies_bus_off_error_frame(monkeypatch):
    monkeypatch.setattr(
        "reachaq.device.can_diagnostics.capture_can_diagnostics",
        lambda channel: {"channel": channel, "interface_state": "DOWN"},
    )
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([_raw_message(0x40, error=True)])

    with pytest.raises(CanTransportReadError) as raised:
        backend.ReceiveMessages(max_count=1, collect_ms=5)

    assert raised.value.category == "bus_off"
    assert raised.value.diagnostics["interface_state"] == "DOWN"


# Controller-problem + error-counter frames exactly as christielab10's PEAK
# adapter emitted them across a pellet-board reboot on 2026-09-23.
_TX_WARNING_NOTICE = bytes([0x00, 0x08, 0, 0, 0, 0, 0x67, 0x00])
_TX_PASSIVE_NOTICE = bytes([0x00, 0x20, 0, 0, 0, 0, 0x87, 0x00])


def test_receive_logs_controller_state_notices_and_keeps_reading(caplog):
    heartbeat = (JerryCANCmdType.HEARTBEAT << 5) | 0x01
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([
        _raw_message(0x204, _TX_WARNING_NOTICE, error=True),
        _raw_message(0x204, _TX_PASSIVE_NOTICE, error=True),
        _raw_message(heartbeat, b"\xff\0"),
    ])

    with caplog.at_level(logging.WARNING, logger=socketcan_jerrycan.__name__):
        # Ended by the heartbeat, not by a window: the read returns once
        # max_count messages are in, and the window only bounds a read that
        # never gets one. A 5 ms window closed before the heartbeat once in a
        # full rig run (2026-09-29), the two notices' logging taking longer.
        messages = backend.ReceiveMessages(max_count=1, collect_ms=5000)

    assert [m.type for m in messages] == [JerryCANCmdType.HEARTBEAT]
    # In the one read: the notices did not end it.
    assert backend._bus.messages == []
    notices = [r.getMessage() for r in caplog.records if "CAN error notice" in r.getMessage()]
    assert any("controller tx-warning" in n and "tx errors 103" in n for n in notices)
    assert any("controller tx-passive" in n and "tx errors 135" in n for n in notices)


def test_receive_throttles_repeated_identical_notices(caplog):
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([_raw_message(0x204, _TX_WARNING_NOTICE, error=True)] * 5)

    with caplog.at_level(logging.WARNING, logger=socketcan_jerrycan.__name__):
        backend.ReceiveMessages(max_count=10, collect_ms=5)

    assert sum("CAN error notice" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("controller", [0x01, 0x02])
def test_receive_fails_on_controller_overflow(monkeypatch, controller):
    monkeypatch.setattr(
        "reachaq.device.can_diagnostics.capture_can_diagnostics",
        lambda channel: {"channel": channel},
    )
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([
        _raw_message(0x004, bytes([0x00, controller, 0, 0, 0, 0, 0, 0]), error=True)])

    with pytest.raises(CanTransportReadError) as raised:
        backend.ReceiveMessages(max_count=1, collect_ms=5)

    assert raised.value.category == "overflow"


def test_receive_classifies_malformed_frame(monkeypatch):
    monkeypatch.setattr(
        "reachaq.device.can_diagnostics.capture_can_diagnostics",
        lambda channel: {"channel": channel},
    )
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([_raw_message(0x15 << 5, b"\0")])

    with pytest.raises(CanTransportReadError) as raised:
        backend.ReceiveMessages(max_count=1, collect_ms=5)

    assert raised.value.category == "malformed_frame"


def _malformed_analog_block(address):
    # 8 channels x 3 scans claims 24 samples; one frame holds 16, so decode
    # raises ValueError.
    payload = struct.pack("<BBBBH16H", 0, 8, 3, 0, 1000, *range(16))
    return _raw_message((JerryCANCmdType.ANALOG_BLOCK << 5) | address, payload + b"\x00")


def _pellet_heartbeat():
    return _raw_message(JerryCANCmdType.HEARTBEAT << 5, b"\xff\0")


@pytest.mark.parametrize("address", [0x08, 0x04])  # board types 2 and 1; 0x04 is bit 2 only
def test_receive_skips_undecodable_frame_from_another_board_and_keeps_reading(address):
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([_malformed_analog_block(address), _pellet_heartbeat()])

    messages = backend.ReceiveMessages(max_count=1, collect_ms=5000)

    assert [(m.type, m.dst_id) for m in messages] == [(JerryCANCmdType.HEARTBEAT, 0x00)]
    # Ended by the heartbeat after the malformed frame was read past.
    assert backend._bus.messages == []


@pytest.mark.parametrize("address", [0x00, 0x03])  # board type 0: nodes 0 and 3
def test_receive_still_fails_on_undecodable_frame_from_the_pellet(monkeypatch, address):
    monkeypatch.setattr(
        "reachaq.device.can_diagnostics.capture_can_diagnostics",
        lambda channel: {"channel": channel},
    )
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([_malformed_analog_block(address), _pellet_heartbeat()])

    with pytest.raises(CanTransportReadError) as raised:
        backend.ReceiveMessages(max_count=1, collect_ms=5000)

    assert raised.value.category == "malformed_frame"


def test_receive_warns_once_per_address_about_undecodable_foreign_frames(caplog):
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([
        _malformed_analog_block(0x08),
        _malformed_analog_block(0x08),
        _malformed_analog_block(0x09),
        _pellet_heartbeat(),
    ])

    with caplog.at_level(logging.WARNING, logger=socketcan_jerrycan.__name__):
        messages = backend.ReceiveMessages(max_count=1, collect_ms=5000)

    assert [m.type for m in messages] == [JerryCANCmdType.HEARTBEAT]
    assert backend._bus.messages == []
    warned = [r.getMessage() for r in caplog.records if "undecodable" in r.getMessage()]
    assert len(warned) == 2
    assert any("address 8" in w for w in warned)
    assert any("address 9" in w for w in warned)


def test_receive_preserves_can_operation_error_and_enetdown(monkeypatch):
    class CanOperationError(Exception):
        def __init__(self):
            super().__init__("Network is down")
            self.error_code = errno.ENETDOWN

    monkeypatch.setattr(
        "reachaq.device.can_diagnostics.capture_can_diagnostics",
        lambda channel: {"channel": channel},
    )
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = mock.Mock()
    backend._bus.recv.side_effect = CanOperationError()

    with pytest.raises(CanTransportReadError) as raised:
        backend.ReceiveMessages()

    assert raised.value.category == "network_down"
    assert raised.value.error_code == errno.ENETDOWN
    assert raised.value.__cause__.__class__.__name__ == "CanOperationError"


def test_pellet_runtime_retains_active_status_and_ignores_retired_servo(caplog):
    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(kind="socketcan", fd=True),
    )

    analog = JerryCANMsg()
    analog.type = JerryCANCmdType.ANALOG_OUT
    analog.dst_id = 0
    rgb = JerryCANMsg()
    rgb.type = JerryCANCmdType.RGB_LED
    rgb.dst_id = 0

    assert isinstance(interface._translate(analog), AnalogOutput)
    assert isinstance(interface._translate(rgb), ColorLed)

    retained_motors = []
    with caplog.at_level(logging.WARNING):
        for motor_id in range(3):
            status = JerryCANMsg()
            status.type = JerryCANCmdType.SERVO_STATUS
            status.dst_id = 0
            status.servo_status.motor_id = motor_id
            translated = interface._translate(status)
            if translated is not None:
                retained_motors.append(translated.motor)

    assert retained_motors == [
        Motor.PELLET_COVER_SERVO,
        Motor.PELLET_LOAD_SERVO,
    ]
    assert "Unknown motor id" not in caplog.text

    status.servo_status.motor_id = 3
    assert interface._translate(status) is None
    assert "isa_servo=True motor_id=3" in caplog.text


def test_send_refuses_large_jerrycan_payload_without_fd():
    backend = SocketCanJerryCAN(CanTransportConfiguration(
        kind="socketcan",
        fd=False,
    ))
    backend._bus = FakeBus()
    message = JerryCANMsg()
    message.type = JerryCANCmdType.CFG_READ
    message.cfg_read.type = JerryCANCfgMsg.Type.STEPPER

    assert backend.SendMessage(message, 0) == -errno.EMSGSIZE
    assert backend._bus.sent == []


def test_send_uses_can_fd_with_bitrate_switch(monkeypatch):
    fake_can = types.SimpleNamespace(
        Message=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "can", fake_can)
    backend = SocketCanJerryCAN(CanTransportConfiguration(
        kind="socketcan",
        fd=True,
    ))
    backend._bus = FakeBus()
    message = JerryCANMsg()
    message.type = JerryCANCmdType.CFG_READ
    message.cfg_read.type = JerryCANCfgMsg.Type.STEPPER

    assert backend.SendMessage(message, 0) == 0
    assert len(backend._bus.sent) == 1
    assert backend._bus.sent[0].is_fd is True
    assert backend._bus.sent[0].bitrate_switch is True


def test_can_interface_does_not_query_configuration_without_board_address(monkeypatch):
    class MissingBoardBus:
        close_called = False

        def Open(self):
            return 0

        def Close(self):
            self.close_called = True
            return 0

        def ReceiveMessages(self, max_count, collect_ms):
            del max_count, collect_ms
            return []

    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(
            kind=CanTransportKind.SOCKETCAN,
            fd=True,
        ),
    )
    # This unit test exercises missing-board cleanup, not cross-process channel
    # ownership. Keep it deterministic when a real reachAQ app owns can0.
    interface._channel_ownership = mock.Mock()
    interface._jc = MissingBoardBus()
    query_called = []
    monkeypatch.setattr(interface, "_query_configuration", lambda: query_called.append(True))
    monkeypatch.setattr(can_interface, "get_perf_now", lambda: 0.0)
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(
        can_interface,
        "time",
        types.SimpleNamespace(perf_counter=lambda: next(clock)),
    )

    assert interface.open() is False
    assert interface._jc.close_called is True
    assert query_called == []


def test_stepper_status_frame_decodes_the_status_byte():
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.STEPPER_STATUS, 0)  # 0x2C0
    payload = struct.pack("<BBBBff", 1, 0x03, 0, 1, 2.5, 3.0)

    message = socketcan_jerrycan.decode_frame(can_id, payload)

    assert message.stepper_status.motor_id == 1
    assert message.stepper_status.status == 0x03
    assert message.stepper_status.limit_switch == 1
    assert message.stepper_status.position == 2.5


@pytest.mark.parametrize("payload, error", [
    (b"\xff\xff\xff\xff", -1),
    (b"\x8c\xff\xff\xff", -116),
    (b"\xf5\xff\xff\xff", -11),
    (b"\x00\x00\x00\x00", 0),
])
def test_ack_error_decodes_as_a_signed_int32(payload, error):
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.ACKNOWLEDGE, 0)
    message = socketcan_jerrycan.decode_frame(can_id, payload + bytes([42]))
    assert message.ack.error == error
    assert message.uuid == 42


# -- pellet 2.4.0: STEPPER_STOP, STEPPER_FAULT, BOOT_CHAIN (jerrycan_types.h) -----------

def _socketcan_interface():
    return CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(kind=CanTransportKind.SOCKETCAN, fd=True),
    )


def test_stepper_stop_encodes_its_type_mask_and_uuid():
    message = JerryCANMsg()
    message.type = JerryCANCmdType.STEPPER_STOP
    message.uuid = 0
    message.stepper_stop.motor_mask = 0x05

    can_id, payload = socketcan_jerrycan.encode_frame(message, 0)

    assert JerryCANCmdType.STEPPER_STOP == 0x2C
    assert can_id == 0x2C << 5
    assert payload == b"\x05\x00"  # 1-byte motor_mask, then the uuid
    decoded = socketcan_jerrycan.decode_frame(can_id, payload)
    assert decoded.type == JerryCANCmdType.STEPPER_STOP
    assert decoded.stepper_stop.motor_mask == 0x05
    assert decoded.uuid == 0


def test_stepper_fault_decodes_its_20_bytes_and_the_faulted_uuid():
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.STEPPER_FAULT, 0)
    payload = struct.pack("<BBBBiiff", 2, 2, 0x02, 0, 571, 1142, 6.25, 192.0)
    assert len(payload) == 20

    message = socketcan_jerrycan.decode_frame(can_id, payload + bytes([77]))

    assert JerryCANCmdType.STEPPER_FAULT == 0x2D
    assert message.uuid == 77
    fault = message.stepper_fault
    assert (fault.motor_id, fault.reason, fault.flags) == (2, 2, 0x02)
    assert (fault.pulses_issued, fault.pulses_planned) == (571, 1142)
    assert fault.believed_position == 6.25
    assert fault.pulses_per_turn == 192.0


def test_the_receive_filter_admits_stepper_fault_frames():
    filters = socketcan_jerrycan._jerrycan_filters()
    assert {"can_id": 0x2D << 5, "can_mask": 0x7E0, "extended": False} in filters


def test_boot_chain_response_decodes_its_10_bytes_with_the_uuid_at_offset_11():
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.BOOTLOADER_RESPONSE, 0)
    flags = (socketcan_jerrycan.BOOT_CHAIN_STAGE0 | socketcan_jerrycan.BOOT_CHAIN_MCUBOOT_VALID
             | socketcan_jerrycan.BOOT_CHAIN_HEADER_OK)
    payload = struct.pack("<BBI4sB", 7, flags, 12, b"\xde\xad\xbe\xef", 0x2A)
    assert len(payload) == 11

    message = socketcan_jerrycan.decode_frame(can_id, payload + bytes([9]))

    response = message.bootloader_response
    assert response.type == socketcan_jerrycan.JerryCANBootloaderCmd.SubCommand.BOOT_CHAIN
    assert response.boot_chain.flags == flags
    assert response.boot_chain.desc_sequence == 12
    assert response.boot_chain.mcuboot_sha256_prefix == b"\xde\xad\xbe\xef"
    assert response.boot_chain.running_build_num == 0x2A
    assert message.uuid == 9


def test_boot_chain_read_error_flag_decodes():
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.BOOTLOADER_RESPONSE, 0)
    payload = struct.pack("<BBI4sB", 7, socketcan_jerrycan.BOOT_CHAIN_READ_ERROR, 0, bytes(4), 0)

    message = socketcan_jerrycan.decode_frame(can_id, payload + b"\x00")

    assert message.bootloader_response.boot_chain.flags == 0x80


def test_a_legacy_version_response_still_reads_its_uuid_at_offset_7():
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.BOOTLOADER_RESPONSE, 0)
    payload = struct.pack("<BBBBBBB", 0, 2, 4, 0, 2, 3, 1)

    message = socketcan_jerrycan.decode_frame(can_id, payload + bytes([5]))

    assert message.bootloader_response.version.running_version_minor == 4
    assert message.uuid == 5


def test_a_boot_chain_response_from_the_pellet_does_not_end_the_reader(caplog):
    # Before 2.4.0 hosts raised "Malformed JerryCAN frame" on subcommand 7, which ended
    # the reader and with it the pellet connection.
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    boot_chain = struct.pack("<BBI4sB", 7, 0x07, 3, b"\x01\x02\x03\x04", 1) + b"\x00"
    backend._bus = FakeBus([
        _raw_message(JerryCANCmdType.BOOTLOADER_RESPONSE << 5, boot_chain),
        _pellet_heartbeat(),
    ])

    messages = backend.ReceiveMessages(max_count=2, collect_ms=5000)

    assert [m.type for m in messages] == [JerryCANCmdType.BOOTLOADER_RESPONSE, JerryCANCmdType.HEARTBEAT]
    interface = _socketcan_interface()
    with caplog.at_level(logging.INFO, logger="reachaq.device.can_interface"):
        assert interface._translate(messages[0]) is None
    assert "pellet boot chain: flags=0x07 desc_sequence=3 mcuboot_sha256_prefix=01020304" in caplog.text


def test_an_unknown_bootloader_subcommand_is_kept_as_a_number_and_dropped(caplog):
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([
        _raw_message(JerryCANCmdType.BOOTLOADER_RESPONSE << 5, bytes([0x42, 1, 2, 3, 4, 5, 6, 0])),
        _pellet_heartbeat(),
    ])

    messages = backend.ReceiveMessages(max_count=2, collect_ms=5000)

    assert messages[0].bootloader_response.type == 0x42
    with caplog.at_level(logging.INFO, logger="reachaq.device.can_interface"):
        assert _socketcan_interface()._translate(messages[0]) is None
    assert "unknown subcommand 66" in caplog.text
    assert messages[1].type == JerryCANCmdType.HEARTBEAT


def test_can_interface_translates_a_stepper_fault():
    from reachaq.device import StepperFault
    from reachaq.device.stepper_motor import turns_to_mm

    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.STEPPER_FAULT, 0)
    payload = struct.pack("<BBBBiiff", 2, 2, 0x02, 0, 571, 1142, 2.5, 192.0) + bytes([31])
    message = socketcan_jerrycan.decode_frame(can_id, payload)

    fault = _socketcan_interface()._translate(message)

    assert isinstance(fault, StepperFault)
    assert fault.target == Target.PELLET_DEVICE
    assert fault.motor == Motor.PELLET_Z_MOTOR
    assert fault.reason_name == "LIMIT" and fault.toward_switch and not fault.is_homing
    assert (fault.pulses_issued, fault.pulses_planned, fault.uuid) == (571, 1142, 31)
    assert fault.believed_position_mm == pytest.approx(turns_to_mm(2.5))
    assert not fault.needs_home_only
    # Raw counts only, never "lost steps" (design I8: bench step 1b has not run).
    assert "571 pulses issued of 1142 planned" in fault.describe()
    assert "lost" not in fault.describe()


def test_can_interface_records_the_pellet_capabilities_and_bit_9():
    interface = _socketcan_interface()
    assert interface.board_capabilities is None
    assert not interface.has_stepper_motion_safety
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.CAPABILITIES_RESPONSE, 0)

    for caps, motion_safety in ((0x17, False), (0x217, True)):
        message = socketcan_jerrycan.decode_frame(can_id, struct.pack("<BII", 1, caps, 5) + b"\x00")
        interface._translate(message)
        assert interface.board_capabilities == caps
        assert interface.has_stepper_motion_safety is motion_safety


def test_stepper_stop_goes_out_with_uuid_0_and_leaves_the_command_uuids_alone():
    interface = _socketcan_interface()
    interface._pellet_addr = 0
    sent = []
    interface._jc = types.SimpleNamespace(StepperStop=lambda dst, mask, uuid: sent.append((dst, mask, uuid)) or 0)
    before = CanInterface.uuid()

    assert interface.stepper_stop(0) is True

    assert sent == [(0, 0, 0)]
    assert CanInterface.uuid() == before


def test_a_backend_without_stepper_stop_reports_no_motion_safety():
    # pyjerrycan has no StepperStop: the host stays on ESTOP even with bit 9.
    interface = _socketcan_interface()
    interface._jc = types.SimpleNamespace()
    interface._board_capabilities = 0x217

    assert not interface.has_stepper_motion_safety
    assert interface.stepper_stop(0) is False


# -- per-axis travel clamp (design H8) -----------------------------------------------

def _clamp_interface(max_pos):
    from reachaq.device import StepperConfig

    interface = _socketcan_interface()
    interface._pellet_addr = 0
    moves = []

    def stepper_move(addr, motor_id, position, velocity, acceleration, abs_or_rel, save, uuid):
        moves.append((motor_id, position, int(abs_or_rel), save))
        return 0

    interface._jc = types.SimpleNamespace(StepperMove=stepper_move)
    config = StepperConfig.from_dict({} if max_pos is None else {"min_pos": 0, "max_pos": max_pos})
    interface.z_config = config
    interface.x_config = StepperConfig.from_dict({"min_pos": 0, "max_pos": 35.0})
    return interface, moves


def test_an_absolute_target_past_max_pos_is_clamped_to_it():
    from reachaq.device.stepper_motor import mm_to_turns

    interface, moves = _clamp_interface(30.0)

    assert interface.move_motor_z(33.0)
    assert interface.move_motor_x(36.0)
    assert interface.move_motor_z(12.0)

    assert moves[0][1] == pytest.approx(mm_to_turns(30.0))  # z: 30 mm, as the 2.4.0 cap
    assert moves[1][1] == pytest.approx(mm_to_turns(35.0))  # x: its own 35 mm
    assert moves[2][1] == pytest.approx(mm_to_turns(12.0))  # inside the travel: unchanged


def test_a_saved_send_position_past_max_pos_is_clamped_too():
    from reachaq.device.stepper_motor import mm_to_turns

    interface, moves = _clamp_interface(30.0)

    assert interface.set_motor_z(31.0)

    assert moves == [(2, pytest.approx(mm_to_turns(30.0)), 0, True)]


def test_a_relative_move_is_held_to_max_pos():
    from reachaq.core import Offset3DTuple
    from reachaq.device.stepper_motor import mm_to_turns

    interface, moves = _clamp_interface(30.0)
    interface._last_positions = Offset3DTuple(0.0, 0.0, 25.0)

    assert interface.move_motor_z(10.0, relative=True)

    assert moves[0][1] == pytest.approx(mm_to_turns(5.0))  # 25 + 10 held to 30


def test_without_max_pos_the_absolute_15_turn_cap_stays():
    interface, moves = _clamp_interface(None)

    assert interface.move_motor_z(100.0)

    assert moves[0][1] == pytest.approx(15.0)


# -- review I1: a mid-session reboot or firmware change forgets bit 9 ------------------

def _status_frame(*, boot_id=None, sequence=1):
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.STEPPER_STATUS, 0)
    payload = struct.pack("<BBBBff", 2, 0x08, 0, 0, 1.0, 1.0) + b"\x00"
    if boot_id is not None:
        payload += struct.pack("<BBBBIIQ", 0xA7, 1, 0, 0, boot_id, sequence, 1000 * sequence)
    return socketcan_jerrycan.decode_frame(can_id, payload)


def _bit_9_interface():
    interface = _socketcan_interface()
    interface._pellet_addr = 0
    asked = []
    interface._jc = types.SimpleNamespace(
        StepperStop=lambda *args: 0,
        RequestCapabilities=lambda dst: asked.append("capabilities") or 0,
        BootloaderCommand=lambda dst, subcmd: asked.append("version") or 0,
    )
    interface._board_capabilities = 0x217
    return interface, asked


def test_a_pellet_reboot_forgets_bit_9_and_asks_again(caplog):
    interface, asked = _bit_9_interface()
    interface._translate(_status_frame(boot_id=0x1111, sequence=1))
    interface._translate(_status_frame(boot_id=0x1111, sequence=2))
    assert interface.has_stepper_motion_safety and asked == []

    with caplog.at_level(logging.WARNING, logger="reachaq.device.can_interface"):
        interface._translate(_status_frame(boot_id=0x2222, sequence=0))

    assert interface.board_capabilities is None
    assert not interface.has_stepper_motion_safety, "stops are ESTOP until the board answers"
    assert asked == ["capabilities", "version"]
    assert "rebooted" in caplog.text


def test_a_board_reverted_to_2_3_1_is_seen_by_its_missing_timing_trailer():
    # 2.3.1 sends no timing trailer, so its boot id is never seen: the missing trailer is.
    interface, asked = _bit_9_interface()
    interface._translate(_status_frame(boot_id=0x1111))

    interface._translate(_status_frame(boot_id=None))

    assert interface.board_capabilities is None and not interface.has_stepper_motion_safety
    assert asked == ["capabilities", "version"]
    # The board's answer decides again: 2.3.1 reports 0x04, so it stays on ESTOP.
    can_id = socketcan_jerrycan.encode_can_id(JerryCANCmdType.CAPABILITIES_RESPONSE, 0)
    interface._translate(socketcan_jerrycan.decode_frame(can_id, struct.pack("<BII", 1, 0x04, 9) + b"\x00"))
    assert interface.board_capabilities == 0x04 and not interface.has_stepper_motion_safety


def test_a_board_without_timing_trailers_is_not_mistaken_for_a_reboot():
    interface, asked = _bit_9_interface()
    interface._board_capabilities = 0x204  # bit 9 without the trailer bit

    interface._translate(_status_frame(boot_id=None))

    assert interface.board_capabilities == 0x204 and asked == []


# -- review m8: a backend that cannot send STOP refuses a bit 9 board -----------------

def test_a_backend_without_stop_refuses_to_open_a_bit_9_board(monkeypatch):
    from reachaq.device import BoardCapabilities
    from reachaq.device.can_interface import PelletBackendUnsupportedError

    class NoStopBus:  # pyjerrycan's JerryCAN has no StepperStop
        close_called = False

        def Open(self):
            return 0

        def Close(self):
            self.close_called = True
            return 0

        def ReceiveMessages(self, max_count, collect_ms):
            return [_status_frame(boot_id=None)]

        def RequestCapabilities(self, dst):
            return 0

    interface = _socketcan_interface()
    interface._channel_ownership = mock.Mock()
    interface._jc = NoStopBus()
    monkeypatch.setattr(interface, "_query_configuration", lambda: None)
    monkeypatch.setattr(interface, "_translate", lambda message: None)
    monkeypatch.setattr(interface, "_assign_address", lambda message: setattr(interface, "_pellet_addr", 0))
    monkeypatch.setattr(interface, "get_response", lambda *args, **kwargs: BoardCapabilities(
        target=Target.PELLET_DEVICE, wire_schema_version=1, capabilities=0x217, boot_id=1))

    with pytest.raises(PelletBackendUnsupportedError, match="cannot send STEPPER_STOP"):
        interface.open()

    assert interface._jc.close_called and not interface.is_open


def test_a_backend_without_stop_still_opens_a_2_3_1_board(monkeypatch):
    from reachaq.device import BoardCapabilities

    interface = _socketcan_interface()
    interface._jc = types.SimpleNamespace()

    interface._refuse_backend_without_stop(0x04)  # no raise
    interface._jc = types.SimpleNamespace(StepperStop=lambda *a: 0)
    interface._refuse_backend_without_stop(0x217)  # SocketCAN: no raise
