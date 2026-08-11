import errno
import logging
import socket
import sys
import types
from unittest import mock

import pytest

from autotrainer.device import (
    AnalogOutput,
    CanTransportConfiguration,
    CanTransportKind,
    ColorLed,
    Motor,
    ServoStatus,
    Target,
)
from autotrainer.device import can_interface
from autotrainer.device import socketcan_jerrycan
from autotrainer.device.can_interface import CanInterface
from autotrainer.device.socketcan_jerrycan import (
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


def test_receive_classifies_bus_off_error_frame(monkeypatch):
    monkeypatch.setattr(
        "autotrainer.device.can_diagnostics.capture_can_diagnostics",
        lambda channel: {"channel": channel, "interface_state": "DOWN"},
    )
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([_raw_message(0x40, error=True)])

    with pytest.raises(CanTransportReadError) as raised:
        backend.ReceiveMessages(max_count=1, collect_ms=5)

    assert raised.value.category == "bus_off"
    assert raised.value.diagnostics["interface_state"] == "DOWN"


def test_receive_classifies_malformed_frame(monkeypatch):
    monkeypatch.setattr(
        "autotrainer.device.can_diagnostics.capture_can_diagnostics",
        lambda channel: {"channel": channel},
    )
    backend = SocketCanJerryCAN(CanTransportConfiguration(kind="socketcan", fd=True))
    backend._bus = FakeBus([_raw_message(0x15 << 5, b"\0")])

    with pytest.raises(CanTransportReadError) as raised:
        backend.ReceiveMessages(max_count=1, collect_ms=5)

    assert raised.value.category == "malformed_frame"


def test_receive_preserves_can_operation_error_and_enetdown(monkeypatch):
    class CanOperationError(Exception):
        def __init__(self):
            super().__init__("Network is down")
            self.error_code = errno.ENETDOWN

    monkeypatch.setattr(
        "autotrainer.device.can_diagnostics.capture_can_diagnostics",
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
