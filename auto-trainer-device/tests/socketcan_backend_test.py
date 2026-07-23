import errno
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


def test_close_marks_interface_closed_even_if_backend_close_fails():
    interface = object.__new__(CanInterface)
    interface._is_open = True
    interface._jc = mock.Mock()
    interface._jc.Close.side_effect = RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        interface.close()

    assert interface.is_open is False


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
    assert captured["ignore_rx_error_frames"] is True
    assert captured["can_filters"]
    assert "bitrate" not in captured
    assert "data_bitrate" not in captured


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


def test_receive_ignores_unsupported_and_malformed_frames():
    valid_id = (JerryCANCmdType.HEARTBEAT << 5) | 0x01
    backend = SocketCanJerryCAN(CanTransportConfiguration(
        kind="socketcan",
        fd=True,
    ))
    backend._bus = FakeBus([
        _raw_message(0, error=True),
        _raw_message(valid_id, remote=True),
        _raw_message(valid_id, extended=True),
        _raw_message(0x15 << 5, b"\0"),
        _raw_message(valid_id, b"\xff\0"),
    ])

    messages = backend.ReceiveMessages(max_count=1, collect_ms=5)

    assert len(messages) == 1
    assert messages[0].type == JerryCANCmdType.HEARTBEAT
    assert messages[0].dst_id == 0x01


def test_pellet_only_runtime_ignores_unused_status_messages():
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

    assert interface._translate(analog) is None
    assert interface._translate(rgb) is None

    retained_motors = []
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
    assert Motor.TUNNEL_GATE_SERVO not in interface._pellet_board_last_status_perf_c


def test_full_runtime_retains_auxiliary_status_messages():
    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE, Target.MAGNET_DEVICE),
        can_transport=CanTransportConfiguration(kind="socketcan", fd=True),
    )

    analog = JerryCANMsg()
    analog.type = JerryCANCmdType.ANALOG_OUT
    analog.dst_id = 0
    rgb = JerryCANMsg()
    rgb.type = JerryCANCmdType.RGB_LED
    rgb.dst_id = 0
    gate = JerryCANMsg()
    gate.type = JerryCANCmdType.SERVO_STATUS
    gate.dst_id = 0
    gate.servo_status.motor_id = 2

    assert isinstance(interface._translate(analog), AnalogOutput)
    assert isinstance(interface._translate(rgb), ColorLed)
    translated_gate = interface._translate(gate)
    assert isinstance(translated_gate, ServoStatus)
    assert translated_gate.motor == Motor.TUNNEL_GATE_SERVO


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
    monkeypatch.setattr(can_interface.time, "perf_counter", lambda: next(clock))

    assert interface.open() is False
    assert interface._jc.close_called is True
    assert query_called == []
