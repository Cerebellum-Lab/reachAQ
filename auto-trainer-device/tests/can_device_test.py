import contextlib
import logging
import math
import queue
import threading
import time
import uuid
from functools import partial
from unittest import mock

import pytest
from autotrainer.core import RawValueHolder

from autotrainer.core.message import SystemStatusMessageKind, SystemCommandKind
from autotrainer.device import (
    CanDevice,
    DeviceApi,
    Target,
    LoadCellReading,
    PressureReading,
    SensorStatus,
    MagnetDigitalInputs,
    Motor,
    StepperStatus,
    ServoStatus,
    ServoConfig,
    StepperConfig,
    MotorSteps,
    DeviceConnection,
    MotorConfigurationFile,
)
from autotrainer.device.can_device import (
    default_move_retract,
    default_load_pellet,
    default_send_pellet,
)
from autotrainer.device.device_connection import _REQUEST_DISCONNECT

_expected = None

def data_callback(kind: int, response):
    assert kind == _expected
    del response  # uncheck atm


def test_device_connection_throttles_empty_reads():
    """An idle CAN backend must not spin and starve the application's GUI thread."""
    interface = mock.Mock()
    interface.can_read.return_value = True
    interface.read.return_value = []
    interface.is_open = True

    device = mock.Mock()
    device.device_interface = interface
    connection = DeviceConnection(device, message_queue=queue.Queue())
    connection._collect_ms = 5

    thread = threading.Thread(target=connection._run_connected)
    thread.start()
    time.sleep(0.055)
    connection._cmd_queue.put((_REQUEST_DISCONNECT, None, None))
    thread.join(1)

    assert not thread.is_alive()
    # Disconnect commands are intentionally checked every 250 ms, so include
    # that interval while still proving this was a throttled poll, not a spin.
    assert 2 <= interface.read.call_count <= 80


def test_pellet_only_connection_skips_unused_motor_configurations():
    device = CanDevice(
        api=DeviceApi(message_callback=data_callback),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
    )
    connection = DeviceConnection(device, message_queue=queue.Queue())
    connection.send_message = mock.Mock()

    @contextlib.contextmanager
    def no_wait(*args, **kwargs):
        yield

    connection.await_acknowledge = no_wait
    connection.use_motor_configurations(MotorConfigurationFile())

    configured_motors = {
        call.args[1][0]
        for call in connection.send_message.call_args_list
    }
    assert Motor.TUNNEL_MAGNET_SERVO not in configured_motors
    assert Motor.TUNNEL_GATE_SERVO not in configured_motors
    assert Motor.TUNNEL_FAN_SERVO not in configured_motors
    assert {
        Motor.PELLET_X_MOTOR,
        Motor.PELLET_Y_MOTOR,
        Motor.PELLET_Z_MOTOR,
        Motor.PELLET_LOAD_SERVO,
        Motor.PELLET_COVER_SERVO,
    } <= configured_motors


@pytest.fixture
def expected_tok() -> RawValueHolder:
    value = RawValueHolder(value=None)
    return value


def api_msg_cb(msg_kind, data, *, event, tokens_acked, expected_tok: RawValueHolder):
    # print(msg_kind, data)
    if msg_kind == SystemStatusMessageKind.ACKNOWLEDGE:
        tok, perf_c = data
        tokens_acked.append(tok)
        if tok is not None and expected_tok is not None and tok == expected_tok.value:
            expected_tok.value = None
            if event is not None:
                event.set()


@pytest.fixture
def tokens_acked():
    return []


@pytest.fixture
def expected_tok_event():
    return threading.Event()


@pytest.fixture
def device_conn(device):
    msg_q = queue.Queue()
    msg_cb = device.api.message_callback
    dc = DeviceConnection(device, message_queue=msg_q)
    dc.request_connect()
    device.api.message_callback = msg_cb
    try:
        yield dc
    finally:
        dc.request_disconnect()


@pytest.fixture  # (scope="module")
def device(expected_tok_event, expected_tok, tokens_acked) -> CanDevice:  # noqa
    device = CanDevice(api=DeviceApi(message_callback=data_callback), force_emulation=True)
    # unneeded, at least with emulation iface:
    # device._interface.magnet_address = 0x40
    # device._interface.pellet_address = 0x01
    # device.notify_message(_REQUEST_CONNECT)
    device.api.message_callback = partial(
        api_msg_cb,
        tokens_acked=tokens_acked,
        expected_tok=expected_tok,
        event=expected_tok_event,
    )
    device.connect()
    device.device_interface.open()
    try:
        yield device  # noqa
    finally:
        device.disconnect()


@pytest.mark.parametrize("kind, tag, data", [
    (SystemCommandKind.REQUEST_VERSION, 101, None),
    (SystemCommandKind.UPDATE_SCALE_TARE, 102, None),
    (SystemCommandKind.SET_X, 103, 10),
    (SystemCommandKind.SET_Y, 104, 15),
    (SystemCommandKind.SET_Z, 105, 20),
    (SystemCommandKind.MOVE_X, 106, 10),
    (SystemCommandKind.MOVE_Y, 106, 15),
    (SystemCommandKind.MOVE_Z, 108, 20),
    (SystemCommandKind.MOVE_MAGNET_SERVO, 109, 25),
    (SystemCommandKind.MOVE_GATE_SERVO, 110, 30),
    (SystemCommandKind.MOVE_LOAD_SERVO, 111, 35),
    (SystemCommandKind.MOVE_COVER_SERVO, 112, 40),
    (SystemCommandKind.SEND_HOME, 113, None),
    (SystemCommandKind.LOAD_PELLET, 114, None),
    (SystemCommandKind.SEND_PELLET, 115, None),
    (SystemCommandKind.RELEASE_PELLET, 116, None),
    (SystemCommandKind.COVER_PELLET, 117, None),
    (SystemCommandKind.PLAY_TONE, 118, None),
    (SystemCommandKind.DELAY, 119, 0.5),
    (SystemCommandKind.READ_MOTOR_CONFIGURATION, 120, Motor.PELLET_X_MOTOR),
    (SystemCommandKind.WRITE_MOTOR_CONFIGURATION, 121, (Motor.PELLET_X_MOTOR, StepperConfig())),
    (SystemCommandKind.SEND_FIXED_XYZ, 122, None),
    (SystemCommandKind.SEND_FIXED_XYZ, 123, None),
    (SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE, 124, default_move_retract()),
])
def test_notify_command(device, kind, tag, data):
    device.notify_message(kind, data, tag)


@pytest.mark.parametrize("data, kind", [
    (LoadCellReading(target=Target.MAGNET_DEVICE, load=13), None),
    (PressureReading(Target.MAGNET_DEVICE, pressure=14), None),
    (SensorStatus(Target.PELLET_DEVICE, temperature_c=27.3, humidity_percent=64.2), None),
    (MagnetDigitalInputs(Target.MAGNET_DEVICE, continuity_0=False, continuity_1=True), None),
    (StepperStatus(Target.PELLET_DEVICE, Motor.PELLET_X_MOTOR, 10, 2.0, False),
     SystemStatusMessageKind.PELLET_MOTOR_X),
    (ServoStatus(Target.PELLET_DEVICE, Motor.PELLET_LOAD_SERVO, 40),
     SystemStatusMessageKind.PELLET_LOAD),
    (ServoConfig(Target.MAGNET_DEVICE, Motor.PELLET_X_MOTOR, 0, 0, 0, 0, 0, 0),
     SystemStatusMessageKind.MOTOR_CONFIGURATION),
    (StepperConfig(Target.PELLET_DEVICE, Motor.PELLET_X_MOTOR, 0, 0, 0, 0, False),
     SystemStatusMessageKind.MOTOR_CONFIGURATION),
])
def test_notify_data(device, data, kind):
    global _expected

    if kind:
        _expected = kind

    device.notify_data([data])


@pytest.mark.parametrize("kind,data", (
    (SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE, default_move_retract()),
    (SystemCommandKind.SET_LOAD_PELLET_PROCEDURE, default_load_pellet()),
    (SystemCommandKind.SET_SEND_PELLET_PROCEDURE, default_send_pellet()),
))
def test_set_procedures(
    expected_tok,
    expected_tok_event,
    tokens_acked,
    device,
    kind,
    data,
):
    ctx = uuid.uuid4()
    expected_tok.value = ctx
    device.notify_message(kind, data, context=ctx)
    expected_tok_event.wait(3)  # should be quite faster
    assert ctx in tokens_acked


def test_move_relative(
    expected_tok,
    expected_tok_event,
    tokens_acked,
    device,
    device_conn,
):
    # we rely on that on start:
    dev_positions = device.device_interface._positions  # noqa
    assert dev_positions[Motor.PELLET_X_MOTOR] == 0
    assert dev_positions[Motor.PELLET_Y_MOTOR] == 0
    ctx = uuid.uuid4()
    expected_tok.value = ctx
    device.notify_message(SystemCommandKind.SEND_RETRACT, None, context=ctx)
    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    tokens_acked.clear()
    # emulation iface doesn't check motor limits, so the result position is 0 + retract_offset,
    # default one being -15, so we get -15 :
    assert math.isclose(dev_positions[Motor.PELLET_Y_MOTOR], -15, abs_tol=0.1)
    #
    expected_tok_event.clear()
    expected_tok.value = ctx
    device.notify_message(SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE,
                          MotorSteps("custom", [{'y_rel': 20}, {'x_rel': -5}]),
                          context=ctx)
    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    tokens_acked.clear()
    expected_tok_event.clear()
    expected_tok.value = ctx
    device.notify_message(SystemCommandKind.SEND_RETRACT, None, context=ctx)
    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    # tokens_acked.clear()
    assert math.isclose(dev_positions[Motor.PELLET_X_MOTOR], -5, abs_tol=0.1)  # -5
    assert math.isclose(dev_positions[Motor.PELLET_Y_MOTOR], 5, abs_tol=0.1)  # -15 + 20 == 5


def test_can_connect_twice(device, caplog):
    with caplog.at_level(logging.DEBUG):
        device.connect()
    assert "CAN command Handler thread already alive" in caplog.text
    assert device.connected
    assert device.device_interface.is_open is True


def test_rel_move_succeed_after_uuid_ack_timeout(
    expected_tok,
    expected_tok_event,
    tokens_acked,
    device,
    device_conn,
    monkeypatch,
    caplog,
):
    orig_move_y = device.device_interface.move_motor_y
    def ret_move(*args, **kwargs):
        # consume one uuid, but don't insert ack into return messages as with emulation iface
        device.device_interface.next_uuid()
        # restore orig move:
        device.device_interface.move_motor_y = orig_move_y
        # return True to fake command written to CAN bus ok:
        return True
    m = mock.MagicMock()
    m.side_effect = ret_move
    device.device_interface.move_motor_y = m
    ctx = uuid.uuid4()
    expected_tok.value = ctx
    device.default_command_ack_timeout_duration = 0.5

    ack_timeout_engaged = False
    ack_timeout_engaged_count = 0
    def dev_prop_changed(name, value, old):
        if name == device.UUID_ACK_TIMEOUT_ENGAGED:
            nonlocal ack_timeout_engaged, ack_timeout_engaged_count
            ack_timeout_engaged = value
            if value:
                ack_timeout_engaged_count += 1

    device.property_changed += dev_prop_changed

    device.notify_message(SystemCommandKind.SEND_RETRACT, None, context=ctx)

    expected_tok_event.wait(3)
    assert ctx in tokens_acked
    assert not ack_timeout_engaged
    assert ack_timeout_engaged_count == 1
    assert device._commands_handler_thread.is_alive()
