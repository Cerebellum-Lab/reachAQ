import itertools
import logging
import queue
import threading
from types import SimpleNamespace

import pytest

from autotrainer.core import ObservableObject, SystemCommandKind, SystemStatusMessageKind
from autotrainer.device import DeviceApi, DeviceConnection
from autotrainer.device.can_device import CanDevice
from tools.acquisition.model import hardware_model as hardware_model_module
from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


def _device(seen):
    device = object.__new__(CanDevice)
    device._interface = SimpleNamespace(uuid=lambda: 5)
    device._want_exit = threading.Event()
    device._commands_handler_thread = None
    device._commands_queue = queue.Queue()
    device.latency_observer = lambda stage, fields: seen.append((stage, fields))
    return device


def test_queuing_a_command_with_a_token_is_observed():
    seen = []
    device = _device(seen)

    device.notify_message(SystemCommandKind.SEND_PELLET, None, "tok-1")

    stage, fields = seen[0]
    assert stage == "enqueue"
    assert fields["token"] == "tok-1"
    assert fields["kind"] == "SEND_PELLET"
    assert device._commands_queue.get_nowait() == (SystemCommandKind.SEND_PELLET, None, "tok-1")


def test_an_ack_is_observed_with_its_uuid_and_kernel_time():
    seen = []
    device = _device(seen)

    device._handle_ack(SimpleNamespace(target="pellet", uuid=12, perf_c=3.5,
                                       timestamp_ns=2_000_000_000))

    stage, fields = seen[0]
    assert stage == "ack"
    assert fields["can_uuid"] == 12
    assert fields["perf"] == 3.5
    assert fields["kernel_wall"] == 2.0


def test_a_failing_observer_never_reaches_the_command_path():
    device = _device([])

    def broken(*_args):
        raise RuntimeError("recorder down")

    device.latency_observer = broken
    device.notify_message(SystemCommandKind.SEND_PELLET, None, "tok-2")
    assert device._commands_queue.qsize() == 1


def test_the_hardware_model_forwards_the_observer_to_its_device():
    model = object.__new__(HardwareModel)
    model._can_device = SimpleNamespace(latency_observer=None)

    def observer(stage, fields):
        return None

    model.set_can_latency_observer(observer)
    assert model._can_device.latency_observer is observer


def _run_through_the_loop(kind, data, observer):
    """Queue one tokened command on an emulated device and wait for its acknowledgement.

    The unit tests above never start the command thread, so only this reaches the
    hooks inside the real handler loop (dequeue, send) and the reader's ack.
    """
    token = "loop-token"
    acknowledged = threading.Event()

    def on_message(message_kind, payload):
        if message_kind == SystemStatusMessageKind.ACKNOWLEDGE and payload[0] == token:
            acknowledged.set()

    device = CanDevice(api=DeviceApi(message_callback=on_message), force_emulation=True)
    device.latency_observer = observer
    device.connect()
    device.device_interface.open()
    connection = DeviceConnection(device, message_queue=queue.Queue())
    connection.request_connect()
    device.api.message_callback = on_message
    try:
        device.notify_message(kind, data, context=token)
        return token, acknowledged.wait(3)
    finally:
        connection.request_disconnect()
        device.disconnect()


def test_a_command_with_a_uuid_is_stamped_at_every_stage_in_the_real_loop():
    seen = []

    token, acknowledged = _run_through_the_loop(
        SystemCommandKind.SET_X, 10, lambda stage, fields: seen.append((stage, dict(fields))))

    assert acknowledged
    stages = [stage for stage, _ in seen]
    # The ack is recorded on the bus reader's thread, which can beat the command
    # thread's send row by microseconds, so only the order before them is fixed.
    assert stages[:2] == ["enqueue", "dequeue"]
    assert sorted(stages[2:]) == ["ack", "send"]
    by_stage = dict(seen)
    enqueue, dequeue, send, ack = (by_stage[s] for s in ("enqueue", "dequeue", "send", "ack"))
    assert enqueue["token"] == dequeue["token"] == send["token"] == token
    assert send["kind"] == "SET_X"
    # The uuid is what joins the token to the board's ack.
    assert send["can_uuid"] is not None
    assert ack["can_uuid"] == send["can_uuid"]
    assert enqueue["perf"] <= dequeue["perf"] <= send["perf"] <= send["perf_end"]


def test_a_command_without_a_uuid_is_stamped_but_has_nothing_to_bind_to_an_ack():
    seen = []

    token, acknowledged = _run_through_the_loop(
        SystemCommandKind.REQUEST_VERSION, None,
        lambda stage, fields: seen.append((stage, dict(fields))))

    assert acknowledged
    send = [fields for stage, fields in seen if stage == "send"]
    assert len(send) == 1 and send[0]["token"] == token
    assert send[0]["can_uuid"] is None


def test_a_failing_observer_does_not_stop_the_real_loop_acknowledging():
    def broken(*_args):
        raise RuntimeError("recorder down")

    _token, acknowledged = _run_through_the_loop(SystemCommandKind.SET_X, 10, broken)

    assert acknowledged


class _Hardware(ObservableObject):
    def __init__(self):
        super().__init__(("device_event",))
        self.observer = None

    def set_can_latency_observer(self, observer):
        self.observer = observer


def test_the_recorder_registers_its_latency_log_with_the_hardware_model():
    hardware = _Hardware()
    laser = ObservableObject(("trace_received",))
    recorder = SessionDataRecorder(object(), laser, hardware_model=hardware)
    try:
        assert hardware.observer == recorder.latency_events.record_can
    finally:
        recorder.close()


def _sending_model(order, sent):
    """A HardwareModel with only what _send_with_token touches.

    The transport is a stub that logs into `order`, so a test can say what came
    before the command reached it."""
    model = object.__new__(HardwareModel)
    model._lock = threading.RLock()
    model._pending_tokens = {}
    model._command_outcomes = {}
    model._refresh_cmd_in_progress = lambda commands: None

    def send_command(device, cmd, data=None, context=None):
        order.append("send_command")
        sent.append((device, cmd, data, context))
        return True

    model._send_command = send_command
    return model


def test_the_token_stage_is_reported_before_the_command_reaches_the_transport(monkeypatch):
    order, sent, rows = [], [], []
    model = _sending_model(order, sent)
    clock = itertools.count(100.0, 1.0)
    monkeypatch.setattr(hardware_model_module, "get_perf_now", lambda: next(clock))

    def observer(stage, fields):
        order.append("observer")
        rows.append((stage, dict(fields)))

    model.set_can_latency_observer(observer)
    device = object()

    token = model._send_with_token(device, SystemCommandKind.SET_X, 10)

    # Exactly the contract the recorder reads: stage, token, kind, perf.
    assert rows == [("token", {"token": str(token), "kind": "SET_X", "perf": 100.0})]
    assert order == ["observer", "send_command"]
    # The transport gets the same id, so CanDevice's enqueue/send rows (which
    # carry str(context)) join to this one.
    assert sent == [(device, SystemCommandKind.SET_X, 10, token)]
    # One clock read serves the pending-command record and the latency row.
    assert model._pending_tokens[token] == (SystemCommandKind.SET_X, 100.0)


def test_a_raising_token_observer_never_stops_the_command(caplog):
    order, sent = [], []
    model = _sending_model(order, sent)

    def broken(stage, fields):
        order.append("observer")
        raise RuntimeError("recorder down")

    model.set_can_latency_observer(broken)
    device = object()

    with caplog.at_level(logging.ERROR):
        token = model._send_with_token(device, SystemCommandKind.SET_X, 10)

    assert token is not None
    assert order == ["observer", "send_command"]
    assert sent == [(device, SystemCommandKind.SET_X, 10, token)]
    assert token in model._pending_tokens
    assert "CAN latency observer failed" in caplog.text


@pytest.mark.parametrize("clear_after_setting", [False, True], ids=["init_default", "cleared"])
def test_a_model_with_no_observer_sends_without_reporting_or_logging(clear_after_setting, caplog):
    order, sent = [], []
    model = _sending_model(order, sent)
    model._can_latency_observer = None  # what HardwareModel.__init__ leaves
    if clear_after_setting:
        model.set_can_latency_observer(lambda stage, fields: order.append("observer"))
        model.set_can_latency_observer(None)
    device = object()

    with caplog.at_level(logging.DEBUG):
        token = model._send_with_token(device, SystemCommandKind.SET_X, 10)

    assert token is not None
    assert order == ["send_command"]
    # A None observer must be skipped, not called and its TypeError swallowed.
    assert "CAN latency observer failed" not in caplog.text


class _ConnectStopped(Exception):
    pass


def _connect_until_the_connection_is_built(hardware, monkeypatch, built):
    """Run the real HardwareModel connect, under the harness's emulated CAN,
    up to the point it builds the DeviceConnection.

    A full connect cannot finish on emulation: the emulated board never reports
    a firmware version, so connect stops at "Pellet firmware version was not
    reported" and leaves its reader thread running. The device is created and
    wired before that point, so the DeviceConnection is replaced with a stub
    that records what the device looked like when the connection was built."""

    def build(device, *_args, **_kwargs):
        built.append((device, device.latency_observer))
        raise _ConnectStopped

    monkeypatch.setattr(hardware_model_module, "DeviceConnection", build)
    with pytest.raises(_ConnectStopped):
        hardware.connect(queue.Queue())


def test_connecting_hands_the_models_observer_to_the_new_device(hardware_model, monkeypatch):
    def observer(stage, fields):
        return None

    hardware_model.set_can_latency_observer(observer)
    built = []
    try:
        _connect_until_the_connection_is_built(hardware_model, monkeypatch, built)

        (device, observer_when_built), = built
        assert isinstance(device, CanDevice)
        assert hardware_model._can_device is device
        # Already in place when the connection starts, so no command can go
        # out unobserved.
        assert observer_when_built is observer
        assert device.latency_observer is observer
    finally:
        hardware_model.disconnect()


def test_a_reconnect_gives_the_replacement_device_the_observer_too(hardware_model, monkeypatch):
    def observer(stage, fields):
        return None

    hardware_model.set_can_latency_observer(observer)
    built = []
    try:
        for _ in range(2):
            _connect_until_the_connection_is_built(hardware_model, monkeypatch, built)

        (first, first_observer), (second, second_observer) = built
        assert second is not first
        assert first_observer is observer
        assert second_observer is observer
    finally:
        hardware_model.disconnect()
