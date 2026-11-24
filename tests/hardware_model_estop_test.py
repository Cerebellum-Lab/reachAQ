"""HardwareModel's side of the pellet ESTOP: when it stops the stage, and what a stop means.

After an ESTOP every axis needs a home, the reported position is unknown, and
every move, send, load and sequence is refused until the operator's home
succeeds; that home releases the board's latch first. A teardown stops a stage
that may be moving, and leaves an idle one alone.
"""

import math
import threading
import time
from queue import Queue
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import pytest

from autotrainer.core import Motor, SystemCommandKind
from autotrainer.core.message import SystemStatusMessageKind
from autotrainer.device import CanDevice, CommandRefusal, DeviceApi, RefusedCommandClass, Target
from autotrainer.device.device_interface import Acknowledge

import tools.acquisition.model.hardware_model as hardware_model_module
from tools.acquisition.model.hardware_model import HardwareModel, OPERATOR_STOP_REASON

_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)


class _Subscribers:
    def __iadd__(self, handler):
        return self

    def __isub__(self, handler):
        return self


class _Device:
    """Stands in for CanDevice: records ESTOPs, releases and sends, in one ordered log."""

    def __init__(self, model, log, *, moving=False):
        self._model = model
        self.log = log
        self.motion_possibly_in_flight = moving
        self.property_changed = _Subscribers()

    def estop(self, reason):
        self.log.append(("estop", reason))
        self._model._on_device_estop(reason, _XYZ, True)
        return True

    def release_estop(self):
        self.log.append(("release",))


def _wired(hardware_model, *, moving=False):
    """A model whose device and connection are fakes sharing one log."""
    log = []
    hardware_model._can_device = _Device(hardware_model, log, moving=moving)
    hardware_model._device_conn = SimpleNamespace(
        send_message=lambda kind, data=None, context=None: log.append(("send", kind, context)),
    )
    return log


def _sent(log):
    return [entry[1] for entry in log if entry[0] == "send"]


def _stopped(hardware_model, reason="MOVE_Z was not acknowledged within 3 s"):
    log = _wired(hardware_model)
    assert hardware_model.estop(reason)
    return log


# -- what a stop means ---------------------------------------------------------

def test_a_stop_marks_every_axis_needing_home_and_the_position_unknown(hardware_model):
    changes = []
    hardware_model.property_changed += lambda name, value, old: changes.append((name, value))
    hardware_model._last_motor_coordinates = hardware_model._last_motor_coordinates.replace(x=1.0, y=2.0, z=15.5)

    _stopped(hardware_model)

    assert hardware_model.estop_active
    assert hardware_model.pellet_axes_needing_home == ("X", "Y", "Z")
    assert all(math.isnan(v) for v in hardware_model.last_position)
    record = hardware_model.pellet_estop
    assert record["reason"] == "MOVE_Z was not acknowledged within 3 s"
    assert record["axes"] == ["X", "Y", "Z"]
    assert record["operator"] is False and record["sent"] is True
    assert (HardwareModel.PELLET_ESTOP_PROPERTY, record) in changes


def test_the_operators_stop_is_recorded_as_the_operators(hardware_model):
    _stopped(hardware_model, OPERATOR_STOP_REASON)

    assert hardware_model.pellet_estop["operator"] is True
    assert hardware_model.pellet_axes_needing_home == ("X", "Y", "Z")


def test_position_reports_are_ignored_while_stopped(hardware_model):
    from autotrainer.core import MessageHandler

    _stopped(hardware_model)
    status = SimpleNamespace(position=15.49, send_position=15.5, driver_fault=False)

    hardware_model._message_handler_property_changed(MessageHandler.STEPPER_Z_PROPERTY, status, None)

    assert math.isnan(hardware_model.last_position.z)


def test_a_stop_without_a_board_is_logged_and_not_sent(hardware_model, caplog):
    import logging

    with caplog.at_level(logging.ERROR):
        assert hardware_model.estop("test") is False

    assert not hardware_model.estop_active
    assert any("ESTOP not sent" in r.getMessage() for r in caplog.records)


# -- refused until homed --------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda hm: hm.move_x(1.0),
    lambda hm: hm.move_z(15.5),
    lambda hm: hm.load_pellet(),
    lambda hm: hm.send_pellet(),
    lambda hm: hm.send_retract(),
    lambda hm: hm.release_pellet(),
    lambda hm: hm.cover_pellet(),
    lambda hm: hm.send_to_limits(),
    lambda hm: hm.send_home(),
], ids=["move_x", "move_z", "load", "send", "retract", "release", "cover", "limits", "automatic-home"])
def test_motion_is_refused_while_stopped(hardware_model, call):
    log = _stopped(hardware_model)

    assert call(hardware_model) is None
    assert _sent(log) == []


def test_a_set_still_goes_out_while_stopped(hardware_model):
    # SET_X/Y/Z only store the send position on the board.
    log = _stopped(hardware_model)

    assert hardware_model.set_x(2.0) is not None
    assert _sent(log) == [SystemCommandKind.SET_X]


def test_the_operators_home_releases_the_latch_first_and_clears_the_stop(hardware_model):
    log = _stopped(hardware_model)

    token = hardware_model.home_after_estop()

    assert token is not None
    assert log[1:] == [("release",), ("send", SystemCommandKind.SEND_HOME, token)]
    assert hardware_model.estop_active, "not cleared before the board acknowledges the home"

    hardware_model._ack_received(token)

    assert not hardware_model.estop_active
    assert hardware_model.pellet_estop is None
    assert hardware_model.pellet_axes_needing_home == ()
    assert hardware_model.move_x(1.0) is not None


def test_a_refused_home_after_the_stop_leaves_the_stage_stopped(hardware_model):
    _stopped(hardware_model)
    token = hardware_model.home_after_estop()

    refusal = CommandRefusal(
        token=token, error=-1, command=SystemCommandKind.SEND_HOME, step="home",
        command_class=RefusedCommandClass.STEPPER_HOME, motors=(Motor.PELLET_Y_MOTOR,),
    )
    hardware_model._command_refused(token, refusal=refusal)

    assert hardware_model.estop_active
    assert hardware_model.move_x(1.0) is None


def test_a_home_sent_before_the_stop_does_not_clear_it(hardware_model):
    log = _wired(hardware_model)
    old_home = hardware_model.send_home()
    hardware_model.estop("test")

    hardware_model._ack_received(old_home)

    assert hardware_model.estop_active
    assert hardware_model.pellet_axes_needing_home == ("X", "Y", "Z")
    assert _sent(log) == [SystemCommandKind.SEND_HOME]


def test_a_command_the_stop_ended_ends_as_stopped(hardware_model):
    _wired(hardware_model)
    token = hardware_model.move_z(15.5)
    failures = []
    hardware_model.command_failed += failures.append
    refusal = CommandRefusal(
        token=token, error=-1, command=SystemCommandKind.MOVE_Z, step="MOVE_Z",
        command_class=RefusedCommandClass.STEPPER_MOVE, motors=(Motor.PELLET_Z_MOTOR,),
        estop_reason="MOVE_Z was not acknowledged within 3 s",
    )

    hardware_model._command_refused(token, refusal=refusal)

    outcome, error = hardware_model._command_outcomes[token]
    assert outcome == "stopped"
    assert error.startswith("Pellet stage stopped: MOVE_Z was not acknowledged within 3 s.")
    with pytest.raises(RuntimeError, match="ended as stopped"):
        hardware_model.wait_pending_command_acked(token, timeout=0.01)
    assert failures[0].category == "estop"
    assert token not in hardware_model.pending_tokens


# -- teardown and timeouts ------------------------------------------------------

def test_a_teardown_stops_a_moving_stage_before_the_device_is_told_to_disconnect(hardware_model):
    log = _wired(hardware_model, moving=True)
    hardware_model._device_conn = SimpleNamespace(
        request_disconnect=lambda: log.append(("request_disconnect",)),
        join=lambda: True,
    )

    hardware_model._disconnect_transport_locked(command_outcome="cancelled")

    assert [entry[0] for entry in log] == ["estop", "request_disconnect"]
    assert log[0][1] == "the CAN connection was closed during a move"


def test_an_idle_disconnect_sends_no_estop(hardware_model):
    log = _wired(hardware_model, moving=False)
    hardware_model._device_conn = SimpleNamespace(
        request_disconnect=lambda: log.append(("request_disconnect",)),
        join=lambda: True,
    )

    hardware_model.disconnect()

    assert log == [("request_disconnect",)]
    assert not hardware_model.estop_active


def test_a_move_that_times_out_while_in_flight_stops_the_stage(hardware_model):
    log = _wired(hardware_model, moving=True)
    token = hardware_model.move_z(15.5)

    with pytest.raises(RuntimeError, match="timeout"):
        hardware_model.wait_pending_command_acked(token, timeout=0.01)

    assert log[-1] == ("estop", "MOVE_Z was not acknowledged within 0.01 s")


def test_a_set_that_times_out_stops_nothing(hardware_model):
    log = _wired(hardware_model, moving=True)
    token = hardware_model.set_x(1.0)

    hardware_model.wait_pending_command_acked(token, timeout=0.01, raise_on_timeout=False)

    assert [entry for entry in log if entry[0] == "estop"] == []


# -- connect ---------------------------------------------------------------------

class _ConnectDevice:
    """CanDevice for connect(): records the release, in order with the connection's sends."""

    log = None
    default_command_ack_timeout_duration = 0.1
    default_board_status_timeout_delay = 1.0

    def __init__(self, **kwargs):
        self.property_changed = _Subscribers()
        self.latency_observer = None
        self.device_interface = None
        self.estop_callback = kwargs.get("estop_callback")

    def release_estop(self):
        type(self).log.append("release")


class _ConnectConnection:
    log = None
    model = None

    def __init__(self, can_device, cmd_queue, **kwargs):
        self.device = SimpleNamespace(connected=True)
        self._api = SimpleNamespace(message_callback=None)

    def request_connect(self):
        pass

    def wait_connected(self, *, timeout):
        pass

    def await_acknowledge(self, tokens, timeout):
        import contextlib

        @contextlib.contextmanager
        def ack():
            yield
        return ack()

    def send_message(self, kind, data=None, context=None):
        type(self).log.append(kind)
        if kind is SystemCommandKind.REQUEST_VERSION:
            type(self).model._firmware_version_received.set()

    def load_default_motor_config(self):
        return None

    def load_default_move_config(self):
        return None

    def request_disconnect(self):
        pass

    def join(self):
        return True


@pytest.fixture
def fake_connect(hardware_model, monkeypatch):
    log = []
    monkeypatch.setattr(_ConnectDevice, "log", log)
    monkeypatch.setattr(_ConnectConnection, "log", log)
    monkeypatch.setattr(_ConnectConnection, "model", hardware_model)
    monkeypatch.setattr(hardware_model_module, "CanDevice", _ConnectDevice)
    monkeypatch.setattr(hardware_model_module, "DeviceConnection", _ConnectConnection)
    monkeypatch.setattr(HardwareModel, "pellet_commands_allowed", property(lambda self: True))
    monkeypatch.setattr(HardwareModel, "_request_board_capabilities", lambda self, device: None)
    monkeypatch.setattr(HardwareModel, "_check_timedout_commands_handler", lambda self: None)
    return log


def test_the_connect_home_releases_the_latch_first_and_clears_a_stop(hardware_model, fake_connect):
    hardware_model._on_device_estop("an earlier stop", _XYZ, True)

    hardware_model.connect(Queue())

    assert fake_connect.index("release") < fake_connect.index(SystemCommandKind.SEND_HOME)
    assert not hardware_model.estop_active
    assert hardware_model.pellet_axes_needing_home == ()


def test_the_device_reports_its_estops_to_the_model(hardware_model, fake_connect):
    hardware_model.connect(Queue())

    callback = hardware_model._can_device.estop_callback
    callback("a test stop", _XYZ, True)

    assert hardware_model.pellet_estop["reason"] == "a test stop"


def test_a_recovery_while_stopped_leaves_the_latch_for_the_operator(hardware_model, fake_connect):
    hardware_model._on_device_estop("an earlier stop", _XYZ, True)

    hardware_model.connect(Queue(), _recovery=True)

    assert "release" not in fake_connect


# -- end to end, with the emulated board ------------------------------------------

def test_a_move_never_acknowledged_on_the_emulated_board_stops_the_stage_until_the_operator_homes(
    hardware_model,
):
    # The hardware_model fixture freezes get_perf_now, so the device's own ack timeout
    # never runs out here: the waiter's timeout is the one that stops the stage
    # (CanDevice's own is can_device_estop_test's). The ESTOP then ends the move as
    # stopped, unsent again, and the operator's home releases and clears it.
    def route(kind, data):
        if kind == SystemStatusMessageKind.ACKNOWLEDGE:
            hardware_model._ack_received(data[0])
        elif kind == SystemStatusMessageKind.COMMAND_REFUSED:
            hardware_model._command_refused(data[0], refusal=data[2])

    device = CanDevice(api=DeviceApi(message_callback=route), force_emulation=True,
                       required_targets=(Target.PELLET_DEVICE,),
                       estop_callback=hardware_model._on_device_estop)
    interface = device.device_interface
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            for message in interface.read(64):
                if isinstance(message, Acknowledge):
                    device._handle_ack(message)
            time.sleep(0.001)

    interface.open()
    device.connect()
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    hardware_model._can_device = device
    hardware_model._device_conn = SimpleNamespace(
        send_message=lambda kind, data=None, context=None: device.notify_message(kind, data, context))
    try:
        moves = []

        def lost_move(*args, **kwargs):
            moves.append(args)
            interface.next_uuid()
            return True

        interface.move_motor_z = lost_move
        device._init_handlers()

        token = hardware_model.move_z(15.5)
        deadline = time.monotonic() + 2.0
        while not device.motion_possibly_in_flight and time.monotonic() < deadline:
            time.sleep(0.005)
        with pytest.raises(RuntimeError, match="timeout"):
            hardware_model.wait_pending_command_acked(token, timeout=0.2)

        assert interface.estop_frames == [True]
        assert hardware_model.estop_active
        assert hardware_model.pellet_estop["reason"] == "MOVE_Z was not acknowledged within 0.2 s"
        deadline = time.monotonic() + 2.0
        while device.motion_possibly_in_flight and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not device.motion_possibly_in_flight, "the loop ended the stopped move"
        assert len(moves) == 1
        assert hardware_model.move_x(1.0) is None

        home = hardware_model.home_after_estop()
        hardware_model.wait_pending_command_acked(home, timeout=2.0)

        assert interface.estop_frames == [True, False]
        assert not hardware_model.estop_active
        assert hardware_model.pellet_axes_needing_home == ()
    finally:
        stop.set()
        reader.join(1)
        hardware_model._device_conn = None
        hardware_model._can_device = None
        device.disconnect()
        interface.close()
