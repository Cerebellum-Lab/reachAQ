"""A CAN recovery whose home the board refuses must leave Refresh Hardware able to reconnect.

Runs the real HardwareModel connect over the emulated CAN device. The board's
refusal is an acknowledgement carrying an errno, as pellet firmware 2.3.1
sends it, so it travels CanDevice and DeviceConnection.await_acknowledge as it
would from a real board.
"""

import threading
from queue import Queue
from unittest import mock

import pytest

from autotrainer.core import Motor, SystemMessageHandler
from autotrainer.device import EmulationInterface
from autotrainer.device.device_interface import Acknowledge
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.hardware_model import HardwareModel

_EIO = -5
_WORKING_HOME = EmulationInterface.stepper_home


def _refusing_home(self, motor: Motor):
    """EmulationInterface.stepper_home for a board with a driver fault."""
    self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid(), error=_EIO))
    return self._is_open


@pytest.fixture
def system_msg_handler():
    """A running handler on the real clock.

    Not top_fixtures' system_msg_handler: it pulls in mock_get_perf_now, and
    DeviceConnection's reader reads its disconnect request only once
    get_perf_now() passes a time.perf_counter() deadline, so under the fake
    clock the reader never stops and every recovery fails to close it.
    """
    handler = SystemMessageHandler(Queue())
    handler.start()
    try:
        yield handler
    finally:
        handler.request_terminate()
        handler.wait_terminated()


@pytest.fixture
def hardware(system_msg_handler, monkeypatch):
    """A HardwareModel that records the _recovery flag of every _connect_locked."""
    calls = []
    connect_locked = HardwareModel._connect_locked

    def spy(self, cmd_queue, *, _recovery, _generation):
        calls.append(_recovery)
        return connect_locked(self, cmd_queue, _recovery=_recovery, _generation=_generation)

    monkeypatch.setattr(HardwareModel, "_connect_locked", spy)
    model = HardwareModel(system_msg_handler)
    model.connect_calls = calls
    try:
        yield model
    finally:
        model.disconnect()


def _app_for(hardware, system_msg_handler):
    app = object.__new__(AppModel)
    app._hardware = hardware
    app._system_message_handler = system_msg_handler
    return app


def _when_state(hardware, state, error_prefix=""):
    """An event set once the CAN connection state is published as state.

    The published state is what the operator sees, so it is what a test waits
    for: the recovery's device events go out before it closes the transport.
    """
    reached = threading.Event()

    def changed(name, value, _previous):
        if (
            name == HardwareModel.CAN_CONNECTION_STATE_PROPERTY
            and value["state"] == state
            and value["error"].startswith(error_prefix)
        ):
            reached.set()

    hardware.property_changed += changed
    return reached


def _recover_into_a_refused_home(hardware, system_msg_handler, monkeypatch):
    """Connect, fail the bus, have the recovery's home refused, then fix the board."""
    assert hardware.connect(system_msg_handler.input_queue) is True
    assert hardware.connect_calls == [False]

    stopped = _when_state(hardware, "failed", "CAN recovery stopped")
    monkeypatch.setattr(EmulationInterface, "stepper_home", _refusing_home)
    hardware.report_can_watchdog_failure("reader stalled (test)")
    assert stopped.wait(10), "the recovery did not stop on the refused home"
    assert hardware.connect_calls == [False, True]

    # The operator releases the E-stop before pressing anything.
    monkeypatch.setattr(EmulationInterface, "stepper_home", _WORKING_HOME)


def test_refused_recovery_home_closes_the_session(hardware, system_msg_handler, monkeypatch):
    _recover_into_a_refused_home(hardware, system_msg_handler, monkeypatch)

    state = hardware._can_connection_state
    assert state["state"] == "failed"
    assert "CAN recovery stopped" in state["error"]
    assert hardware.connected is False, (
        "after the refused recovery the CAN session is still reported connected; "
        f"axes needing home={hardware.pellet_axes_needing_home}"
    )


def test_refresh_after_a_refused_recovery_reconnects_and_homes(
    hardware, system_msg_handler, monkeypatch,
):
    _recover_into_a_refused_home(hardware, system_msg_handler, monkeypatch)
    app = _app_for(hardware, system_msg_handler)

    result = app._initialize_pellet_controller_for_refresh()

    assert hardware.connect_calls == [False, True, False], (
        f"Refresh Hardware did not reconnect: _connect_locked calls={hardware.connect_calls}, "
        f"refresh result={result.state!r} {result.info!r}, "
        f"CAN state={hardware._can_connection_state}"
    )
    assert hardware._can_connection_state["state"] == "ready"
    assert hardware.pellet_axes_needing_home == ()


def test_subsystem_retry_after_a_refused_recovery_reconnects_and_homes(
    hardware, system_msg_handler, monkeypatch,
):
    _recover_into_a_refused_home(hardware, system_msg_handler, monkeypatch)
    app = _app_for(hardware, system_msg_handler)
    app._begin_subsystem_start = mock.Mock(return_value=1)
    app._set_subsystem_status = mock.Mock()

    ready = app._start_can_domain(wait_connected=True)

    assert hardware.connect_calls == [False, True, False], (
        f"the CAN/pellet retry did not reconnect: _connect_locked calls={hardware.connect_calls}, "
        f"returned {ready}, last status {app._set_subsystem_status.call_args}, "
        f"CAN state={hardware._can_connection_state}"
    )
    assert hardware._can_connection_state["state"] == "ready"


def test_a_bus_failure_after_refresh_starts_a_new_recovery(
    hardware, system_msg_handler, monkeypatch,
):
    """The stopped recovery's claim on recovery ends with the operator's reconnect."""
    _recover_into_a_refused_home(hardware, system_msg_handler, monkeypatch)
    _app_for(hardware, system_msg_handler)._initialize_pellet_controller_for_refresh()
    recovered = _when_state(hardware, "ready")

    hardware.report_can_watchdog_failure("reader stalled again (test)")

    assert recovered.wait(10), (
        "a bus failure after Refresh Hardware was not recovered: "
        f"recovery still owned by {hardware._can_recovery_failure!r}, "
        f"_connect_locked calls={hardware.connect_calls}"
    )
    assert hardware.connect_calls == [False, True, False, True]
    assert hardware._can_connection_state["state"] == "ready"


def test_refresh_after_an_exhausted_recovery_reconnects(
    hardware, system_msg_handler, monkeypatch,
):
    """Same exit shape: the last attempt fails after the transport is up."""
    assert hardware.connect(system_msg_handler.input_queue) is True

    exhausted = _when_state(hardware, "failed", "CAN recovery exhausted")
    real_request_version = EmulationInterface.request_version
    monkeypatch.setattr(EmulationInterface, "request_version", lambda self: self._is_open)
    hardware.report_can_watchdog_failure("reader stalled (test)")
    assert exhausted.wait(20), "the recovery did not exhaust"
    monkeypatch.setattr(EmulationInterface, "request_version", real_request_version)
    connected_after_exhaustion = hardware.connected

    result = _app_for(hardware, system_msg_handler)._initialize_pellet_controller_for_refresh()

    assert hardware.connect_calls == [False, True, True, True, False], (
        f"Refresh Hardware did not reconnect: connected after exhaustion={connected_after_exhaustion}, "
        f"_connect_locked calls={hardware.connect_calls}, refresh result={result.state!r}, "
        f"CAN state={hardware._can_connection_state}"
    )


def test_control_a_refused_startup_home_is_retried_by_refresh(
    hardware, system_msg_handler, monkeypatch,
):
    """The startup path: the refusal raises out of connect and refresh shuts CAN down."""
    app = _app_for(hardware, system_msg_handler)
    monkeypatch.setattr(EmulationInterface, "stepper_home", _refusing_home)

    first = app._initialize_pellet_controller_for_refresh()
    assert first.state == "error", first.info
    assert hardware.connected is False

    monkeypatch.setattr(EmulationInterface, "stepper_home", _WORKING_HOME)
    second = app._initialize_pellet_controller_for_refresh()

    assert hardware.connect_calls == [False, False]
    assert second.state == "ok", second.info
    assert hardware._can_connection_state["state"] == "ready"
    assert hardware.pellet_axes_needing_home == ()
