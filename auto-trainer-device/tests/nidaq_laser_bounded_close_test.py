"""The laser controller's lock is never held across a driver call.

connect_terms and disconnect_terms ran under _operation_lock, and a
calibration ramp started its tasks under it. A driver that hung in one of
those held the lock, and close() then blocked at the lock before it could
mark the controller closed, with every cancel and abort behind it.

The hangs here are the fake's sick-driver modes (nidaq_daqmx_fake); nothing
touches a driver or a board.
"""

import threading
import time

import pytest

from autotrainer.device import (
    LaserCalibrationRamp,
    LaserChannelId,
    LaserControllerStillClosing,
    LaserPulseTrain,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser

from nidaq_daqmx_fake import FakeDaqmx, rig_lasers


TRIGGER_ROUTE = ("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")
AO_CLOCK_ROUTE = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")
RAMP = LaserCalibrationRamp(
    channel_id=LaserChannelId.LASER_1, start_volts=0.0, stop_volts=5.0,
    steps=3, samples_per_step=10, timeout_seconds=5.0,
    # 2 samples at 100 kHz: these short steps would keep none of 600 us.
    settle_seconds=20e-6)
PULSE = LaserPulseTrain(
    channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0, duration_ms=1.0)


def _in_thread(function, *args):
    outcome = []

    def run():
        try:
            outcome.append(function(*args))
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _sick(monkeypatch, *calls):
    daq = FakeDaqmx(hang=calls)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    return daq


def test_a_hung_disconnect_holds_nothing_that_marks_the_controller_closed(monkeypatch):
    daq = _sick(monkeypatch, "disconnect_terms")
    try:
        controller = NidaqLaserController(rig_lasers(
            trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))
        first_close, first_outcome = _in_thread(controller.close)
        assert daq.hanging.wait(5.0), "close() did not reach disconnect_terms"
        # Already closed as the driver call runs.
        assert controller._closed

        # Whatever asks whether it is closed is told at once, not after the
        # driver: a ramp, a pulse, and a second close.
        ramp, ramp_outcome = _in_thread(controller.run_calibration_ramp, RAMP)
        pulse, pulse_outcome = _in_thread(controller.run_pulse_train, PULSE)
        second_close, second_outcome = _in_thread(controller.close)
        for thread in (ramp, pulse, second_close):
            thread.join(2.0)
            assert not thread.is_alive()
        assert "closed" in str(ramp_outcome[0])
        assert "closed" in str(pulse_outcome[0])
        assert second_outcome == [None]
        assert first_close.is_alive()

        daq.hang_released.set()
        first_close.join(5.0)
        assert first_outcome == [None]
        # Released once, by the close that took it.
        assert daq.disconnected == [TRIGGER_ROUTE]
        assert controller._trigger_routes == []
    finally:
        daq.hang_released.set()


def test_a_route_connected_as_the_controller_closes_is_undone(monkeypatch):
    # The ramp puts its clock on the backplane and the driver hangs there.
    # close() goes on, and the route, once made, is the ramp's to undo:
    # close() had released only the routes it held.
    monkeypatch.setattr(nidaq_laser, "_CALIBRATION_RELEASE_TIMEOUT_S", 0.2)
    daq = _sick(monkeypatch, "connect_terms")
    try:
        controller = NidaqLaserController(rig_lasers())
        ramp, ramp_outcome = _in_thread(controller.run_calibration_ramp, RAMP)
        assert daq.hanging.wait(5.0), "the ramp did not reach connect_terms"

        close, close_outcome = _in_thread(controller.close)
        close.join(5.0)

        assert not close.is_alive()
        # The ramp had not let go when close() stopped waiting: an error,
        # named for as long as the ramp lasts.
        error, = close_outcome
        assert "calibration ramp release" in str(error)
        assert controller.work_left_running() == ("the calibration ramp",)
        daq.hang_released.set()
        ramp.join(5.0)
        assert not ramp.is_alive()
        assert controller.work_left_running() == ()
        assert "closed" in str(ramp_outcome[0])
        assert daq.connected == [AO_CLOCK_ROUTE]
        assert daq.disconnected == [AO_CLOCK_ROUTE]
        assert controller._trigger_routes == []
        assert not any(task.started for task in daq.tasks)
    finally:
        daq.hang_released.set()


def test_two_callers_needing_one_backplane_clock_route_share_it(monkeypatch):
    # A guard test, which passed before the change it sits beside: a route
    # another caller is still connecting is waited for and reused, not
    # connected a second time. The route is the calibration ramp's: the
    # 6713's ao/SampleClock on backplaneClockLine, which both the ramp's
    # diode input and its PMT line take from the 6221 side.
    daq = _sick(monkeypatch, "connect_terms")
    try:
        controller = NidaqLaserController(rig_lasers())
        first = controller._shared_clock_for
        routes = []
        one, one_outcome = _in_thread(
            lambda: first("PXI1Slot5", "/PXI1Slot4/ao/SampleClock", line="PXI_Trig1",
                          added=routes))
        assert daq.hanging.wait(5.0)
        two, two_outcome = _in_thread(
            lambda: first("PXI1Slot5", "/PXI1Slot4/ao/SampleClock", line="PXI_Trig1"))
        two.join(0.5)
        assert two.is_alive(), "the second caller did not wait for the first"

        daq.hang_released.set()
        one.join(5.0)
        two.join(5.0)

        assert one_outcome == two_outcome == ["/PXI1Slot5/PXI_Trig1"]
        assert daq.connected == [AO_CLOCK_ROUTE]
        assert routes == [AO_CLOCK_ROUTE]
        assert controller._trigger_routes == [AO_CLOCK_ROUTE]
    finally:
        daq.hang_released.set()


def test_a_caller_waiting_on_a_route_is_woken_by_the_close(monkeypatch):
    # A caller waiting for another's route to settle slept out its wait
    # before it found the controller closed: close() woke nobody.
    monkeypatch.setattr(nidaq_laser, "_ROUTE_SETTLE_WAIT_S", 30.0)
    daq = _sick(monkeypatch, "connect_terms")
    try:
        controller = NidaqLaserController(rig_lasers())
        clock_for = controller._shared_clock_for
        one, _one_outcome = _in_thread(
            lambda: clock_for("PXI1Slot5", "/PXI1Slot4/ao/SampleClock", line="PXI_Trig1"))
        assert daq.hanging.wait(5.0)
        two, two_outcome = _in_thread(
            lambda: clock_for("PXI1Slot5", "/PXI1Slot4/ao/SampleClock", line="PXI_Trig1"))
        two.join(0.3)
        assert two.is_alive(), "the second caller did not wait for the first"

        controller.close()
        two.join(2.0)

        assert not two.is_alive(), "the close did not wake the waiting caller"
        assert "closed" in str(two_outcome[0])
    finally:
        daq.hang_released.set()


def test_a_controller_that_fails_to_open_closes_within_a_bound(monkeypatch):
    # Opening fails part-way, and the controller closes what it had opened.
    # That close had no bound: a driver hung in it hung the Run start, or the
    # ramp, behind it. It gives up past its bound now, and says, on the
    # error, when it ends.
    monkeypatch.setattr(nidaq_laser, "_FAILED_OPEN_CLOSE_TIMEOUT_S", 0.3)
    daq = _sick(monkeypatch, "disconnect_terms")
    daq.held_wait_limit = 3.0
    connect = NidaqLaserController._connect_trigger_route

    def connect_then_fail(self, channel):
        connect(self, channel)
        raise RuntimeError("DAQmx refused the laser's shutter line")

    monkeypatch.setattr(NidaqLaserController, "_connect_trigger_route", connect_then_fail)
    try:
        started = time.monotonic()
        with pytest.raises(LaserControllerStillClosing,
                           match="refused the laser's shutter line") as refused:
            NidaqLaserController(rig_lasers(
                trigger_source="/PXI1Slot4/PXI_Trig0",
                trigger_route_source="/PXI1Slot5/PFI0"))

        assert time.monotonic() - started < 2.0
        assert daq.hung == ["disconnect_terms"]
        # A type of its own, which a caller finds however the error is
        # wrapped; the open's own error is its cause.
        assert "refused the laser's shutter line" in str(refused.value.__cause__)
        still_closing = refused.value.still_closing
        assert not still_closing.is_set()
        daq.hang_released.set()
        assert still_closing.wait(5.0)
        assert daq.disconnected == [TRIGGER_ROUTE]
    finally:
        daq.hang_released.set()


def test_a_controller_that_fails_to_open_and_closes_says_nothing_more(monkeypatch):
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    connect = NidaqLaserController._connect_trigger_route

    def connect_then_fail(self, channel):
        connect(self, channel)
        raise RuntimeError("DAQmx refused the laser's shutter line")

    monkeypatch.setattr(NidaqLaserController, "_connect_trigger_route", connect_then_fail)
    with pytest.raises(RuntimeError, match="refused the laser's shutter line") as refused:
        NidaqLaserController(rig_lasers(
            trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))

    assert not isinstance(refused.value, LaserControllerStillClosing)
    assert daq.disconnected == [TRIGGER_ROUTE]


def test_a_normal_close_is_unchanged(monkeypatch):
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))
    controller.run_pulse_train(PULSE)

    started = time.monotonic()
    controller.close()

    assert time.monotonic() - started < 1.0
    assert daq.disconnected == [TRIGGER_ROUTE]
    assert daq.task("laser_1_shutter").writes[-1] is False
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao")[-1] == 0.0
    with pytest.raises(RuntimeError, match="closed"):
        controller.run_pulse_train(PULSE)



def test_a_route_another_call_leaves_pending_is_given_up_on(monkeypatch):
    # A caller waiting for another's route to settle waited as long as that
    # one's driver call took: without end, if it hung.
    monkeypatch.setattr(nidaq_laser, "_ROUTE_PENDING_GIVE_UP_S", 0.3, raising=False)
    monkeypatch.setattr(nidaq_laser, "_ROUTE_SETTLE_WAIT_S", 0.05)
    daq = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(rig_lasers())
    with controller._operation_lock:
        controller._pending_routes[AO_CLOCK_ROUTE] = "connect"
    try:
        waiter, outcome = _in_thread(lambda: controller._shared_clock_for(
            "PXI1Slot5", "/PXI1Slot4/ao/SampleClock", line="PXI_Trig1"))
        waiter.join(3.0)

        assert not waiter.is_alive(), "it waited on the pending route without end"
        error, = outcome
        assert isinstance(error, RuntimeError)
        assert "/PXI1Slot4/ao/SampleClock -> /PXI1Slot4/PXI_Trig1" in str(error)
        assert daq.connected == []
    finally:
        with controller._operation_lock:
            controller._settle_route(AO_CLOCK_ROUTE)
