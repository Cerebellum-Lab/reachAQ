"""The NI-DAQ laser calibration ramp, against a stand-in for NI-DAQmx.

Nothing here touches a driver or a board: nidaqmx is replaced by the fake in
nidaq_daqmx_fake, which records what is done to each task and every write,
refuses a channel another started task reserves (as DAQmx does, at -50103),
and records the terminals routed.
"""

import logging
import threading
import time

import pytest

from autotrainer.device import (
    LaserCalibrationRamp,
    LaserChannelId,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser

from nidaq_daqmx_fake import (
    FakeDaqmx as _FakeDaqmx,
    rig_lasers as _rig_lasers,
)


RAMP = LaserCalibrationRamp(
    channel_id=LaserChannelId.LASER_1,
    start_volts=0.0,
    stop_volts=5.0,
    steps=3,
    samples_per_step=10,
    timeout_seconds=5.0,
)


@pytest.fixture
def daq(monkeypatch):
    fake = _FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: fake)
    return fake


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met")
        time.sleep(0.01)


def test_closing_the_controller_mid_ramp_stops_the_ramp_and_resets_the_laser(daq):
    # reachAQ closing mid-ramp closes the ramp's controller from its own
    # thread while the ramp thread waits inside DAQmx. The ramp's tasks hold
    # the analog output, so the command reset in close() failed at -50103,
    # and the 6713 kept its last sample with the shutter open.
    daq.block_wait = True
    controller = NidaqLaserController(_rig_lasers())
    outcome = []

    def ramp():
        try:
            outcome.append(controller.run_calibration_ramp(RAMP))
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=ramp)
    thread.start()
    _wait_for(lambda: any(
        task.name.endswith("calibration_ao") and task.started for task in daq.tasks))
    shutter = daq.task("laser_1_shutter")
    assert shutter.writes[-1] is True

    controller.close()
    thread.join(5.0)

    assert not thread.is_alive()
    assert ("laser_1_calibration_ao", "abort") in daq.controlled
    assert ("laser_1_calibration_ai", "abort") in daq.controlled
    # The command went back to its minimum and the shutter closed.
    assert daq.task("laser_1_manual_ao").writes == [0.0]
    assert shutter.writes[-1] is False
    error, = outcome
    assert isinstance(error, RuntimeError) and "aborted" in str(error)
    # The ramp's clock route, released once, by whichever let go of it.
    route = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")
    assert daq.disconnected.count(route) == 1
    assert controller._trigger_routes == []


def _ramp_in_thread(controller):
    outcome = []

    def ramp():
        try:
            outcome.append(controller.run_calibration_ramp(RAMP))
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=ramp)
    thread.start()
    return thread, outcome


def test_close_waits_for_the_ramp_to_let_go_of_the_output_before_resetting_it(monkeypatch):
    # close() aborted the ramp and reset the laser at once, and the ramp's own
    # cleanup skipped its reset once it found the controller closed. Were the
    # abort not to release ao0, the reset failed at -50103 and the 6713 held
    # the last ramp sample. The pulse path waits for its owner first; so now
    # does the ramp's.
    daq = _FakeDaqmx(block_wait=True, abort_releases=False)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(_rig_lasers())
    thread, outcome = _ramp_in_thread(controller)
    _wait_for(lambda: any(
        task.name.endswith("calibration_ao") and task.started for task in daq.tasks))
    shutter = daq.task("laser_1_shutter")

    controller.close()
    thread.join(5.0)

    assert not thread.is_alive()
    # The ramp's own closed-branch write, from its own thread. The last
    # manual_ao task is close()'s own reset, which says nothing about the
    # ramp's: this passed with that branch deleted.
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao", thread=thread) == [0.0]
    # Then close()'s, after the ramp let go.
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao")[-2:] == [0.0, 0.0]
    assert shutter.writes[-1] is False
    assert daq.reserved == {}
    error, = outcome
    assert "aborted" in str(error)


def test_a_ramp_that_outlives_the_wait_still_puts_the_command_back(monkeypatch):
    # An abort that neither releases ao0 nor returns the ramp's wait: close()
    # gives up waiting, its reset is refused at -50103, and it raises. When
    # the driver later lets the ramp go, the ramp's own cleanup is what puts
    # the command back to its minimum.
    monkeypatch.setattr(nidaq_laser, "_CALIBRATION_RELEASE_TIMEOUT_S", 0.1)
    daq = _FakeDaqmx(block_wait=True, abort_releases=False, abort_unblocks=False,
                     hold_waits=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(_rig_lasers())
    thread, outcome = _ramp_in_thread(controller)
    _wait_for(lambda: any(
        task.name.endswith("calibration_ao") and task.started for task in daq.tasks))
    shutter = daq.task("laser_1_shutter")

    with pytest.raises(RuntimeError, match="channel 1 reset"):
        controller.close()
    assert thread.is_alive()
    assert shutter.writes[-1] is False
    daq.waits_released.set()
    thread.join(5.0)

    assert not thread.is_alive()
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao", thread=thread) == [0.0]
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao")[-1] == 0.0
    assert daq.reserved == {}
    error, = outcome
    assert "aborted" in str(error)


def test_a_close_between_holding_the_tasks_and_routing_the_clock_leaves_no_route(daq):
    # The ramp counted the routes it found and released those after the
    # count, from its own thread, while close() released all of them from
    # another: a close in between emptied the list, the ramp's clock route
    # was then added at index 0, and "since 1" released nothing. Now the
    # ramp releases the routes it added, by name, and adds none once closed.
    trigger_route = ("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")
    controller = NidaqLaserController(_rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))
    assert daq.connected == [trigger_route]
    shared_clock_for = controller._shared_clock_for
    closer = threading.Thread(target=controller.close)

    def close_first(*args, **kwargs):
        if not closer.is_alive() and not controller._closed:
            closer.start()
            _wait_for(lambda: controller._closed)
            # Done, or waiting for this ramp to let go: either way closed.
            closer.join(1.0)
        return shared_clock_for(*args, **kwargs)

    controller._shared_clock_for = close_first

    with pytest.raises(RuntimeError, match="closed"):
        controller.run_calibration_ramp(RAMP)
    closer.join(5.0)

    assert not closer.is_alive()
    assert all(daq.disconnected.count(route) == 1 for route in daq.connected)
    assert controller._trigger_routes == []
    assert not any(task.started for task in daq.tasks)


def test_a_failed_input_task_leaves_the_ramps_output_task_closed(daq, monkeypatch):
    # The ramp created its output and input tasks before its cleanup scope:
    # an input task that failed to create left the output task open.
    controller = NidaqLaserController(_rig_lasers())

    def refuse(_channel):
        raise RuntimeError("DAQmx -200170: the physical channel does not exist")

    monkeypatch.setattr(controller, "_create_calibration_input_task", refuse)

    with pytest.raises(RuntimeError, match="-200170"):
        controller.run_calibration_ramp(RAMP)

    assert daq.task("laser_1_calibration_ao").closed
    assert daq.reserved == {}
    # And the laser was put back as after any ramp.
    assert daq.task("laser_1_manual_ao").writes == [0.0]


def test_a_ramp_on_a_closed_controller_starts_nothing(daq):
    # A close that came first has reset the outputs; the ramp must not then
    # drive the laser after it.
    controller = NidaqLaserController(_rig_lasers())
    controller.close()

    with pytest.raises(RuntimeError, match="closed"):
        controller.run_calibration_ramp(RAMP)

    assert not any(task.started for task in daq.tasks)


# ------------------------------------------------------ the cross-board clock


def test_the_ramp_clocks_its_inputs_over_a_backplane_line(daq):
    # The AI task on the 6221 was clocked from /PXI1Slot4/ao/SampleClock by
    # name, a route DAQmx makes across the backplane only by reserving a
    # line - which it refuses on this unidentified chassis, -89125. The clock
    # is put on PXI_Trig1 on the 6713 and read as the 6221's PXI_Trig1, as
    # a pulse train's shared clock is.
    import dataclasses

    controller = NidaqLaserController(_rig_lasers())

    points = controller.run_calibration_ramp(
        dataclasses.replace(RAMP, enable_pmt_shutter=True))

    assert len(points) == 3
    route = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")
    assert daq.connected == [route]
    assert daq.task("laser_1_calibration_ai").timing_kwargs["source"] == "/PXI1Slot5/PXI_Trig1"
    assert daq.task("laser_pmt_shutter_calibration_do").timing_kwargs["source"] == (
        "/PXI1Slot5/PXI_Trig1")
    # Held for the ramp alone.
    assert daq.disconnected == [route]
    assert controller._trigger_routes == []


def test_the_ramp_releases_its_clock_route_when_it_fails(monkeypatch):
    daq = _FakeDaqmx(failing_task="calibration_ai")
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    controller = NidaqLaserController(_rig_lasers())

    with pytest.raises(RuntimeError, match="refused to start"):
        controller.run_calibration_ramp(RAMP)

    route = ("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")
    assert daq.connected == [route]
    assert daq.disconnected == [route]
    assert controller._trigger_routes == []


def test_the_ramp_keeps_the_controllers_own_trigger_route(daq):
    controller = NidaqLaserController(_rig_lasers(
        trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0"))
    trigger_route = ("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")
    assert daq.connected == [trigger_route]

    controller.run_calibration_ramp(RAMP)

    assert daq.disconnected == [("/PXI1Slot4/ao/SampleClock", "/PXI1Slot4/PXI_Trig1")]
    assert controller._trigger_routes == [trigger_route]


def test_a_ramp_on_one_board_needs_no_route(daq):
    controller = NidaqLaserController(_rig_lasers(analog_output="PXI1Slot5/ao0"))

    controller.run_calibration_ramp(RAMP)

    assert daq.connected == []
    assert daq.task("laser_1_calibration_ai").timing_kwargs["source"] == "/PXI1Slot5/ao/SampleClock"



# ------------------------------------------- close()'s abort and the ramp's own


def _ramp_in_thread(controller):
    outcome = []

    def ramp():
        try:
            outcome.append(controller.run_calibration_ramp(RAMP))
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=ramp, daemon=True)
    thread.start()
    return thread, outcome


def _wait_until_ramping(daq):
    _wait_for(lambda: any(
        task.name.endswith("calibration_ao") and task.started for task in daq.tasks))


def _abort_log(caplog):
    return [record for record in caplog.records
            if record.levelno > logging.DEBUG and "abort" in record.getMessage()]


def test_an_abort_the_ramps_own_cleanup_overtakes_is_not_a_failure(daq, caplog):
    # close() aborted the ramp's tasks one by one. The first abort let the
    # ramp's wait return, and its finally stopped and closed both tasks
    # while close() was still on its way to the second: that abort met a
    # closed task, and close() reported a "calibration task abort" failure
    # on a laser that was reset.
    daq.block_wait = True
    daq.hold_waits = True
    controller = NidaqLaserController(_rig_lasers())
    ramp, ramp_outcome = _ramp_in_thread(controller)
    _wait_until_ramping(daq)
    at_input, go_on = threading.Event(), threading.Event()

    def hold_the_second_abort(task):
        if task.name.endswith("calibration_ai"):
            at_input.set()
            go_on.wait(5.0)

    daq.before_control = hold_the_second_abort
    with caplog.at_level(logging.DEBUG):
        closer = threading.Thread(target=controller.close)
        closer.start()
        assert at_input.wait(5.0), "close() did not reach the second abort"
        # The first abort let the ramp go; its own cleanup closes both tasks.
        ramp.join(5.0)
        assert not ramp.is_alive()
        assert daq.task("laser_1_calibration_ai").closed
        go_on.set()
        closer.join(5.0)

    assert not closer.is_alive()
    assert _abort_log(caplog) == []
    assert "aborted" in str(ramp_outcome[0])
    # The laser was put back.
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="manual_ao")[-1] == 0.0


def test_a_task_the_ramp_has_let_go_of_is_not_aborted(daq, caplog):
    # close() took its snapshot while the ramp held its tasks, then the ramp
    # ended and let go of them before the aborts were made.
    daq.block_wait = True
    daq.hold_waits = True
    controller = NidaqLaserController(_rig_lasers())
    ramp, ramp_outcome = _ramp_in_thread(controller)
    _wait_until_ramping(daq)
    abort = controller._abort_calibration_ramp
    snapshot_taken, go_on = threading.Event(), threading.Event()

    def abort_after_the_ramp_ends(tasks):
        snapshot_taken.set()
        go_on.wait(5.0)
        return abort(tasks)

    controller._abort_calibration_ramp = abort_after_the_ramp_ends
    with caplog.at_level(logging.DEBUG):
        closer = threading.Thread(target=controller.close)
        closer.start()
        assert snapshot_taken.wait(5.0)
        daq.waits_released.set()
        ramp.join(5.0)
        assert not ramp.is_alive()
        go_on.set()
        closer.join(5.0)

    assert not closer.is_alive()
    assert daq.controlled == []
    assert _abort_log(caplog) == []


def test_an_abort_that_fails_on_a_task_the_ramp_still_holds_is_reported(monkeypatch, caplog):
    # The ramp stays in its wait, holding both tasks, whatever is aborted.
    daq = _FakeDaqmx(block_wait=True, hold_waits=True, abort_unblocks=False)
    daq.failing_abort = "calibration_ai"
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    monkeypatch.setattr(nidaq_laser, "_CALIBRATION_RELEASE_TIMEOUT_S", 0.1)
    controller = NidaqLaserController(_rig_lasers())
    ramp, _ramp_outcome = _ramp_in_thread(controller)
    _wait_until_ramping(daq)
    try:
        with pytest.raises(RuntimeError, match="calibration task abort"):
            controller.close()
        assert any(record.levelname == "ERROR"
                   and "Failed to abort a NI-DAQ laser calibration task"
                   in record.getMessage() for record in caplog.records)
    finally:
        daq.waits_released.set()
        ramp.join(5.0)
