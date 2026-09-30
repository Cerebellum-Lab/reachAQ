"""How the wiring tool judges each point from what moved and what it drove.

A laser's inputs are checked by holding its command and seeing which follow
it. Its board-trigger readback, laserN_trigger, is named for the laser but
carries the board's STIM line, not the command: held against the command it
read as "silent" however well it was wired. christielab10 acquires its
readbacks as custom channels, laserN_trigger_readback, which are the same.
"""

import importlib.util
import sys
from pathlib import Path

from tools.acquisition.model.nidaq_wiring_verification import (
    CONFIRMED,
    DRIVER,
    OPAQUE,
    SILENT,
    UNEXPECTED,
    UNTESTED,
    WiringPoint,
)

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "hardware" / "verify_nidaq_wiring.py"
_spec = importlib.util.spec_from_file_location("verify_nidaq_wiring", _TOOL)
verify = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("verify_nidaq_wiring", verify)
_spec.loader.exec_module(verify)

from nidaq_wiring_phases_test import _Nidaqmx  # noqa: E402


DIODE = WiringPoint("laser1_diode", "PXI1Slot5/ai8", "analog stream input")
COPY = WiringPoint("laser1_command_copy", "PXI1Slot5/ai3", "analog stream input")
COMMAND = WiringPoint("laser1_command", "PXI1Slot4/ao0", "laser command output",
                      kind=DRIVER)
SHUTTER = WiringPoint("laser1_shutter", "PXI1Slot5/port0/line4", "laser shutter output",
                      kind=OPAQUE, opaque_reason="digital output with no readback path")
TONE = WiringPoint("tone1", "PXI1Slot5/port0/line0", "digital stream input")
ANALOG_READBACK = WiringPoint("laser1_trigger", "PXI1Slot5/ai9", "analog stream input")
DIGITAL_READBACK = WiringPoint("laser1_trigger", "PXI1Slot5/port0/line7",
                               "digital stream input")
RIG_READBACK = WiringPoint("laser1_trigger_readback", "PXI1Slot5/ai9", "analog stream input")
TRIGGER_IN = WiringPoint("laser1_trigger_in", "/PXI1Slot5/PFI0", "stimulus trigger input")

NOT_CHECKED = ("trigger readback: not checked against the command (it follows "
               "the board STIM)")


def _judge(points, levels, *, observed=None, board_outputs_exercised=True):
    """Hold laser 1's command at `levels`, then judge every point."""
    observed_by_point = dict(observed or {})
    analog_points = [point for point in points
                     if point.kind != OPAQUE and "/ai" in point.physical_channel]
    verify.record_laser_responses(analog_points, 1, levels, observed_by_point)
    checks = verify.judge_points(points, observed_by_point, {1}, board_outputs_exercised)
    return {check.name: check for check in checks}


def test_a_lasers_inputs_follow_its_command_and_the_command_is_confirmed_by_them():
    checks = _judge((DIODE, COPY, COMMAND, SHUTTER),
                    {"PXI1Slot5/ai8": 1.0, "PXI1Slot5/ai3": 1.0})

    assert checks["laser1_diode"].status == CONFIRMED
    assert checks["laser1_command_copy"].status == CONFIRMED
    assert checks["laser1_command"].status == CONFIRMED
    assert checks["laser1_command"].detail == "drove laser1_command_copy"
    assert checks["laser1_shutter"].status == UNTESTED


def test_an_input_of_the_laser_that_stays_put_is_silent():
    checks = _judge((DIODE, COPY, COMMAND), {"PXI1Slot5/ai8": 0.0, "PXI1Slot5/ai3": 1.0})

    assert checks["laser1_diode"].status == SILENT
    assert checks["laser1_diode"].detail == (
        "its driver was exercised and it did not respond")


def test_another_input_that_follows_the_command_is_unexpected():
    stray = WiringPoint("pressure", "PXI1Slot5/ai1", "analog stream input")

    checks = _judge((DIODE, stray), {"PXI1Slot5/ai8": 1.0, "PXI1Slot5/ai1": 0.8})

    assert checks["pressure"].status == UNEXPECTED
    assert checks["pressure"].detail == (
        "responded to laser 1, which it is not assigned to, at 0.800 V")


def test_an_analog_trigger_readback_is_not_judged_by_the_command():
    # Wired right, it stays put while the command is held: it follows the
    # board STIM. It read as silent.
    checks = _judge((DIODE, ANALOG_READBACK), {"PXI1Slot5/ai8": 1.0, "PXI1Slot5/ai9": 0.0})

    assert checks["laser1_trigger"].status == UNTESTED
    assert checks["laser1_trigger"].detail == NOT_CHECKED
    assert checks["laser1_diode"].status == CONFIRMED


def test_christielab10s_readback_is_not_judged_by_the_command():
    # The rig acquires its readbacks as custom channels, laserN_trigger_readback
    # (christielab10_nidaq_blocks.yaml), which the name match first missed.
    for readback in (RIG_READBACK, WiringPoint(
            "laser2_trigger_readback", "PXI1Slot5/ai10", "analog stream input")):
        checks = _judge((DIODE, readback), {"PXI1Slot5/ai8": 1.0})

        assert checks[readback.name].status == UNTESTED
        assert checks[readback.name].detail == NOT_CHECKED


def test_a_digital_trigger_readback_the_board_sweep_did_not_raise_is_silent():
    # The sweep holds each board STIM high and reads every port0 line: a
    # digital readback that did not rise is a missing cable.
    checks = _judge((DIODE, DIGITAL_READBACK), {"PXI1Slot5/ai8": 1.0})

    assert checks["laser1_trigger"].status == SILENT
    assert checks["laser1_trigger"].detail == (
        "its driver was exercised and it did not respond")


def test_a_digital_trigger_readback_with_no_board_sweep_is_not_checked():
    checks = _judge((DIODE, DIGITAL_READBACK), {"PXI1Slot5/ai8": 1.0},
                    board_outputs_exercised=False)

    assert checks["laser1_trigger"].status == UNTESTED
    assert checks["laser1_trigger"].detail == NOT_CHECKED


def test_a_trigger_readback_that_follows_the_command_is_unexpected():
    # Not confirmed by the command, which it has no business following.
    checks = _judge((DIODE, ANALOG_READBACK), {"PXI1Slot5/ai8": 1.0, "PXI1Slot5/ai9": 0.9})

    assert checks["laser1_trigger"].status == UNEXPECTED
    assert "which it is not assigned to" in checks["laser1_trigger"].detail


def test_a_trigger_readback_the_board_stim_raised_stays_confirmed():
    observed = {DIGITAL_READBACK.fingerprint: (CONFIRMED, "held high by board STIM3")}

    checks = _judge((DIODE, DIGITAL_READBACK), {"PXI1Slot5/ai8": 1.0}, observed=observed)

    assert checks["laser1_trigger"].status == CONFIRMED
    assert checks["laser1_trigger"].detail == "held high by board STIM3"


def test_a_trigger_input_is_judged_by_the_board_sweep_not_the_command():
    # The trigger route's input, the PFI the board STIM arrives on. Taken as
    # following the laser command, it read silent with CAN down, where
    # nothing had driven it.
    checks = _judge((DIODE, TRIGGER_IN), {"PXI1Slot5/ai8": 1.0},
                    board_outputs_exercised=False)
    assert checks["laser1_trigger_in"].status == UNTESTED
    assert checks["laser1_trigger_in"].detail == "nothing in this run drives it"

    checks = _judge((DIODE, TRIGGER_IN), {"PXI1Slot5/ai8": 1.0})
    assert checks["laser1_trigger_in"].status == SILENT


def test_what_each_phase_exercises():
    assert verify.driver_exercised(DIODE, {1}, False)
    assert not verify.driver_exercised(DIODE, set(), True)
    assert verify.driver_exercised(TONE, set(), True)
    assert not verify.driver_exercised(TONE, set(), False)
    # A trigger readback is never exercised by a laser; a digital one is by
    # the board sweep, an analog one by nothing here.
    assert not verify.driver_exercised(DIGITAL_READBACK, {1}, False)
    assert verify.driver_exercised(DIGITAL_READBACK, set(), True)
    assert not verify.driver_exercised(ANALOG_READBACK, {1}, True)
    assert not verify.driver_exercised(RIG_READBACK, {1}, True)
    assert not verify.driver_exercised(TRIGGER_IN, {1}, False)
    assert verify.driver_exercised(TRIGGER_IN, set(), True)


def test_the_observe_window_watches_only_what_it_can_read(capsys):
    # It reads the digital ports only: an analog readback listed as watched
    # could never be seen to change.
    ports = {"PXI1Slot5/port0": [0, 0, 0b1000_0000], "PXI1Slot5/port1": [0],
             "PXI1Slot5/port2": [0]}
    nidaqmx = _Nidaqmx(ports)
    confirmed = {}

    verify.observe_undriven(
        nidaqmx, (RIG_READBACK, DIGITAL_READBACK), 0.05,
        {RIG_READBACK.fingerprint, DIGITAL_READBACK.fingerprint},
        lambda point, detail: confirmed.__setitem__(point.name, detail))

    watching = [line for line in capsys.readouterr().out.splitlines() if "watching" in line]
    assert watching == [
        "--- watching 1 undriven line(s) for 0.05s: laser1_trigger ---"]
    assert list(confirmed) == ["laser1_trigger"]
