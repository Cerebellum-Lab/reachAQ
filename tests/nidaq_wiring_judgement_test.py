"""How the wiring tool judges each point from what moved and what it drove.

A laser's inputs are checked by holding its command and seeing which follow
it. Its board-trigger readback, laserN_trigger, is named for the laser but
carries the board's STIM line, not the command: held against the command it
read as "silent" however well it was wired.
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


def test_a_trigger_readback_is_not_judged_by_the_command():
    # Wired right, it stays put while the command is held: it follows the
    # board STIM. It read as silent.
    for readback in (ANALOG_READBACK, DIGITAL_READBACK):
        checks = _judge((DIODE, readback), {"PXI1Slot5/ai8": 1.0, "PXI1Slot5/ai9": 0.0})

        assert checks["laser1_trigger"].status == UNTESTED
        assert checks["laser1_trigger"].detail == NOT_CHECKED
        assert checks["laser1_diode"].status == CONFIRMED


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


def test_a_trigger_readback_is_left_for_the_observe_window():
    # Nothing in the driven phases checks it, so --observe watches it.
    assert not verify.driver_exercised(DIGITAL_READBACK, {1}, True)
    assert verify.driver_exercised(DIODE, {1}, False)
    assert verify.driver_exercised(TONE, set(), True)
    assert not verify.driver_exercised(TONE, set(), False)
