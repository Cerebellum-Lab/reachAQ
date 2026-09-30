"""tools/hardware/validate_laser_hardware.py builds the ramp it is asked for.

Nothing here opens a laser: only the command line and the ramp it makes.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from autotrainer.device import LaserChannelId

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "hardware" / "validate_laser_hardware.py"
_spec = importlib.util.spec_from_file_location("validate_laser_hardware", _TOOL)
cli = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("validate_laser_hardware", cli)
_spec.loader.exec_module(cli)


def _ramp(*options):
    args = cli._parse_args(["--config", "system_configuration.yaml", "--action", "ramp", *options])
    return cli._calibration_ramp(args, LaserChannelId.LASER_1)


def test_the_ramp_takes_its_settle_window_from_the_command_line_in_microseconds():
    ramp = _ramp("--samples-per-step", "50", "--settle-us", "120")

    assert (ramp.samples_per_step, ramp.settle_seconds) == (50, pytest.approx(120e-6))


def test_the_ramp_settles_600_us_of_500_sample_steps_unless_told_otherwise():
    ramp = _ramp()

    assert (ramp.samples_per_step, ramp.settle_seconds) == (500, pytest.approx(600e-6))
