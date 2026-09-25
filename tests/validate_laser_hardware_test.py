"""tools/hardware/validate_laser_hardware.py builds the ramp it is asked for.

Nothing here opens a laser: only the command line and the ramp it makes.
"""

import importlib.util
import sys
from pathlib import Path

from autotrainer.device import LaserChannelId

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "hardware" / "validate_laser_hardware.py"
_spec = importlib.util.spec_from_file_location("validate_laser_hardware", _TOOL)
cli = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("validate_laser_hardware", cli)
_spec.loader.exec_module(cli)


def _ramp(*options):
    args = cli._parse_args(["--config", "system_configuration.yaml", "--action", "ramp", *options])
    return cli._calibration_ramp(args, LaserChannelId.LASER_1)


def test_the_ramp_takes_its_settle_window_from_the_command_line():
    ramp = _ramp("--samples-per-step", "50", "--settle-samples", "12")

    assert (ramp.samples_per_step, ramp.settle_samples) == (50, 12)


def test_the_ramp_settles_a_fifth_of_each_step_unless_told_otherwise():
    assert _ramp().settle_samples == 20
