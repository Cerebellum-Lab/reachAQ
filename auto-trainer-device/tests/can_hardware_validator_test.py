import argparse

import pytest

from autotrainer.core import Motor
from tools.hardware.validate_can_hardware import (
    _motor_config_from_file,
    _require_config_write_allowed,
)


def test_motor_config_from_file_selects_only_requested_pellet_motor(tmp_path):
    config_path = tmp_path / "motor_config.yaml"
    config_path.write_text(
        """
pellet:
  load:
    min_pos: 1
    max_pos: 119
    min_pwm: 801
    max_pwm: 2299
    max_vel: 499
    max_acc: 4999
  barrier:
    min_pos: 2
    max_pos: 118
    min_pwm: 901
    max_pwm: 2099
    max_vel: 498
    max_acc: 4998
""",
        encoding="utf-8",
    )

    cover = _motor_config_from_file(config_path, Motor.PELLET_COVER_SERVO)
    load = _motor_config_from_file(config_path, Motor.PELLET_LOAD_SERVO)

    assert cover.motor is Motor.PELLET_COVER_SERVO
    assert cover.minimum_position == 2
    assert cover.maximum_pwm_duration == 2099
    assert load.motor is Motor.PELLET_LOAD_SERVO
    assert load.minimum_position == 1
    assert load.maximum_pwm_duration == 2299


def test_config_write_requires_explicit_opt_in():
    with pytest.raises(RuntimeError, match="--allow-config-write"):
        _require_config_write_allowed(argparse.Namespace(allow_config_write=False))

    _require_config_write_allowed(argparse.Namespace(allow_config_write=True))
