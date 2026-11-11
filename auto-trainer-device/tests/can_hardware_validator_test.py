import argparse
import sys

import pytest

from autotrainer.core import Motor
from autotrainer.device.device_interface import Acknowledge
import tools.hardware.validate_can_hardware as validator
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


class _InterfaceThatAnswers:
    """Stands in for CanInterface: opens, finds the board, and answers one ACK."""

    pellet_address = 0
    answers = ()

    def __init__(self, **kwargs):
        self._pending = list(type(self).answers)

    def open(self):
        return True

    def are_addresses_valid(self):
        return True

    def stepper_home(self, motor):
        return True

    def read(self, max_count, *, collect_ms=0):
        out, self._pending = self._pending, []
        return out

    def close(self):
        pass


def _run_home(monkeypatch, error):
    monkeypatch.setattr(_InterfaceThatAnswers, "answers", (Acknowledge(uuid=7, error=error),))
    monkeypatch.setattr(validator, "CanInterface", _InterfaceThatAnswers)
    monkeypatch.setattr(sys, "argv", [
        "validate_can_hardware.py", "--channel", "test-bus", "--action", "home",
        "--motor", "y", "--allow-motion", "--listen-seconds", "0.05",
    ])
    return validator.main()


def test_listen_reports_a_refused_command_and_main_exits_nonzero(monkeypatch, capsys):
    assert _run_home(monkeypatch, -5) == 1
    err = capsys.readouterr().err
    assert "EIO" in err and "(-5)" in err and "uuid 7" in err


@pytest.mark.parametrize("error", [0, -11])
def test_an_ack_of_zero_or_eagain_is_not_a_refusal_and_main_exits_zero(monkeypatch, capsys, error):
    assert _run_home(monkeypatch, error) == 0
    assert "refused" not in capsys.readouterr().err
