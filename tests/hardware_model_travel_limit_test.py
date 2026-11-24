"""HardwareModel refuses a pellet x/y/z SET or MOVE outside the configured travel
(min_pos / max_pos in motor_config.yaml), sends nothing and logs an error."""
import logging
import math
from types import SimpleNamespace
from unittest import mock

import pytest

from autotrainer.core import Offset3DTuple, SystemCommandKind
from autotrainer.core.message import SystemDataArgsKwargs
from autotrainer.device import MotorConfigurationFile
from tools.acquisition.model.hardware_model import HardwareModel

TRAVEL = {a: {"min_pos": 0, "max_pos": 35} for a in "xyz"}


def _hardware(pellet=None, *, set_at=(10.0, 10.0, 10.0), motor_at=(10.0, 10.0, 10.0)):
    hardware = object.__new__(HardwareModel)
    hardware._motor_configurations = (
        None if pellet is None else MotorConfigurationFile.from_yaml_dict({"pellet": pellet}, source=None)
    )
    hardware._last_requested_set_coordinates = Offset3DTuple(*set_at)
    hardware._last_motor_coordinates = Offset3DTuple(*motor_at)
    hardware._device_conn = mock.sentinel.device
    hardware._on_property_changed = mock.Mock()
    hardware._send_with_token = mock.Mock(return_value=mock.sentinel.token)
    return hardware


def _refusals(caplog):
    return [r for r in caplog.records if r.levelno == logging.ERROR and "refused" in r.getMessage()]


@pytest.mark.parametrize("axis, cmd", [("x", "SET_X"), ("y", "SET_Y"), ("z", "SET_Z")])
def test_a_set_above_max_is_refused_sends_nothing_and_logs(caplog, axis, cmd):
    hardware = _hardware(TRAVEL)
    with caplog.at_level(logging.ERROR):
        assert getattr(hardware, f"set_{axis}")(35.5) is None
    hardware._send_with_token.assert_not_called()
    hardware._on_property_changed.assert_not_called()
    assert hardware._last_requested_set_coordinates == Offset3DTuple(10, 10, 10)
    assert [r.getMessage() for r in _refusals(caplog)] == [
        f"Axis-{axis.upper()}: move to 35.500 refused: outside the configured travel 0-35 mm"
    ]


def test_a_set_at_the_limits_is_sent():
    hardware = _hardware(TRAVEL)
    assert hardware.set_x(35) is mock.sentinel.token
    assert hardware.set_y(0) is mock.sentinel.token
    assert hardware._send_with_token.call_count == 2


def test_a_set_below_zero_still_clamps_to_zero():
    hardware = _hardware(TRAVEL, set_at=(4.0, 10.0, 10.0))
    assert hardware.set_x(-3) is mock.sentinel.token
    hardware._send_with_token.assert_called_once_with(
        mock.sentinel.device, SystemCommandKind.SET_X, SystemDataArgsKwargs(0, relative=False))
    hardware._send_with_token.reset_mock()
    # x is now 0; a relative SET below 0 clamps too
    assert hardware.set_x(-9, absolute=False) is mock.sentinel.token
    hardware._send_with_token.assert_called_once_with(
        mock.sentinel.device, SystemCommandKind.SET_X, SystemDataArgsKwargs(0, relative=True))


def test_a_clamped_set_still_below_a_positive_min_is_refused(caplog):
    hardware = _hardware({"x": {"min_pos": 2.5, "max_pos": 35}})
    with caplog.at_level(logging.ERROR):
        assert hardware.set_x(-1) is None
    hardware._send_with_token.assert_not_called()
    assert "Axis-X: move to 0.000 refused: outside the configured travel 2.5-35 mm" in caplog.text


def test_a_relative_set_into_out_of_range_is_refused(caplog):
    hardware = _hardware(TRAVEL, set_at=(10.0, 30.0, 10.0))
    with caplog.at_level(logging.ERROR):
        assert hardware.set_y(6, absolute=False) is None
    hardware._send_with_token.assert_not_called()
    assert "Axis-Y: move to 36.000 refused" in caplog.text
    assert hardware.set_y(5, absolute=False) is mock.sentinel.token


@pytest.mark.parametrize("value", [35.001, -0.5])
def test_a_move_outside_the_travel_is_refused(caplog, value):
    hardware = _hardware(TRAVEL)
    with caplog.at_level(logging.ERROR):
        assert hardware.move_z(value) is None
    hardware._send_with_token.assert_not_called()
    assert f"Axis-Z: move to {value:.3f} refused: outside the configured travel 0-35 mm" in caplog.text


def test_a_relative_move_out_of_range_is_refused(caplog):
    hardware = _hardware(TRAVEL, motor_at=(3.0, 10.0, 10.0))
    with caplog.at_level(logging.ERROR):
        assert hardware.move_x(-4, absolute=False) is None
        assert hardware.move_x(33, absolute=False) is None
    hardware._send_with_token.assert_not_called()
    assert len(_refusals(caplog)) == 2
    assert hardware.move_x(-3, absolute=False) is mock.sentinel.token
    hardware._send_with_token.assert_called_once_with(mock.sentinel.device, SystemCommandKind.MOVE_X, 0.0)


def test_one_sided_travel_limits_only_that_side(caplog):
    hardware = _hardware({"x": {"max_pos": 35}, "y": {"min_pos": 5}})
    assert hardware.move_x(-2) is mock.sentinel.token  # no min on x
    assert hardware.move_y(500) is mock.sentinel.token  # no max on y
    with caplog.at_level(logging.ERROR):
        assert hardware.move_x(36) is None
        assert hardware.move_y(4) is None
    assert "outside the configured travel -inf-35 mm" in caplog.text
    assert "outside the configured travel 5-inf mm" in caplog.text


@pytest.mark.parametrize("pellet", [None, {}, {"x": {"max_vel": 120}}])
def test_without_travel_limits_the_commands_sent_are_as_before(pellet):
    hardware = _hardware(pellet, set_at=(4.0, 10.0, 10.0), motor_at=(3.0, 10.0, 10.0))
    hardware.set_x(500)
    hardware.set_x(-600)
    hardware.set_y(-20, absolute=False)
    hardware.move_x(900)
    hardware.move_z(-50)
    hardware.move_x(40, absolute=False)
    dev = mock.sentinel.device
    assert hardware._send_with_token.call_args_list == [
        mock.call(dev, SystemCommandKind.SET_X, SystemDataArgsKwargs(500, relative=False)),
        mock.call(dev, SystemCommandKind.SET_X, SystemDataArgsKwargs(0, relative=False)),
        mock.call(dev, SystemCommandKind.SET_Y, SystemDataArgsKwargs(-10.0, relative=True)),
        mock.call(dev, SystemCommandKind.MOVE_X, 900),
        mock.call(dev, SystemCommandKind.MOVE_Z, -50),
        mock.call(dev, SystemCommandKind.MOVE_X, 43.0),
    ]


def test_a_dcs_set_reaching_out_of_range_is_refused(caplog):
    hardware = _hardware(TRAVEL)
    hardware._check_dcs_cfg = lambda: SimpleNamespace(
        diamond_to_motor=lambda d: Offset3DTuple(d.x + 30, d.y + 30, d.z + 30))
    with caplog.at_level(logging.ERROR):
        assert hardware.set_dcs_x(6) is None
        assert hardware.set_dcs_z(5.5) is None
    hardware._send_with_token.assert_not_called()
    assert "Axis-X: move to 36.000 refused" in caplog.text
    assert "Axis-Z: move to 35.500 refused" in caplog.text
    assert hardware.set_dcs_y(4) is mock.sentinel.token


def test_a_nan_target_is_refused_when_limits_are_configured(caplog):
    hardware = _hardware(TRAVEL)
    with caplog.at_level(logging.ERROR):
        assert hardware.move_x(math.nan) is None
    hardware._send_with_token.assert_not_called()
