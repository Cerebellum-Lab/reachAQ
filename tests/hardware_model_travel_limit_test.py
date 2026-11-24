"""HardwareModel refuses a pellet x/y/z SET or MOVE outside the configured travel
(min_pos / max_pos in motor_config.yaml), sends nothing and logs an error."""
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from autotrainer.core import Offset3DTuple, SystemCommandKind
from autotrainer.core.message import SystemDataArgsKwargs
from autotrainer.device import CompoundMovements, MotorConfigurationFile, MotorSteps
from tools.acquisition.model.hardware_model import HardwareModel, _pellet_sequences

TRAVEL = {a: {"min_pos": 0, "max_pos": 35} for a in "xyz"}


def _hardware(pellet=None, *, set_at=(10.0, 10.0, 10.0), motor_at=(10.0, 10.0, 10.0)):
    hardware = object.__new__(HardwareModel)
    hardware._motor_configurations = (
        None if pellet is None else MotorConfigurationFile.from_yaml_dict({"pellet": pellet}, source=None)
    )
    hardware._last_requested_set_coordinates = Offset3DTuple(*set_at)
    hardware._last_motor_coordinates = Offset3DTuple(*motor_at)
    hardware._last_motor_send_coordinates = Offset3DTuple(math.nan, math.nan, math.nan)
    hardware._pellet_sequences = _pellet_sequences()
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
    # x is now 0; a relative SET below 0 clamps too, and is sent as the absolute 0 it checked
    assert hardware.set_x(-9, absolute=False) is mock.sentinel.token
    hardware._send_with_token.assert_called_once_with(
        mock.sentinel.device, SystemCommandKind.SET_X, SystemDataArgsKwargs(0, relative=False))


def test_with_travel_a_relative_set_is_sent_as_the_absolute_target_it_checked():
    # The board adds a relative SET to its own reported send position (y 34 here), not to the
    # requested one the host checked (y 20, a SET the board refused): 34 + 5 would leave the
    # 35 mm travel. So the checked absolute target, 25, is sent.
    hardware = _hardware(TRAVEL, set_at=(10.0, 20.0, 10.0))
    hardware._last_motor_send_coordinates = Offset3DTuple(10.0, 34.0, 10.0)
    assert hardware.set_y(5, absolute=False) is mock.sentinel.token
    hardware._send_with_token.assert_called_once_with(
        mock.sentinel.device, SystemCommandKind.SET_Y, SystemDataArgsKwargs(25.0, relative=False))
    assert hardware._last_requested_set_coordinates.y == 25.0


def test_an_axis_without_travel_still_sends_a_relative_set_relative():
    hardware = _hardware({"x": {"min_pos": 0, "max_pos": 35}}, set_at=(10.0, 20.0, 10.0))
    assert hardware.set_y(5, absolute=False) is mock.sentinel.token
    hardware._send_with_token.assert_called_once_with(
        mock.sentinel.device, SystemCommandKind.SET_Y, SystemDataArgsKwargs(5, relative=True))


def test_a_send_relative_step_is_checked_from_the_reported_send_position_too(caplog):
    # CanDevice adds send_*_rel to the board's reported send position, which can differ from
    # the requested one: both bases are checked.
    hardware = _with_sequence(_hardware(TRAVEL, set_at=(10.0, 20.0, 10.0)), "move_retract", [{"send_y_rel": 2}])
    hardware._last_motor_send_coordinates = Offset3DTuple(10.0, 34.0, 10.0)
    with caplog.at_level(logging.ERROR):
        assert hardware.send_retract() is None
    hardware._send_with_token.assert_not_called()
    assert "Axis-Y: move to 36.000 refused" in caplog.text
    assert "Axis-Y: move to 22.000" not in caplog.text


def test_predefined_send_reaching_out_of_range_on_the_reported_send_position_is_refused(caplog):
    hardware = _with_sequence(_hardware(TRAVEL, set_at=(10.0, 20.0, 10.0)), "send_pellet", [{"predefined": "send"}])
    hardware._last_motor_send_coordinates = Offset3DTuple(10.0, 35.5, 10.0)
    with caplog.at_level(logging.ERROR):
        assert hardware.send_pellet() is None
    assert "Axis-Y: move to 35.500 refused" in caplog.text


def test_predefined_send_to_an_unknown_send_position_is_refused_with_travel(caplog):
    # Nothing SET since the connect and no send position reported: the board would go to a
    # saved position nobody has checked.
    hardware = _with_sequence(_hardware(TRAVEL, set_at=(math.nan,) * 3), "send_pellet", [{"predefined": "send"}])
    with caplog.at_level(logging.ERROR):
        assert hardware.send_pellet() is None
    hardware._send_with_token.assert_not_called()
    assert "Pellet sequence send_pellet refused: the send position of X, Y, Z is not known" in caplog.text
    # Without travel it is sent as before.
    hardware = _with_sequence(_hardware(None, set_at=(math.nan,) * 3), "send_pellet", [{"predefined": "send"}])
    assert hardware.send_pellet() is mock.sentinel.token


def test_an_xyz_outside_the_travel_is_refused_whole_before_any_set(caplog):
    hardware = _hardware(TRAVEL)
    with caplog.at_level(logging.ERROR):
        assert hardware.refuse_xyz_outside_travel(Offset3DTuple(5.0, 36.2, 40.0)) is True
    assert "Axis-Y: move to 36.200 refused" in caplog.text
    assert "Axis-Z: move to 40.000 refused" in caplog.text
    hardware._send_with_token.assert_not_called()
    # The SET's clamp to 0 applies: -2 is a SET of 0, inside the travel.
    assert hardware.refuse_xyz_outside_travel(Offset3DTuple(-2.0, 35.0, 0.0)) is False
    assert _hardware(None).refuse_xyz_outside_travel(Offset3DTuple(99.0, 99.0, 99.0)) is False


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


# The compound sequences (move_config.yaml) are run step by step by CanDevice, so HardwareModel
# checks every stepper target of one before it triggers it, and refuses the whole sequence.

MOVE_CONFIG = CompoundMovements.from_file(Path(__file__).with_name("move_config.yaml"))


def _with_sequence(hardware, name, steps):
    hardware._pellet_sequences[name] = MotorSteps(name, steps)
    return hardware


@pytest.mark.parametrize("method, name, cmd", [
    ("load_pellet", "load_pellet", SystemCommandKind.LOAD_PELLET),
    ("send_pellet", "send_pellet", SystemCommandKind.SEND_PELLET),
    ("send_retract", "move_retract", SystemCommandKind.SEND_RETRACT),
    ("cover_pellet", "cover_pellet", SystemCommandKind.COVER_PELLET),
    ("release_pellet", "release_pellet", SystemCommandKind.RELEASE_PELLET),
])
def test_a_sequence_with_a_step_outside_the_travel_is_refused_whole(caplog, method, name, cmd):
    hardware = _with_sequence(_hardware(TRAVEL), name, [{"y": 0.0}, {"x": 36.0}, {"z": 5.0}])
    with caplog.at_level(logging.ERROR):
        assert getattr(hardware, method)() is None
    hardware._send_with_token.assert_not_called()
    assert "Axis-X: move to 36.000 refused: outside the configured travel 0-35 mm" in caplog.text
    assert f"Pellet sequence {name} refused" in caplog.text

    hardware = _with_sequence(_hardware(TRAVEL), name, [{"y": 0.0}, {"x": 35.0}, {"z": 5.0}])
    assert getattr(hardware, method)() is mock.sentinel.token
    assert hardware._send_with_token.call_args[0][:2] == (mock.sentinel.device, cmd)


def test_every_out_of_range_step_is_logged():
    hardware = _with_sequence(_hardware(TRAVEL), "load_pellet", [{"x": 36.0}, {"z_rel": 30}])
    with mock.patch("tools.acquisition.model.hardware_model.logger") as logger:
        assert hardware.load_pellet() is None
    messages = [c.args[0] for c in logger.error.call_args_list]
    # z_rel 30 from z 10 is clamped to CanInterface's 15 turns (36.548 mm): still refused.
    assert messages == [
        "Axis-X: move to 36.000 refused: outside the configured travel 0-35 mm",
        "Axis-Z: move to 36.548 refused: outside the configured travel 0-35 mm",
        "Pellet sequence %s refused: a step lies outside the configured travel",
    ]


def test_the_christielab10_load_pellet_and_retract_pass():
    hardware = _hardware(TRAVEL, motor_at=(3.0, 10.0, 4.0))
    hardware._pellet_sequences = _pellet_sequences(MOVE_CONFIG)
    assert hardware.load_pellet() is mock.sentinel.token
    # y_rel -25 from y 10 is clamped to 0 by CanInterface, as before: inside the travel.
    assert hardware.send_retract() is mock.sentinel.token


def test_a_relative_sequence_step_is_checked_from_the_reported_position(caplog):
    hardware = _with_sequence(_hardware(TRAVEL, motor_at=(3.0, 30.0, 4.0)), "move_retract", [{"y_rel": 6}])
    with caplog.at_level(logging.ERROR):
        assert hardware.send_retract() is None
    assert "Axis-Y: move to 36.000 refused" in caplog.text


def test_a_step_relative_to_an_unknown_position_is_not_checked():
    hardware = _with_sequence(_hardware(TRAVEL, motor_at=(math.nan,) * 3), "move_retract", [{"y_rel": 60}])
    assert hardware.send_retract() is mock.sentinel.token


def test_the_send_position_is_the_requested_one_else_the_reported_one(caplog):
    steps = [{"send_z_rel": 0}, {"send_x_rel": 0}, {"send_y_rel": 6}]
    hardware = _with_sequence(_hardware(TRAVEL, set_at=(10.0, 30.0, 10.0)), "move_retract", steps)
    with caplog.at_level(logging.ERROR):
        assert hardware.send_retract() is None
    assert "Axis-Y: move to 36.000 refused" in caplog.text

    hardware = _with_sequence(_hardware(TRAVEL, set_at=(math.nan,) * 3), "send_pellet",
                              [{"predefined": "send"}])
    hardware._last_motor_send_coordinates = Offset3DTuple(10.0, 10.0, 35.2)
    assert hardware.send_pellet() is None
    hardware._last_motor_send_coordinates = Offset3DTuple(10.0, 10.0, 35.0)
    assert hardware.send_pellet() is mock.sentinel.token


def test_without_travel_limits_sequences_are_sent_as_before():
    hardware = _with_sequence(_hardware(None), "load_pellet", [{"x": 36.0}])
    assert hardware.load_pellet() is mock.sentinel.token
    hardware = _with_sequence(_hardware({"x": {"max_vel": 100}}), "load_pellet", [{"x": 36.0}])
    assert hardware.load_pellet() is mock.sentinel.token


def test_the_sequences_are_move_config_s_else_the_built_in_ones():
    sequences = _pellet_sequences(MOVE_CONFIG)
    assert sequences["load_pellet"] is MOVE_CONFIG.load_pellet
    assert sequences["move_retract"] is MOVE_CONFIG.move_retract
    assert sequences["cover_pellet"] is MOVE_CONFIG.cover_pellet
    empty = CompoundMovements.from_yaml_dict({"actions": {"load_pellet": []}})
    assert _pellet_sequences(empty)["load_pellet"].steps == _pellet_sequences()["load_pellet"].steps
    assert _pellet_sequences(None)["move_retract"].name == "move_retract"


def test_a_drift_moving_the_send_position_out_of_range_is_refused(caplog):
    hardware = _hardware(TRAVEL, set_at=(34.0, 10.0, 1.0))
    with caplog.at_level(logging.ERROR):
        hardware.set_motors_drift(Offset3DTuple(-2.0, 0.0, 0.0))
        hardware.set_motors_drift(Offset3DTuple(0.0, 0.0, 1.5))
    hardware._send_with_token.assert_not_called()
    assert "Axis-X: move to 36.000 refused" in caplog.text
    assert "Axis-Z: move to -0.500 refused" in caplog.text
    hardware.set_motors_drift(Offset3DTuple(1.0, 0.0, 0.5))
    assert [c.args[1] for c in hardware._send_with_token.call_args_list] == [
        SystemCommandKind.SET_MOTOR_DRIFT, SystemCommandKind.SET_X, SystemCommandKind.SET_Y, SystemCommandKind.SET_Z]
