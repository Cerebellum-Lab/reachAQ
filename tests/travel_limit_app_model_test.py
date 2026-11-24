"""AppModel sends a pellet x/y/z position as a whole or not at all: an axis outside the
configured travel refuses the whole position before any axis is SET, so the stage is never
left with a mix of the new position and the previous one."""
import logging
import math
from types import SimpleNamespace
from unittest import mock

import pytest

from autotrainer.core import Offset3DTuple
from autotrainer.device import MotorConfigurationFile
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.hardware_model import HardwareModel, _pellet_sequences

TRAVEL = {a: {"min_pos": 0, "max_pos": 35} for a in "xyz"}


def _hardware():
    hardware = object.__new__(HardwareModel)
    hardware._motor_configurations = MotorConfigurationFile.from_yaml_dict({"pellet": TRAVEL}, source=None)
    hardware._last_requested_set_coordinates = Offset3DTuple(10.0, 10.0, 10.0)
    hardware._last_motor_coordinates = Offset3DTuple(10.0, 10.0, 10.0)
    hardware._last_motor_send_coordinates = Offset3DTuple(math.nan, math.nan, math.nan)
    hardware._pellet_sequences = _pellet_sequences()
    hardware._device_conn = SimpleNamespace(device=SimpleNamespace(connected=True))
    hardware._on_property_changed = mock.Mock()
    hardware._send_with_token = mock.Mock(return_value=mock.sentinel.token)
    hardware.wait_pending_command_acked = mock.Mock()
    return hardware


def _app_model(hardware):
    return SimpleNamespace(
        _hardware=hardware,
        _behavior=SimpleNamespace(algorithm=SimpleNamespace(diamond_triangle_config=None)),
        _dcs_held_off=False,
        _protocol_positioning_active=False,
        on_error=mock.Mock(),
    )


def _animal(x, y, z):
    return SimpleNamespace(name="m1", pellet_x=x, pellet_y=y, pellet_z=z, is_pellet_dcs=False)


def test_an_animal_base_position_outside_the_travel_sets_no_axis(caplog):
    hardware = _hardware()
    with caplog.at_level(logging.ERROR):
        AppModel._set_animal_base_positions(_app_model(hardware), _animal(5.0, 36.2, 7.0))
    hardware._send_with_token.assert_not_called()
    assert tuple(hardware._last_requested_set_coordinates) == (10.0, 10.0, 10.0)
    assert "Axis-Y: move to 36.200 refused" in caplog.text
    assert "Not sending m1's pellet position" in caplog.text


def test_an_animal_base_position_inside_the_travel_sets_every_axis():
    hardware = _hardware()
    AppModel._set_animal_base_positions(_app_model(hardware), _animal(5.0, 35.0, 7.0))
    assert hardware._send_with_token.call_count == 3
    assert tuple(hardware._last_requested_set_coordinates) == (5.0, 35.0, 7.0)


def test_a_protocol_target_outside_the_travel_moves_no_axis():
    hardware = _hardware()
    app_model = _app_model(hardware)
    with pytest.raises(RuntimeError, match="outside the configured travel"):
        AppModel._move_protocol_motor_target(app_model, Offset3DTuple(5.0, 36.2, 7.0))
    hardware._send_with_token.assert_not_called()
    hardware.wait_pending_command_acked.assert_not_called()
    assert app_model._protocol_positioning_active is False


def test_a_protocol_target_inside_the_travel_sets_and_acks_every_axis():
    hardware = _hardware()
    AppModel._move_protocol_motor_target(_app_model(hardware), Offset3DTuple(5.0, 35.0, 7.0))
    assert hardware._send_with_token.call_count == 3
    assert hardware.wait_pending_command_acked.call_count == 3


def test_a_refused_base_position_blocks_presentation_and_tells_the_operator(caplog):
    hardware = _hardware()
    app_model = _app_model(hardware)
    AppModel._set_animal_base_positions(app_model, _animal(5.0, 36.2, 7.0))

    (title, message), _ = app_model.on_error.call_args
    assert "refused" in title.lower()
    assert "m1" in message and "outside the configured travel" in message
    with caplog.at_level(logging.ERROR):
        assert hardware.send_pellet() is None
    hardware._send_with_token.assert_not_called()
    assert "Pellet presentation refused" in caplog.text
    assert "m1" in caplog.text


def test_an_accepted_base_position_clears_the_block():
    hardware = _hardware()
    app_model = _app_model(hardware)
    AppModel._set_animal_base_positions(app_model, _animal(5.0, 36.2, 7.0))
    AppModel._set_animal_base_positions(app_model, _animal(5.0, 30.0, 7.0))
    hardware._send_with_token.reset_mock()
    assert hardware.send_pellet() is mock.sentinel.token


def test_the_operator_setting_every_axis_clears_the_block_and_a_protocol_target_does_not():
    hardware = _hardware()
    app_model = _app_model(hardware)
    AppModel._set_animal_base_positions(app_model, _animal(5.0, 36.2, 7.0))
    AppModel._move_protocol_motor_target(app_model, Offset3DTuple(5.0, 30.0, 7.0))
    assert hardware.send_pellet() is None
    hardware.set_x(5.0, sender="UI-Set-Button")
    hardware.set_y(30.0, sender="UI-Set-Button")
    assert hardware.send_pellet() is None  # z not set yet
    hardware.set_z(7.0, sender="UI-Set-Button")
    hardware._send_with_token.reset_mock()
    assert hardware.send_pellet() is mock.sentinel.token
