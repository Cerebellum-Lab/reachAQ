"""The intersession shift is applied to every axis or to none: one shifted axis outside the
configured travel leaves the whole send position unshifted."""
import logging
from types import SimpleNamespace
from unittest import mock

from autotrainer.behavior import SystemMachine, PelletDeviceProtocol
from autotrainer.core import Offset3DTuple
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig


def _machine(refused):
    device = mock.create_autospec(PelletDeviceProtocol)
    device.refuse_shift_outside_travel.return_value = refused
    algo = SimpleNamespace(intersession_pellet_shift_enabled=True, diamond_triangle_config=None)
    return SimpleNamespace(_algorithm=algo, _pellet_device=device), device


def test_a_shift_with_an_axis_outside_the_travel_shifts_no_axis(caplog):
    machine, device = _machine(refused=True)
    with caplog.at_level(logging.ERROR):
        SystemMachine._apply_processed_shift_xyz(machine, Offset3DTuple(1.0, 6.0, 0.0))
    flips = DiamondTriangleOffsetConfig.flips_motor_diamond
    (delta,), _ = device.refuse_shift_outside_travel.call_args
    assert tuple(delta) == (1.0 * flips[0], 6.0 * flips[1], 0.0)
    device.set_x.assert_not_called()
    device.set_y.assert_not_called()
    device.set_z.assert_not_called()
    assert "Not applying the pellet send_position shift" in caplog.text


def test_a_shift_inside_the_travel_is_applied_as_before():
    machine, device = _machine(refused=False)
    SystemMachine._apply_processed_shift_xyz(machine, Offset3DTuple(1.0, 6.0, 0.0))
    flips = DiamondTriangleOffsetConfig.flips_motor_diamond
    device.set_x.assert_called_once_with(1.0 * flips[0], absolute=False, sender="processed_shift_xyz")
    device.set_y.assert_called_once_with(6.0 * flips[1], absolute=False, sender="processed_shift_xyz")
    device.set_z.assert_not_called()
