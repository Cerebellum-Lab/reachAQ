import numpy as np

from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from reachaq.core import Offset3DTuple

import pytest


@pytest.fixture()
def diamond_triangle_config():
    return DiamondTriangleOffsetConfig(
        used_position=Offset3DTuple(8, 15, 10),
        measured_offset=Offset3DTuple(5, -7, -9),
        version=DiamondTriangleOffsetConfig.current_config_version,
    )


def test_flips_as_expected(diamond_triangle_config):
    dcs = diamond_triangle_config
    assert dcs.flips_motor_diamond == (1, -1, -1)
    # when motor increases/move starting from its home position:
    # X -> DCS increases as well
    # Y -> DCS decreases
    # Z -> DCS decreases


def test_flips_are_consistent(diamond_triangle_config):
    # also:
    dcs = diamond_triangle_config
    assert dcs.flips_motor_diamond == dcs.flips_inference_motor * dcs.flips_inference_diamond
    assert dcs.flips_inference_motor == dcs.flips_inference_diamond * dcs.flips_motor_diamond


@pytest.mark.parametrize("motor_xyz", [
    (-3, -9, 7),
    (0, 0, 0),
    (1, 2, 3),
    (4, 5, 6),
    (15, 8, 12),
])
@pytest.mark.parametrize("used_pos", [
    (0, 0, 0),
    (3, 5, 9),
    (15, 0, 5),
])
@pytest.mark.parametrize("measured_off", [
    (8, -30, -8),
    (6, -33, -6),
    (-4, 0, 8),
])
def test_transform_coordinates(motor_xyz, used_pos, measured_off):
    motor_xyz = Offset3DTuple(motor_xyz)
    used_pos = Offset3DTuple(used_pos)
    measured_off = Offset3DTuple(measured_off)
    cfg = DiamondTriangleOffsetConfig(used_position=used_pos, measured_offset=measured_off,
                                      version=DiamondTriangleOffsetConfig.current_config_version)
    motor_xyz = Offset3DTuple(motor_xyz)
    diamond_xyz = cfg.motor_to_diamond(motor_xyz)
    assert np.isclose(cfg.diamond_to_motor(diamond_xyz), motor_xyz).all()
    inference_xyz = cfg.motor_to_inference(motor_xyz)
    assert np.isclose(cfg.inference_to_motor(inference_xyz), motor_xyz).all()
    assert np.isclose(cfg.inference_to_diamond(inference_xyz), diamond_xyz).all()
    assert np.isclose(cfg.inference_to_motor(cfg.diamond_to_inference(diamond_xyz)), motor_xyz).all()
    orig = Offset3DTuple(0, 0, 0)
    assert cfg.motor_to_diamond(cfg.inference_to_motor(orig)) == cfg.inference_to_diamond(orig)
    assert cfg.motor_to_diamond(cfg.inference_to_motor(motor_xyz)) == cfg.inference_to_diamond(motor_xyz)
    assert cfg.motor_to_inference(cfg.inference_to_motor(motor_xyz)) == motor_xyz
    assert (
        cfg.diamond_to_inference(cfg.measured_offset)
        - cfg.motor_to_inference(cfg.used_position)
    ) == (0, 0, 0)
    assert cfg.motor_to_diamond(cfg.used_position) == cfg.measured_offset
    assert cfg.motor_to_inference(cfg.used_position) - cfg.diamond_to_inference(cfg.measured_offset) == (0, 0, 0)
    assert cfg.diamond_to_motor(cfg.measured_offset) == cfg.used_position
    assert cfg.motor_to_diamond(cfg.used_position) == cfg.measured_offset
