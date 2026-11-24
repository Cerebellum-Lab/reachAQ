import copy
from pathlib import Path

import pytest

from reachaq.core import Offset3DTuple
from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig

zero_offset = Offset3DTuple(0, 0, 0)
one_offset = Offset3DTuple(1, 1, 1)


def test_diamond_triangle_offset_config_with_args_fail():
    with pytest.raises(TypeError):
        DiamondTriangleOffsetConfig(zero_offset, zero_offset)  # noqa


def test_diamond_triangle_offset_config_kwargs():
    cfg = DiamondTriangleOffsetConfig(used_position=zero_offset, measured_offset=one_offset)
    assert cfg.used_position == zero_offset
    assert cfg.measured_offset == one_offset


@pytest.mark.parametrize("bad_path", [None, Path("/must/absolutely/really/not/exists/forever")])
def test_load_diamond_triangle_offset_bad_path(bad_path):
    assert DiamondTriangleOffsetConfig.load_config(bad_path) is None


def test_save_diamond_triangle(tmp_path):
    dst = tmp_path.joinpath(DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH.name)
    cfg = DiamondTriangleOffsetConfig(
        used_position=zero_offset,
        measured_offset=one_offset,
        diamond_coord=one_offset + one_offset,
        raw_diamond_coord=one_offset + one_offset + one_offset,
    )
    cfg.to_file(dst)
    cfg2 = DiamondTriangleOffsetConfig.load_config(dst)
    assert isinstance(cfg2.used_position, Offset3DTuple)
    assert isinstance(cfg2.measured_offset, Offset3DTuple)
    assert isinstance(cfg2.diamond_coord, Offset3DTuple)
    #
    assert cfg2 == cfg
    #
    cfg2.used_position += one_offset
    assert cfg2 != cfg
    #
    cfg2 = copy.deepcopy(cfg)
    assert cfg2 == cfg
    cfg2.raw_diamond_coord -= one_offset
    assert cfg2 != cfg
