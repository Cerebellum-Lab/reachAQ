"""A DCS configuration names the 3D calibration it was measured against.

The id is written only when there is one, so a DCS measured against a legacy calibration
saves exactly as before, and a file written before the id existed still loads.
"""

import yaml

from reachaq.core import Offset3DTuple
from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig

_ID = "0123456789abcdef0123456789abcdef"


def _config(**overrides) -> DiamondTriangleOffsetConfig:
    values = dict(
        used_position=Offset3DTuple(7.0, 4.5, 6.5),
        measured_offset=Offset3DTuple(4.7, -34.0, -8.4),
        diamond_coord=Offset3DTuple(0, 0, 0),
        raw_diamond_coord=Offset3DTuple(3.6, -19.3, 0.7),
        version=DiamondTriangleOffsetConfig.current_config_version,
    )
    values.update(overrides)
    return DiamondTriangleOffsetConfig(**values)


def test_calibration_id_round_trip(tmp_path):
    path = tmp_path / "diamond_triangle_offset.yaml"
    cfg = _config(calibration_id=_ID)
    cfg.to_file(path)

    assert yaml.safe_load(path.read_text())["calibration_id"] == _ID
    loaded = DiamondTriangleOffsetConfig.from_file(path)
    assert loaded.calibration_id == _ID
    assert loaded == cfg
    assert loaded.fully_valid and loaded.version == 2
    # The id is part of what the configuration is: another calibration's is another config.
    assert loaded != _config(calibration_id="f" * 32)


def test_old_yaml_without_id_loads(tmp_path):
    path = tmp_path / "diamond_triangle_offset.yaml"
    # Written before the id existed, as christielab10's file is.
    path.write_text(
        "diamond_coord: [0, 0, 0]\n"
        "raw_diamond_coord: [3.6, -19.3, 0.7]\n"
        "measured_offset: [4.7, -34.0, -8.4]\n"
        "used_position: [7.0, 4.5, 6.5]\n"
        "version: 2\n"
    )
    loaded = DiamondTriangleOffsetConfig.load_config(path)
    assert loaded.calibration_id is None
    assert loaded.fully_valid
    assert loaded == _config()


def test_no_id_is_not_written(tmp_path):
    path = tmp_path / "diamond_triangle_offset.yaml"
    _config().to_file(path)
    assert "calibration_id" not in yaml.safe_load(path.read_text())
    assert DiamondTriangleOffsetConfig.from_file(path).calibration_id is None
