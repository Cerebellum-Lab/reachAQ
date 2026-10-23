"""A stale DCS configuration is held off its consumers but stays loaded.

AppModel decides when the DCS configuration is stale (measured against another 3D
calibration) and calls set_dcs_suppressed. The consumers read diamond_triangle_config,
which is then None, as if there were none; loaded_diamond_triangle_config keeps the file's
content for what must still use it (the 3D-calibration capture's motor path).
"""

import pytest

from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoProps
from autotrainer.core import Offset3DTuple
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig


@pytest.fixture
def algo(monkeypatch, tmp_path) -> BehaviorAlgorithm:
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    return BehaviorAlgorithm(diamond_triangle_offset_config_path=tmp_path / "diamond_triangle_offset.yaml")


def _config(calibration_id=None) -> DiamondTriangleOffsetConfig:
    return DiamondTriangleOffsetConfig(
        used_position=Offset3DTuple(1, 2, 3),
        measured_offset=Offset3DTuple(3, 30, -8),
        diamond_coord=Offset3DTuple(0, 0, 0),
        raw_diamond_coord=Offset3DTuple(1, 1, 1),
        version=DiamondTriangleOffsetConfig.current_config_version,
        calibration_id=calibration_id,
    )


def _dcs_events(algo):
    events = []

    def on_changed(name, value, prev):
        if name == BehaviorAlgoProps.DIAMOND_TRIANGLE_CONFIG:
            events.append((value, prev))

    algo.property_changed += on_changed
    return events


def test_suppression_hides_the_config_and_fires_the_change(algo):
    cfg = _config()
    algo.diamond_triangle_config = cfg
    events = _dcs_events(algo)

    algo.set_dcs_suppressed(True)
    assert algo.diamond_triangle_config is None
    assert algo.loaded_diamond_triangle_config is cfg
    assert events == [(None, cfg)]

    # Only a change is announced.
    algo.set_dcs_suppressed(True)
    assert events == [(None, cfg)]

    algo.set_dcs_suppressed(False)
    assert algo.diamond_triangle_config is cfg
    assert events == [(None, cfg), (cfg, None)]


def test_a_reload_keeps_the_suppression(algo, tmp_path):
    algo.set_dcs_suppressed(True)
    events = _dcs_events(algo)
    _config("a" * 32).to_file(algo.diamond_triangle_offset_config_path)

    algo.reload_diamond_triangle_config()
    assert algo.loaded_diamond_triangle_config == _config("a" * 32)
    assert algo.diamond_triangle_config is None
    # Consumers saw nothing new: they had None and still have None.
    assert events == []

    # Setting it directly (as the DCS calibration does) keeps it too.
    algo.diamond_triangle_config = _config("b" * 32)
    assert algo.diamond_triangle_config is None
    assert algo.loaded_diamond_triangle_config == _config("b" * 32)


def test_suppression_holds_off_drift_monitoring(algo):
    algo.diamond_triangle_config = _config()
    algo.set_dcs_suppressed(True)
    algo.handle_diamond_triangle_offset(Offset3DTuple(0.5, 1, -1), Offset3DTuple(8, 6, 5))
    assert algo.get_diamond_triangle_drifts() is None

    algo.set_dcs_suppressed(False)
    algo.handle_diamond_triangle_offset(Offset3DTuple(0.5, 1, -1), Offset3DTuple(8, 6, 5))
    assert algo.get_diamond_triangle_drifts() is not None
