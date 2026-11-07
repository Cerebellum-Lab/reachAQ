from autotrainer.core import Offset3DTuple
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core.pose_elements import SceneElement
from autotrainer.inference import PoseResponse

from tools.acquisition.model.coordinate_model import CoordinateModel


def _configuration():
    zero = Offset3DTuple.get_zero()
    return DiamondTriangleOffsetConfig(
        used_position=zero,
        measured_offset=zero,
        diamond_coord=zero,
        raw_diamond_coord=zero,
    )


def _response(location):
    return PoseResponse(
        locations_3d={SceneElement.Diamond: location},
        raw_loc_3d={SceneElement.Diamond: location},
    )


def test_valid_coordinate_clears_warning_state():
    model = CoordinateModel()
    model.warned_bad = True
    model.triggered_bad = True

    assert model.validate_pose(
        _response(Offset3DTuple.get_zero()),
        _configuration(),
        inference_started_perf=0.0,
        paused=False,
        perf_now=10.0,
    ) is None
    assert model.previous_valid_perf == 10.0
    assert model.warned_bad is False
    assert model.triggered_bad is False


def test_invalid_coordinate_reports_only_once_when_enabled():
    model = CoordinateModel()
    model.report_error = True
    bad = _response(Offset3DTuple(10.0, 0.0, 0.0))

    message = model.validate_pose(
        bad,
        _configuration(),
        inference_started_perf=0.0,
        paused=False,
        perf_now=10.0,
    )
    assert "Could not ensure valid diamond position" in message
    assert model.validate_pose(
        bad,
        _configuration(),
        inference_started_perf=0.0,
        paused=False,
        perf_now=11.0,
    ) is None
