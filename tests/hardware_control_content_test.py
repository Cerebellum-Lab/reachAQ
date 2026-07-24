import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import Offset3DTuple  # noqa: E402
from autotrainer.model import EnvironmentProvider, HardwareVersion  # noqa: E402
from tools.acquisition.model.hardware_model import HardwareModel  # noqa: E402
from tools.acquisition.view.hardware_control_content import (  # noqa: E402
    HardwareControlContent,
    _format_motor_feedback,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_alogus_hardware_control_uses_full_motor_travel_and_transforms_it(
    qapp,
    app_model,
    monkeypatch,
):
    monkeypatch.delenv("AUTOTRAINER_HARDWARE_VERSION", raising=False)
    EnvironmentProvider.set_hardware_version(None)
    assert EnvironmentProvider.hardware_version() == HardwareVersion.ALOGUS_V1
    content = HardwareControlContent(app_model)
    try:
        assert content._travel_limits == {
            "x": (0, 35),
            "y": (0, 35),
            "z": (0, 35),
        }

        config = app_model.behavior.algorithm.diamond_triangle_config
        motor_min = Offset3DTuple(0, 0, 0)
        motor_max = Offset3DTuple(35, 35, 35)
        if config is not None and config.fully_valid:
            expected_min = config.motor_to_diamond(motor_min)
            expected_max = config.motor_to_diamond(motor_max)
        else:
            expected_min = motor_min
            expected_max = motor_max

        for index, spinbox in enumerate((content._x_pos, content._y_pos, content._z_pos)):
            assert spinbox.minimum() == pytest.approx(
                min(expected_min[index], expected_max[index])
            )
            assert spinbox.maximum() == pytest.approx(
                max(expected_min[index], expected_max[index])
            )
    finally:
        EnvironmentProvider.set_hardware_version(None)
        content.deleteLater()
        qapp.processEvents()


def test_motor_feedback_line_uses_requested_compact_format():
    text = _format_motor_feedback("LIVE", Offset3DTuple(2.3, 25.1, -0.7))

    assert text == "• LIVE •   X 2.3 | Y 25.1 | Z −0.7 mm"


def test_motor_feedback_line_tracks_board_position_and_connection_state(
    qapp,
    app_model,
):
    hardware = app_model.hardware
    content = HardwareControlContent(app_model)
    try:
        assert content._motor_feedback_label.text() == (
            "• DISCONNECTED •   X — | Y — | Z — mm"
        )

        motor_position = Offset3DTuple(2.3, 25.1, -0.7)
        hardware._last_motor_coordinates = motor_position
        hardware._pellet_version = "1.2.5"
        content._on_hardware_model_property_changed(
            HardwareModel.POS_XYZ,
            motor_position,
            Offset3DTuple(math.nan, math.nan, math.nan),
        )

        config = app_model.behavior.algorithm.diamond_triangle_config
        expected_position = motor_position
        if config is not None and config.fully_valid:
            expected_position = config.motor_to_diamond(motor_position)
        assert content._motor_feedback_label.text() == _format_motor_feedback(
            "LIVE",
            expected_position,
        )

        hardware._device_pellet_status_timeout_engaged = True
        content._on_hardware_model_property_changed(
            HardwareModel.DEVICE_PELLET_STATUS_TIMEOUT_ENGAGED,
            True,
            False,
        )
        assert content._motor_feedback_label.text() == _format_motor_feedback(
            "STALE",
            expected_position,
        )

        hardware._pellet_version = ""
        content._on_hardware_model_property_changed(
            HardwareModel.PELLET_VERSION_PROPERTY,
            "",
            "1.2.5",
        )
        assert content._motor_feedback_label.text() == _format_motor_feedback(
            "DISCONNECTED",
            expected_position,
        )
    finally:
        content.deleteLater()
        qapp.processEvents()
