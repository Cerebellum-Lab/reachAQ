import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import Offset3DTuple  # noqa: E402
from autotrainer.model import EnvironmentProvider, HardwareVersion  # noqa: E402
from tools.acquisition.view.hardware_control_content import HardwareControlContent  # noqa: E402


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
