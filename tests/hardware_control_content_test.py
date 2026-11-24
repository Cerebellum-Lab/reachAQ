import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from autotrainer.core import Offset3DTuple  # noqa: E402
from tools.acquisition.model.hardware_model import HardwareModel  # noqa: E402
from tools.acquisition.view.hardware_control_content import (  # noqa: E402
    HardwareControlContent,
    _format_motor_feedback,
    _motor_to_ui_position,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_hardware_control_uses_home_relative_zero_to_35_range(
    qapp,
    app_model,
):
    content = HardwareControlContent(app_model)
    try:
        for spinbox in (content._x_pos, content._y_pos, content._z_pos):
            assert spinbox.minimum() == pytest.approx(0)
            assert spinbox.maximum() == pytest.approx(35)
    finally:
        content.deleteLater()
        qapp.processEvents()


def test_motor_feedback_line_uses_requested_compact_format():
    text = _format_motor_feedback("LIVE", Offset3DTuple(2.3, 25.1, -0.7))

    assert text == "• LIVE •   X 2.3 | Y 25.1 | Z −0.7 mm"


def test_home_relative_mapping_uses_board_coordinates_and_clamps_feedback():
    position = _motor_to_ui_position(Offset3DTuple(2.3, 9.9, 40.0))

    assert position == Offset3DTuple(2.3, 9.9, 35.0)


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
        assert content._motor_set_feedback_label.text() == (
            "• SET •   X — | Y — | Z — mm"
        )

        motor_position = Offset3DTuple(2.3, 25.1, 0.7)
        hardware._last_motor_coordinates = motor_position
        hardware._last_motor_send_coordinates = Offset3DTuple(4.0, 15.0, 6.0)
        hardware._pellet_version = "1.2.5"
        content._on_hardware_model_property_changed(
            HardwareModel.POS_XYZ,
            motor_position,
            Offset3DTuple(math.nan, math.nan, math.nan),
        )

        expected_position = motor_position
        assert content._motor_feedback_label.text() == _format_motor_feedback(
            "LIVE",
            expected_position,
        )
        assert content._motor_set_feedback_label.text() == (
            "• SET •   X 4.0 | Y 15.0 | Z 6.0 mm"
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
        hardware._last_motor_coordinates = Offset3DTuple(
            math.nan, math.nan, math.nan
        )
        hardware._last_motor_send_coordinates = Offset3DTuple(
            math.nan, math.nan, math.nan
        )
        hardware._device_pellet_status_timeout_engaged = False
        hardware._pellet_version = ""
        content.deleteLater()
        qapp.processEvents()


def test_set_button_sends_home_relative_board_coordinate_unchanged(
    qapp,
    app_model,
    monkeypatch,
):
    hardware = app_model.hardware
    received = []
    monkeypatch.setattr(
        hardware,
        "set_y",
        lambda value, sender: received.append((value, sender)),
    )
    content = HardwareControlContent(app_model)
    try:
        content.setEnabled(True)
        content._y_pos.setValue(5.0)
        content._y_set_button.click()

        assert received == [(5.0, "UI-Set-Button")]
    finally:
        content.deleteLater()
        qapp.processEvents()


def test_no_selected_animal_displays_ui_home(
    qapp,
    app_model,
):
    content = HardwareControlContent(app_model)
    try:
        content.set_selected_animal(None)

        assert content._x_pos.value() == pytest.approx(0)
        assert content._y_pos.value() == pytest.approx(0)
        assert content._z_pos.value() == pytest.approx(0)
    finally:
        content.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("coord", "xyz")
def test_position_spinboxes_accept_typed_values(
    qapp,
    app_model,
    coord,
):
    content = HardwareControlContent(app_model)
    try:
        content.setEnabled(True)
        content.show()
        spinbox = getattr(content, f"_{coord}_pos")
        spinbox.setValue(2.3)

        QTest.mouseClick(spinbox.lineEdit(), Qt.MouseButton.LeftButton)
        qapp.processEvents()
        QTest.keyClicks(spinbox, "25")
        QTest.keyClick(spinbox, Qt.Key.Key_Enter)

        assert spinbox.value() == pytest.approx(25.0)
    finally:
        content.close()
        content.deleteLater()
        qapp.processEvents()


def test_pending_command_and_connection_share_one_availability_rule(
    qapp,
    app_model,
):
    hardware = app_model.hardware
    content = HardwareControlContent(app_model)
    try:
        content._capture_active = True
        hardware._pellet_version = "test"
        hardware._firmware_compatibility = hardware._firmware_policy.evaluate("2.0.0")
        hardware._pending_tokens.clear()

        content._refresh_enabled_state()
        assert content.isEnabled()
        assert content._x_set_button.isEnabled()

        hardware._pending_tokens["operation"] = ("moving", 0.0)
        content._refresh_enabled_state()
        assert content.isEnabled()
        assert not content._x_set_button.isEnabled()

        hardware._pellet_version = ""
        content._refresh_enabled_state()
        assert not content.isEnabled()
    finally:
        hardware._pending_tokens.clear()
        hardware._pellet_version = ""
        content.deleteLater()
        qapp.processEvents()


def _connected(content, hardware):
    content._capture_active = True
    hardware._pellet_version = "test"
    hardware._firmware_compatibility = hardware._firmware_policy.evaluate("2.0.0")


def _released(content, hardware, qapp):
    hardware._pending_tokens.clear()
    hardware._pellet_version = ""
    hardware._estop_record = None
    content.deleteLater()
    qapp.processEvents()


def test_stop_stays_enabled_while_a_command_is_running(qapp, app_model):
    hardware = app_model.hardware
    content = HardwareControlContent(app_model)
    try:
        _connected(content, hardware)
        hardware._pending_tokens["operation"] = ("moving", 0.0)
        content._refresh_enabled_state()

        assert not content._x_set_button.isEnabled()
        assert content._stop_button.isEnabled()
        assert content._stop_button not in content._commands_widgets

        content._capture_active = False  # outside a capture as well
        content._refresh_enabled_state()
        assert content._stop_button.isEnabled()
    finally:
        _released(content, hardware, qapp)


def test_stop_calls_the_hardware_models_estop_directly(qapp, app_model, monkeypatch):
    hardware = app_model.hardware
    content = HardwareControlContent(app_model)
    calls = []
    monkeypatch.setattr(hardware, "estop", lambda reason: calls.append(reason) or True)
    pellet_machine = app_model.behavior.system_machine.pellet
    monkeypatch.setattr(pellet_machine, "move_home", lambda **kwargs: calls.append(("move_home", kwargs)))
    try:
        _connected(content, hardware)
        hardware._pending_tokens["operation"] = ("moving", 0.0)
        content._refresh_enabled_state()

        QTest.mouseClick(content._stop_button, Qt.MouseButton.LeftButton)

        assert calls == ["operator STOP"]
    finally:
        _released(content, hardware, qapp)


def _home_button(content):
    from PySide6.QtWidgets import QPushButton

    return next(b for b in content.findChildren(QPushButton) if b.text() == "Home")


def test_home_after_a_stop_releases_the_latch_through_the_hardware_model(qapp, app_model, monkeypatch):
    hardware = app_model.hardware
    content = HardwareControlContent(app_model)
    calls = []
    monkeypatch.setattr(hardware, "home_after_estop", lambda: calls.append("home_after_estop"))
    pellet_machine = app_model.behavior.system_machine.pellet
    monkeypatch.setattr(pellet_machine, "move_home", lambda **kwargs: calls.append("move_home"))
    try:
        _connected(content, hardware)
        content._refresh_enabled_state()
        QTest.mouseClick(_home_button(content), Qt.MouseButton.LeftButton)
        hardware._estop_record = {"reason": "operator STOP"}
        QTest.mouseClick(_home_button(content), Qt.MouseButton.LeftButton)

        assert calls == ["move_home", "home_after_estop"]
    finally:
        _released(content, hardware, qapp)
