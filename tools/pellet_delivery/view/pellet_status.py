import math
from typing import Optional

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFormLayout

from autotrainer.core import Offset3DTuple, get_verbose_logger
from autotrainer.pyside import CardWidget
from autotrainer.pyside.xyz_label import XYZQLabel
from tools.pellet_delivery.model.app_model import AppModel


logger = get_verbose_logger(__name__)

_NO_UPDATES = "(no updates)"


def opt_float_to_txt(value: Optional[float], *, n_digits: int = 1):
    if value is None:
        return _NO_UPDATES
    return f"{value:.0{n_digits}f}"


def create_position_panel():
    layout = QFormLayout()
    layout.setHorizontalSpacing(8)
    x_y_z_device = XYZQLabel(_NO_UPDATES)
    layout.addRow("Stepper[motor] X/Y/Z (mm):", x_y_z_device)
    # ◈
    x_y_z_diamond = XYZQLabel(_NO_UPDATES)
    layout.addRow("Stepper[diamo] X/Y/Z (mm):", x_y_z_diamond)

    layout.setContentsMargins(8, 8, 8, 8)

    panel = CardWidget(title="Motor Location", content_layout=layout)

    return x_y_z_device, x_y_z_diamond, panel

def create_send_position_panel():
    layout = QFormLayout()
    layout.setHorizontalSpacing(8)

    x_y_z_device = XYZQLabel(_NO_UPDATES)
    layout.addRow("Send[motor] X/Y/Z (mm):", x_y_z_device)

    x_y_z_diamond = XYZQLabel(_NO_UPDATES)
    layout.addRow("Send[diamo] X/Y/Z (mm):", x_y_z_diamond)

    layout.setContentsMargins(8, 8, 8, 8)

    panel = CardWidget(title="Send Location", content_layout=layout)

    return x_y_z_device, x_y_z_diamond, panel


def create_servo_panel():
    layout = QFormLayout()
    layout.setHorizontalSpacing(8)

    load_arm = QLabel(_NO_UPDATES)
    cover_arm = QLabel(_NO_UPDATES)

    layout.addRow("Load Arm (\u00b0):", load_arm)
    layout.addRow("Cover Arm (\u00b0):", cover_arm)

    layout.setContentsMargins(8, 8, 8, 8)

    panel = CardWidget(title="Servos", content_layout=layout)

    return load_arm, cover_arm, panel


class PelletStatus(QWidget):
    def __init__(self, app_model: AppModel):
        super().__init__()

        self._app_model = app_model
        app_model.property_changed += self._model_property_changed

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self._xyz_device, self._xyz_diamond, panel = create_position_panel()
        layout.addWidget(panel)

        self._send_xyz_device, self._send_xyz_diamond, panel = create_send_position_panel()
        layout.addWidget(panel)

        self._load_arm, self._cover_arm, panel = create_servo_panel()
        layout.addWidget(panel)

        self.setLayout(layout)

    def _model_property_changed(self, name: str, value, _old_value):
        app_model = self._app_model
        if name == "is_connected":
            if value:
                reset_prop = self._model_property_changed
                reset_prop("xyz", app_model.xyz, None)
                reset_prop("send_xyz", app_model.send_xyz, None)
                reset_prop("load_arm", app_model.load_arm, None)
                reset_prop("cover_arm", app_model.cover_arm, None)
            else:
                for xyz_label in self._xyz_device, self._xyz_diamond, self._send_xyz_device, self._send_xyz_diamond:
                    xyz_label.setText(_NO_UPDATES)
                self._load_arm.setText(_NO_UPDATES)
                self._cover_arm.setText(_NO_UPDATES)
            return

        if app_model.is_connected:
            if name == "xyz":
                self._xyz_device.update_coordinate(value)
                self._xyz_diamond.update_coordinate(app_model.to_diamond_coordinates(value))

            elif name == "send_xyz":
                self._send_xyz_device.update_coordinate(value)
                self._send_xyz_diamond.update_coordinate(app_model.to_diamond_coordinates(value))

            elif name in {'x', 'y', 'z'}:
                d = {name: value}
                self._xyz_device.update_coordinate(**d)
                cur_xyz = app_model.xyz.replace(**d)
                self._xyz_diamond.update_coordinate(app_model.to_diamond_coordinates(cur_xyz))

            elif name in {'send_x', 'send_y', 'send_z'}:
                d = {name[-1]: value}
                self._send_xyz_device.update_coordinate(**d)
                cur_send_xyz = app_model.send_xyz.replace(**d)
                self._send_xyz_diamond.update_coordinate(app_model.to_diamond_coordinates(cur_send_xyz))

            elif name == "load_arm":
                self._load_arm.setText(opt_float_to_txt(value))

            elif name == "cover_arm":
                self._cover_arm.setText(opt_float_to_txt(value))

