
from typing import Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QSpinBox, \
    QLayout, QVBoxLayout, QFileDialog, QFrame, QDoubleSpinBox, QComboBox, QSizePolicy

import qtawesome as qta

from autotrainer.core import Offset3DTuple
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.message import Motor
from autotrainer.device import is_servo
from autotrainer.device.coordinate_system import CoordinateSystem, COORDINATE_SYSTEMS
from autotrainer.pyside import Separator, CardWidget

from tools.pellet_delivery.model.app_model import AppModel
from autotrainer.pyside import MotorConfigDialog


logger = get_verbose_logger(__name__)

_MIN_CONTROL_BUTTON_WIDTH = 120


def add_position(label: str, s_min: float, s_max: float) -> Tuple[
    QHBoxLayout, QDoubleSpinBox, QPushButton, QPushButton, QLabel
]:
    position_layout = QHBoxLayout()
    position_layout.setContentsMargins(8, 8, 8, 8)
    position_layout.setSpacing(8)

    q_label = QLabel(label)
    position_layout.addWidget(q_label, 0)

    pos = QDoubleSpinBox()
    pos.setMinimumWidth(40)
    pos.setRange(s_min, s_max)
    pos.setDecimals(1)
    pos.setSingleStep(0.2)
    pos.setWrapping(False)
    position_layout.addWidget(pos, 0)

    move_button = QPushButton("Move")
    position_layout.addWidget(move_button, 0)

    set_button = QPushButton("Set")
    position_layout.addWidget(set_button, 0)

    return position_layout, pos, move_button, set_button, q_label


class PelletControl(QWidget):
    def __init__(self, app_model: AppModel):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        self._app_model: AppModel = app_model

        self._config_dialog = None

        self._app_model.property_changed += self._model_property_changed

        layout = QVBoxLayout()
        layout.addLayout(self._create_button_layout())
        layout.addWidget(Separator("#dedede"))
        layout.addLayout(self._create_move_layout())

        panel = CardWidget(title="Control", content_layout=layout)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(panel)

        self.setLayout(layout)
        self.setEnabled(False)

    def _create_button_layout(self):
        b_layout = QHBoxLayout()
        b_layout.setContentsMargins(2, 2, 2, 2)
        b_layout.setSpacing(8)

        button = self._home_button = QPushButton("Home")
        button.setMinimumWidth(_MIN_CONTROL_BUTTON_WIDTH)
        button.clicked.connect(self._app_model.send_home)
        b_layout.addWidget(button)

        b_layout.addStretch(1)

        button = self._load_button = QPushButton("Load")
        button.setMinimumWidth(_MIN_CONTROL_BUTTON_WIDTH)
        button.clicked.connect(self._app_model.load_pellet)
        b_layout.addWidget(button)

        button = self._send_button = QPushButton("Send")
        button.setMinimumWidth(_MIN_CONTROL_BUTTON_WIDTH)
        button.clicked.connect(self._app_model.send_pellet)
        b_layout.addWidget(button)

        button = self._retract_button = QPushButton("Retract")
        button.setMinimumWidth(_MIN_CONTROL_BUTTON_WIDTH)
        button.clicked.connect(self._app_model.move_retract)
        b_layout.addWidget(button)

        b_layout.addStretch(1)

        button = self._release_button = QPushButton("Release")
        button.setMinimumWidth(_MIN_CONTROL_BUTTON_WIDTH)
        button.clicked.connect(self._app_model.release_pellet)
        b_layout.addWidget(button)

        button = self._cover_button = QPushButton("Cover")
        button.setMinimumWidth(_MIN_CONTROL_BUTTON_WIDTH)
        button.clicked.connect(self._app_model.cover_pellet)
        b_layout.addWidget(button)

        b_layout.addStretch(1)

        self._move_file_button = QPushButton("")
        folder_icon = qta.icon('fa5s.folder-open')
        self._move_file_button.setIcon(folder_icon)
        self._move_file_button.clicked.connect(lambda: self._load_move_file())
        b_layout.addWidget(self._move_file_button)

        self._config_button = QPushButton("")
        gear_icon = qta.icon('fa5s.cog')  # Font Awesome 5 Solid cog icon
        self._config_button.setIcon(gear_icon)
        self._config_button.clicked.connect(lambda: self._update_config())
        b_layout.addWidget(self._config_button)

        return b_layout

    def _create_move_layout(self):
        s_layout = QHBoxLayout()
        s_layout.setContentsMargins(2, 2, 2, 2)

        # NB: following X/Y/Z pos labels text are anyway reset after/below when we set the coordinate system
        p_layout, self._x_pos, moveButton, setButton, self._x_label = add_position("X", -10, 10)
        moveButton.clicked.connect(self._move_x)
        setButton.clicked.connect(self._set_x)
        s_layout.addLayout(p_layout)

        s_layout.addStretch(1)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        s_layout.addWidget(line)

        s_layout.addStretch(1)

        p_layout, self._y_pos, moveButton, setButton, self._y_label = add_position("Y", -10, 10)
        moveButton.clicked.connect(self._move_y)
        setButton.clicked.connect(self._set_y)
        s_layout.addLayout(p_layout)

        s_layout.addStretch(1)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        s_layout.addWidget(line)

        s_layout.addStretch(1)

        p_layout, self._z_pos, moveButton, setButton, self._z_label = add_position("Z", -10, 10)
        moveButton.clicked.connect(self._move_z)
        setButton.clicked.connect(self._set_z)
        s_layout.addLayout(p_layout)

        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(0, 5, 0, 0)

        combo = self._combo_coordinate_system = QComboBox()
        for coord_system in COORDINATE_SYSTEMS:
            combo.addItem(coord_system.value, coord_system)
        def select_coordinate(_: int):
            self._refresh_pellet_xyz_control()
        combo.setCurrentIndex(0)
        combo.currentIndexChanged.connect(select_coordinate)
        select_coordinate(0)  # ensure we set as when switched to
        combo_l = QHBoxLayout()
        combo_l.setContentsMargins(8, 0, 0, 0)
        combo_l.setSpacing(5)
        combo_l.addWidget(QLabel("Coordinate system:"))
        combo_l.addWidget(combo)
        combo_l.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v_layout.addLayout(combo_l)
        v_layout.addLayout(s_layout)

        return v_layout

    def _to_motor(self, xyz: Offset3DTuple) -> Offset3DTuple:
        return (
            self._app_model.to_motor_coordinates(xyz)
                if self._combo_coordinate_system.currentData() == CoordinateSystem.Diamond
            else xyz
        )

    def _set_x(self):
        self._app_model.set_x(self._to_motor(Offset3DTuple(self._x_pos.value(), 0, 0)).x)

    def _set_y(self):
        self._app_model.set_y(self._to_motor(Offset3DTuple(0, self._y_pos.value(), 0)).y)

    def _set_z(self):
        self._app_model.set_z(self._to_motor(Offset3DTuple(0, 0, self._z_pos.value())).z)

    def _move_x(self):
        self._app_model.move_x(self._to_motor(Offset3DTuple(self._x_pos.value(), 0, 0)).x)

    def _move_y(self):
        self._app_model.move_y(self._to_motor(Offset3DTuple(0, self._y_pos.value(), 0)).y)

    def _move_z(self):
        self._app_model.move_z(self._to_motor(Offset3DTuple(0, 0, self._z_pos.value())).z)

    def _refresh_pellet_xyz_control(self):
        cur_coordinate_system = self._combo_coordinate_system.currentData()
        app_model = self._app_model
        limits = app_model.travel_limits
        if limits is None:
            min_xyz = max_xyz = None
        else:
            min_xyz = Offset3DTuple(*(limits[c][0] for c in 'xyz'))
            max_xyz = Offset3DTuple(*(limits[c][1] for c in 'xyz'))
        if cur_coordinate_system == CoordinateSystem.Motor:
            value = app_model.xyz
        elif cur_coordinate_system == CoordinateSystem.Diamond:
            cur_coordinate_system = "Diamo"  # for the x/y/z labels
            if limits is not None:
                min_xyz = app_model.to_diamond_coordinates(min_xyz)
                max_xyz = app_model.to_diamond_coordinates(max_xyz)
            value = app_model.to_diamond_coordinates(app_model.xyz)
        else:
            raise RuntimeError(f"Unhandled {cur_coordinate_system}")
        #
        for idx, pos in enumerate((self._x_pos, self._y_pos, self._z_pos)):
            if limits is not None:
                v1, v2 = min_xyz[idx], max_xyz[idx]
                r = min(v1, v2), max(v1, v2)
                pos.setRange(*r)
            else:
                r = None
            pos.blockSignals(True)
            pos.setValue(value[idx])
            pos.blockSignals(False)
            logger.debug("setting %s -> %s", pos, r)

        for idx, label in enumerate((self._x_label, self._y_label, self._z_label)):
            c = "XYZ"[idx]
            label.setText(f"<b>{c}</b>-{cur_coordinate_system} (mm):")

    def _model_property_changed(self, name: str, value, _old_value):
        if name == "travel_limits":
            logger.debug("got & applying travel_limits: %s", value)
            if value is not None:
                self._refresh_pellet_xyz_control()
        elif name == "config":
            if self._config_dialog is not None:
                if is_servo(value.motor):
                    self._config_dialog.update_servo_config(value)
                else:
                    self._config_dialog.update_stepper_config(value)
        elif name in {'x', 'y', 'z'}:
            self._refresh_pellet_xyz_control()
        elif name == "is_connected":
            if value:
                self._refresh_pellet_xyz_control()

    def _update_config(self):
        self._config_dialog = MotorConfigDialog(self)
        self._config_dialog.motor_selected.connect(self._on_motor_selected)
        self._config_dialog.accepted.connect(self._on_config_accepted)
        self._config_dialog.rejected.connect(lambda: setattr(self, '_config_dialog', None))

        self._config_dialog.setModal(True)
        self._config_dialog.show()

    def _on_config_accepted(self):
        if self._config_dialog.config is not None:
            self._app_model.set_config((self._config_dialog.config.motor,
                                        self._config_dialog.config))

        self._config_dialog = None

    def _on_motor_selected(self, motor: Motor):
        self._app_model.get_config(motor)

    def _load_move_file(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Open File")
        dialog.setFileMode(QFileDialog.ExistingFile)  # Allow only one file to be selected

        # Optionally set filters for file types
        dialog.setNameFilter("All Files (*)")

        # Show the dialog and wait for user's response
        if dialog.exec_():
            # Get the selected file path(s) - will be a list
            selected_files = dialog.selectedFiles()
            # Return the first (and likely only) file

            self._app_model.load_move_file(selected_files[0])
