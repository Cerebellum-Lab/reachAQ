from functools import partial

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QWidget, QGridLayout, QFormLayout, QSizePolicy

from autotrainer.core import MessageHandler, Offset3DTuple
from autotrainer.core.logging import get_verbose_logger

from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from autotrainer.pyside.xyz_label import XYZQLabel

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.hardware_model import HardwareModel


logger = get_verbose_logger(__name__)


class HardwareStatusContent(ContentWidget):

    head_magnet_changed = Signal(float, name="head_magnet_changed")
    pellet_x_changed = Signal(float, name="pellet_x_changed")
    pellet_y_changed = Signal(float, name="pellet_y_changed")
    pellet_z_changed = Signal(float, name="pellet_z_changed")
    send_x_changed = Signal(float, name="send_x_changed")
    send_y_changed = Signal(float, name="send_y_changed")
    send_z_changed = Signal(float, name="send_z_changed")
    load_arm_changed = Signal(float, name="load_arm_changed")
    cover_arm_changed = Signal(float, name="cover_arm_changed")

    def __init__(self, app_model: AppModel):
        super().__init__()

        self._app_model = app_model
        self._message_handler = app_model.message_handler

        self._card_widget = CardWidget(title="Hardware Status")
        # self._card_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._message_handler.property_changed += self._model_property_changed

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(2, 2, 0, 0)
        layout.setHorizontalSpacing(8)
        content_layout.addLayout(layout)

        cur_row = cur_col = 0

        label = self._tunnel_section_label = QLabel("<b>Tunnel:</b>")
        layout.addWidget(label, cur_row, cur_col)
        cur_row += 1

        self._head_magnet_label = QLabel("Head magnet (%):")
        layout.addWidget(self._head_magnet_label, cur_row, cur_col)
        self._head_magnet = QLabel("(no updates)")
        layout.addWidget(self._head_magnet, cur_row, cur_col + 1)
        cur_row += 1

        #

        label = QLabel("<b>Pellet:</b>")
        label.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(label, cur_row, cur_col)
        cur_row += 1

        layout.addWidget(QLabel("XYZ (mm):"), cur_row, cur_col)
        label = self._pellet_xyz = XYZQLabel()
        label.setObjectName("PelletXYZ")
        layout.addWidget(label, cur_row, cur_col + 1)
        cur_row += 1

        layout.addWidget(QLabel("Send XYZ (mm):"), cur_row, cur_col)
        label = self._send_pellet_xyz = XYZQLabel()
        label.setObjectName("SendXYZ")
        layout.addWidget(label, cur_row, cur_col + 1)
        cur_row += 1

        layout.addWidget(QLabel("Load Arm (\u00b0):"), cur_row, cur_col)
        self._load_arm = QLabel("(no updates)")
        layout.addWidget(self._load_arm, cur_row, cur_col + 1)
        cur_row += 1

        layout.addWidget(QLabel("Cover Arm (\u00b0):"), cur_row, cur_col)
        self._cover_arm = QLabel("(no updates)")
        layout.addWidget(self._cover_arm, cur_row, cur_col + 1)

        content = QWidget()
        # content.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        content.setContentsMargins(8, 4, 4, 4)
        content.setLayout(content_layout)

        self._card_widget.setContentWidget(content)

        #
        # Final layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._card_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)
        # self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)

        def xyz_update(xyz_label: XYZQLabel, coord: str, value):
            # logger.debug("xyz_update: %s %s -> %s", xyz_label.objectName(), coord, value)
            # NB: this function is called repeatedly over and over again,
            # at the freq of the motor-status CAN/system messages.
            algo = app_model.behavior.algorithm
            cfg = algo.diamond_triangle_config
            coord_idx = "xyz".index(coord)
            assert coord_idx in (0, 1, 2)
            t = [0, 0, 0]
            # NB: atm the coordinate systems are presumed to have all their **same axis** parallel one to another.
            # i.e. we suppose that a change to X in one coordinate system won't change the value of Y and Z in the other coordinate system,
            # and same for Y and Z respectively.
            t[coord_idx] = value
            motor_coord = Offset3DTuple(*t)
            if cfg is not None and cfg.fully_valid:
                xyz_value = cfg.motor_to_diamond(motor_coord)
                suffix = None
            else:
                xyz_value = motor_coord
                suffix = " @ MotorCoordSystem"
            coord_value = getattr(xyz_value, coord)
            xyz_label.update_coordinate(**{coord: coord_value}, suffix=suffix)

        self.head_magnet_changed.connect(lambda x: self._head_magnet.setText(str(round(x, 1))))
        self.pellet_x_changed.connect(partial(xyz_update, self._pellet_xyz, "x"))
        self.pellet_y_changed.connect(partial(xyz_update, self._pellet_xyz, "y"))
        self.pellet_z_changed.connect(partial(xyz_update, self._pellet_xyz, "z"))
        self.send_x_changed.connect(partial(xyz_update, self._send_pellet_xyz, "x"))
        self.send_y_changed.connect(partial(xyz_update, self._send_pellet_xyz, "y"))
        self.send_z_changed.connect(partial(xyz_update, self._send_pellet_xyz, "z"))
        self.load_arm_changed.connect(lambda x: self._load_arm.setText(str(round(x, 1))))
        self.cover_arm_changed.connect(lambda x: self._cover_arm.setText(str(round(x, 1))))
        app_model.hardware.property_changed += self._on_hardware_model_property_changed
        self._update_tunnel_headfix_visibility(app_model.hardware.tunnel_headfix_enabled)

    def _update_tunnel_headfix_visibility(self, is_enabled: bool):
        self._tunnel_section_label.setVisible(is_enabled)
        self._head_magnet_label.setVisible(is_enabled)
        self._head_magnet.setVisible(is_enabled)

    @invoke_method
    def _on_hardware_model_property_changed(self, property_name: str, value, _):
        if property_name == HardwareModel.TUNNEL_HEADFIX_ENABLED:
            self._update_tunnel_headfix_visibility(value)

    @invoke_method
    def _model_property_changed(self, property_name: str, value, _):
        # If any of the values may be coming from a different thread (e.g., the device), a signal is generally needed
        # rather than direct set/update.
        if property_name == MessageHandler.HEAD_MAGNET_INTENSITY_PROPERTY:
            self.head_magnet_changed.emit(value)

        elif property_name == MessageHandler.STEPPER_X_PROPERTY:
            self.pellet_x_changed.emit(value.position)
            self.send_x_changed.emit(value.send_position)

        elif property_name == MessageHandler.STEPPER_Y_PROPERTY:
            self.pellet_y_changed.emit(value.position)
            self.send_y_changed.emit(value.send_position)

        elif property_name == MessageHandler.STEPPER_Z_PROPERTY:
            self.pellet_z_changed.emit(value.position)
            self.send_z_changed.emit(value.send_position)

        elif property_name == MessageHandler.LOAD_ARM_ANGLE_PROPERTY:
            self.load_arm_changed.emit(value)

        elif property_name == MessageHandler.COVER_ARM_ANGLE_PROPERTY:
            self.cover_arm_changed.emit(value)
