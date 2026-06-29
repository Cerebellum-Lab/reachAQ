from typing import Optional, Callable
from functools import partial

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QLabel, QSpinBox, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QFormLayout, QStackedLayout, QSizePolicy, QComboBox, QDoubleSpinBox)

from autotrainer.behavior.behavior_algorithm import BehaviorAlgoProps
from autotrainer.core import AnimalSubject, Offset3DTuple
from autotrainer.core.logging import get_verbose_logger

from autotrainer.model import EnvironmentProvider, HardwareVersion

from autotrainer.pyside import CardWidget
from autotrainer.pyside.StackedContent import StackedLayout
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import AppModelStatus

from tools.acquisition.model.hardware_model import HardwareModel


logger = get_verbose_logger(__name__)


# TODO: This is just to see if the behavior is correct.  They should end up somewhere that any application or script can
#  access.
_anshutz_travel_limits = {
    "x": (-10, 10),
    "y": (-10, 10),
    "z": (-10, 10),
}

_alogus_travel_limits = {
    "x": (0, 35),
    "y": (0, 35),
    "z": (0, 35),
}


class HardwareControlContent(ContentWidget):

    position_changed = Signal(int, name="position_changed")
    command_changed = Signal(str, name="command_changed")

    def __init__(self, app_model: AppModel):
        super().__init__()

        self._app_model = app_model
        self._hardware_model = app_model.hardware

        self._commands_widgets = []
        add_cmd_widget = self._commands_widgets.append

        def log_hardware_cmd(cmd: Callable):
            orig_func = getattr(cmd, "_orig_func_qualname", cmd)
            logger.verbose("User-control: Executing %s", orig_func)
            return cmd()

        # Header
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        layout.setSpacing(8)

        self._tunnel_version_label = QLabel("Tunnel:")
        layout.addWidget(self._tunnel_version_label)
        self._tunnel_version = QLabel("(unknown version)")
        layout.addWidget(self._tunnel_version)

        layout.addWidget(QLabel("Pellet:"))
        self._pellet_version = QLabel("(unknown version)")
        layout.addWidget(self._pellet_version)

        self._card_widget = CardWidget(title="Hardware Control", header_right_layout=layout)
        # self._card_widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)

        if EnvironmentProvider.hardware_version() == HardwareVersion.ANSHUTZ:
            self._travel_limits = _anshutz_travel_limits
        else:
            self._travel_limits = _alogus_travel_limits

        layout = QGridLayout()
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        label = self._tunnel_section_label = QLabel("<b>Tunnel</b>")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, 0, 0)
        vbox = QVBoxLayout()
        label = QLabel("<b>Pellet Release Location (mm)</b>")
        label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(label)
        label = self._is_motor_cs_label = QLabel("<b>@ MotorCoordSystem</b>")
        if app_model.behavior.algorithm.diamond_triangle_config is not None:
            label.hide()
        label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(label)
        layout.addLayout(vbox, 0, 2)

        label = QLabel("<b>Compound Move</b>")
        label.setAlignment(Qt.AlignCenter)
        label.setContentsMargins(0, 0, 0, 4)  # ensure small margin below
        layout.addWidget(label, 0, 4)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 4, 0, 0)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(4)

        spinbox = self._head_magnet_position_spinbox = QSpinBox()
        add_cmd_widget(spinbox)
        spinbox.setValue(0)
        spinbox.setMaximum(100)
        spinbox.setWrapping(False)
        button = self._head_magnet_move_button = QPushButton("Move")
        add_cmd_widget(button)
        def head_magnet_intensity_updated():
            self._hardware_model.update_head_magnet_intensity(self._head_magnet_position_spinbox.value())
        button.clicked.connect(lambda: log_hardware_cmd(head_magnet_intensity_updated))

        right_layout = QHBoxLayout()
        right_layout.setSpacing(4)
        right_layout.addWidget(self._head_magnet_position_spinbox)
        right_layout.addWidget(self._head_magnet_move_button)
        self._head_magnet_row_label = QLabel("Head magnet intensity (%):")
        form_layout.addRow(self._head_magnet_row_label, right_layout)

        button = self._tare_button = QPushButton("Tare")
        add_cmd_widget(button)
        button.clicked.connect(lambda: log_hardware_cmd(self._hardware_model.tare_load_cell))
        self._load_cell_row_label = QLabel("Load cell:")
        form_layout.addRow(self._load_cell_row_label, self._tare_button)

        layout.addLayout(form_layout, 1, 0)

        algo = app_model.behavior.algorithm

        def set_xyz(coord: str):
            value = getattr(self, f"_{coord}_pos").value()
            coord_idx = "xyz".index(coord)
            assert coord_idx in (0, 1, 2)
            t = [self._x_pos.value(), self._y_pos.value(), self._z_pos.value()]
            t[coord_idx] = value
            xyz = Offset3DTuple(*t)
            cfg = algo.diamond_triangle_config
            if cfg is not None:
                xyz = cfg.diamond_to_motor(xyz)
            value = xyz[coord_idx]
            meth = getattr(self._hardware_model, f"set_{coord}")
            meth(value, sender="UI-Set-Button")

        sub_layout = QGridLayout()
        sub_layout.setContentsMargins(0, 0, 0, 0)
        sub_layout.setHorizontalSpacing(4)
        sub_layout.setVerticalSpacing(4)
        sub_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        row = col = 0

        def add_coord(coord: str):
            nonlocal row
            pos = QDoubleSpinBox()
            add_cmd_widget(pos)
            pos.setValue(0)
            pos.setContentsMargins(0, 0, 0, 0)
            pos.setMinimumWidth(60)
            pos.setDecimals(1)
            pos.setSingleStep(0.5)
            pos.setAlignment(Qt.AlignmentFlag.AlignRight)
            range_label = QLabel()
            set_button = QPushButton("Set")
            add_cmd_widget(set_button)
            set_button.clicked.connect(partial(set_xyz, coord))
            sub_layout.addWidget(QLabel(f"{coord.upper()} :"), row, col)
            sub_layout.addWidget(pos, row, col + 1)
            sub_layout.addWidget(range_label, row, col + 2)
            sub_layout.addWidget(set_button, row, col + 3)
            row += 1
            return pos, set_button, range_label

        self._x_pos, self._x_set_button, self._x_range_label = add_coord('x')
        self._y_pos, self._y_set_button, self._y_range_label = add_coord('y')
        self._z_pos, self._z_set_button, self._z_range_label = add_coord('z')

        layout.addLayout(sub_layout, 1, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self._set_pos_limits()

        #

        pellet_machine = app_model.behavior.system_machine.pellet

        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)
        #
        button = QPushButton("Home")
        add_cmd_widget(button)
        button.clicked.connect(
            lambda: log_hardware_cmd(partial(pellet_machine.move_home, force=True)))
        button_layout.addWidget(button)
        #
        button = QPushButton("Load")
        add_cmd_widget(button)
        button.clicked.connect(
            lambda: log_hardware_cmd(partial(pellet_machine.load_pellet, force=True, reason="manual_button")))
        button_layout.addWidget(button)
        #
        button = QPushButton("Send")
        add_cmd_widget(button)
        button.clicked.connect(lambda: log_hardware_cmd(partial(pellet_machine.send_pellet, force=True)))
        button_layout.addWidget(button)
        #
        button = QPushButton("Retract")
        add_cmd_widget(button)
        button.clicked.connect(lambda: log_hardware_cmd(partial(pellet_machine.move_retract, force=True)))
        button_layout.addWidget(button)
        #
        button = QPushButton("Release")
        add_cmd_widget(button)
        button.clicked.connect(
            lambda: log_hardware_cmd(partial(pellet_machine.release_pellet, force=True)))
        button_layout.addWidget(button)
        #
        button = QPushButton("Cover")
        add_cmd_widget(button)
        button.clicked.connect(
            lambda: log_hardware_cmd(partial(pellet_machine.cover_pellet, force=True)))
        button_layout.addWidget(button)
        layout.addLayout(button_layout, 1, 4)

        # central layout/widget
        widget = QWidget()
        widget.setLayout(layout)
        # widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Pre)
        self._card_widget.setContentWidget(widget)

        # Footer
        self._basic_footer = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Command in progress:"))
        self._command_label = QLabel("None")
        layout.addWidget(self._command_label)
        self._basic_footer.setLayout(layout)
        self._stack_layout = StackedLayout()
        self._stack_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._stack_layout.addWidget(self._basic_footer)
        widget = QWidget()
        widget.setLayout(self._stack_layout)
        self._card_widget.footer.setContent(widget)

        # Final layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._card_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)
        # self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.setEnabled(False)

        self.command_changed.connect(lambda x: self._command_label.setText(x))

        app_model.behavior.algorithm.property_changed += self._on_algo_property_changed
        app_model.property_changed += self._on_app_model_property_changed
        self._hardware_model.property_changed += self._on_hardware_model_property_changed
        self._update_tunnel_headfix_visibility(self._hardware_model.tunnel_headfix_enabled)

    def _set_pos_limits(self):
        limits = self._travel_limits
        if limits is not None:
            algo = self._app_model.behavior.algorithm
            min_xyz = Offset3DTuple(*(limits[c][0] for c in 'xyz'))
            max_xyz = Offset3DTuple(*(limits[c][1] for c in 'xyz'))
            diamond_triangle_cfg = algo.diamond_triangle_config
            if diamond_triangle_cfg is not None and diamond_triangle_cfg.fully_valid:
                min_xyz = diamond_triangle_cfg.motor_to_diamond(min_xyz)
                max_xyz = diamond_triangle_cfg.motor_to_diamond(max_xyz)
            for idx, pos in enumerate((self._x_pos, self._y_pos, self._z_pos)):
                c = "xyz"[idx]
                v1, v2 = min_xyz[idx], max_xyz[idx]
                r = min(v1, v2), max(v1, v2)
                pos.setRange(*r)
                getattr(self, f"_{c}_range_label").setText(f"[ {r[0]:>5.1f} : {r[1]:<5.1f}]")

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        pass

    @invoke_method
    def set_selected_animal(self, animal: Optional[AnimalSubject]):
        self._set_pos_limits()
        algo = self._app_model.behavior.algorithm
        cfg = algo.diamond_triangle_config
        if cfg is None or not cfg.fully_valid:
            logger.notice("Displaying animal data with Motor coordinate system")
            self._is_motor_cs_label.show()
            if animal is None:
                xyz = Offset3DTuple(0, 0, 0)
            else:
                if animal.is_pellet_dcs:
                    animal.pellet_x = animal.pellet_y = animal.pellet_z = 0
                    animal.is_pellet_dcs = False
                xyz = Offset3DTuple(animal.pellet_x, animal.pellet_y, animal.pellet_z)
        else:
            logger.debug("Displaying animal data with Diamond coordinate system")
            self._is_motor_cs_label.hide()
            if animal is None:
                xyz = Offset3DTuple(0, 0, 0)
                xyz = cfg.motor_to_diamond(xyz)
            else:
                xyz = Offset3DTuple(animal.pellet_x, animal.pellet_y, animal.pellet_z)
                if not animal.is_pellet_dcs:
                    xyz = cfg.motor_to_diamond(xyz)

        for widget, value in (
            (self._x_pos, xyz.x),
            (self._y_pos, xyz.y),
            (self._z_pos, xyz.z),
        ):
            assert isinstance(widget, (QDoubleSpinBox, QSpinBox))
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        self.update()

    def _update_head_magnet_position(self):
        self._hardware_model.update_head_magnet_intensity(self._head_magnet_position_spinbox.value())

    def _update_title(self, value: str):
        if value:
            if value.find("emulator") != -1:
                self._card_widget.header.setTitle("Hardware Control: Alogus Emulation")
            else:
                self._card_widget.header.setTitle(f"Hardware Control: {EnvironmentProvider.hardware_version()}")
        else:
            self._card_widget.header.setTitle("Hardware Control")

    def _update_tunnel_headfix_visibility(self, is_enabled: bool):
        for widget in (
            self._tunnel_version_label,
            self._tunnel_version,
            self._tunnel_section_label,
            self._head_magnet_row_label,
            self._head_magnet_position_spinbox,
            self._head_magnet_move_button,
            self._load_cell_row_label,
            self._tare_button,
        ):
            widget.setVisible(is_enabled)

    def set_commands_enabled(self, enabled: bool = True):
        for widget in self._commands_widgets:
            widget.setEnabled(enabled)

    @invoke_method
    def _on_app_model_property_changed(self, name, value, _):
        app_model = self._app_model
        if name == app_model.Props.STATUS:
            self.setEnabled(value != AppModelStatus.IDLE)

    @invoke_method
    def _on_hardware_model_property_changed(self, property_name: str, value, _):
        if property_name == HardwareModel.TUNNEL_VERSION_PROPERTY:
            self._update_title(value)
            if value:
                self._tunnel_version.setText(value.replace("emulator", "").strip())
            else:
                self._tunnel_version.setText("(unknown version)")

        elif property_name == HardwareModel.TUNNEL_HEADFIX_ENABLED:
            self._update_tunnel_headfix_visibility(value)

        elif property_name == HardwareModel.PELLET_VERSION_PROPERTY:
            self._update_title(value)
            if value:
                self._pellet_version.setText(value.replace("emulator", "").strip())
                self.setEnabled(True)
            else:
                self._pellet_version.setText("(unknown version)")
                self.setEnabled(False)
                self.command_changed.emit("None")

        elif property_name == HardwareModel.PENDING_COMMAND_PROPERTY:
            if value is not None:
                self.command_changed.emit(value)
                self.set_commands_enabled(False)
            else:
                self.command_changed.emit("None")
                self.set_commands_enabled(True)

    @invoke_method
    def _on_algo_property_changed(self, name, value, _):
        if name == BehaviorAlgoProps.DIAMOND_TRIANGLE_CONFIG:
            # force execute set-selected-animal
            self.set_selected_animal(self._app_model.selected_animal)
            # this will set as desired the UI elements.
