import math
from typing import Optional, Callable
from functools import partial

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QLabel, QSpinBox, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QDoubleSpinBox)

from autotrainer.behavior.behavior_algorithm import BehaviorAlgoProps
from autotrainer.core import AnimalSubject, Offset3DTuple
from autotrainer.core.logging import get_verbose_logger

from autotrainer.model import EnvironmentProvider

from autotrainer.pyside import CardWidget
from autotrainer.pyside.StackedContent import StackedLayout
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import AppModelStatus

from tools.acquisition.model.hardware_model import HardwareModel


logger = get_verbose_logger(__name__)


_UI_POSITION_MIN = 0.0
_UI_POSITION_MAX = 35.0


def _format_motor_feedback_value(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    formatted = f"{value:.1f}"
    if formatted == "-0.0":
        formatted = "0.0"
    return formatted.replace("-", "−", 1)


def _format_motor_feedback(status: str, position: Offset3DTuple) -> str:
    x, y, z = (_format_motor_feedback_value(value) for value in position)
    return f"• {status} •   X {x} | Y {y} | Z {z} mm"


def _motor_to_ui_value(value: float) -> float:
    if not math.isfinite(value):
        return value
    return min(_UI_POSITION_MAX, max(_UI_POSITION_MIN, value))


def _motor_to_ui_position(position: Offset3DTuple) -> Offset3DTuple:
    return Offset3DTuple(*(
        _motor_to_ui_value(value)
        for value in position
    ))


class _EditablePositionSpinBox(QDoubleSpinBox):
    """A position editor whose first keyboard input replaces the current value."""

    def __init__(self):
        super().__init__()
        self._replace_on_next_key = False

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._replace_on_next_key = True

    def keyPressEvent(self, event):
        if self._replace_on_next_key and (
            event.text()
            or event.key() in {Qt.Key.Key_Backspace, Qt.Key.Key_Delete}
        ):
            self.selectAll()
        self._replace_on_next_key = False
        super().keyPressEvent(event)


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

        layout.addWidget(QLabel("Pellet:"))
        self._pellet_version = QLabel("(unknown version)")
        layout.addWidget(self._pellet_version)

        self._card_widget = CardWidget(title="Hardware Control", header_right_layout=layout)
        # self._card_widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)

        layout = QGridLayout()
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        label = QLabel("<b>Pellet Release Location (mm)</b>")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, 0, 0)

        label = QLabel("<b>Compound Move</b>")
        label.setAlignment(Qt.AlignCenter)
        label.setContentsMargins(0, 0, 0, 4)  # ensure small margin below
        layout.addWidget(label, 0, 4)

        def set_xyz(coord: str):
            value = getattr(self, f"_{coord}_pos").value()
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
            pos = _EditablePositionSpinBox()
            add_cmd_widget(pos)
            pos.setReadOnly(False)
            pos.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            pos.lineEdit().setReadOnly(False)
            pos.setValue(0)
            pos.setContentsMargins(0, 0, 0, 0)
            pos.setMinimumWidth(60)
            pos.setDecimals(1)
            pos.setSingleStep(0.5)
            pos.setRange(_UI_POSITION_MIN, _UI_POSITION_MAX)
            pos.setAlignment(Qt.AlignmentFlag.AlignRight)
            range_label = QLabel(
                f"[ {_UI_POSITION_MIN:>5.1f} : {_UI_POSITION_MAX:<5.1f}]"
            )
            set_button = QPushButton("Set")
            add_cmd_widget(set_button)
            set_button.clicked.connect(partial(set_xyz, coord))
            sub_layout.addWidget(QLabel(f"{coord.upper()} :"), row, col)
            sub_layout.addWidget(pos, row, col + 1)
            sub_layout.addWidget(range_label, row, col + 2)
            sub_layout.addWidget(set_button, row, col + 3)
            row += 1
            return pos, set_button

        self._x_pos, self._x_set_button = add_coord('x')
        self._y_pos, self._y_set_button = add_coord('y')
        self._z_pos, self._z_set_button = add_coord('z')

        self._motor_feedback_label = QLabel()
        self._motor_feedback_label.setObjectName("motorFeedbackLabel")
        self._motor_feedback_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sub_layout.addWidget(self._motor_feedback_label, row, col, 1, 4)
        row += 1

        self._motor_set_feedback_label = QLabel()
        self._motor_set_feedback_label.setObjectName("motorSetFeedbackLabel")
        self._motor_set_feedback_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sub_layout.addWidget(self._motor_set_feedback_label, row, col, 1, 4)

        layout.addLayout(sub_layout, 1, 2, alignment=Qt.AlignmentFlag.AlignTop)

        self._update_motor_feedback()

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

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        pass

    @invoke_method
    def set_selected_animal(self, animal: Optional[AnimalSubject]):
        cfg = self._app_model.behavior.algorithm.diamond_triangle_config
        if animal is None:
            xyz = Offset3DTuple(0, 0, 0)
        else:
            xyz = Offset3DTuple(animal.pellet_x, animal.pellet_y, animal.pellet_z)
            if animal.is_pellet_dcs:
                if cfg is not None and cfg.fully_valid:
                    xyz = cfg.diamond_to_motor(xyz)
                else:
                    logger.notice(
                        "Cannot display the animal's Diamond coordinates without "
                        "a valid calibration; displaying UI home"
                    )
                    xyz = Offset3DTuple(0, 0, 0)

        xyz = _motor_to_ui_position(xyz)

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

    def _update_motor_feedback(self):
        position = self._hardware_model.last_position
        if position is None:
            position = Offset3DTuple(math.nan, math.nan, math.nan)
        position = _motor_to_ui_position(position)
        set_position = _motor_to_ui_position(
            self._hardware_model.motor_send_coordinates
        )

        if not self._hardware_model.pellet_version:
            status = "DISCONNECTED"
            color = "#6b7280"
        elif self._hardware_model.pellet_status_timeout_engaged:
            status = "STALE"
            color = "#9a6700"
        elif all(math.isfinite(value) for value in position):
            status = "LIVE"
            color = "#1a7f37"
        else:
            status = "STALE"
            color = "#9a6700"

        self._motor_feedback_label.setText(_format_motor_feedback(status, position))
        self._motor_feedback_label.setStyleSheet(f"color: {color};")
        self._motor_set_feedback_label.setText(
            _format_motor_feedback("SET", set_position)
        )
        self._motor_set_feedback_label.setStyleSheet("color: #2563a8;")

    def _update_title(self, value: str):
        if value:
            if value.find("emulator") != -1:
                self._card_widget.header.setTitle("Hardware Control: Alogus Emulation")
            else:
                self._card_widget.header.setTitle(f"Hardware Control: {EnvironmentProvider.hardware_version()}")
        else:
            self._card_widget.header.setTitle("Hardware Control")

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
        if property_name == HardwareModel.PELLET_VERSION_PROPERTY:
            self._update_title(value)
            if value:
                self._pellet_version.setText(value.replace("emulator", "").strip())
                self.setEnabled(True)
            else:
                self._pellet_version.setText("(unknown version)")
                self.setEnabled(False)
                self.command_changed.emit("None")
            self._update_motor_feedback()

        elif property_name in {
            HardwareModel.POS_XYZ,
            HardwareModel.SEND_X,
            HardwareModel.SEND_Y,
            HardwareModel.SEND_Z,
            HardwareModel.DEVICE_PELLET_STATUS_TIMEOUT_ENGAGED,
        }:
            self._update_motor_feedback()

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
