import math

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QLabel, QWidget, QVBoxLayout,
                               QHBoxLayout, QStackedLayout, QGridLayout, QPushButton, QSizePolicy)

from autotrainer.inference.analysis import IntersessionResponse
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoProps
from autotrainer.behavior.pellet_shift import ShiftXYZHandler
from autotrainer.pyside import CardWidget, QSwitch
from autotrainer.pyside.StackedContent import StackedLayout
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from autotrainer.pyside.xyz_label import XYZQLabel
from autotrainer.pyside.DayTotalCount import DailyAndTotalCountsLabel

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.inference_model import InferenceModel
from tools.acquisition.model.behavior_model import BehaviorModel
from tools.acquisition.model.hardware_model import HardwareModel


class BehaviorContent(ContentWidget):

    status_changed = Signal(str, name="status_changed")

    def __init__(self,
                 app_model: AppModel,
                 behavior_model: BehaviorModel,
                 inference_model: InferenceModel):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        system_machine = behavior_model.system_machine
        algo = system_machine.algorithm
        pellet_machine = system_machine.pellet
        intersession_machine = system_machine.intersession

        self._app_model = app_model
        self._behavior_model = behavior_model
        self._inference_model = inference_model
        self._analysis = behavior_model.analysis

        self._inference_status = QLabel("")
        card = self._card_widget = CardWidget(title="Behavior", header_right_layout=self._inference_status)

        hbox_main_layout = QHBoxLayout()
        hbox_main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        hbox_main_layout.setContentsMargins(4, 4, 4, 4)
        hbox_main_layout.setSpacing(16)

        left_main_layout = QVBoxLayout()
        left_main_layout.setContentsMargins(0, 0, 0, 0)
        left_main_layout.setSpacing(0)

        left_layout = self._left_layout = QGridLayout()
        left_main_layout.addLayout(left_layout)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setHorizontalSpacing(8)
        left_layout.setVerticalSpacing(4)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        right_layout = self._right_layout = QGridLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setHorizontalSpacing(8)
        right_layout.setVerticalSpacing(4)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        left_cur_row = 0

        label = QLabel("States")
        label.setStyleSheet("font-weight: bold;")
        label.setContentsMargins(0, 0, 0, 4)
        left_layout.addWidget(label, left_cur_row, 0)
        left_cur_row += 1

        # allows to not have the behavior content constantly resize on width when any of the states below changes
        left_layout.setColumnMinimumWidth(1, 90)
        # left_layout.setColumnStretch(1, 1)

        left_layout.addWidget(QLabel("System:"), left_cur_row, 0)
        label = self._system_machine_state_label = QLabel(self._behavior_model.system_machine.state)
        left_layout.addWidget(label, left_cur_row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        left_cur_row += 1

        left_layout.addWidget(QLabel("Pellet:"), left_cur_row, 0)
        label = self._pellet_machine_state_label = QLabel(self._behavior_model.system_machine.pellet.state)
        left_layout.addWidget(label, left_cur_row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        left_cur_row += 1

        left_layout.addWidget(QLabel("Intertrial:"), left_cur_row, 0)
        label = self._intersession_state_label = QLabel(self._behavior_model.system_machine.intersession.state)
        label.setContentsMargins(0, 0, 0, 4)
        left_layout.addWidget(label, left_cur_row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        left_cur_row += 1

        left_layout.addWidget(QLabel("Intertrial Analysis:"), left_cur_row, 0)
        toggle = self._intersession_toggle = QSwitch()
        toggle.stateChanged.connect(self._intersession_toggle_state_changed)
        toggle.setToolTip(
            "Enables reach detection and segmentation after each trial where the mouse is seen.  This may modify "
            "pellet counts and adjust the pellet delivery position.")
        left_layout.addWidget(toggle, left_cur_row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        left_cur_row += 1

        label = self._head_fixation_label = QLabel("Auto-Clamp:")
        left_layout.addWidget(label, left_cur_row, 0)
        toggle = self._head_fixation_toggle = QSwitch()
        toggle.stateChanged.connect(self._head_fixation_toggle_state_changed)
        toggle.setToolTip("Enables automatic magnet adjustment to 100% when the headbar detector is triggered.")
        left_layout.addWidget(toggle, left_cur_row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        left_cur_row += 1

        label = self._head_magnet_baseline_title_label = QLabel("Head Magnet Baseline:")
        left_layout.addWidget(label, left_cur_row, 0)
        label = self._head_magnet_baseline_label = QLabel(f"{self._behavior_model.algorithm.baseline_intensity:.1f}%")
        label.setContentsMargins(0, 4, 0, 0)
        left_layout.addWidget(self._head_magnet_baseline_label, left_cur_row, 1)
        left_cur_row += 1
        button = self._make_baseline_button = QPushButton("Make Current Position Baseline")
        button.setContentsMargins(0, 0, 0, 0)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        button.clicked.connect(self._make_position_baseline)
        left_main_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        #

        right_cur_row = 0
        label = QLabel("<b>Pellet Counts</b>")
        right_layout.addWidget(label, right_cur_row, 0)
        label = QLabel("<b>day / total</b>")
        right_layout.addWidget(label, right_cur_row, 1)
        right_cur_row += 1

        right_layout.addWidget(QLabel("Presented:"), right_cur_row, 0)
        self._pellets_presented_label = DailyAndTotalCountsLabel(day=algo.pellets_presented_day, total=algo.pellets_presented_day)
        right_layout.addWidget(self._pellets_presented_label, right_cur_row, 1)
        right_cur_row += 1

        right_layout.addWidget(QLabel("Reaches:"), right_cur_row, 0)
        label = self._pellet_reaches_label = DailyAndTotalCountsLabel(day=algo.pellet_reaches_day,
                                                                      total=algo.pellet_reaches_total)
        right_layout.addWidget(label, right_cur_row, 1)
        right_cur_row += 1

        right_layout.addWidget(QLabel("Success:"), right_cur_row, 0)
        label = self._successful_reaches_label = DailyAndTotalCountsLabel(day=algo.successful_reaches_day,
                                                                          total=algo.successful_reaches_total)
        right_layout.addWidget(label, right_cur_row, 1)
        right_cur_row += 1

        right_layout.addWidget(QLabel("Consumed:"), right_cur_row, 0)
        self._pellets_consumed_label = DailyAndTotalCountsLabel(day=algo.pellet_consumed_day, total=algo.pellet_consumed_total)
        right_layout.addWidget(self._pellets_consumed_label, right_cur_row, 1)
        right_cur_row += 1

        label = QLabel("<b>Pellet Shift XYZ</b>")
        label.setContentsMargins(0, 8, 0, 4)
        right_layout.addWidget(label, right_cur_row, 0)
        label = QLabel("<b>mm</b>")
        label.setContentsMargins(0, 8, 0, 4)
        right_layout.addWidget(label, right_cur_row, 1)
        right_cur_row += 1

        right_layout.addWidget(QLabel("Prev. trial:"), right_cur_row, 0)
        label = self._prev_pellet_shift_label = XYZQLabel(n_digits=2)
        right_layout.addWidget(label, right_cur_row, 1)
        right_cur_row += 1

        right_layout.addWidget(QLabel("Prev. processed:"), right_cur_row, 0)
        label = self._prev_processed_pellet_shift_label = XYZQLabel()
        right_layout.addWidget(label, right_cur_row, 1)

        for r_idx in range(right_layout.rowCount()):
            i = right_layout.itemAtPosition(r_idx, 1)
            if i:
                w = i.widget()
                if isinstance(w, (QLabel, )):
                    w.setAlignment(Qt.AlignmentFlag.AlignRight)

        #

        hbox_main_layout.addLayout(left_main_layout)
        hbox_main_layout.addLayout(right_layout)

        #

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        content.setContentsMargins(4, 4, 4, 4)
        content.setLayout(hbox_main_layout)

        card.setContentWidget(content)

        # Footer
        self._basic_footer = QWidget()
        self._basic_footer.setContentsMargins(0, 0, 0, 0)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(QLabel("Inference model:"))
        self._model_location_label = QLabel("")
        layout.addWidget(self._model_location_label)

        self._basic_footer.setLayout(layout)

        self._stack_layout = StackedLayout()
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

        self._inference_status.setText(f"Inference: {inference_model.status}")
        self._intersession_toggle.setChecked(behavior_model.algorithm.intersession_enabled)

        self._inference_model_property_changed("model_location", inference_model.model_location, None)
        #

        inference_model.property_changed += self._inference_model_property_changed
        app_model.hardware.property_changed += self._hardware_model_property_changed
        system_machine.shift_xyz_handler.property_changed += self._shift_xyz_property_changed

        system_machine.events.state_changed += lambda old, new: self._system_machine_state_label.setText(new)
        pellet_machine.events.state_changed += lambda old, new: self._pellet_machine_state_label.setText(new)
        intersession_machine.events.state_changed += lambda old, new: self._intersession_state_label.setText(new)

        algo.property_changed += self._algorithm_property_changed
        self.status_changed.connect(self._inference_status.setText)
        self.set_is_editable(False)
        self._update_tunnel_headfix_visibility(app_model.hardware.tunnel_headfix_enabled)

    def set_is_editable(self, is_editable: bool):
        self._stack_layout.setCurrentIndex(1 if is_editable else 0)

    def set_is_capture_active(self, is_active: bool):
        pass

    def _intersession_toggle_state_changed(self, x: int):
        self._behavior_model.algorithm.intersession_enabled = x != 0

    def _head_fixation_toggle_state_changed(self, x: int):
        self._behavior_model.algorithm.head_fixation_enabled = x != 0

    def _make_position_baseline(self):
        self._behavior_model.use_current_head_magnet_position_as_baseline()
        self._app_model.save_configuration()

    def _update_tunnel_headfix_visibility(self, is_enabled: bool):
        if not is_enabled and self._head_fixation_toggle.isChecked():
            self._behavior_model.algorithm.head_fixation_enabled = False
        for widget in (
            self._head_fixation_label,
            self._head_fixation_toggle,
            self._head_magnet_baseline_title_label,
            self._head_magnet_baseline_label,
            self._make_baseline_button,
        ):
            widget.setVisible(is_enabled)

    @invoke_method
    def _hardware_model_property_changed(self, name, value, _):
        if name == HardwareModel.TUNNEL_HEADFIX_ENABLED:
            self._update_tunnel_headfix_visibility(value)

    @invoke_method
    def _algorithm_property_changed(self, name, value, _):
        props = BehaviorAlgoProps
        if name == props.INTERSESSION_ENABLED:
            self._intersession_toggle.setChecked(value)

        elif name == props.BASELINE_INTENSITY:
            self._head_magnet_baseline_label.setText(f"{value:.1f}%")

        elif name == props.HEAD_FIXATION_ENABLED:
            self._head_fixation_toggle.setChecked(value)

        elif name == props.DAY_PELLET_COUNT:
            self._pellets_consumed_label.update_values(day=value)

        elif name == props.TOTAL_PELLET_COUNT:
            self._pellets_consumed_label.update_values(total=value)

        elif name == props.DAY_PELLET_PRESENTED:
            self._pellets_presented_label.update_values(day=value)

        elif name == props.TOTAL_PELLET_PRESENTED:
            self._pellets_presented_label.update_values(total=value)

        elif name == props.DAY_PELLET_REACHES:
            self._pellet_reaches_label.update_values(day=value)

        elif name == props.TOTAL_PELLET_REACHES:
            self._pellet_reaches_label.update_values(total=value)

        elif name == props.DAY_SUCCESSFUL_REACHES:
            self._successful_reaches_label.update_values(day=value)

        elif name == props.TOTAL_SUCCESSFUL_REACHES:
            self._successful_reaches_label.update_values(total=value)

    @invoke_method
    def _inference_model_property_changed(self, name, value, _):
        props = self._inference_model
        if name == props.IS_ENABLED:
            self._intersession_toggle.setEnabled(value)
        elif name == props.STATUS:
            self.status_changed.emit(f"Inference: {value}")
        elif name == props.MODEL_LOCATION:
            if value is not None and len(value) > 0:
                self._model_location_label.setText(value)
            else:
                self._model_location_label.setText("Inference model not specified")

    @invoke_method
    def _shift_xyz_property_changed(self, name, value, _):
        if name == ShiftXYZHandler.LAST_SHIFT_XYZ:
            self._prev_pellet_shift_label.update_coordinate(value)
        elif name == ShiftXYZHandler.LAST_PROCESSED_SHIFT_XYZ:
            self._prev_processed_pellet_shift_label.update_coordinate(value)
