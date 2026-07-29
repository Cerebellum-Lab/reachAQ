import inspect
import copy
import itertools
import logging
import math
import platform
from datetime import date

import verboselogs
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QFormLayout, QLineEdit, QComboBox, QLabel, QHBoxLayout, QPushButton,
                               QFileDialog, QTabWidget, QVBoxLayout, QCheckBox, QDoubleSpinBox, QSpinBox, QGridLayout,
                               QLayout, QSizePolicy)

from autotrainer.core.configuration.behavior_configuration import HeadClampConfiguration, PelletDeliveryConfiguration, \
    HeadClampReleaseMode
from autotrainer.core.logging import get_verbose_logger
from autotrainer.pyside import QSwitch

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.user_preferences import UserPreferences

logger = get_verbose_logger(__name__)


_DELAY_OR_DURATION_MAX_VALUE = 999_999  # in seconds, ~277 hours, ~= 11.5 days


def apply_size_policy(tab, klasses):
    # tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    for childs in itertools.chain(map(lambda c: tab.findChildren(c), klasses)):
        for child in childs:
            child.setSizePolicy(
                QSizePolicy.Policy.Fixed if isinstance(child, QSwitch) else QSizePolicy.Policy.MinimumExpanding,
                QSizePolicy.Policy.Fixed
            )


def set_row_col_visible(layout: QGridLayout, row: int, col: int, visible: bool):
    """
    Sets the visibility of all widgets in a specific row of a QGridLayout.
    """
    target_col = col
    for col in range(layout.columnCount()):
        if not (target_col <= col <= target_col + 1):
            continue
        # itemAtPosition returns a QLayoutItem
        item = layout.itemAtPosition(row, col)
        if item:
            widget = item.widget()
            if widget:
                widget.setVisible(visible)
            # If the item is a layout (e.g., QHBoxLayout nested inside the grid),
            # you would need to iterate through its children as well.
            # For this example, we assume widgets are added directly.


def refresh_enabled(callbacks):
    for cb in callbacks:
        cb()


class PreferencesContent(QWidget):

    def __init__(self, preferences: UserPreferences, app_model: AppModel):
        super(PreferencesContent, self).__init__(None)

        self._preferences = preferences
        self._app_model = app_model

        tabs = self._tabs = QTabWidget(self)

        self._general_tab = self._create_general_tab()
        tabs.addTab(self._general_tab, "General")

        self._behavior_tab = self._create_behavior_tab()
        tabs.addTab(self._behavior_tab, "Behavior")

        self._advanced_tab = self._create_advanced_tab()
        tabs.addTab(self._advanced_tab, "Advanced")

        layout = QVBoxLayout()
        layout.addWidget(self._tabs)

        self.setLayout(layout)

        self._update_tab_sizes()
        tabs.currentChanged.connect(self._update_tab_sizes)

    def _update_tab_sizes(self):
        tabs = self._tabs
        cur_idx = tabs.currentIndex()
        for i in range(tabs.count()):
            widget = tabs.widget(i)
            if i != cur_idx:
                widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            else:
                # Set desired size policy for the active tab
                widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # Ensure the layout updates
        cur_widget = tabs.currentWidget()
        cur_widget.updateGeometry()
        tabs.minimumSize = cur_widget.minimumSizeHint
        self.updateGeometry()
        self.update()

    def _create_general_tab(self):
        form_layout = QFormLayout(None)

        self._device_id_label = QLabel()
        plat_node_name = platform.node()
        self._device_id_label.setText(plat_node_name)

        self._device_id_edit = QLineEdit(None, None)
        self._device_id_edit.setText(self._preferences.serial_number)
        self._device_id_edit.textChanged.connect(self._device_id_changed)

        self._data_location_edit = QLineEdit(None, None)
        self._data_location_edit.setText(self._app_model.output_location)
        self._data_location_edit.textChanged.connect(self._data_location_changed)

        self._animal_location_edit = QLineEdit(None, None)
        self._animal_location_edit.setText(self._preferences.animal_location)
        self._animal_location_edit.textChanged.connect(self._animal_location_changed)

        form_layout.addRow("Device Id:", self._device_id_label)

        toggle = self._toggle_use_alternate_device_id = QSwitch()
        form_layout.addRow("Use alternate:", toggle)
        def on_use_alternate_device_id_toggled(value: int):
            toggled = value != 0
            idx = form_layout.getWidgetPosition(self._device_id_label)[0]
            form_layout.itemAt(idx).widget().setStyleSheet("" if toggled else "font-weight: bold;")
            for w in (self._device_id_edit, self._label_warning_device_id):
                idx = form_layout.getWidgetPosition(w)[0]
                form_layout.setRowVisible(idx, toggled)
            if not toggled:
                self._device_id_edit.setText(plat_node_name)

        form_layout.addRow("<b>Alternate Device Id:</b>", self._device_id_edit)
        self._label_warning_device_id = QLabel("<b>Some services may not operate as expected with an alternate Device Id</b>")
        form_layout.addRow("<b>Warning:</b>", self._label_warning_device_id)

        is_alternate_device_id = plat_node_name != self._preferences.serial_number
        toggle.setChecked(is_alternate_device_id)
        on_use_alternate_device_id_toggled(is_alternate_device_id)
        toggle.stateChanged.connect(on_use_alternate_device_id_toggled)

        layout = QHBoxLayout()
        layout.addWidget(self._data_location_edit)
        button = QPushButton("Select...")
        button.clicked.connect(lambda: self._browse_for_location("data"))
        layout.addWidget(button)

        form_layout.addRow("Data location:", layout)

        layout = QHBoxLayout()
        layout.addWidget(self._animal_location_edit)
        button = QPushButton("Select...")
        button.clicked.connect(lambda: self._browse_for_location("animal"))
        layout.addWidget(button)

        form_layout.addRow("Animal location:", layout)

        tab = QWidget(None)
        tab.setLayout(form_layout)
        apply_size_policy(tab, (QSwitch, QSpinBox, QDoubleSpinBox))

        return tab

    def _create_behavior_tab(self):
        app_model = self._app_model
        behavior = app_model.behavior
        analysis = behavior.analysis
        algo = behavior.algorithm
        tunnel_headfix_enabled = app_model.hardware.tunnel_headfix_enabled
        if not tunnel_headfix_enabled:
            algo.head_fixation_enabled = False
            algo.active_config.head_clamp.enabled = False
            analysis.auto_tunnel_sweep_monitor.config.enabled = False
            analysis.auto_tunnel_sweep_monitor.stop()

        states_refresh = []
        add_enabled_state = states_refresh.append
        def refresh_enabled_states():
            for r in states_refresh:
                r()

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        top_layout = QVBoxLayout()
        top_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        analysis_layout = QHBoxLayout()
        analysis_layout.addWidget(QLabel("Live inference:"))
        toggle = self._inference_enabled_toggle = QSwitch()
        toggle.setToolTip(
            "Enables real-time pose inference during live trials. Requires a CUDA-capable NVIDIA GPU, "
            "the proprietary NVIDIA driver, and a working TensorFlow GPU runtime."
        )
        toggle.setChecked(app_model.inference.is_enabled)
        def inference_enabled_state_changed(x: int):
            enabled = x != 0
            app_model.inference.is_enabled = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(inference_enabled_state_changed)  # after setChecked
        analysis_layout.addWidget(toggle)
        analysis_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        top_layout.addLayout(analysis_layout)
        #
        inference_model_layout = QHBoxLayout()
        inference_model_layout.addWidget(QLabel("Inference model:"))

        line_edit = self._inference_model_edit = QLineEdit(None, None)
        add_enabled_state(lambda: self._inference_model_edit.setEnabled(self._inference_enabled_toggle.isChecked()))
        line_edit.setText(self._app_model.inference.model_location)
        line_edit.textChanged.connect(self._inference_model_changed)
        inference_model_layout.addWidget(self._inference_model_edit)

        button = self._select_model_button = QPushButton("Select...")
        self._select_model_button.setEnabled(self._inference_enabled_toggle.isChecked())
        button.clicked.connect(lambda: self._browse_for_location("inference_model"))
        inference_model_layout.addWidget(button)
        inference_model_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        #
        top_layout.addLayout(inference_model_layout)
        main_layout.addLayout(top_layout)
        #
        cur_row = 0
        cur_col = 0
        left_grid_layout = QGridLayout()
        left_grid_layout.setContentsMargins(0, 6, 0, 0)
        left_grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        left_grid_layout.setSpacing(2)
        left_grid_layout.setHorizontalSpacing(10)
        tunnel_headfix_left_rows = []

        grids_hbox_layout = QHBoxLayout()
        grids_hbox_layout.setContentsMargins(0, 0, 0, 0)

        left_grid_layout.addWidget(QLabel("<b>Deliver Pellets:</b>"), cur_row, cur_col)
        toggle = self._deliver_pellet_toggle = QSwitch()
        add_enabled_state(lambda: self._deliver_pellet_toggle.setEnabled(self._inference_enabled_toggle.isChecked()))
        toggle.setToolTip(
            "Enables pellet load-send-release cycles based on pellet detection and related factors.")
        toggle.setChecked(algo.pellet_delivery_enabled)
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        def deliver_pellet_state_changed(x: int):
            enabled = x != 0
            algo.pellet_delivery_enabled = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(deliver_pellet_state_changed)
        cur_row += 1
        #
        left_grid_layout.addWidget(QLabel("Retract Enabled"), cur_row, cur_col)
        toggle = QSwitch()
        toggle.setChecked(algo.active_config.pellet_delivery.retract_enabled)
        add_enabled_state(
            lambda t=toggle: t.setEnabled(self._deliver_pellet_toggle.isEnabled() and self._deliver_pellet_toggle.isChecked()))
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        def retract_enabled_changed(x: int):
            enabled = x != 0
            algo.active_config.pellet_delivery.retract_enabled = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(retract_enabled_changed)
        cur_row += 1

        left_grid_layout.addWidget(QLabel("Pellet Send Wait Delay"), cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        spinbox.setToolTip("Delay before send-pellet after start-recording")
        spinbox.setRange(0, _DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setValue(algo.active_config.pellet_delivery.pellet_send_wait_delay)
        add_enabled_state(lambda s=spinbox:
            s.setEnabled(
                self._deliver_pellet_toggle.isEnabled()
                and self._deliver_pellet_toggle.isChecked()
                and algo.active_config.pellet_delivery.retract_enabled
                and (not algo.active_config.head_clamp.enabled
                     or not algo.active_config.head_clamp.wait_engaged_before_send_pellet)
            ))
        def on_pellet_send_delay_changed(value: float):
            algo.active_config.pellet_delivery.pellet_send_wait_delay = value
        spinbox.valueChanged.connect(on_pellet_send_delay_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        # pelletDelivery:maxPelletMissingSeconds
        left_grid_layout.addWidget(QLabel("Pellet missing seconds:"), cur_row, cur_col)
        spinbox = self._deliver_pellet_missing_seconds_spinbox = QDoubleSpinBox()
        spinbox.setToolTip("Delay pellet missing after which load pellet can be executed")
        add_enabled_state(lambda: self._deliver_pellet_missing_seconds_spinbox.setEnabled(
            self._deliver_pellet_toggle.isEnabled() and self._deliver_pellet_toggle.isChecked()
        ))
        spinbox.setValue(algo.pellet_missing_time)
        spinbox.setDecimals(2)
        spinbox.setSingleStep(0.05)
        def max_pellet_missing_seconds_changed(value):
            algo.pellet_missing_time = value
        spinbox.valueChanged.connect(max_pellet_missing_seconds_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        left_grid_layout.addWidget(QLabel("<b>Cover Pellets:</b>"), cur_row, cur_col)
        toggle = self._pellet_cover_toggle = QSwitch()
        toggle.setToolTip(
            "Covers the pellet when the mouse is not in the tunnel. "
            "Release then generates a tone when the tunnel is entered.")
        add_enabled_state(lambda: self._pellet_cover_toggle.setEnabled(
            self._deliver_pellet_toggle.isEnabled() and self._deliver_pellet_toggle.isChecked()
        ))
        toggle.setChecked(algo.pellet_cover_enabled)
        def pellet_cover_toggle_state_changed(x: int):
            enabled = x != 0
            algo.pellet_cover_enabled = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(pellet_cover_toggle_state_changed)
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1
        #
        left_grid_layout.addWidget(QLabel("Y DCS (mm) :"), cur_row, cur_col)
        spinbox = self._uncover_delay_spinbox = QDoubleSpinBox()
        spinbox.setToolTip("Min Y DCS for all hand parts")
        add_enabled_state(lambda s=spinbox, t=self._pellet_cover_toggle:
            s.setEnabled(t.isChecked())
        )
        spinbox.setValue(algo.pellet_uncover_y_dcs)
        spinbox.setMinimum(-30)
        spinbox.setMaximum(30)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(0.5)
        def pellet_uncover_y_dcs_changed(value):
            algo.pellet_uncover_y_dcs = value
        spinbox.valueChanged.connect(pellet_uncover_y_dcs_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        left_grid_layout.addWidget(QLabel("duration (sec.) :"), cur_row, cur_col)
        spinbox = self._uncover_delay_spinbox = QDoubleSpinBox()
        spinbox.setToolTip("Duration with min Y DCS valid before trigger uncover")
        add_enabled_state(lambda s=spinbox, t=self._pellet_cover_toggle:
                          s.setEnabled(t.isChecked())
                          )
        spinbox.setValue(algo.pellet_uncover_delay)
        spinbox.setMinimum(0)
        spinbox.setMaximum(5)
        spinbox.setDecimals(2)
        spinbox.setSingleStep(0.1)
        def pellet_uncover_delay_changed(value):
            algo.pellet_uncover_delay = value
        spinbox.valueChanged.connect(pellet_uncover_delay_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        shift_xyz_cfg = algo.active_config.shift_xyz_handler
        left_grid_layout.addWidget(QLabel("<b>Intertrial Pellet Shift:</b>"), cur_row, cur_col)
        toggle = self._intersession_pellet_shift_toggle = QSwitch()
        toggle.setToolTip("Enables adjustment of the pellet delivery position based on post trial reach analysis.")
        add_enabled_state(lambda t=toggle: t.setEnabled(self._inference_enabled_toggle.isChecked()))
        toggle.setChecked(algo.intersession_pellet_shift_enabled)
        def allow_intersession_shift_toggle_state_changed(x: int):
            enabled = x != 0
            algo.intersession_pellet_shift_enabled = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(allow_intersession_shift_toggle_state_changed)
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1

        left_grid_layout.addWidget(QLabel("Use Minimum Reach Fail"), cur_row, cur_col)
        toggle = self._use_minimum_reach_fail_toggle = QSwitch()
        toggle.setChecked(shift_xyz_cfg.use_reach_buffer)
        add_enabled_state(lambda t=toggle: t.setEnabled(
            self._intersession_pellet_shift_toggle.isChecked() and self._inference_enabled_toggle.isChecked()))
        def use_minimum_reach_fail_changed(x: int):
            enabled = x != 0
            algo.active_config.shift_xyz_handler.use_reach_buffer = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(use_minimum_reach_fail_changed)
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1

        left_grid_layout.addWidget(QLabel("Minimum Reach Fail"), cur_row, cur_col)
        spinbox = QSpinBox()
        add_enabled_state(
            lambda s=spinbox: s.setEnabled(
                self._use_minimum_reach_fail_toggle.isChecked()
                and self._use_minimum_reach_fail_toggle.isEnabled()
            ))
        spinbox.setValue(algo.active_config.shift_xyz_handler.buffer.minimum_reach_fail)
        spinbox.setRange(2, 99)
        def minimum_reach_fail_changed(value: int):
            algo.active_config.shift_xyz_handler.buffer.minimum_reach_fail = value
        spinbox.valueChanged.connect(minimum_reach_fail_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        left_grid_layout.addWidget(QLabel("Use Tongue Eaten"), cur_row, cur_col)
        toggle = QSwitch()
        toggle.setChecked(shift_xyz_cfg.use_tongue_eaten)
        add_enabled_state(
            lambda t=toggle: t.setEnabled(
                self._intersession_pellet_shift_toggle.isChecked()
                and self._inference_enabled_toggle.isChecked()
            ))
        def use_tongue_eaten_changed(x: int):
            enabled = x != 0
            algo.active_config.shift_xyz_handler.use_tongue_eaten = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(use_tongue_eaten_changed)
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1
        #
        
        left_grid_layout.addWidget(QLabel("<b>Home On Excessive Drift:</b>"), cur_row, cur_col)
        toggle = QSwitch()
        add_enabled_state(
            lambda t=toggle: t.setEnabled(self._inference_enabled_toggle.isChecked()))
        toggle.setChecked(algo.home_on_excessive_drift_distance_config.enabled)
        def home_on_excessive_toggle_changed(value: int):
            enabled = value != 0
            algo.home_on_excessive_drift_distance_config.enabled = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(home_on_excessive_toggle_changed)
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1

        left_grid_layout.addWidget(QLabel("Excessive distance threshold (mm) :"), cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox, t=toggle: s.setEnabled(t.isEnabled() and t.isChecked()))
        spinbox.setRange(0, 99)
        spinbox.setValue(algo.home_on_excessive_drift_distance_config.excessive_distance_threshold)
        def excessive_distance_threshold_changed(value):
            algo.home_on_excessive_drift_distance_config.excessive_distance_threshold = value
        spinbox.valueChanged.connect(excessive_distance_threshold_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        # grid_layout.addWidget(QLabel("<b>Auto-correct motors drift:</b>"), cur_row, cur_col)
        # toggle = self._auto_correct_motors_drift_toggle = QSwitch()
        # add_enabled_state(lambda: self._auto_correct_motors_drift_toggle.setEnabled(self._inference_enabled_toggle.isChecked()))
        # toggle.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # toggle.setChecked(self._app_model.behavior.algorithm.auto_correct_motors_drift)
        # def auto_correct_motors_drift_toggle_changed(value: int):
        #     enabled = value != 0
        #     logger.verbose("auto_correct_motors_drift_toggle_changed: %s", enabled)
        #     self._app_model.behavior.algorithm.auto_correct_motors_drift = enabled
        # toggle.stateChanged.connect(auto_correct_motors_drift_toggle_changed)
        # grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        # cur_row += 1
        #
        left_grid_layout.addWidget(QLabel("<b>Triangle-pellet distance too far detection:</b>"), cur_row, cur_col)
        toggle = QSwitch()
        add_enabled_state(lambda t=toggle: t.setEnabled(self._inference_enabled_toggle.isChecked()))
        toggle.setChecked(algo.use_triangle_pellet_distance_too_far)
        def use_triangle_pellet_distance_changed(value):
            enabled = value != 0
            algo.use_triangle_pellet_distance_too_far = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(use_triangle_pellet_distance_changed)
        left_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1
        #
        left_grid_layout.addWidget(QLabel("Maximum expected distance (mm):"), cur_row, cur_col)
        spinbox = self._triangle_pellet_expected_distance_spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox, t=toggle: s.setEnabled(t.isEnabled() and t.isChecked()))
        spinbox.setRange(0, 99)
        spinbox.setValue(algo.triangle_pellet_expected_distance)
        def triangle_pellet_expected_distance_changed(value):
            algo.triangle_pellet_expected_distance = value
        spinbox.valueChanged.connect(triangle_pellet_expected_distance_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        left_grid_layout.addWidget(QLabel("Triangle-Pellet diff too far threshold (mm):"), cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox, t=toggle: s.setEnabled(t.isEnabled() and t.isChecked()))
        spinbox.setRange(0, 20)
        spinbox.setValue(algo.triangle_pellet_diff_too_far_threshold)
        def triangle_pellet_diff_too_far_threshold_changed(value):
            algo.triangle_pellet_diff_too_far_threshold = value
        spinbox.valueChanged.connect(triangle_pellet_diff_too_far_threshold_changed)
        left_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        # right part:
        right_grid_layout = QGridLayout()
        right_grid_layout.setContentsMargins(2, 6, 0, 0)
        right_grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        right_grid_layout.setSpacing(2)
        right_grid_layout.setHorizontalSpacing(10)
        cur_row = 0
        cur_col = 0

        # headClamp: autoClampReleaseToneFreq
        label = QLabel("<b>Auto-Clamp:</b>")
        headclamp_cfg = algo.active_config.head_clamp
        right_grid_layout.addWidget(label, cur_row, cur_col)
        toggle = auto_clamp_enabled_toggle = QSwitch()
        toggle.setChecked(algo.head_fixation_enabled)
        # auto-clamp enabled:
        def toggle_changed(value):
            enabled = value != 0
            algo.head_fixation_enabled = enabled
            algo.active_config.head_clamp.enabled = enabled
            refresh_enabled_states()
        toggle.stateChanged.connect(toggle_changed)
        right_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1
        #
        right_grid_layout.addWidget(QLabel("Wait Engaged Before Send-Pellet:"), cur_row, cur_col)
        toggle = QSwitch()
        toggle.setChecked(algo.active_config.head_clamp.wait_engaged_before_send_pellet)
        add_enabled_state(lambda t=toggle:
            t.setEnabled(auto_clamp_enabled_toggle.isChecked()
                         and algo.active_config.pellet_delivery.retract_enabled))
        def update_wait_engaged_before_send_pellet(value):
            toggled = value != 0
            algo.active_config.head_clamp.wait_engaged_before_send_pellet = toggled
            refresh_enabled_states()
        toggle.stateChanged.connect(update_wait_engaged_before_send_pellet)
        right_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1
        #
        right_grid_layout.addWidget(QLabel("Threshold:"), cur_row, cur_col)
        spinbox = QDoubleSpinBox(None)
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setMinimum(0)
        spinbox.setMaximum(1023)
        spinbox.setWrapping(False)
        spinbox.setValue(analysis.headbar_pressure_monitor.engaged_threshold)
        def update_headbar_pressure_threshold(value):
            analysis.headbar_pressure_monitor.engaged_threshold = value
        spinbox.valueChanged.connect(update_headbar_pressure_threshold)
        spinbox.setToolTip("A value that adjusts the sensitivity of the headbar detector for it to be considered engaged.")
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        label = QLabel("PreRelease duration (sec.):")
        tooltip = "Set to 0 to disable/skip the pre-release intermediate step"
        label.setToolTip(tooltip)
        right_grid_layout.addWidget(label, cur_row, cur_col)
        spinbox = pre_release_dur_spinbox = QDoubleSpinBox(None)
        spinbox.setToolTip(tooltip)
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setRange(0, _DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(1)
        spinbox.setValue(algo.head_clamp_config.prerelease_duration)
        def auto_clamp_prerelease_duration_chanded(value):
            algo.head_clamp_config.prerelease_duration = value
            refresh_enabled_states()
        spinbox.valueChanged.connect(auto_clamp_prerelease_duration_chanded)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        right_grid_layout.addWidget(QLabel("PreRelease intensity (%):"), cur_row, cur_col)
        spinbox = QDoubleSpinBox(None)
        add_enabled_state(lambda s=spinbox, s2=pre_release_dur_spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked() and s2.value() > 0))
        spinbox.setRange(0, 100)
        spinbox.setDecimals(0)
        spinbox.setSingleStep(1)
        spinbox.setValue(algo.head_clamp_config.prerelease_intensity)
        def auto_clamp_prerelease_intensity_chanded(value):
            algo.head_clamp_config.prerelease_intensity = value
        spinbox.valueChanged.connect(auto_clamp_prerelease_intensity_chanded)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        right_grid_layout.addWidget(QLabel("Release tone freq (Hz) :"), cur_row, cur_col)
        spinbox = QSpinBox()
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setMinimum(0)
        spinbox.setMaximum(_DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setValue(algo.auto_clamp_release_tone_freq)
        def auto_clamp_release_tone_freq_changed(value):
            algo.auto_clamp_release_tone_freq = value
        spinbox.valueChanged.connect(auto_clamp_release_tone_freq_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        # headClamp:autoClampReleaseToneDelay
        right_grid_layout.addWidget(QLabel("Release tone delay (sec.) :"), cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setValue(algo.auto_clamp_release_tone_delay)
        def auto_clamp_release_tone_delay_changed(value):
            algo.auto_clamp_release_tone_delay = value
        spinbox.valueChanged.connect(auto_clamp_release_tone_delay_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        label = QLabel("Before re-engage delay (sec.):")
        tooltip = "Delay to wait before allow re-engage auto-clamp again"
        label.setToolTip(tooltip)
        right_grid_layout.addWidget(label, cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        spinbox.setToolTip(tooltip)
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setRange(0, _DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setDecimals(1)
        spinbox.setValue(algo.auto_clamp_before_reengage_delay)
        def auto_clamp_before_reengage_delay_changed(value):
            algo.auto_clamp_before_reengage_delay = value
        spinbox.valueChanged.connect(auto_clamp_before_reengage_delay_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        # release mode
        def show_headclamp_release_mode(mode):
            is_activity = mode == HeadClampReleaseMode.ACTIVITY
            for r in activity_rows:
                set_row_col_visible(right_grid_layout, *r, is_activity)
            is_fixed_duration = mode == HeadClampReleaseMode.FIXED_DURATION
            for r in fixed_duration_rows:
                set_row_col_visible(right_grid_layout, *r, is_fixed_duration)

        combo = QComboBox()
        add_enabled_state(lambda e=combo: e.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        for i in HeadClampReleaseMode:
            combo.addItem(i.value)
        combo.setCurrentText(headclamp_cfg.release_mode)
        def headclamp_release_mode_changed(idx, c=combo):
            mode = HeadClampReleaseMode(c.itemText(idx))  # this ensure we have known mode
            algo.active_config.head_clamp.release_mode = mode.value  # we use the value to store in config
            set_fixed_duration_value(algo.active_config.head_clamp.fixed_duration_release_delay)
            show_headclamp_release_mode(mode)
        combo.currentIndexChanged.connect(headclamp_release_mode_changed)
        right_grid_layout.addWidget(QLabel("Release Mode:"), cur_row, cur_col)
        right_grid_layout.addWidget(combo, cur_row, cur_col + 1)
        cur_row += 1

        release_mode_item_indent_px = 12
        # headClamp:autoClampNoActivityReleaseDelay
        label = QLabel("No-activity release delay (sec.) :")
        label.setContentsMargins(release_mode_item_indent_px, 0, 0, 0)
        right_grid_layout.addWidget(label, cur_row, cur_col)
        activity_rows = [(cur_row, cur_col)]
        spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setRange(0, _DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setValue(algo.auto_clamp_no_activity_release_delay)
        def auto_clamp_no_activity_release_delay_changed(value):
            algo.auto_clamp_no_activity_release_delay = value
        spinbox.valueChanged.connect(auto_clamp_no_activity_release_delay_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1

        # headClamp:autoClampReleaseLoadCount
        label = QLabel("Release load count:")
        label.setContentsMargins(release_mode_item_indent_px, 0, 0, 0)
        right_grid_layout.addWidget(label, cur_row, cur_col)
        activity_rows.append((cur_row, cur_col))
        spinbox = QSpinBox()
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setMinimum(0)
        spinbox.setMaximum(1_000_000)
        spinbox.setValue(algo.auto_clamp_release_load_count)
        def auto_clamp_release_load_count_changed(value):
            algo.auto_clamp_release_load_count = value
        spinbox.valueChanged.connect(auto_clamp_release_load_count_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        label = QLabel("Fixed duration:")
        label.setContentsMargins(release_mode_item_indent_px, 0, 0, 0)
        right_grid_layout.addWidget(label, cur_row, cur_col)
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        widget.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(hbox)
        spinbox = QDoubleSpinBox()
        combo = QComboBox()
        add_enabled_state(lambda c=combo: c.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        hbox.addWidget(spinbox, stretch=1)
        hbox.addWidget(combo)
        add_enabled_state(lambda s=spinbox: s.setEnabled(auto_clamp_enabled_toggle.isChecked()))
        spinbox.setMinimum(0)
        spinbox.setMaximum(60 * 60 * 60)  # 60 hours
        spinbox.setDecimals(3)
        def fixed_duration_value_changed(value, s=spinbox, c=combo):
            unit = c.currentText()
            if unit == "hours":
                value *= 3600
            elif unit == "minutes":
                value *= 60
            else:
                assert unit == "seconds"
            # set_fixed_duration_value(value, s, c)  auto-set to different unit eventually conveniently
            algo.active_config.head_clamp.fixed_duration_release_delay = value
        spinbox.valueChanged.connect(fixed_duration_value_changed)
        combo.addItems(["seconds", "minutes", "hours"])
        def set_fixed_duration_value(value: float, s=spinbox, c=combo):
            # arg value must be in seconds
            unit = "seconds"
            if value >= 60:
                value /= 60
                unit = "minutes"
                if value >= 60:
                    value /= 60
                    unit = "hours"
            s.blockSignals(True)
            s.setValue(value)
            s.blockSignals(False)
            c.blockSignals(True)
            c.setCurrentText(unit)
            c.blockSignals(False)

        set_fixed_duration_value(algo.active_config.head_clamp.fixed_duration_release_delay)

        def fixed_duration_unit_changed(unit, s=spinbox):
            value = algo.active_config.head_clamp.fixed_duration_release_delay
            if unit == "hours":
                value /= 3600
            elif unit == "minutes":
                value /= 60
            else:
                assert unit == "seconds"
            s.blockSignals(True)
            s.setValue(value)
            s.blockSignals(False)
        combo.currentTextChanged.connect(fixed_duration_unit_changed)
        fixed_duration_rows = [(cur_row, cur_col)]
        right_grid_layout.addWidget(widget, cur_row, cur_col + 1)
        cur_row += 1

        show_headclamp_release_mode(algo.active_config.head_clamp.release_mode)

        #
        right_grid_layout.addWidget(QLabel("<b>Tunnel Sweep:</b>"), cur_row, cur_col)
        toggle = self._tunnel_auto_sweep_toggle = QSwitch()
        toggle.setChecked(analysis.auto_tunnel_sweep_monitor.config.enabled)
        def toggled(x: int):
            enabled = x != 0
            analysis.auto_tunnel_sweep_monitor.config.enabled = enabled
            if enabled:
                analysis.auto_tunnel_sweep_monitor.restart()
            else:
                analysis.auto_tunnel_sweep_monitor.stop()
            refresh_enabled_states()
        toggle.stateChanged.connect(toggled)
        right_grid_layout.addWidget(toggle, cur_row, cur_col + 1)
        cur_row += 1
        #
        right_grid_layout.addWidget(QLabel("Pellet Misplaced Trigger Delay (sec.)"), cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox, t=toggle: s.setEnabled(t.isChecked()))
        spinbox.setRange(0, _DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setDecimals(1)
        spinbox.setValue(analysis.auto_tunnel_sweep_monitor.config.misplaced_trigger_delay)
        def value_changed(value):
            analysis.auto_tunnel_sweep_monitor.config.misplaced_trigger_delay = value
        spinbox.valueChanged.connect(value_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        right_grid_layout.addWidget(QLabel("Rate Limit Delay (sec.)"), cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox, t=toggle: s.setEnabled(t.isChecked()))
        spinbox.setRange(0, _DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setDecimals(0)
        spinbox.setValue(analysis.auto_tunnel_sweep_monitor.config.rate_limit_delay)
        def value_changed(value):
            analysis.auto_tunnel_sweep_monitor.config.rate_limit_delay = value
        spinbox.valueChanged.connect(value_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        right_grid_layout.addWidget(QLabel("Tunnel FAN ON duration (sec.)"), cur_row, cur_col)
        spinbox = QDoubleSpinBox()
        add_enabled_state(lambda s=spinbox, t=toggle: s.setEnabled(t.isChecked()))
        spinbox.setRange(0, _DELAY_OR_DURATION_MAX_VALUE)
        spinbox.setDecimals(1)
        spinbox.setValue(analysis.auto_tunnel_sweep_monitor.config.tunnel_fan_on_duration)
        def value_changed(value):
            analysis.auto_tunnel_sweep_monitor.config.tunnel_fan_on_duration = value
        spinbox.valueChanged.connect(value_changed)
        right_grid_layout.addWidget(spinbox, cur_row, cur_col + 1)
        cur_row += 1
        #
        # to enable/disable the inference dependant sub-widgets:
        refresh_enabled_states()

        #
        for row in tunnel_headfix_left_rows:
            set_row_col_visible(left_grid_layout, row, 0, tunnel_headfix_enabled)
            set_row_col_visible(left_grid_layout, row, 1, tunnel_headfix_enabled)

        right_widget = QWidget()
        right_widget.setLayout(right_grid_layout)
        right_widget.setVisible(tunnel_headfix_enabled)

        grids_hbox_layout.addLayout(left_grid_layout)
        grids_hbox_layout.addWidget(right_widget, stretch=1)
        main_layout.addLayout(grids_hbox_layout)
        #
        tab = QWidget()
        tab.setLayout(main_layout)
        apply_size_policy(tab, (QSwitch, QSpinBox, QDoubleSpinBox))

        return tab

    def _create_advanced_tab(self):
        combo_log_level = self._log_level_combobox = QComboBox(None)
        combo_log_level.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for display, lvl in (
                ("Success", verboselogs.SUCCESS),  # 0
                ("Warning", logging.WARNING),  # 1
                ("Notice", verboselogs.NOTICE),  # 2
                ("Info", logging.INFO),  # 3
                ("Verbose", verboselogs.VERBOSE),  # 4
                ("Debug", logging.DEBUG),  # 5
                ("Spam", verboselogs.SPAM),  # 6
        ):
            combo_log_level.addItem(display, lvl)

        levels_to_idx = {
            verboselogs.SUCCESS: 0,
            logging.WARNING: 1,
            verboselogs.NOTICE: 2,
            logging.INFO: 3,
            verboselogs.VERBOSE: 4,
            logging.DEBUG: 5,
            verboselogs.SPAM: 6,
        }

        log_level_idx = levels_to_idx.get(self._preferences.log_level)  # default to preferences.log_level
        if log_level_idx is None:
            log_level_idx = min(levels_to_idx.items(), key=lambda i: abs(self._preferences.log_level - i[0]))[1]
        combo_log_level.setCurrentIndex(log_level_idx)
        combo_log_level.currentIndexChanged.connect(self._log_level_changed)

        self._log_location_edit = QLineEdit(None, None)
        self._log_location_edit.setText(self._preferences.log_location)
        self._log_location_edit.textChanged.connect(self._log_location_changed)

        form_layout = QFormLayout(None)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        layout = QHBoxLayout()
        layout.addWidget(self._log_location_edit)
        button = QPushButton("Select...")
        button.clicked.connect(lambda: self._browse_for_location("log"))
        layout.addWidget(button)

        form_layout.addRow("Log location:", layout)

        form_layout.addRow("", QLabel("Log location change requires restart.  Leave blank for default location."))
        form_layout.addRow("Log level:", self._log_level_combobox)

        # May want these back with some future additions.
        # form_layout.addRow(QWidget())
        # form_layout.addRow(ATSeparator())
        # form_layout.addRow(QWidget())

        tab = QWidget(None)
        tab.setLayout(form_layout)

        return tab

    def _device_id_changed(self, value: str):
        self._preferences.serial_number = value

    def _data_location_changed(self, value: str):
        self._app_model.output_location = value

    def _animal_location_changed(self, value: str):
        self._preferences.animal_location = value

    def _inference_model_changed(self, value: str):
        self._app_model.inference.model_location = value

    def _log_level_changed(self, value):
        # logging.root.debug("_log_level_changed: %s", value)
        # print("%s" % (repr_all_loggers(),))
        if value != -1:
            new_level = self._log_level_combobox.itemData(value)
            self._preferences.log_level = new_level
            # get_console_handler().setLevel(new_level)

    def _log_location_changed(self, value: str):
        self._preferences.log_location = value

    def _browse_for_location(self, which: str):
        if which == "animal":
            initial_location = self._preferences.animal_location
        else:
            initial_location = self._preferences.log_location

        dirname = QFileDialog.getExistingDirectory(self, "Select Directory", initial_location)

        if len(dirname) > 0:
            if which == "animal":
                self._animal_location_edit.setText(dirname)
            elif which == "data":
                self._data_location_edit.setText(dirname)
            elif which == "inference_model":
                self._inference_model_edit.setText(dirname)
            else:
                self._log_location_edit.setText(dirname)
