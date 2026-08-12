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
                               QTabWidget, QVBoxLayout, QCheckBox, QDoubleSpinBox, QSpinBox, QGridLayout,
                               QLayout, QSizePolicy, QGroupBox, QFileDialog)

from autotrainer.behavior.pellet_trial import (
    AttemptAssignmentPolicy,
    RetrySettingsPolicy,
    TrialCountBasis,
    TrialOutcome,
)
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.configuration import SessionControlConfiguration
from autotrainer.pyside import QSwitch
from autotrainer.pyside.content_widget import invoke_method

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.user_preferences import UserPreferences
from tools.acquisition.view.softmouse_publication_controller import (
    SoftMousePublicationController,
)

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

        self._animal_metadata_tab = self._create_animal_metadata_tab()
        tabs.addTab(self._animal_metadata_tab, "Animal metadata")

        self._advanced_tab = self._create_advanced_tab()
        tabs.addTab(self._advanced_tab, "Advanced")

        layout = QVBoxLayout()
        layout.addWidget(self._tabs)

        self.setLayout(layout)

        self._update_tab_sizes()
        tabs.currentChanged.connect(self._update_tab_sizes)

        self._session_status_callback = self._on_session_status_changed
        self._app_model.property_changed += self._session_status_callback
        self.destroyed.connect(self._unsubscribe_session_status)
        self._set_session_mutable(
            self._app_model.session_recording_status
            is SessionRecordingStatus.READY
        )

    def _unsubscribe_session_status(self, *_args):
        try:
            self._app_model.property_changed -= self._session_status_callback
        except (KeyError, ValueError):
            pass

    def _on_session_status_changed(self, name, value, _old):
        if name == self._app_model.Props.SESSION_RECORDING_STATUS:
            self._set_session_mutable(value is SessionRecordingStatus.READY)

    @invoke_method
    def _set_session_mutable(self, enabled: bool) -> None:
        self._tabs.setEnabled(bool(enabled))

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
        algo = behavior.algorithm

        states_refresh = []
        add_enabled_state = states_refresh.append
        def refresh_enabled_states():
            for r in states_refresh:
                r()

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(self._create_session_control_group(algo))

        top_layout = QVBoxLayout()
        top_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        analysis_layout = QHBoxLayout()
        analysis_layout.addWidget(QLabel("Live inference:"))
        toggle = self._inference_enabled_toggle = QSwitch()
        toggle.setToolTip(
            "Enables real-time pose inference during live preview and recording sessions. Requires a CUDA-capable NVIDIA GPU, "
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
            "Cover pellets until the configured hand-position release condition is met.")
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
        toggle.setToolTip(
            "Allow live trial analysis to recommend or automatically apply "
            "pellet shifts to future pellet trials."
        )
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
        # Tunnel/head-fix controls were removed from reachAQ. Keep this page
        # focused on pellet automation, inference, and calibration behavior.
        refresh_enabled_states()
        grids_hbox_layout.addLayout(left_grid_layout)
        main_layout.addLayout(grids_hbox_layout)
        tab = QWidget()
        tab.setLayout(main_layout)
        apply_size_policy(tab, (QSwitch, QSpinBox, QDoubleSpinBox))
        return tab

    def _create_animal_metadata_tab(self):
        form = QFormLayout(None)
        form.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._softmouse_manifest_edit = QLineEdit()
        self._softmouse_manifest_edit.setText(
            self._preferences.softmouse_manifest_path
        )
        self._softmouse_manifest_edit.setReadOnly(True)
        form.addRow("Shared publication:", self._softmouse_manifest_edit)

        external = QLineEdit("Physical Tag")
        external.setReadOnly(True)
        form.addRow("Permanent external ID:", external)
        rfid = QLineEdit("Plate ID")
        rfid.setReadOnly(True)
        form.addRow("RFID field:", rfid)

        name_field = self._softmouse_name_field = QComboBox()
        name_field.setEditable(True)
        for header in (
            "Physical Tag",
            "Plate ID",
            "Cage Tag",
            "Comment",
        ):
            name_field.addItem(header)
        current = self._preferences.softmouse_name_column
        index = name_field.findText(current)
        if index < 0:
            name_field.addItem(current)
            index = name_field.findText(current)
        name_field.setCurrentIndex(index)
        name_field.currentTextChanged.connect(
            lambda value: setattr(self._preferences, "softmouse_name_column", value)
        )
        form.addRow("New-animal name field:", name_field)

        nightly = QCheckBox("Refresh this computer's local cache daily")
        nightly.setChecked(self._preferences.softmouse_nightly_refresh)
        nightly.toggled.connect(
            lambda value: setattr(
                self._preferences, "softmouse_nightly_refresh", value
            )
        )
        form.addRow("", nightly)
        self._softmouse_editable_controls = (name_field, nightly)

        buttons = QHBoxLayout()
        self._softmouse_refresh_button = QPushButton("Refresh local cache")
        self._softmouse_refresh_button.clicked.connect(self._refresh_animal_metadata)
        buttons.addWidget(self._softmouse_refresh_button)
        form.addRow("", buttons)

        self._softmouse_publication_controller = (
            SoftMousePublicationController.for_model(
                self._app_model,
                QtCore.QCoreApplication.instance(),
            )
        )
        self._softmouse_publish_button = QPushButton("Sync from SoftMouse")
        self._softmouse_publish_button.setToolTip(
            "Download the current Christie animal list, safely replace the shared "
            "Isilon publication, and refresh this computer's local cache."
        )
        self._softmouse_publish_button.clicked.connect(
            self._softmouse_publication_controller.start
        )
        form.addRow("SoftMouse:", self._softmouse_publish_button)
        self._softmouse_publication_status = QLabel(
            self._softmouse_publication_controller.status
        )
        self._softmouse_publication_status.setWordWrap(True)
        form.addRow("Publication:", self._softmouse_publication_status)

        self._softmouse_status = QLabel(self._app_model.animal_metadata_status)
        self._softmouse_status.setWordWrap(True)
        form.addRow("Status:", self._softmouse_status)
        self._softmouse_counts = QLabel("No import preview available")
        self._softmouse_counts.setWordWrap(True)
        form.addRow("Rows:", self._softmouse_counts)
        self._softmouse_mapped_columns = QLabel("—")
        self._softmouse_mapped_columns.setWordWrap(True)
        form.addRow("Mapped columns:", self._softmouse_mapped_columns)
        self._softmouse_ignored_columns = QLabel("—")
        self._softmouse_ignored_columns.setWordWrap(True)
        form.addRow("Other columns:", self._softmouse_ignored_columns)
        note = QLabel(
            "Imports are complete/unfiltered snapshots. Rows without RFID and "
            "animals whose State is Ended are intentionally excluded."
        )
        note.setWordWrap(True)
        form.addRow("", note)

        self._animal_metadata_model_callback = (
            self._on_animal_metadata_model_changed
        )
        self._app_model.property_changed += self._animal_metadata_model_callback
        self._softmouse_publication_controller.status_changed.connect(
            self._softmouse_publication_status.setText
        )
        self._softmouse_publication_controller.running_changed.connect(
            self._update_softmouse_button_states
        )
        self._update_softmouse_button_states()
        tab = QWidget(None)
        tab.setLayout(form)
        def unsubscribe(*_args):
            try:
                self._app_model.property_changed -= (
                    self._animal_metadata_model_callback
                )
            except (KeyError, ValueError):
                pass
        tab.destroyed.connect(unsubscribe)
        if self._app_model.animal_metadata_preview is not None:
            self._show_animal_metadata_preview(
                self._app_model.animal_metadata_preview
            )
        return tab

    @invoke_method
    def _on_animal_metadata_model_changed(self, name, value, _old):
        if name == self._app_model.Props.ANIMAL_METADATA_STATUS:
            self._softmouse_status.setText(value)
        elif name == self._app_model.Props.ANIMAL_METADATA_PREVIEW:
            self._show_animal_metadata_preview(value)
        elif name in {
            self._app_model.Props.SESSION_RECORDING_STATUS,
            self._app_model.Props.ANIMAL_METADATA_REFRESH_BUSY,
        }:
            self._update_softmouse_button_states()

    def _show_animal_metadata_preview(self, preview):
        batch = preview.batch
        self._softmouse_counts.setText(
            f"{batch.total_source_rows} total · {batch.accepted_rows} tagged · "
            f"{batch.ignored_missing_rfid_rows} without RFID · "
            f"{batch.ignored_ended_rows} ended"
        )
        self._softmouse_mapped_columns.setText(
            ", ".join(preview.mapped_columns) or "—"
        )
        self._softmouse_ignored_columns.setText(
            ", ".join(preview.ignored_columns) or "—"
        )
        for header in preview.source_headers:
            if header and self._softmouse_name_field.findText(header) < 0:
                self._softmouse_name_field.addItem(header)

    def _update_softmouse_button_states(self, _running=None):
        ready = (
            self._app_model.session_recording_status
            is SessionRecordingStatus.READY
        )
        publishing = self._softmouse_publication_controller.is_running
        refreshing = self._app_model.animal_metadata_refresh_busy
        enabled = ready and not publishing and not refreshing
        self._softmouse_refresh_button.setEnabled(enabled)
        self._softmouse_publish_button.setEnabled(enabled)
        for control in self._softmouse_editable_controls:
            control.setEnabled(enabled)

    def _apply_animal_metadata_preferences(self):
        try:
            self._app_model.apply_animal_metadata_preferences()
        except Exception as exc:
            self._softmouse_status.setText(f"Configuration failed: {exc}")

    def _refresh_animal_metadata(self):
        try:
            self._app_model.apply_animal_metadata_preferences()
            self._app_model.request_animal_metadata_refresh("manual button")
        except Exception as exc:
            self._softmouse_status.setText(f"Refresh failed: {exc}")

    def _create_session_control_group(self, algo):
        """Build the operator-facing controls for continuous recording sessions."""
        config = algo.active_config.session_control
        group = QGroupBox("Recording session")
        form = QFormLayout(group)

        automatic_cycles = QSwitch()
        automatic_cycles.setChecked(config.automatic_pellet_cycles_enabled)
        automatic_cycles.setToolTip(
            "Automatically repeat pellet delivery trials while recording."
        )
        automatic_cycles.stateChanged.connect(
            lambda value: self._app_model.update_session_control_option(
                "automatic_pellet_cycles_enabled",
                value != 0,
            )
        )
        form.addRow("Automatic pellet cycles:", automatic_cycles)

        automatic_protocol = QSwitch()
        automatic_protocol.setChecked(
            config.automatic_protocol_advance_enabled
        )
        automatic_protocol.setToolTip(
            "Automatically advance or fall back when the selected protocol's "
            "trial criteria are met."
        )
        automatic_protocol.stateChanged.connect(
            lambda value: self._app_model.set_automatic_protocol_advance_enabled(
                value != 0
            )
        )
        form.addRow("Automatic protocol advance:", automatic_protocol)

        intertrial_analysis = QSwitch()
        intertrial_analysis.setChecked(config.intertrial_analysis_enabled)
        intertrial_analysis.setToolTip(
            "Analyze each completed pellet trial from buffered live tracking. "
            "This does not reread camera frames or run the pose model again."
        )
        intertrial_analysis.stateChanged.connect(
            lambda value: self._app_model.set_intertrial_analysis_enabled(
                value != 0
            )
        )
        form.addRow("Live intertrial analysis:", intertrial_analysis)

        analysis_progression = QComboBox()
        analysis_progression.addItem("Continue while analyzing", "continue")
        analysis_progression.addItem("Wait for each analysis", "wait")
        analysis_progression.setCurrentIndex(
            analysis_progression.findData(config.intertrial_progression_mode)
        )
        analysis_progression.currentIndexChanged.connect(
            lambda _index: self._app_model.update_session_control_option(
                "intertrial_progression_mode",
                analysis_progression.currentData(),
            )
        )
        intertrial_analysis.toggled.connect(analysis_progression.setEnabled)
        analysis_progression.setEnabled(intertrial_analysis.isChecked())
        form.addRow("After each pellet trial:", analysis_progression)

        retry_outcomes = QWidget()
        retry_outcomes_layout = QHBoxLayout(retry_outcomes)
        retry_outcomes_layout.setContentsMargins(0, 0, 0, 0)
        retry_outcomes_layout.setSpacing(8)

        def set_retry_outcome(outcome: TrialOutcome, enabled: bool) -> None:
            selected = set(config.behavioral_retry_outcomes)
            if enabled:
                selected.add(outcome.value)
            else:
                selected.discard(outcome.value)
            self._app_model.update_session_control_option(
                "behavioral_retry_outcomes",
                tuple(
                candidate.value
                for candidate in TrialOutcome
                if candidate.value in selected
                ),
            )

        analysis_retry_checkboxes = []
        for outcome, display_name in (
            (TrialOutcome.NO_REACH, "No reach"),
            (TrialOutcome.FAILURE, "Failed reach"),
            (TrialOutcome.PELLET_MISSING, "Pellet missing"),
        ):
            checkbox = QCheckBox(display_name)
            checkbox.setChecked(outcome.value in config.behavioral_retry_outcomes)
            checkbox.toggled.connect(
                lambda checked, item=outcome: set_retry_outcome(item, checked)
            )
            if outcome is not TrialOutcome.PELLET_MISSING:
                intertrial_analysis.toggled.connect(checkbox.setEnabled)
                checkbox.setEnabled(intertrial_analysis.isChecked())
                analysis_retry_checkboxes.append(checkbox)
            retry_outcomes_layout.addWidget(checkbox)
        retry_outcomes_layout.addStretch(1)
        form.addRow("Retry after outcome:", retry_outcomes)

        def analysis_enabled_changed(enabled: bool) -> None:
            if not enabled:
                for checkbox in analysis_retry_checkboxes:
                    checkbox.setChecked(False)

        intertrial_analysis.toggled.connect(analysis_enabled_changed)

        attempt_policy = QComboBox()
        for policy in AttemptAssignmentPolicy:
            attempt_policy.addItem(policy.display_name, policy.value)
        attempt_policy.setCurrentIndex(
            attempt_policy.findData(config.attempt_assignment)
        )
        attempt_policy.currentIndexChanged.connect(
            lambda _index: self._app_model.update_session_control_option(
                "attempt_assignment",
                attempt_policy.currentData(),
            )
        )
        form.addRow("Failed-attempt handling:", attempt_policy)

        retry_settings = QComboBox()
        for policy in RetrySettingsPolicy:
            retry_settings.addItem(policy.display_name, policy.value)
        retry_settings.setCurrentIndex(
            retry_settings.findData(config.retry_settings)
        )
        retry_settings.currentIndexChanged.connect(
            lambda _index: self._app_model.update_session_control_option(
                "retry_settings",
                retry_settings.currentData(),
            )
        )
        form.addRow("Retry settings:", retry_settings)

        count_basis = QComboBox()
        for basis in TrialCountBasis:
            count_basis.addItem(basis.display_name, basis.value)
        scored_index = count_basis.findData(TrialCountBasis.SCORED.value)

        def set_scored_basis_enabled(enabled: bool) -> None:
            item = count_basis.model().item(scored_index)
            item.setEnabled(enabled)
            count_basis.setItemData(
                scored_index,
                (
                    "Uses finalized live intertrial results."
                    if enabled
                    else "Enable live intertrial analysis to stop by scored trials."
                ),
                Qt.ToolTipRole,
            )

        set_scored_basis_enabled(intertrial_analysis.isChecked())
        intertrial_analysis.toggled.connect(set_scored_basis_enabled)
        count_basis.setCurrentIndex(count_basis.findData(config.trial_count_basis))
        count_basis.currentIndexChanged.connect(
            lambda _index: self._app_model.update_session_control_option(
                "trial_count_basis",
                count_basis.currentData(),
            )
        )
        form.addRow("Trial-limit count:", count_basis)

        counted_outcomes = QWidget()
        counted_outcomes_layout = QHBoxLayout(counted_outcomes)
        counted_outcomes_layout.setContentsMargins(0, 0, 0, 0)
        counted_outcomes_layout.setSpacing(8)

        def set_counted_outcome(outcome: TrialOutcome, enabled: bool) -> None:
            selected = set(config.counted_trial_outcomes)
            if enabled:
                selected.add(outcome.value)
            else:
                selected.discard(outcome.value)
            self._app_model.update_session_control_option(
                "counted_trial_outcomes",
                tuple(
                candidate.value
                for candidate in TrialOutcome
                if candidate.value in selected
                and candidate is not TrialOutcome.HARDWARE_ERROR
                and candidate is not TrialOutcome.PENDING_ANALYSIS
                ),
            )

        for outcome, display_name in (
            (TrialOutcome.SUCCESS, "Success"),
            (TrialOutcome.FAILURE, "Failed reach"),
            (TrialOutcome.PELLET_MISSING, "Pellet missing"),
            (TrialOutcome.NO_REACH, "No reach"),
            (TrialOutcome.INCOMPLETE, "Incomplete trial"),
            (TrialOutcome.ABORTED, "Aborted trial"),
        ):
            checkbox = QCheckBox(display_name)
            checkbox.setChecked(outcome.value in config.counted_trial_outcomes)
            checkbox.toggled.connect(
                lambda checked, item=outcome: set_counted_outcome(
                    item,
                    checked,
                )
            )
            counted_outcomes_layout.addWidget(checkbox)
        counted_outcomes_layout.addStretch(1)
        form.addRow("Counted outcomes:", counted_outcomes)

        duration_limit = QDoubleSpinBox()
        duration_limit.setRange(
            0, SessionControlConfiguration.MAX_DURATION_SECONDS,
        )
        duration_limit.setDecimals(1)
        duration_limit.setSingleStep(10)
        duration_limit.setSpecialValueText("No limit")
        duration_limit.setSuffix(" s")
        duration_limit.setValue(config.duration_limit_seconds or 0)
        duration_limit.valueChanged.connect(
            lambda value: self._app_model.update_session_control_option(
                "duration_limit_seconds",
                value if value > 0 else None,
            )
        )
        form.addRow("Stop after duration:", duration_limit)

        trial_limit = QSpinBox()
        trial_limit.setRange(0, 999_999)
        trial_limit.setSpecialValueText("No limit")
        trial_limit.setValue(config.trial_limit or 0)
        trial_limit.valueChanged.connect(
            lambda value: self._app_model.update_session_control_option(
                "trial_limit",
                value if value > 0 else None,
            )
        )
        form.addRow("Stop after trial count:", trial_limit)

        stop_on_protocol = QSwitch()
        stop_on_protocol.setChecked(config.stop_on_protocol_complete)
        stop_on_protocol.stateChanged.connect(
            lambda value: self._app_model.update_session_control_option(
                "stop_on_protocol_complete",
                value != 0,
            )
        )
        form.addRow("Stop when protocol finishes:", stop_on_protocol)

        drain_timeout = QDoubleSpinBox()
        drain_timeout.setRange(1, 300)
        drain_timeout.setDecimals(1)
        drain_timeout.setSuffix(" s")
        drain_timeout.setToolTip(
            "Maximum time to finish the active pellet trial after an automatic "
            "stop condition. Only expiry of this timer is an error."
        )
        drain_timeout.setValue(config.stop_drain_timeout_seconds)
        drain_timeout.valueChanged.connect(
            lambda value: self._app_model.update_session_control_option(
                "stop_drain_timeout_seconds",
                value,
            )
        )
        form.addRow("Finish-current-trial timeout:", drain_timeout)

        apply_size_policy(
            group,
            (QSwitch, QCheckBox, QSpinBox, QDoubleSpinBox),
        )
        return group

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
        elif which == "data":
            initial_location = self._app_model.output_location
        elif which == "inference_model":
            initial_location = self._app_model.inference.model_location
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
