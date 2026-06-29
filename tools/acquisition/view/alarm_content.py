import dataclasses
from functools import partial
from typing import Dict, Callable

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QLabel, QVBoxLayout, QFormLayout, QSizePolicy, QScrollArea, QWidget, QHBoxLayout, QLayout

from autotrainer.behavior.behavior_algorithm import BehaviorAlgoProps
from autotrainer.core import LoadCellMonitor, get_verbose_logger
from autotrainer.core.analysis import EmergencyAlarmMonitor
from autotrainer.core.analysis.alarm_detector import AlarmDetector
from autotrainer.core.analysis.audio_spectrum_monitor import AudioSpectrumThrashMonitor
from autotrainer.core.analysis.global_animal_presence_monitor import GlobalAnimalPresenceAlarm
from autotrainer.core.configuration.alarm_configuration import EmergencyAlarmConfiguration
from autotrainer.core.configuration.alarm_detector import AlarmDetectorConfig
from autotrainer.pyside import CardWidget, StatusIcon
from autotrainer.pyside.content_widget import ContentWidget, invoke_method

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.hardware_model import HardwareModel

logger = get_verbose_logger(__name__)


@dataclasses.dataclass
class AlarmContentContext:
    alarm: AlarmDetector
    property_changed_cb: Callable
    icon: StatusIcon
    label: QLabel


def make_alarm_icon(name):
    return StatusIcon(
        on_icon='fa5s.bell',
        on_disabled_icon='fa5.bell',
        off_icon='fa5.bell',
        off_disabled_icon='fa5.bell',
        on_color='red',
        on_disabled_color='red',
        off_color='black',
        off_disabled_color='gray',
        name=name,
    )


def make_icon(size: int = 18, off_color: str = "black", name: str = "NA"):
    ico = StatusIcon(on_icon='fa5s.bell', off_icon='fa5.bell', on_color='red', off_color=off_color, size=size,
                     name=name)
    return ico


def make_label(txt: str) -> QLabel:
    lbl = QLabel(txt)
    palette = lbl.palette()
    palette.setColor(
        QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText, QColor("black")
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("gray")
    )
    # NB: Inactive != Disabled, later has to be used. Same for WindowText vs Text roles.
    lbl.setPalette(palette)
    return lbl


class AlarmContent(ContentWidget):
    """
    Widget to display alarm content.
    """

    load_cell_thrashing_changed = Signal(bool, name="load_cell_thrashing_changed")
    audio_thrashing_changed = Signal(bool, name="audio_thrashing_changed")
    front_door_changed = Signal(bool, name="front_door_changed")
    slide_door_changed = Signal(bool, name="slide_door_changed")
    device_ack_timeout_changed = Signal(bool)

    def __init__(self, app_model: AppModel, hardware_model: HardwareModel):
        super().__init__()

        self._app_model = app_model
        self._hardware_model = hardware_model

        card = self._card_widget = CardWidget(header_background_color="red")
        card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        card.header.setTitle("Alarms & Detectors", color="white")

        content_layout = QHBoxLayout()
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        content_layout.setSpacing(16)
        content_layout.setContentsMargins(8, 4, 8, 4)

        form_layout_alarms = self._form_layout_alarms = QFormLayout()
        form_layout_alarms.setHorizontalSpacing(8)
        form_layout_alarms.setVerticalSpacing(4)
        form_layout_alarms.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        analysis = app_model.behavior.analysis

        self._alarms: Dict[str, AlarmContentContext] = {}

        label = QLabel("<b>Alarms</b>")
        label.setContentsMargins(0, 0, 0, 4)
        form_layout_alarms.addRow(label, None)

        make_detector_icon = partial(make_icon, off_color="black")

        self.register_alarm("Thrashing", analysis.animal_thrashing_alarm)
        self.register_alarm("Mouse Missing", analysis.presence_in_cage_alarm)
        self.register_alarm("External Doors", analysis.external_doors_alarm)
        self.register_alarm("Device Comm. Error", analysis.device_comm_alarm)
        self.register_alarm("Animal Immobile", analysis.global_animal_presence_alarm)
        self.register_alarm("System Maintenance", analysis.system_maintenance_alarm)
        self.register_alarm("System Fault", analysis.system_fault_alarm)

        #

        form_layout_detectors = QFormLayout()
        form_layout_detectors.setHorizontalSpacing(16)
        form_layout_detectors.setVerticalSpacing(4)
        form_layout_detectors.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        label = QLabel("<b>Detectors</b>")
        label.setContentsMargins(0, 0, 0, 4)
        form_layout_detectors.addRow(label, None)

        self._load_cell_thrash_status = make_detector_icon(name="det-load-cell-thrash")
        self._load_cell_thrash_label = make_label("Load Cell Thrash:")
        form_layout_detectors.addRow(self._load_cell_thrash_label, self._load_cell_thrash_status)
        self.load_cell_thrashing_changed.connect(self._load_cell_thrash_status.setStatus)

        self._audio_spectrum_status = make_detector_icon(name="det-audio")
        form_layout_detectors.addRow("Audio:", self._audio_spectrum_status)
        self.audio_thrashing_changed.connect(self._audio_spectrum_status.setStatus)

        self._front_door_status = StatusIcon.doorIcon(name="det-front-door")
        form_layout_detectors.addRow("Front Door:", self._front_door_status)
        self.front_door_changed.connect(self._front_door_status.setStatus)

        self._slide_door_status = StatusIcon.doorIcon(name="det-slide-door")
        form_layout_detectors.addRow("Sliding Door:", self._slide_door_status)
        self.slide_door_changed.connect(self._slide_door_status.setStatus)

        icon = self._device_ack_timeout_status = make_detector_icon(name="det-device-ack")
        form_layout_detectors.addRow("Device Ack Timeout:", icon)
        self.device_ack_timeout_changed.connect(icon.setStatus)

        icon = self._pellet_misplaced_status = make_detector_icon(name="pellet-misplaced")
        form_layout_detectors.addRow("Pellet Misplaced:", icon)

        icon = self._pellets_before_refill_status = make_detector_icon(name="pellet-before-refill")
        form_layout_detectors.addRow("Pellets Before Refill:", icon)

        icon = self._consecutive_failed_loads_status = make_detector_icon(name="consecutive-failed-loads")
        form_layout_detectors.addRow("Cons. Failed Loads:", icon)

        content_layout.addLayout(form_layout_alarms)
        content_layout.addLayout(form_layout_detectors)

        content_widget = QWidget()
        content_widget.setLayout(content_layout)

        card.setContentWidget(content_widget)
        card.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.addWidget(card)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

        hardware_model.property_changed += self._hardware_model_property_changed
        #
        analysis.load_cell_monitor.property_changed += self._load_cell_property_changed
        analysis.audio_thrashing_monitor.property_changed += self._audio_thrashing_property_changed
        # emergency alarm controls 3 sub-alarms:
        analysis.pellet_misplaced_monitor.property_changed += self._pellet_misplaced_property_changed
        analysis.system_maintenance_alarm.property_changed += self._system_maint_mon_property_changed
        self._update_tunnel_headfix_visibility(hardware_model.tunnel_headfix_enabled)

    def _update_tunnel_headfix_visibility(self, is_enabled: bool):
        self._load_cell_thrash_label.setVisible(is_enabled)
        self._load_cell_thrash_status.setVisible(is_enabled)

    @invoke_method
    def _on_alarm_prop_changed(self, ctx: AlarmContentContext, name: str, value, _):
        self._refresh_alarm(ctx)

    def register_alarm(self, label: str, alarm: AlarmDetector):
        self.unregister_alarm(alarm.name)
        ctx = self._alarms[alarm.name] = AlarmContentContext(
            alarm=alarm,
            property_changed_cb=None,
            icon=make_alarm_icon(alarm.name),
            label=make_label(f"{label}:"),
        )
        ctx.property_changed_cb = partial(self._on_alarm_prop_changed, ctx)
        self._refresh_alarm(ctx)
        self._form_layout_alarms.addRow(ctx.label, ctx.icon)
        alarm.property_changed += ctx.property_changed_cb

    def unregister_alarm(self, name: str):
        ctx = self._alarms.pop(name, None)
        if ctx is None:
            return
        ctx.alarm.property_changed -= ctx.property_changed_cb
        self._form_layout_alarms.removeRow(ctx.label)

    def set_is_capture_active(self, is_editable: bool):
        self._card_widget.setEnabled(is_editable)

    @invoke_method
    def _hardware_model_property_changed(self, name: str, value, _):
        if name == HardwareModel.FRONT_DOOR_PROPERTY:
            self.front_door_changed.emit(value)
        elif name == HardwareModel.SLIDE_DOOR_PROPERTY:
            self.slide_door_changed.emit(value)
        elif name == HardwareModel.DEVICE_ACK_TIMEOUT_ENGAGED:
            self.device_ack_timeout_changed.emit(value)
        elif name == HardwareModel.TUNNEL_HEADFIX_ENABLED:
            self._update_tunnel_headfix_visibility(value)

    @invoke_method
    def _load_cell_property_changed(self, name: str, new_value, _):
        if name == LoadCellMonitor.IS_THRASHING_DETECTED_PROPERTY:
            self.load_cell_thrashing_changed.emit(new_value)

    @invoke_method
    def _audio_thrashing_property_changed(self, name, new_value, _):
        if name == AudioSpectrumThrashMonitor.IS_ENGAGED:
            self.audio_thrashing_changed.emit(new_value)

    def _refresh_alarm(self, ctx: AlarmContentContext):
        cfg = ctx.alarm.config
        ctx.label.setEnabled(cfg.use)
        ctx.icon.setInUse(cfg.use)
        ctx.icon.setStatus(ctx.alarm.is_engaged)

    @invoke_method
    def _refresh_alarm_icons_labels(self):
        for ctx in self._alarms.values():
            self._refresh_alarm(ctx)

    @invoke_method
    def _pellet_misplaced_property_changed(self, name, value, _):
        mon = self._app_model.analysis.pellet_misplaced_monitor
        if name == mon.IS_ENGAGED:
            self._pellet_misplaced_status.setStatus(value)

    @invoke_method
    def _system_maint_mon_property_changed(self, name, value, _):
        mon = self._app_model.analysis.system_maintenance_alarm
        if name == mon.MAX_PELLET_LOADED_ENGAGED:
            self._pellets_before_refill_status.setStatus(value)
        elif name == mon.MAX_CONSECUTIVE_FAILED_LOAD_ENGAGED:
            self._consecutive_failed_loads_status.setStatus(value)
