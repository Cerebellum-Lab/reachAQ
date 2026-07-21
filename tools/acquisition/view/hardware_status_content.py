from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from autotrainer.core import MessageHandler
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.device import CanTransportConfiguration
from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method

from tools.acquisition.model.laser_model import LaserModel
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel


class HardwareStatusContent(ContentWidget):

    def __init__(self, app_model):
        super().__init__()

        self._app_model = app_model
        self._message_handler = app_model.message_handler
        self._pellet_version = "(unknown)"
        self._enabled_labels: Dict[str, QLabel] = {}
        self._device_labels: Dict[str, QLabel] = {}
        self._info_labels: Dict[str, QLabel] = {}
        self._header_labels: Dict[str, QLabel] = {}
        self._refresh_widgets = []

        self.setObjectName("HardwareStatusContent")
        self.setStyleSheet(
            "#HardwareStatusContent QLabel {color: #2f343a;}"
            "#HardwareStatusContent QLabel#StatusHeader {"
            "color: #5b6470; font-weight: 600; padding-bottom: 2px;"
            "border-bottom: 1px solid #d6d9de;"
            "}"
            "#HardwareStatusContent QLabel#StatusDevice {color: #20242a; font-weight: 600;}"
            "#HardwareStatusContent QLabel#StatusEnabled {color: #20242a;}"
            "#HardwareStatusContent QLabel#StatusInfo {color: #2f343a;}"
            "#HardwareStatusContent QLabel#RefreshEnabled,"
            "#HardwareStatusContent QLabel#RefreshDevice,"
            "#HardwareStatusContent QLabel#RefreshInfo {color: #8a5a00; font-weight: 600;}"
        )

        self._card_widget = CardWidget(title="Hardware Status")
        self._message_handler.property_changed += self._model_property_changed

        layout = self._grid_layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(8, 5, 8, 7)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(2)
        layout.setColumnMinimumWidth(0, 62)
        layout.setColumnMinimumWidth(1, 92)
        layout.setColumnStretch(2, 1)

        self._row = 0
        self._add_header_row()
        self._add_refresh_row()
        for key, title in (
            ("cameras", "Cameras"),
            ("nidaq", "NI-DAQ"),
            ("can", "CAN Bus"),
            ("pellet", "Pellet"),
            ("laser", "Laser"),
        ):
            self._add_status_row(key, title)

        content = QWidget()
        content.setContentsMargins(0, 0, 0, 0)
        content.setLayout(layout)
        self._card_widget.setContentWidget(content)

        container_layout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        container_layout.addWidget(self._card_widget)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        self.setLayout(container_layout)

        for camera in app_model.cameras:
            camera.property_changed += self._on_camera_property_changed
        app_model.property_changed += self._on_app_model_property_changed
        app_model.hardware.property_changed += self._on_hardware_model_property_changed
        app_model.laser.property_changed += self._on_laser_property_changed
        app_model.nidaq_signal_monitor.property_changed += self._on_nidaq_property_changed
        app_model.configuration_loaded_event += self._on_configuration_loaded

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(1000)
        self._refresh_status()
        self.set_hardware_refreshing(False)

    def _add_refresh_row(self) -> None:
        enabled = QLabel("...")
        enabled.setObjectName("RefreshEnabled")
        enabled.setAlignment(Qt.AlignmentFlag.AlignCenter)
        device = QLabel("Hardware scan")
        device.setObjectName("RefreshDevice")
        info = QLabel("Scanning for devices and refreshing bindings...")
        info.setObjectName("RefreshInfo")
        self._grid_layout.addWidget(enabled, self._row, 0)
        self._grid_layout.addWidget(device, self._row, 1)
        self._grid_layout.addWidget(info, self._row, 2)
        self._refresh_widgets = [enabled, device, info]
        self._row += 1

    def _add_header_row(self) -> None:
        for col, text in enumerate(("Enabled", "Devices", "Info")):
            label = QLabel(text)
            label.setObjectName("StatusHeader")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if col == 0:
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid_layout.addWidget(label, self._row, col)
            self._header_labels[text.lower()] = label
        self._row += 1

    def _add_status_row(self, key: str, title: str) -> None:
        enabled = QLabel("-")
        enabled.setObjectName("StatusEnabled")
        enabled.setAlignment(Qt.AlignmentFlag.AlignCenter)
        enabled.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        device = QLabel(title)
        device.setObjectName("StatusDevice")
        device.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info = QLabel("-")
        info.setObjectName("StatusInfo")
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._grid_layout.addWidget(enabled, self._row, 0)
        self._grid_layout.addWidget(device, self._row, 1)
        self._grid_layout.addWidget(info, self._row, 2)
        self._enabled_labels[key] = enabled
        self._device_labels[key] = device
        self._info_labels[key] = info
        self._row += 1

    @invoke_method
    def set_hardware_refreshing(self, is_refreshing: bool) -> None:
        for widget in self._refresh_widgets:
            widget.setVisible(is_refreshing)

    def _refresh_status(self) -> None:
        self._refresh_camera_status()
        self._refresh_daq_status()
        self._refresh_can_status()
        self._refresh_pellet_status()
        self._refresh_laser_status()

    def _refresh_camera_status(self) -> None:
        cameras = list(self._app_model.cameras)
        enabled_count = sum(1 for camera in cameras if camera.is_enabled)
        self._set_label_status(
            self._enabled_labels["cameras"],
            f"{enabled_count}/{len(cameras)}",
            "ok" if enabled_count else "disabled",
        )
        scan_info, scan_state = self._scan_info("cameras")
        bindings = ", ".join(self._camera_binding_text(camera) for camera in cameras) or "none configured"
        self._set_info("cameras", f"{scan_info}; bindings: {bindings}", scan_state)

    def _refresh_daq_status(self) -> None:
        hardware = self._app_model.hardware
        monitor = self._app_model.nidaq_signal_monitor
        is_enabled = hardware.nidaq_enabled
        self._set_label_status(
            self._enabled_labels["nidaq"],
            self._yes_no(is_enabled),
            "ok" if is_enabled else "disabled",
        )
        scan_info, scan_state = self._scan_info("nidaq")
        configured_names = self._daq_device_names()
        configured_text = ", ".join(configured_names) if configured_names else "none"
        if monitor.is_starting:
            use_state = "starting stream"
        elif monitor.is_running:
            use_state = "streaming"
        elif is_enabled:
            use_state = "idle"
        else:
            use_state = "disabled"
        self._set_info(
            "nidaq",
            f"{scan_info}; configured: {configured_text}; {use_state}",
            scan_state,
        )

    def _refresh_can_status(self) -> None:
        hardware = self._app_model.hardware
        is_enabled = hardware.can_enabled
        self._set_label_status(
            self._enabled_labels["can"],
            self._yes_no(is_enabled),
            "ok" if is_enabled else "disabled",
        )
        scan_info, scan_state = self._scan_info("can")
        if is_enabled:
            connection_state = "connected" if hardware.connected else "idle"
            info = f"{scan_info}; {self._can_transport_text()}; {connection_state}"
        else:
            info = f"{scan_info}; disabled"
        self._set_info("can", info, scan_state)

    def _refresh_pellet_status(self) -> None:
        hardware = self._app_model.hardware
        is_enabled = hardware.pellet_controller_enabled
        self._set_label_status(
            self._enabled_labels["pellet"],
            self._yes_no(is_enabled),
            "ok" if is_enabled else "disabled",
        )
        scan_info, scan_state = self._scan_info("pellet")
        if not is_enabled:
            self._set_info("pellet", f"{scan_info}; disabled", scan_state)
            return
        connection_state = "connected" if hardware.connected else "idle"
        detail = f"{scan_info}; controller {connection_state}"
        if self._pellet_version and self._pellet_version != "(unknown)":
            detail = f"{detail}; fw {self._pellet_version}"
        self._set_info("pellet", detail, scan_state)

    def _refresh_laser_status(self) -> None:
        laser = self._app_model.laser
        configuration = laser.configuration
        is_enabled = configuration.backend != "disabled"
        self._set_label_status(
            self._enabled_labels["laser"],
            self._yes_no(is_enabled),
            "ok" if is_enabled else "disabled",
        )
        scan_info, scan_state = self._scan_info("laser")
        if not is_enabled:
            self._set_info("laser", scan_info, scan_state)
            return
        connection_state = "connected" if laser.is_connected else "configured"
        channel_count = len(configuration.channels)
        channel_text = "1 channel" if channel_count == 1 else f"{channel_count} channels"
        self._set_info(
            "laser",
            f"{scan_info}; {channel_text}; {connection_state}",
            scan_state,
        )

    def _camera_binding_text(self, camera) -> str:
        if not camera.is_enabled:
            return f"{camera.name}=off"
        source = camera.camera_source
        source_name = source.name if source is not None and source.name else "(not bound)"
        status = camera.capture_process_status
        if status == CaptureProcessStatus.RUNNING:
            state = "running"
        elif status == CaptureProcessStatus.FAILED:
            state = "failed"
        else:
            state = "idle"
        return f"{camera.name}={source_name} {state}"

    def _daq_device_names(self) -> Tuple[str, ...]:
        configured_names = getattr(self._app_model, "configured_nidaq_device_names", None)
        if configured_names is not None:
            return tuple(configured_names)

        names = []
        nidaq_ports = self._app_model.nidaq_ports
        if nidaq_ports.device_name:
            names.append(nidaq_ports.device_name)
        for channel in self._app_model.nidaq_signal_monitor.configuration.channels:
            device_name = self._device_name_from_channel(channel.physical_channel)
            if device_name and device_name not in names:
                names.append(device_name)
        for channel in self._app_model.laser.configuration.channels:
            for physical_channel in (
                channel.analog_output,
                channel.diode_input,
                channel.shutter_output,
                channel.command_copy_input,
            ):
                device_name = self._device_name_from_channel(physical_channel)
                if device_name and device_name not in names:
                    names.append(device_name)
        return tuple(names)

    def _scan_info(self, key: str) -> Tuple[str, str]:
        scan_results = getattr(self._app_model, "hardware_scan_results", {})
        entry = scan_results.get(key)
        if entry is None:
            return "Not scanned", "idle"
        return entry.info, entry.state

    def _can_transport_text(self) -> str:
        try:
            transport = CanTransportConfiguration.from_environment()
        except Exception as exc:
            return f"config error: {exc}"
        parts = [transport.kind.value, transport.channel]
        if transport.bitrate:
            parts.append(f"{transport.bitrate} bps")
        if transport.data_bitrate:
            parts.append(f"data {transport.data_bitrate} bps")
        if transport.fd:
            parts.append("CAN-FD")
        return " / ".join(str(part) for part in parts if part)

    @staticmethod
    def _device_name_from_channel(channel_name: Optional[str]) -> Optional[str]:
        if not channel_name:
            return None
        parts = channel_name.strip().lstrip("/").split("/")
        if len(parts) < 2 or not parts[0]:
            return None
        return parts[0]

    @staticmethod
    def _yes_no(value: bool) -> str:
        return "Yes" if value else "No"

    def _set_info(self, key: str, text: str, state: str) -> None:
        self._set_label_status(self._info_labels[key], text, state)

    def _set_label_status(self, label: QLabel, text: str, state: str) -> None:
        label.setText(text)
        label.setToolTip(text)
        if state == "ok":
            label.setStyleSheet("color: #1b6e3c; font-weight: 500;")
        elif state == "error":
            label.setStyleSheet("color: #b00020; font-weight: 600;")
        elif state == "disabled":
            label.setStyleSheet("color: #68717d;")
        else:
            label.setStyleSheet("color: #8a5a00; font-weight: 500;")

    @invoke_method
    def _on_camera_property_changed(self, _property_name: str, _value, _):
        self._refresh_camera_status()

    @invoke_method
    def _on_app_model_property_changed(self, property_name: str, _value, _):
        if property_name == "hardware_scan_results":
            self._refresh_status()

    @invoke_method
    def _on_hardware_model_property_changed(self, _property_name: str, _value, _):
        self._refresh_can_status()
        self._refresh_pellet_status()
        self._refresh_daq_status()

    @invoke_method
    def _on_laser_property_changed(self, property_name: str, _value, _):
        if property_name in (LaserModel.CONFIGURATION, LaserModel.IS_CONNECTED):
            self._refresh_laser_status()

    @invoke_method
    def _on_nidaq_property_changed(self, property_name: str, _value, _):
        if property_name in (
            NidaqSignalMonitorModel.CONFIGURATION,
            NidaqSignalMonitorModel.IS_STARTING,
            NidaqSignalMonitorModel.IS_RUNNING,
            NidaqSignalMonitorModel.STATUS_MESSAGE,
        ):
            self._refresh_daq_status()

    @invoke_method
    def _on_configuration_loaded(self, _configuration):
        self._refresh_status()

    @invoke_method
    def _model_property_changed(self, property_name: str, value, _):
        if property_name == MessageHandler.FIRMWARE_VERSION_PROPERTY:
            version = str(value).lower()
            if "module" in version:
                version = version.replace("module", "").strip()
            if "pellet" in version:
                self._pellet_version = version.replace("pellet", "").replace(":", "").strip() or "(unknown)"
                self._refresh_pellet_status()
