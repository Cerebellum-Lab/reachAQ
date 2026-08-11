import html
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core import CameraId
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState


class _CollapsibleHardwareCategory(QWidget):

    _COLORS = {
        "ok": ("#e8f5ec", "#65a978", "#1b6e3c"),
        "warning": ("#fff4d6", "#d39e00", "#8a5a00"),
        "error": ("#fdebec", "#cf6670", "#b00020"),
        "disabled": ("#f0f2f4", "#c9cdd3", "#68717d"),
        "idle": ("#edf2f7", "#aeb8c4", "#52606d"),
    }

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._enabled_state = "disabled"
        self._detail_state = "idle"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QWidget(self)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(6, 3, 8, 3)
        header_layout.setSpacing(5)

        self.toggle_button = QToolButton(self.header)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setAutoRaise(True)
        header_layout.addWidget(self.toggle_button, stretch=1)

        self.enabled_label = QLabel("Disabled", self.header)
        self.enabled_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.enabled_label)
        layout.addWidget(self.header)

        self.details_scroll = QScrollArea(self)
        self.details_scroll.setFrameShape(QFrame.Shape.StyledPanel)
        self.details_scroll.setWidgetResizable(False)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.details_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.details_scroll.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.details_scroll.setMinimumHeight(72)
        self.details_scroll.setMaximumHeight(170)
        self.details_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.details_scroll.setContentsMargins(20, 3, 4, 5)

        self.details_label = QLabel("-", self.details_scroll)
        self.details_label.setWordWrap(False)
        self.details_label.setTextFormat(Qt.TextFormat.RichText)
        self.details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details_label.setContentsMargins(6, 4, 6, 4)
        self.details_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.details_label.setStyleSheet("font-family: monospace;")
        self.details_scroll.setWidget(self.details_label)
        self.details_label.setVisible(False)
        self.details_scroll.setVisible(False)
        layout.addWidget(self.details_scroll)
        self.details_text = "-"

        self.toggle_button.toggled.connect(self._set_expanded)
        self._apply_status_style()

    @property
    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_enabled(self, text: str, state: str) -> None:
        self.enabled_label.setText(text)
        self.enabled_label.setToolTip(text)
        self._enabled_state = state
        self._apply_status_style()

    def set_details(self, text: str, state: str) -> None:
        self.details_text = text
        lines = text.splitlines() or ["-"]
        header = html.escape(lines[0])
        body = html.escape("\n".join(lines[1:]))
        rich_text = f"<pre><b><u>{header}</u></b>"
        if body:
            rich_text += f"\n{body}"
        rich_text += "</pre>"
        self.details_label.setText(rich_text)
        self.details_label.setToolTip(text)
        self.details_label.adjustSize()
        self._detail_state = state
        self._apply_status_style()

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.details_label.setVisible(expanded)
        self.details_scroll.setVisible(expanded)

    def _combined_state(self) -> str:
        if self._detail_state in ("error", "warning"):
            return self._detail_state
        if self._enabled_state == "disabled":
            return "disabled"
        if self._detail_state in ("idle", "disabled"):
            return "idle"
        if self._detail_state == "ok":
            return "ok"
        return self._enabled_state if self._enabled_state in self._COLORS else "idle"

    def _apply_status_style(self) -> None:
        background, border, foreground = self._COLORS[self._combined_state()]
        self.header.setStyleSheet(
            f"background: {background}; border: 1px solid {border}; border-radius: 3px;"
        )
        self.toggle_button.setStyleSheet(
            f"color: {foreground}; font-weight: 600; border: none; text-align: left;"
        )
        self.enabled_label.setStyleSheet(
            f"color: {foreground}; font-weight: 600; border: none;"
        )
        self.details_label.setStyleSheet(f"color: {foreground}; font-family: monospace;")


class HardwareStatusContent(ContentWidget):

    def __init__(self, app_model):
        super().__init__()

        self._app_model = app_model
        self._enabled_labels: Dict[str, QLabel] = {}
        self._device_labels: Dict[str, QToolButton] = {}
        self._info_labels: Dict[str, QLabel] = {}
        self._category_panels: Dict[str, _CollapsibleHardwareCategory] = {}

        self.setObjectName("HardwareStatusContent")
        self.setStyleSheet("#HardwareStatusContent QLabel {color: #2f343a;}")

        self._refresh_button = QToolButton(self)
        self._refresh_button.setAutoRaise(True)
        self._refresh_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._refresh_button.setVisible(False)
        self._card_widget = CardWidget(
            title="Hardware Status",
            header_right_layout=self._refresh_button,
        )

        layout = self._category_layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(8, 5, 8, 7)
        layout.setSpacing(4)

        layout.addWidget(self._create_hardware_configuration_editor())

        self._refresh_label = QLabel("Scanning hardware and refreshing bindings...")
        self._refresh_label.setStyleSheet("color: #8a5a00; font-weight: 600; padding: 4px;")
        layout.addWidget(self._refresh_label)
        for key, title in (
            ("cameras", "Cameras"),
            ("nidaq", "NI-DAQ"),
            ("can", "CAN Adapter"),
            ("pellet", "Pellet Controller"),
            ("rfid", "RFID Reader"),
            ("gpu", "GPU"),
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

        app_model.property_changed += self._on_app_model_property_changed
        app_model.configuration_loaded_event += self._on_configuration_loaded
        app_model.hardware.property_changed += self._on_hardware_model_property_changed
        self._refresh_status()
        self._sync_hardware_configuration_editor()
        self.set_hardware_refreshing(False)

    def _create_hardware_configuration_editor(self) -> QGroupBox:
        group = QGroupBox("Hardware Configuration")
        form = QFormLayout(group)
        form.setContentsMargins(8, 6, 8, 8)
        self._hardware_enabled_controls = {}
        for key, title in (
            ("can", "CAN adapter"),
            ("pellet", "Pellet controller"),
            ("nidaq", "NI-DAQ"),
            ("rfid", "USB RFID reader"),
        ):
            checkbox = QCheckBox(f"Enable {title}")
            self._hardware_enabled_controls[key] = checkbox
            form.addRow("", checkbox)

        self._rfid_device_edit = QLineEdit()
        self._rfid_device_edit.setPlaceholderText("/dev/serial/by-id/...")
        form.addRow("RFID serial device:", self._rfid_device_edit)

        button_row = QHBoxLayout()
        self._save_hardware_button = QPushButton("Apply and save")
        self._save_hardware_button.clicked.connect(
            self._apply_hardware_configuration_editor
        )
        button_row.addWidget(self._save_hardware_button)
        self._hardware_save_status = QLabel("")
        self._hardware_save_status.setWordWrap(True)
        button_row.addWidget(self._hardware_save_status, stretch=1)
        form.addRow("", button_row)
        return group

    def _sync_hardware_configuration_editor(self) -> None:
        hardware = self._app_model.hardware
        configuration = self._loaded_configuration()
        hardware_configuration = (
            None if configuration is None else configuration.hardware
        )
        values = {
            "can": bool(hardware.can_enabled),
            "pellet": bool(hardware.pellet_controller_enabled),
            "nidaq": bool(hardware.nidaq_enabled),
            "rfid": bool(
                hardware_configuration is not None
                and hardware_configuration.rfid_reader_enabled
            ),
        }
        for key, value in values.items():
            self._hardware_enabled_controls[key].setChecked(value)
        self._rfid_device_edit.setText(
            "" if hardware_configuration is None
            else hardware_configuration.rfid_device
        )

    def _apply_hardware_configuration_editor(self) -> None:
        try:
            message = self._app_model.update_hardware_configuration(
                can_enabled=self._hardware_enabled_controls["can"].isChecked(),
                pellet_controller_enabled=self._hardware_enabled_controls["pellet"].isChecked(),
                nidaq_enabled=self._hardware_enabled_controls["nidaq"].isChecked(),
                rfid_reader_enabled=self._hardware_enabled_controls["rfid"].isChecked(),
                rfid_device=self._rfid_device_edit.text(),
            )
        except Exception as exc:
            self._hardware_save_status.setText(f"Not saved: {exc}")
            self._sync_hardware_configuration_editor()
            return
        self._hardware_save_status.setText(message or "Hardware settings saved")
        self._sync_hardware_configuration_editor()
        self._refresh_status()

    def set_refresh_action(self, action) -> None:
        self._refresh_button.setDefaultAction(action)
        self._refresh_button.setVisible(True)

    def _add_status_row(self, key: str, title: str) -> None:
        panel = _CollapsibleHardwareCategory(title)
        self._category_layout.addWidget(panel)
        self._category_panels[key] = panel
        self._enabled_labels[key] = panel.enabled_label
        self._device_labels[key] = panel.toggle_button
        self._info_labels[key] = panel.details_label

    @invoke_method
    def set_hardware_refreshing(self, is_refreshing: bool) -> None:
        self._refresh_label.setVisible(is_refreshing)

    def _refresh_status(self) -> None:
        self._refresh_camera_status()
        self._refresh_daq_status()
        self._refresh_can_status()
        self._refresh_pellet_status()
        self._refresh_rfid_status()
        self._refresh_gpu_status()
        self._refresh_laser_status()

    def _refresh_camera_status(self) -> None:
        cameras = [
            camera
            for camera in self._app_model.cameras
            if camera.camera_id in (CameraId.Left, CameraId.Right, CameraId.Camera3)
        ]
        enabled_count = sum(1 for camera in cameras if camera.is_enabled)
        self._set_enabled("cameras", enabled_count > 0)
        scan_info, scan_state = self._scan_info("cameras")
        rows = [self._camera_status_row(camera, scan_info, scan_state) for camera in cameras]
        rows.append(self._subsystem_status_row(
            SubsystemId.REACH_SYNCHRONIZATION,
            "reach synchronization",
        ))
        self._set_info(
            "cameras",
            self._format_device_rows(rows, self._scan_notes(scan_info)),
            scan_state,
        )

    def _refresh_daq_status(self) -> None:
        hardware = self._app_model.hardware
        monitor = self._app_model.nidaq_signal_monitor
        is_enabled = hardware.nidaq_enabled
        self._set_enabled("nidaq", is_enabled)
        scan_info, scan_state = self._scan_info("nidaq")
        configured_names = self._daq_device_names()
        if monitor.is_starting:
            use_state = "starting stream"
        elif monitor.is_running:
            use_state = "streaming"
        elif is_enabled:
            use_state = "idle"
        else:
            use_state = "disabled"
        discovered_rows = self._nidaq_scan_rows(scan_info, configured_names)
        discovered_names = {row[0] for row in discovered_rows}
        for name in configured_names:
            if name not in discovered_names:
                if name in scan_info:
                    state = "connected · bound"
                else:
                    state = "missing" if scan_state in ("ok", "warning") else "not scanned"
                discovered_rows.append((name, "configured", state))
        stream_binding = ", ".join(configured_names) if configured_names else "no binding"
        discovered_rows.append(("input stream", stream_binding, use_state))
        discovered_rows.append(self._subsystem_status_row(
            SubsystemId.NIDAQ_STREAM,
            "runtime",
        ))
        discovered_rows.extend(self._daq_binding_rows())
        self._set_info(
            "nidaq",
            self._format_device_rows(discovered_rows, self._scan_notes(scan_info)),
            scan_state,
        )

    def _refresh_can_status(self) -> None:
        hardware = self._app_model.hardware
        is_enabled = hardware.can_enabled
        self._set_enabled("can", is_enabled)
        scan_info, scan_state = self._scan_info("can")
        rows = self._can_scan_rows(scan_info)
        if not rows:
            state = "not scanned" if scan_info == "Not scanned" else scan_state
            rows = [("adapter", self._single_line(scan_info), state)]
        rows.append(("runtime", self._can_binding(), "enabled" if is_enabled else "disabled"))
        rows.append(self._subsystem_status_row(SubsystemId.CAN_PELLET, "health"))
        self._set_info(
            "can",
            self._format_device_rows(rows, self._scan_notes(scan_info)),
            scan_state,
        )

    def _refresh_pellet_status(self) -> None:
        hardware = self._app_model.hardware
        is_enabled = hardware.pellet_controller_enabled
        scan_info, _ = self._scan_info("pellet")
        is_connected = bool(getattr(hardware, "connected", False))
        has_timed_out = bool(
            getattr(hardware, "pellet_status_timeout_engaged", False)
        )
        if not is_enabled:
            state = "disabled"
            panel_state = "disabled"
        elif scan_info == "Not scanned":
            state = "not scanned"
            panel_state = "idle"
        elif has_timed_out:
            state = "connection lost"
            panel_state = "error"
        elif is_connected:
            state = "connected"
            panel_state = "ok"
        else:
            state = "connection failed"
            panel_state = "error"
        self._set_enabled("pellet", is_enabled, panel_state)
        rows = [("pellet", self._pellet_binding(), state)]
        rows.append(self._subsystem_status_row(SubsystemId.CAN_PELLET, "health"))
        version = getattr(hardware, "pellet_version", "")
        rows.append(
            (
                "firmware",
                f"Pellet: {version}" if version else "unknown",
                "reported" if version else "unknown",
            )
        )
        self._set_info(
            "pellet",
            self._format_device_rows(rows, self._scan_notes(scan_info)),
            panel_state,
        )

    def _refresh_gpu_status(self) -> None:
        scan_info, scan_state = self._scan_info("gpu")
        found = scan_state == "ok"
        self._set_enabled("gpu", found, scan_state)
        rows = self._gpu_scan_rows(scan_info, scan_state)
        self._set_info(
            "gpu",
            self._format_device_rows(rows, self._scan_notes(scan_info)),
            scan_state,
        )

    def _refresh_rfid_status(self) -> None:
        reader_status = getattr(self._app_model, "rfid_reader_status", None)
        subsystem = self._subsystem_status(SubsystemId.RFID_READER)
        if subsystem is None:
            enabled = reader_status is not None
            state = "idle" if enabled else "disabled"
            runtime = "status unavailable" if enabled else "reader disabled"
            runtime_state = "idle" if enabled else "disabled"
        else:
            enabled = subsystem.state is not SubsystemState.DISABLED
            state = {
                SubsystemState.READY: "ok",
                SubsystemState.STARTING: "warning",
                SubsystemState.FAILED: "error",
                SubsystemState.BLOCKED: "error",
                SubsystemState.DISABLED: "disabled",
                SubsystemState.STOPPED: "idle",
                SubsystemState.STOPPING: "warning",
            }[subsystem.state]
            runtime = subsystem.error or subsystem.reason or subsystem.state.value
            runtime_state = subsystem.state.value
        device = getattr(reader_status, "device", "") or self._configured_rfid_device()
        rows = [("reader", device or "not configured", runtime_state)]
        rows.append(("runtime", runtime, runtime_state))
        scan = getattr(self._app_model, "rfid_scan_result", None)
        if scan is not None:
            kind = getattr(getattr(scan, "kind", None), "value", "scan")
            rfid = getattr(scan, "rfid", "")
            animal = getattr(scan, "animal", None)
            identity = getattr(animal, "name", "") or rfid or "unknown tag"
            rows.append(("last scan", identity, kind.replace("_", " ")))
        self._set_enabled("rfid", enabled, state)
        self._set_info("rfid", self._format_device_rows(rows), state)

    def _refresh_laser_status(self) -> None:
        laser = self._app_model.laser
        configuration = laser.configuration
        is_enabled = configuration.backend != "disabled"
        self._set_enabled("laser", is_enabled)
        scan_info, scan_state = self._scan_info("laser")
        connection_state = "connected" if laser.is_connected else "configured"
        if not is_enabled:
            rows = [("controller", "backend disabled", "disabled")]
        else:
            rows = []
            for channel in configuration.channels:
                bindings = [
                    f"AO {channel.analog_output}",
                    f"AI {channel.diode_input}",
                    f"shutter {channel.shutter_output}",
                ]
                if channel.command_copy_input:
                    bindings.append(f"copy {channel.command_copy_input}")
                rows.append((f"laser {channel.channel_id.value}", " · ".join(bindings), connection_state))
            timing = "hardware" if configuration.hardware_timed else "software"
            rate = "manual" if configuration.sample_rate_hz is None else f"{configuration.sample_rate_hz:g} Hz"
            rows.append(("timing", f"{timing} · {rate}", connection_state))
        rows.append(self._subsystem_status_row(SubsystemId.LASER, "runtime"))
        self._set_info(
            "laser",
            self._format_device_rows(rows, self._scan_notes(scan_info)),
            scan_state,
        )

    def _camera_status_row(self, camera, scan_info: str, scan_state: str) -> Tuple[str, str, str]:
        if not camera.is_enabled:
            return camera.name, self._camera_binding(camera), "disabled"
        status = camera.capture_process_status
        if status == CaptureProcessStatus.RUNNING:
            state = "running"
        elif status == CaptureProcessStatus.FAILED:
            state = "failed"
        elif scan_info == "Not scanned":
            state = "not scanned"
        elif scan_state in ("warning", "error") and f"missing: {camera.name}" in scan_info:
            state = "missing"
        else:
            state = "idle"
        return camera.name, self._camera_binding(camera), state

    def _subsystem_status_row(self, subsystem_id, label: str) -> Tuple[str, str, str]:
        key = subsystem_id.value if isinstance(subsystem_id, SubsystemId) else str(subsystem_id)
        status = self._subsystem_status(key)
        if status is None:
            return label, "status unavailable", "idle"
        detail = status.error or status.reason or status.state.value
        state = {
            SubsystemState.READY: "ready",
            SubsystemState.STARTING: "starting",
            SubsystemState.STOPPING: "stopping",
            SubsystemState.FAILED: "failed",
            SubsystemState.BLOCKED: "blocked",
            SubsystemState.DISABLED: "disabled",
            SubsystemState.STOPPED: "stopped",
        }[status.state]
        return label, detail, state

    def _subsystem_status(self, subsystem_id):
        key = subsystem_id.value if isinstance(subsystem_id, SubsystemId) else str(subsystem_id)
        try:
            statuses = object.__getattribute__(self._app_model, "subsystem_statuses")
        except (AttributeError, TypeError):
            statuses = {}
        return statuses.get(key)

    def _configured_rfid_device(self) -> str:
        configuration = self._loaded_configuration()
        if configuration is None:
            return ""
        return configuration.hardware.rfid_device

    @staticmethod
    def _camera_binding(camera) -> str:
        source = camera.camera_source
        if source is None:
            return "not bound"
        url = getattr(source, "url", "") or ""
        parsed = urlparse(url)
        if parsed.scheme == "spinnaker" and parsed.hostname:
            return parsed.hostname
        return getattr(source, "name", "") or url or "not bound"

    @classmethod
    def _nidaq_scan_rows(
        cls,
        scan_info: str,
        configured_names: Iterable[str],
    ) -> List[Tuple[str, str, str]]:
        configured = set(configured_names)
        rows = []
        for line in scan_info.splitlines():
            stripped = line.strip()
            if not stripped.startswith("→"):
                continue
            parts = [part.strip() for part in stripped[1:].split("·")]
            name = parts[0]
            if not name:
                continue
            binding = " · ".join(parts[1:]) or "NI-DAQ"
            state = "connected · bound" if name in configured else "connected"
            rows.append((name, binding, state))
        return rows

    @classmethod
    def _can_scan_rows(cls, scan_info: str) -> List[Tuple[str, str, str]]:
        rows = []
        for line in scan_info.splitlines():
            stripped = line.strip()
            if stripped.startswith("✓"):
                parts = [part.strip() for part in stripped[1:].split("·")]
                rows.append((parts[0], " · ".join(parts[1:]) or "adapter", "connected"))
            elif stripped.startswith("→"):
                text = stripped[1:].strip()
                if text.lower().startswith("no can interface"):
                    rows.append(("interface", "none", "missing"))
                    continue
                parts = [part.strip() for part in text.split("·")]
                identity = parts[0].replace(" ↑", "").replace(" ↓", "")
                state = "up" if "↑" in parts[0] else "down" if "↓" in parts[0] else "found"
                rows.append((identity, " · ".join(parts[1:]) or "interface", state))
            elif "↳ app:" in stripped:
                rows.append(("app", stripped.split("app:", 1)[1].strip(), "bound"))
        return rows

    @classmethod
    def _gpu_scan_rows(cls, scan_info: str, scan_state: str) -> List[Tuple[str, str, str]]:
        rows = []
        for line in scan_info.splitlines():
            stripped = line.strip()
            if not stripped.startswith("→"):
                continue
            text = stripped[1:].strip()
            first, separator, details = text.partition(" ")
            rows.append((first, details if separator else "NVIDIA GPU", "found"))
        if not rows:
            state = "not scanned" if scan_info == "Not scanned" else scan_state
            rows.append(("GPU", cls._single_line(scan_info), state))
        return rows

    def _can_binding(self) -> str:
        configuration = self._loaded_configuration()
        hardware = getattr(configuration, "hardware", None)
        if hardware is None:
            return "configured transport"
        return hardware.pellet_identifier or "configured transport"

    def _pellet_binding(self) -> str:
        configuration = self._loaded_configuration()
        hardware = getattr(configuration, "hardware", None)
        identifier = None if hardware is None else hardware.pellet_identifier
        return identifier or "CAN"

    def _loaded_configuration(self):
        try:
            return self._app_model.loaded_configuration
        except Exception:
            return getattr(self._app_model, "__dict__", {}).get("_loaded_configuration")

    @staticmethod
    def _format_device_rows(
        rows: Iterable[Tuple[str, str, str]],
        notes: Iterable[str] = tuple(),
    ) -> str:
        normalized_rows = tuple(
            (
                HardwareStatusContent._single_line(device),
                HardwareStatusContent._single_line(binding),
                HardwareStatusContent._single_line(state),
            )
            for device, binding, state in rows
        )
        device_width = max((len(row[0]) for row in normalized_rows), default=0)
        binding_width = max((len(row[1]) for row in normalized_rows), default=0)
        device_width = max(device_width, len("device"))
        binding_width = max(binding_width, len("binding / model"))
        lines = [
            f"{'device':<{device_width}} | {'binding / model':<{binding_width}} | status"
        ]
        lines.extend(
            f"{device:<{device_width}} | {binding:<{binding_width}} | {state}"
            for device, binding, state in normalized_rows
        )
        lines.extend(f"! {HardwareStatusContent._single_line(note)}" for note in notes)
        return "\n".join(lines)

    @staticmethod
    def _scan_notes(scan_info: str) -> Tuple[str, ...]:
        notes = []
        for line in scan_info.splitlines():
            stripped = line.strip()
            if stripped.startswith("!"):
                note = stripped.lstrip("! ")
            elif "failed" in stripped.lower() or "timed out" in stripped.lower():
                note = stripped
            elif "connection starts" in stripped.lower():
                note = stripped.lstrip("↳ ")
            else:
                continue
            if note and note not in notes:
                notes.append(note)
        return tuple(notes)

    @staticmethod
    def _single_line(value) -> str:
        text = " ".join(str(value or "-").split())
        return text.replace("|", "/")

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

    def _daq_binding_rows(self) -> List[Tuple[str, str, str]]:
        rows = []
        nidaq_ports = self._app_model.nidaq_ports
        for name, physical_channel in vars(nidaq_ports).items():
            if name == "device_name" or not physical_channel:
                continue
            rows.append((name.replace("_", " "), physical_channel, "bound"))
        for channel in self._app_model.nidaq_signal_monitor.configuration.channels:
            rows.append((channel.name, channel.physical_channel, f"{channel.kind} stream"))
        return rows

    def _scan_info(self, key: str) -> Tuple[str, str]:
        scan_results = getattr(self._app_model, "hardware_scan_results", {})
        entry = scan_results.get(key)
        if entry is None:
            return "Not scanned", "idle"
        return entry.info, entry.state

    @staticmethod
    def _device_name_from_channel(channel_name: Optional[str]) -> Optional[str]:
        if not channel_name:
            return None
        parts = channel_name.strip().lstrip("/").split("/")
        if len(parts) < 2 or not parts[0]:
            return None
        return parts[0]

    def _set_enabled(self, key: str, enabled: bool, state: Optional[str] = None) -> None:
        state = state or ("ok" if enabled else "disabled")
        self._category_panels[key].set_enabled(
            "Enabled" if enabled else "Disabled",
            state,
        )

    def _set_info(self, key: str, text: str, state: str) -> None:
        self._category_panels[key].set_details(text, state)

    @invoke_method
    def _on_app_model_property_changed(self, property_name: str, _value, _):
        if property_name in {
            "hardware_scan_results",
            "subsystem_statuses",
            "rfid_reader_status",
            "rfid_scan_result",
        }:
            self._refresh_status()

    @invoke_method
    def _on_hardware_model_property_changed(self, property_name: str, _value, _):
        if property_name in {
            "pellet_version",
            "device_pellet_status_timeout_engaged",
            "can_enabled",
            "pellet_controller_enabled",
            "nidaq_enabled",
        }:
            self._sync_hardware_configuration_editor()
            self._refresh_status()

    @invoke_method
    def _on_configuration_loaded(self, _configuration):
        self._sync_hardware_configuration_editor()
        self._refresh_status()
