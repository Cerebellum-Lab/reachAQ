from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NidaqPortConfiguration,
    SystemConfiguration,
)
from tools.acquisition.model.nidaq_discovery import (
    NidaqDevicePorts,
    device_name_from_channel,
    discover_nidaq_devices,
)


_GENERAL_ROLES: Tuple[Tuple[str, str, str], ...] = (
    ("tone1", "tone1", "do"),
    ("tone2", "tone2", "do"),
    ("tone3_r", "tone3R", "do"),
    ("tone3_l", "tone3L", "do"),
    ("cam_frames", "cam_frames", "di"),
    ("barcode", "barcode", "di"),
)

_LASER_ROLES: Tuple[Tuple[str, str, str], ...] = (
    ("laser_out", "laser_out", "ao"),
    ("diode", "diode", "ai"),
    ("shutter", "shutter", "do"),
    ("laser_copy", "laser_copy", "ai"),
)


class NidaqPortConfigurationDialog(QDialog):
    """Edit reachAQ DAQ role/channel assignments from discovered NI-DAQ devices."""

    def __init__(
        self,
        configuration: SystemConfiguration,
        parent: Optional[QWidget] = None,
        *,
        devices: Optional[Sequence[NidaqDevicePorts]] = None,
        discovery_error: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit DAQ Ports")
        self.setMinimumSize(640, 560)

        self._configuration = configuration
        if devices is None:
            devices, discovery_error = discover_nidaq_devices()
        self._devices = {device.name: device for device in devices}
        self._discovery_error = discovery_error
        self._nidaq_ports = configuration.nidaq_ports
        self._laser_configuration = configuration.laser
        self._general_combos: Dict[str, QComboBox] = {}
        self._laser_combos: Dict[int, Dict[str, QComboBox]] = {}
        self._combo_kinds: Dict[QComboBox, str] = {}
        self._combo_role_names: Dict[QComboBox, str] = {}
        self._missing_current_channels: List[str] = []
        self._refreshing_channel_options = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for device_name in sorted(self._devices):
            self._device_combo.addItem(device_name, device_name)
        device_row.addWidget(self._device_combo, stretch=1)
        root_layout.addLayout(device_row)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        root_layout.addWidget(self._status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        general_group = QGroupBox("DAQ Roles")
        general_layout = QFormLayout(general_group)
        general_layout.setContentsMargins(10, 8, 10, 10)
        general_layout.setSpacing(8)
        for attr_name, label, kind in _GENERAL_ROLES:
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            self._general_combos[attr_name] = combo
            self._register_channel_combo(combo, kind, label)
            general_layout.addRow(f"{label}:", combo)
        scroll_layout.addWidget(general_group)

        self._laser_tabs = QTabWidget()
        self._laser_tabs.setDocumentMode(True)
        self._laser_tabs.setUsesScrollButtons(True)
        for laser_index in range(1, 5):
            tab = QWidget()
            tab_layout = QFormLayout(tab)
            tab_layout.setContentsMargins(10, 10, 10, 10)
            tab_layout.setSpacing(8)
            combos = {}
            for attr_name, label, kind in _LASER_ROLES:
                combo = QComboBox()
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
                combos[attr_name] = combo
                self._register_channel_combo(combo, kind, f"laser{laser_index}.{label}")
                tab_layout.addRow(f"{label}:", combo)
            self._laser_combos[laser_index] = combos
            self._laser_tabs.addTab(tab, f"Laser {laser_index}")
        scroll_layout.addWidget(self._laser_tabs)
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        root_layout.addWidget(scroll, stretch=1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root_layout.addWidget(self._buttons)

        self._device_combo.currentTextChanged.connect(self._populate_port_combos)
        self._select_initial_device()
        self._populate_port_combos()

    @property
    def nidaq_ports(self) -> NidaqPortConfiguration:
        return self._nidaq_ports

    @property
    def laser_configuration(self) -> LaserSystemConfiguration:
        return self._laser_configuration

    def accept(self) -> None:
        try:
            device = self._selected_device()
            if device is None:
                raise RuntimeError("No NI-DAQ device is available for port configuration")
            self._validate_selected_channel_assignments(device)
            self._nidaq_ports = self._build_nidaq_port_configuration(device.name)
            self._laser_configuration = self._build_laser_configuration()
        except Exception as exc:
            QMessageBox.critical(self, "DAQ Port Configuration", str(exc) or exc.__class__.__name__)
            return
        super().accept()

    def _select_initial_device(self) -> None:
        if not self._devices:
            return
        configured_name = self._configuration.nidaq_ports.device_name
        if configured_name is None:
            configured_name = self._infer_configured_device_name()
        index = self._device_combo.findData(configured_name)
        if index < 0:
            index = 0
        self._device_combo.setCurrentIndex(index)

    def _populate_port_combos(self) -> None:
        device = self._selected_device()
        if device is None:
            self._set_all_combos_enabled(False)
            self._set_ok_enabled(False)
            message = self._discovery_error or "No NI-DAQ devices were discovered."
            self._status_label.setText(message)
            self._status_label.setStyleSheet("color: #b00020;")
            return

        self._set_all_combos_enabled(True)
        self._missing_current_channels = []

        current_ports = self._configuration.nidaq_ports
        for attr_name, _label, kind in _GENERAL_ROLES:
            current_value = getattr(current_ports, attr_name)
            missing = self._set_combo_options(
                self._general_combos[attr_name],
                self._options_for_kind(device, kind),
                current_value,
            )
            if missing is not None:
                self._missing_current_channels.append(f"{attr_name}={missing}")

        existing_channels = {
            int(channel.channel_id): channel
            for channel in self._configuration.laser.channels
        }
        for laser_index, combos in self._laser_combos.items():
            channel = existing_channels.get(laser_index)
            current_values = {
                "laser_out": None if channel is None else channel.analog_output,
                "diode": None if channel is None else channel.diode_input,
                "shutter": None if channel is None else channel.shutter_output,
                "laser_copy": None if channel is None else channel.command_copy_input,
            }
            for attr_name, _label, kind in _LASER_ROLES:
                missing = self._set_combo_options(
                    combos[attr_name],
                    self._options_for_kind(device, kind),
                    current_values[attr_name],
                )
                if missing is not None:
                    self._missing_current_channels.append(f"laser{laser_index}.{attr_name}={missing}")

        self._refresh_channel_options()
        self._update_status_label()

    def _register_channel_combo(self, combo: QComboBox, kind: str, role_name: str) -> None:
        self._combo_kinds[combo] = kind
        self._combo_role_names[combo] = role_name
        combo.currentIndexChanged.connect(self._on_channel_combo_changed)

    def _on_channel_combo_changed(self, _index: int) -> None:
        if self._refreshing_channel_options:
            return
        self._missing_current_channels = []
        self._refresh_channel_options()
        self._update_status_label()

    def _refresh_channel_options(self) -> None:
        device = self._selected_device()
        if device is None or self._refreshing_channel_options:
            return

        self._refreshing_channel_options = True
        try:
            for combo, kind in self._combo_kinds.items():
                current_value = self._combo_value(combo)
                used_elsewhere = {
                    value
                    for other_combo in self._combo_kinds
                    if other_combo is not combo
                    for value in (self._combo_value(other_combo),)
                    if value is not None
                }
                all_options = self._options_for_kind(device, kind)
                options = tuple(
                    option
                    for option in all_options
                    if option == current_value or option not in used_elsewhere
                )
                self._set_combo_options(combo, options, current_value)
                has_selectable_channel = len(options) > 0
                combo.setEnabled(has_selectable_channel)
                if len(all_options) == 0:
                    combo.setToolTip(f"The selected device has no {kind.upper()} channels.")
                elif not has_selectable_channel:
                    combo.setToolTip(f"All {kind.upper()} channels on this device are already assigned.")
                else:
                    combo.setToolTip("")
        finally:
            self._refreshing_channel_options = False

    def _update_status_label(self) -> None:
        device = self._selected_device()
        if device is None:
            return
        status = (
            f"Available channels: AO {len(device.analog_outputs)}, AI {len(device.analog_inputs)}, "
            f"DO {len(device.digital_outputs)}, DI {len(device.digital_inputs)}"
        )
        warnings = []
        if self._missing_current_channels:
            warnings.append(
                "Configured channel(s) not available on this device: "
                + ", ".join(self._missing_current_channels)
            )
        unsupported = self._unsupported_selected_channels(device)
        if unsupported:
            warnings.append("Selected channel(s) not supported by this device/type: " + ", ".join(unsupported))
        duplicates = self._duplicate_selected_channels()
        if duplicates:
            warnings.append("Duplicate channel assignment(s): " + ", ".join(duplicates))
        unavailable_kinds = [
            kind.upper()
            for kind, options in (
                ("ao", device.analog_outputs),
                ("ai", device.analog_inputs),
                ("do", device.digital_outputs),
                ("di", device.digital_inputs),
            )
            if len(options) == 0
        ]
        if unavailable_kinds:
            warnings.append("Unsupported channel type(s) on selected device: " + ", ".join(unavailable_kinds))
        if warnings:
            status += "\n" + "\n".join(warnings)
            self._status_label.setStyleSheet("color: #9a6700;")
        else:
            self._status_label.setStyleSheet("")
        self._set_ok_enabled(not unsupported and not duplicates)
        self._status_label.setText(status)

    def _build_nidaq_port_configuration(self, device_name: str) -> NidaqPortConfiguration:
        values = {
            attr_name: self._combo_value(combo)
            for attr_name, combo in self._general_combos.items()
        }
        return NidaqPortConfiguration(device_name=device_name, **values)

    def _build_laser_configuration(self) -> LaserSystemConfiguration:
        existing = {
            int(channel.channel_id): channel
            for channel in self._configuration.laser.channels
        }
        channels = []
        for laser_index in range(1, 5):
            values = {
                attr_name: self._combo_value(combo)
                for attr_name, combo in self._laser_combos[laser_index].items()
            }
            if not any(values.values()):
                continue
            missing = [
                attr_name
                for attr_name in ("laser_out", "diode", "shutter", "laser_copy")
                if values[attr_name] is None
            ]
            if missing:
                raise ValueError(
                    f"Laser {laser_index} has incomplete DAQ mapping: missing {', '.join(missing)}"
                )
            previous = existing.get(laser_index)
            channels.append(
                LaserChannelConfiguration(
                    channel_id=LaserChannelId(laser_index),
                    analog_output=values["laser_out"],
                    diode_input=values["diode"],
                    shutter_output=values["shutter"],
                    auxiliary_output=None if previous is None else previous.auxiliary_output,
                    command_copy_input=values["laser_copy"],
                    trigger_source=None if previous is None else previous.trigger_source,
                    trigger_output=None if previous is None else previous.trigger_output,
                    timing_trigger_output=None if previous is None else previous.timing_trigger_output,
                    minimum_command_volts=0.0 if previous is None else previous.minimum_command_volts,
                    maximum_command_volts=5.0 if previous is None else previous.maximum_command_volts,
                    feedback_scale=1.0 if previous is None else previous.feedback_scale,
                    command_copy_scale=1.0 if previous is None else previous.command_copy_scale,
                )
            )

        current = self._configuration.laser
        backend = current.backend
        if backend != "disabled" and not channels:
            raise ValueError(f"laser backend '{backend}' requires at least one complete laser mapping")
        return LaserSystemConfiguration.from_channels(
            channels,
            hardware_timed=current.hardware_timed,
            sample_rate_hz=current.sample_rate_hz,
            backend=backend,
            pmt_shutter_output=current.pmt_shutter_output,
            trigger_listener_inputs=current.trigger_listener_inputs,
        )

    def _set_all_combos_enabled(self, enabled: bool) -> None:
        for combo in self._general_combos.values():
            combo.setEnabled(enabled)
        for combos in self._laser_combos.values():
            for combo in combos.values():
                combo.setEnabled(enabled)

    def _set_ok_enabled(self, enabled: bool) -> None:
        button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if button is not None:
            button.setEnabled(enabled)

    def _selected_device(self) -> Optional[NidaqDevicePorts]:
        return self._devices.get(self._device_combo.currentData())

    def _options_for_kind(self, device: NidaqDevicePorts, kind: str) -> Tuple[str, ...]:
        if kind == "ao":
            return device.analog_outputs
        if kind == "ai":
            return device.analog_inputs
        if kind == "do":
            return device.digital_outputs
        if kind == "di":
            return device.digital_inputs
        raise ValueError(f"Unsupported NI-DAQ channel kind: {kind}")

    def _set_combo_options(
        self,
        combo: QComboBox,
        options: Iterable[str],
        current_value: Optional[str],
    ) -> Optional[str]:
        options = tuple(options)
        missing_current_value = current_value if current_value and current_value not in options else None
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("", None)
            for option in options:
                combo.addItem(option, option)
            index = combo.findData(current_value)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)
        return missing_current_value

    def _combo_value(self, combo: QComboBox) -> Optional[str]:
        value = combo.currentData()
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return None

    def _selected_channel_entries(self) -> Tuple[Tuple[str, str, str], ...]:
        entries = []
        for combo, kind in self._combo_kinds.items():
            value = self._combo_value(combo)
            if value is not None:
                entries.append((self._combo_role_names[combo], kind, value))
        return tuple(entries)

    def _unsupported_selected_channels(self, device: NidaqDevicePorts) -> List[str]:
        unsupported = []
        for role_name, kind, value in self._selected_channel_entries():
            if value not in self._options_for_kind(device, kind):
                unsupported.append(f"{role_name}={value}")
        return unsupported

    def _duplicate_selected_channels(self) -> List[str]:
        entries = self._selected_channel_entries()
        counts = Counter(value for _role_name, _kind, value in entries)
        duplicates = []
        for value, count in counts.items():
            if count <= 1:
                continue
            roles = [
                role_name
                for role_name, _kind, channel_value in entries
                if channel_value == value
            ]
            duplicates.append(f"{value} ({', '.join(roles)})")
        return duplicates

    def _validate_selected_channel_assignments(self, device: NidaqDevicePorts) -> None:
        unsupported = self._unsupported_selected_channels(device)
        if unsupported:
            raise ValueError(
                "Selected channel(s) are not available for their required type on this device: "
                + ", ".join(unsupported)
            )
        duplicates = self._duplicate_selected_channels()
        if duplicates:
            raise ValueError("Duplicate channel assignment(s) are not allowed: " + ", ".join(duplicates))

    def _infer_configured_device_name(self) -> Optional[str]:
        channels = []
        ports = self._configuration.nidaq_ports
        for attr_name, _label, _kind in _GENERAL_ROLES:
            channels.append(getattr(ports, attr_name))
        for channel in self._configuration.laser.channels:
            channels.extend(
                (
                    channel.analog_output,
                    channel.diode_input,
                    channel.shutter_output,
                    channel.command_copy_input,
                )
            )
        for channel_name in channels:
            device_name = device_name_from_channel(channel_name)
            if device_name in self._devices:
                return device_name
        return None
