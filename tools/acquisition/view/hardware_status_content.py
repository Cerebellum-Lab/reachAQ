from functools import partial
import math
from typing import Dict, Optional

from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from autotrainer.core import MessageHandler, Offset3DTuple
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import CanTransportConfiguration, LaserFeedbackSample
from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from autotrainer.pyside.xyz_label import XYZQLabel

from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.laser_model import LaserModel
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel


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

    def __init__(self, app_model):
        super().__init__()

        self._app_model = app_model
        self._message_handler = app_model.message_handler
        self._camera_status_labels: Dict[object, QLabel] = {}

        self._card_widget = CardWidget(title="Hardware Status")
        self._message_handler.property_changed += self._model_property_changed

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        layout = self._grid_layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        content_layout.addLayout(layout)

        self._row = 0
        self._add_camera_rows()
        self._add_daq_rows()
        self._add_can_rows()
        self._add_pellet_rows()
        self._add_laser_rows()

        content = QWidget()
        content.setContentsMargins(0, 0, 0, 0)
        content.setLayout(content_layout)
        self._card_widget.setContentWidget(content)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._card_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        def xyz_update(xyz_label: XYZQLabel, coord: str, value):
            algo = app_model.behavior.algorithm
            cfg = algo.diamond_triangle_config
            coord_idx = "xyz".index(coord)
            t = [0, 0, 0]
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
        self.load_arm_changed.connect(lambda x: self._load_arm.setText(self._format_number(x, "deg")))
        self.cover_arm_changed.connect(lambda x: self._cover_arm.setText(self._format_number(x, "deg")))

        for camera in app_model.cameras:
            camera.property_changed += self._on_camera_property_changed
        app_model.hardware.property_changed += self._on_hardware_model_property_changed
        app_model.laser.property_changed += self._on_laser_property_changed
        app_model.nidaq_signal_monitor.property_changed += self._on_nidaq_property_changed
        app_model.configuration_loaded_event += self._on_configuration_loaded

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(1000)
        self._refresh_status()

    def _add_section(self, title: str) -> QLabel:
        label = QLabel(f"<b>{title}</b>")
        label.setContentsMargins(0, 6, 0, 0)
        self._grid_layout.addWidget(label, self._row, 0, 1, 2)
        self._row += 1
        return label

    def _add_row(self, label_text: str, value_widget: Optional[QLabel] = None) -> QLabel:
        self._grid_layout.addWidget(QLabel(label_text), self._row, 0)
        label = value_widget or QLabel("(unknown)")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._grid_layout.addWidget(label, self._row, 1)
        self._row += 1
        return label

    def _add_camera_rows(self) -> None:
        self._add_section("Cameras")
        for camera in self._app_model.cameras:
            label = self._add_row(f"{camera.name}:")
            self._camera_status_labels[camera] = label

    def _add_daq_rows(self) -> None:
        self._add_section("NI-DAQ")
        self._daq_use = self._add_row("In use:")
        self._daq_device = self._add_row("Device:")
        self._daq_stream = self._add_row("Signal stream:")
        self._daq_error = self._add_row("Error:")

    def _add_can_rows(self) -> None:
        self._add_section("CAN Bus")
        self._can_use = self._add_row("In use:")
        self._can_transport = self._add_row("Adapter:")
        self._can_connection = self._add_row("Connection:")
        self._can_ack = self._add_row("ACK timeout:")

    def _add_pellet_rows(self) -> None:
        self._add_section("Pellet Controller")
        self._pellet_use = self._add_row("In use:")
        self._pellet_connection = self._add_row("Controller:")
        self._pellet_version = self._add_row("Firmware:")
        self._pellet_timeout = self._add_row("Status timeout:")
        self._pellet_xyz = XYZQLabel()
        self._add_row("XYZ (mm):", self._pellet_xyz)
        self._pellet_xyz.setObjectName("PelletXYZ")
        self._send_pellet_xyz = XYZQLabel()
        self._add_row("Send XYZ (mm):", self._send_pellet_xyz)
        self._send_pellet_xyz.setObjectName("SendXYZ")
        self._load_arm = self._add_row("Load servo:")
        self._cover_arm = self._add_row("Cover servo:")

        self._tunnel_section_label = self._add_section("Tunnel / Headfix")
        self._head_magnet_label = QLabel("Head magnet (%):")
        self._grid_layout.addWidget(self._head_magnet_label, self._row, 0)
        self._head_magnet = QLabel("(no updates)")
        self._grid_layout.addWidget(self._head_magnet, self._row, 1)
        self._row += 1

    def _add_laser_rows(self) -> None:
        self._add_section("Laser")
        self._laser_use = self._add_row("In use:")
        self._laser_backend = self._add_row("Backend:")
        self._laser_connection = self._add_row("Connection:")
        self._laser_channels = self._add_row("Channels:")
        self._laser_feedback = self._add_row("Last feedback:")

    def _refresh_status(self) -> None:
        self._refresh_camera_status()
        self._refresh_daq_status()
        self._refresh_can_status()
        self._refresh_pellet_status()
        self._refresh_laser_status()
        self._update_tunnel_headfix_visibility(self._app_model.hardware.tunnel_headfix_enabled)

    def _refresh_camera_status(self) -> None:
        for camera, label in self._camera_status_labels.items():
            if not camera.is_enabled:
                self._set_label_status(label, "Not in use", "disabled")
                continue
            status = camera.capture_process_status
            if status == CaptureProcessStatus.RUNNING:
                self._set_label_status(label, "Running", "ok")
            elif status == CaptureProcessStatus.FAILED:
                self._set_label_status(label, "Failed", "error")
            elif status == CaptureProcessStatus.UNKNOWN:
                self._set_label_status(label, "Enabled, idle", "idle")
            else:
                self._set_label_status(label, status.name.replace("_", " ").title(), "idle")

    def _refresh_daq_status(self) -> None:
        hardware = self._app_model.hardware
        monitor = self._app_model.nidaq_signal_monitor
        is_enabled = hardware.nidaq_enabled
        self._set_label_status(self._daq_use, self._yes_no(is_enabled), "ok" if is_enabled else "disabled")
        self._daq_device.setText(self._daq_device_name() if is_enabled else "Not in use")
        if not is_enabled:
            self._set_label_status(self._daq_stream, "Not in use", "disabled")
            self._set_label_status(self._daq_error, "", "disabled")
            return
        self._set_label_status(
            self._daq_stream,
            monitor.status_message,
            "ok" if monitor.is_running else "idle",
        )
        self._set_label_status(self._daq_error, monitor.error_message or "", "error" if monitor.error_message else "ok")

    def _refresh_can_status(self) -> None:
        hardware = self._app_model.hardware
        is_enabled = hardware.can_enabled
        self._set_label_status(self._can_use, self._yes_no(is_enabled), "ok" if is_enabled else "disabled")
        self._can_transport.setText(self._can_transport_text() if is_enabled else "Not in use")
        if not is_enabled:
            self._set_label_status(self._can_connection, "Not in use", "disabled")
            self._set_label_status(self._can_ack, "Not in use", "disabled")
            return
        elif hardware.connected:
            self._set_label_status(self._can_connection, "Connected", "ok")
        else:
            self._set_label_status(self._can_connection, "Enabled, idle", "idle")
        self._set_label_status(
            self._can_ack,
            "Timed out" if hardware.device_ack_timeout_engaged else "OK",
            "error" if hardware.device_ack_timeout_engaged else "ok",
        )

    def _refresh_pellet_status(self) -> None:
        hardware = self._app_model.hardware
        is_enabled = hardware.pellet_controller_enabled
        self._set_label_status(self._pellet_use, self._yes_no(is_enabled), "ok" if is_enabled else "disabled")
        if not is_enabled:
            self._set_label_status(self._pellet_connection, "Not in use", "disabled")
            self._set_label_status(self._pellet_version, "Not in use", "disabled")
            self._set_label_status(self._pellet_timeout, "Not in use", "disabled")
            return
        self._set_label_status(
            self._pellet_connection,
            "Connected" if hardware.connected else "Enabled, idle",
            "ok" if hardware.connected else "idle",
        )
        if self._pellet_version.text() in {"", "(unknown)", "Not in use"}:
            self._set_label_status(self._pellet_version, "(unknown)", "idle")
        self._set_label_status(
            self._pellet_timeout,
            "Timed out" if hardware.pellet_status_timeout_engaged else "OK",
            "error" if hardware.pellet_status_timeout_engaged else "ok",
        )
        self._load_arm.setText(self._format_number(hardware.load_arm_position, "deg"))
        self._cover_arm.setText(self._format_number(hardware.cover_arm_position, "deg"))

    def _refresh_laser_status(self) -> None:
        laser = self._app_model.laser
        configuration = laser.configuration
        is_enabled = configuration.backend != "disabled"
        self._set_label_status(self._laser_use, self._yes_no(is_enabled), "ok" if is_enabled else "disabled")
        self._laser_backend.setText(configuration.backend)
        if not is_enabled:
            self._set_label_status(self._laser_connection, "Not in use", "disabled")
            self._laser_channels.setText("0")
            self._laser_feedback.setText("")
            return
        self._set_label_status(
            self._laser_connection,
            "Connected" if laser.is_connected else "Configured, not connected",
            "ok" if laser.is_connected else "idle",
        )
        self._laser_channels.setText(str(len(configuration.channels)))
        self._laser_feedback.setText(self._format_laser_feedback(laser.last_feedback_sample))

    def _daq_device_name(self) -> str:
        nidaq_ports = self._app_model.nidaq_ports
        if nidaq_ports.device_name:
            return nidaq_ports.device_name
        for channel in self._app_model.nidaq_signal_monitor.configuration.channels:
            device_name = self._device_name_from_channel(channel.physical_channel)
            if device_name:
                return device_name
        for channel in self._app_model.laser.configuration.channels:
            for physical_channel in (
                channel.analog_output,
                channel.diode_input,
                channel.shutter_output,
                channel.command_copy_input,
            ):
                device_name = self._device_name_from_channel(physical_channel)
                if device_name:
                    return device_name
        return "(not configured)"

    def _can_transport_text(self) -> str:
        try:
            transport = CanTransportConfiguration.from_environment()
        except Exception as exc:
            return f"Config error: {exc}"
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
    def _format_laser_feedback(sample: Optional[LaserFeedbackSample]) -> str:
        if sample is None:
            return ""
        command_copy = "" if sample.command_copy_volts is None else f", copy {sample.command_copy_volts:.3f} V"
        return (
            f"laser {sample.channel_id.value}: command {sample.command_volts:.3f} V, "
            f"diode {sample.diode_volts:.3f} V{command_copy}"
        )

    @staticmethod
    def _format_number(value: float, suffix: str = "") -> str:
        if value is None or math.isnan(value):
            return "(no updates)"
        text = f"{value:.1f}"
        return text if not suffix else f"{text} {suffix}"

    @staticmethod
    def _yes_no(value: bool) -> str:
        return "Yes" if value else "No"

    def _set_label_status(self, label: QLabel, text: str, state: str) -> None:
        label.setText(text)
        if state == "ok":
            label.setStyleSheet("color: #1b6e3c;")
        elif state == "error":
            label.setStyleSheet("color: #b00020;")
        elif state == "disabled":
            label.setStyleSheet("color: #666;")
        else:
            label.setStyleSheet("color: #7a5c00;")

    def _update_tunnel_headfix_visibility(self, is_enabled: bool):
        self._tunnel_section_label.setVisible(is_enabled)
        self._head_magnet_label.setVisible(is_enabled)
        self._head_magnet.setVisible(is_enabled)

    @invoke_method
    def _on_camera_property_changed(self, _property_name: str, _value, _):
        self._refresh_camera_status()

    @invoke_method
    def _on_hardware_model_property_changed(self, property_name: str, value, _):
        if property_name == HardwareModel.TUNNEL_HEADFIX_ENABLED:
            self._update_tunnel_headfix_visibility(value)
        self._refresh_can_status()
        self._refresh_pellet_status()
        self._refresh_daq_status()

    @invoke_method
    def _on_laser_property_changed(self, property_name: str, _value, _):
        if property_name in (LaserModel.CONFIGURATION, LaserModel.IS_CONNECTED, LaserModel.LAST_FEEDBACK_SAMPLE):
            self._refresh_laser_status()

    @invoke_method
    def _on_nidaq_property_changed(self, property_name: str, _value, _):
        if property_name in (
            NidaqSignalMonitorModel.CONFIGURATION,
            NidaqSignalMonitorModel.IS_RUNNING,
            NidaqSignalMonitorModel.STATUS_MESSAGE,
            NidaqSignalMonitorModel.ERROR_MESSAGE,
        ):
            self._refresh_daq_status()

    @invoke_method
    def _on_configuration_loaded(self, _configuration):
        self._refresh_status()

    @invoke_method
    def _model_property_changed(self, property_name: str, value, _):
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
