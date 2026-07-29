
import logging
import qtawesome as qta
from typing import Tuple

from PySide6.QtCore import Qt, Signal, QTimer, Slot
from PySide6.QtWidgets import (QWidget, QGridLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
                               QCheckBox, QLineEdit, QFileDialog, QPlainTextEdit, QVBoxLayout, QDoubleSpinBox)

from autotrainer.core import PerfMonitor, ProjectInfo, MessageHandler
from autotrainer.core.message import Motor
from autotrainer.device import is_servo
from autotrainer.model import EnvironmentProvider, HardwareVersion
from autotrainer.pyside import PGWidget, CardWidget, TextBoxHandler

from autotrainer.pyside import ConnectionPanel
from autotrainer.pyside import MotorConfigDialog
from tools.head_fix.model.app_model import AppModel

logger = logging.getLogger(__name__)


class MainContent(QWidget):
    connecting = Signal()
    disconnected = Signal()

    def __init__(self, model: AppModel):
        super().__init__()

        self._model = model

        self._config_dialog = None
        self._is_diagnostics_visible = True

        self._plots_by_name = {}

        layout = QVBoxLayout()

        self._connection_panel = ConnectionPanel(model)
        self._connection_panel.connecting.connect(self._connected)
        self._connection_panel.disconnected.connect(self._disconnected)
        layout.addWidget(self._connection_panel)

        self._head_control = self._create_control_panel()
        layout.addWidget(self._head_control)

        self._tone_control = self._create_tone_panel()
        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        layout.addWidget(self._tone_control)

        layout.addWidget(self._create_sensor_panel())

        layout.addWidget(self._create_diagnostics_panel())

        layout.setStretch(3, 1)

        self._layout = layout

        self.setLayout(layout)

        # The combination of PySide6/Qt signals and slots and the Jetson has reasonable performance, but with a basic
        # implementation does not keep up with 100Hz updates.  We don't need to see 100Hz updates on screen, so rather
        # than optimize that pipeline, just use a timer to refresh the data at an acceptable rate.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_data)
        self._timer.start(100)

        self._perf_monitor = PerfMonitor(name="<HeadFixUI>", units="mps", report_window=30)

        handler = TextBoxHandler(self._diagnostics)
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s: %(levelname)s: %(name)s: %(message)s"))
        logging.getLogger("autotrainer").addHandler(handler)

    @Slot()
    def refresh_data(self):
        for name, plot in self._plots_by_name.items():
            try:
                plot.use_cache()
            except Exception as err:
                logger.exception("%s failed: %s", name, err)

    @property
    def is_diagnostics_visible(self) -> bool:
        return self._is_diagnostics_visible

    def on_activated(self):
        self._model.property_changed += self._model_property_changed
        self._model.message_handler.measurement_callback = self._measurements_received
        self._model.message_handler.audio_callback = self._audio_spectrum_received
        self._model.analysis.property_changed += self._analysis_property_changed

    def set_diagnostics_visible(self, is_visible: bool):
        self._diagnostics_panel.setVisible(is_visible)
        self._is_diagnostics_visible = is_visible

        if is_visible:
            self._layout.setStretch(3, 1)
            self._layout.setStretch(2, 0)
        else:
            self._layout.setStretch(3, 0)
            self._layout.setStretch(2, 1)

    def _create_control_panel(self):
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(32)

        position_layout = QHBoxLayout()
        position_layout.setContentsMargins(8, 8, 8, 8)
        position_layout.setSpacing(12)

        position_layout.addWidget(QLabel("Magnet"))

        self._curr_magnet_position = QLabel("(-)")
        position_layout.addWidget(self._curr_magnet_position)

        self._magnet_position = QSpinBox()
        self._magnet_position.setMaximum(120)
        self._magnet_position.setValue(0)
        self._magnet_position.setWrapping(False)
        self._magnet_position.setEnabled(False)
        position_layout.addWidget(self._magnet_position, 0, Qt.AlignLeft)

        self._update_magnet_button = QPushButton("Update")
        self._update_magnet_button.setEnabled(False)
        self._update_magnet_button.clicked.connect(self._set_magnet_position)
        position_layout.addWidget(self._update_magnet_button, 0)

        position_layout.addStretch(1)

        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        position_layout.addWidget(QLabel("Gate"))

        self._curr_gate_position = QLabel("(-)")

        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        position_layout.addWidget(self._curr_gate_position)

        self._gate_position = QSpinBox()
        self._gate_position.setMaximum(120)
        self._gate_position.setValue(0)
        self._gate_position.setWrapping(False)
        self._gate_position.setEnabled(False)

        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        position_layout.addWidget(self._gate_position, 0, Qt.AlignLeft)

        self._update_gate_button = QPushButton("Update")
        self._update_gate_button.setEnabled(False)
        self._update_gate_button.clicked.connect(self._set_gate_position)

        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        position_layout.addWidget(self._update_gate_button, 0)

        self._open_gate = QPushButton("Open Gate")
        self._open_gate.setEnabled(False)
        self._open_gate.clicked.connect(self._model.open_tunnel_gate)

        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        position_layout.addWidget(self._open_gate, 0)

        self._close_gate = QPushButton("Close Gate")
        self._close_gate.setEnabled(False)
        self._close_gate.clicked.connect(self._model.close_tunnel_gate)

        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        position_layout.addWidget(self._close_gate, 0)

        position_layout.addStretch(1)

        self._tare_button = QPushButton("Tare")
        self._tare_button.setEnabled(False)
        self._tare_button.clicked.connect(self._model.tare)
        position_layout.addWidget(self._tare_button, 0)

        self._config_button = QPushButton("")
        gear_icon = qta.icon('fa5s.cog')  # Font Awesome 5 Solid cog icon
        self._config_button.setIcon(gear_icon)
        self._config_button.clicked.connect(lambda: self._update_config())
        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        position_layout.addWidget(self._config_button)

        row_layout.addLayout(position_layout)

        panel = CardWidget(title="Control", background_color=None, header_background_color="#00b6de")
        panel.setContentLayout(row_layout)

        header = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Control")
        title.setStyleSheet("font-weight: bold; color: white")
        layout.addWidget(title)

        layout.addStretch(1)

        header.setLayout(layout)

        panel.header.setContent(header)

        return panel

    def _create_tone_panel(self):
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(8, 8, 8, 8)
        row_layout.setSpacing(16)

        self._set_tone_button = QPushButton("Tone")
        self._set_tone_button.clicked.connect(self._set_tone)
        row_layout.addWidget(self._set_tone_button, 0)

        self._frequency = QDoubleSpinBox()
        self._frequency.setDecimals(0)
        self._frequency.setMaximum(20000)
        self._frequency.setMinimum(1000)
        self._frequency.setValue(5000)
        self._frequency.setSuffix(" Hz")
        row_layout.addWidget(self._frequency, 0)

        self._duration = QDoubleSpinBox()
        self._duration.setMaximum(60)
        self._duration.setMinimum(1)
        self._duration.setValue(2)
        self._duration.setSuffix(" sec")
        row_layout.addWidget(self._duration, 0)

        row_layout.addStretch(1)

        panel = CardWidget(background_color=None, header_background_color="#00b6de")
        panel.setContentLayout(row_layout)

        title = QLabel("Tone Generator")
        title.setStyleSheet("font-weight: bold; color: white")

        panel.header.setContent(title)

        panel.setEnabled(False)

        return panel

    def _create_sensor_panel(self):
        row_layout = QVBoxLayout()

        record_layout = QHBoxLayout()
        record_layout.setContentsMargins(8, 8, 8, 8)
        record_layout.setSpacing(8)

        self._enable_streaming = QCheckBox("Stream measurements")
        self._enable_streaming.setEnabled(True)
        self._enable_streaming.setChecked(self._model.user_settings.stream_enabled)
        self._enable_streaming.stateChanged.connect(lambda x: self._enable_data_stream(x))
        record_layout.addWidget(self._enable_streaming)

        self._record = QCheckBox("Save measurements")
        self._record.setChecked(self._model.user_settings.record_enabled)
        self._record.stateChanged.connect(lambda x: self._update_record_enabled(x))
        record_layout.addWidget(self._record)

        self._record_location = QLineEdit(self._model.user_settings.record_location)
        self._record_location.setMinimumWidth(100)
        record_layout.addWidget(self._record_location, 1)

        self._browse_button = QPushButton("Select...")
        self._browse_button.clicked.connect(self._browse_for_location)
        record_layout.addWidget(self._browse_button)

        row_layout.addLayout(record_layout, 1.0)

        plot_layout = QGridLayout()
        plot_layout.setContentsMargins(8, 0, 8, 0)

        self._headbar_pressure_plot, widget = self._create_plot_widget("Head Bar Pressure (cnts)")
        # self._headbar_pressure_plot.setYRange(-1.0, 1024)
        plot_layout.addWidget(widget, 0, 1)

        self._head_contact_plot, widget = self._create_plot_widget("Head Contact (On/Off)")
        self._head_contact_plot.setYRange(-1, 2)
        plot_layout.addWidget(widget, 2, 0)

        self._audio_spectrum_plot, widget = self._create_plot_widget("Audio Spectrum (dB)")
        assert isinstance(self._audio_spectrum_plot, PGWidget)
        self._audio_spectrum_plot.setXRange(min=0, max=64)
        self._audio_spectrum_plot.setYRange(min=0, max=200)
        ticks = []
        for i in range(0, 64, 10):  # Step through indices
            ticks.append((i, str(i * 1500)))  # Map index to actual x value

        # Set the custom ticks
        ax = self._audio_spectrum_plot.getAxis('bottom')
        ax.setTicks([ticks])

        # if EnvironmentProvider.hardware_version() != HardwareVersion.ANSHUTZ:
        plot_layout.addWidget(widget, 2, 1)

        #

        self._temperature_plot, widget = self._create_plot_widget("Temperature (\u00b0C)")
        self._temperature_plot.setYRange(0, 50)
        plot_layout.addWidget(widget, 3, 0)

        self._humidity_plot, widget = self._create_plot_widget("Relative Humidity (%)")
        self._humidity_plot.setYRange(0, 100)
        plot_layout.addWidget(widget, 3, 1)

        row_layout.addLayout(plot_layout, 1.0)

        panel = CardWidget(background_color=None, header_background_color="#00b6de")
        panel.setContentLayout(row_layout)
        panel.header.setTitle("Sensor Data", "white")

        return panel

    # noinspection PyMethodMayBeStatic

    def _create_plot_widget(self, title: str) -> Tuple[PGWidget, QWidget]:
        plot = PGWidget()
        plot.setBackground(None)
        plot.getPlotItem().getViewBox().setBackgroundColor((220, 220, 220))
        plot.setMaximumHeight(150)
        plot.setTitle(title)

        self._plots_by_name[title] = plot

        # pyqtgraph is not always well-behaved with layouts.  Rather than fight it, use a QWidget for normal behavior.
        widget = QWidget()
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(plot)

        return plot, widget

    def _create_diagnostics_panel(self):
        self._diagnostics = QPlainTextEdit()
        self._diagnostics.setReadOnly(True)
        self._diagnostics.setStyleSheet("border: none")

        panel = CardWidget(header_background_color="#00b6de")
        panel.setContentWidget(self._diagnostics)
        panel.header.setTitle("Logs", "white")

        self._diagnostics_panel = panel

        return panel

    def _model_property_changed(self, name, value, _):
        if name == MessageHandler.HEAD_MAGNET_INTENSITY_PROPERTY:
            self._curr_magnet_position.setText(f"({round(value, 1)})")
        elif name == MessageHandler.HEAD_GATE_PROPERTY:
            self._curr_gate_position.setText(f"({round(value, 1)})")
        elif name == MessageHandler.CONFIG_PROPERTY:
            if self._config_dialog is not None:
                if is_servo(value.motor):
                    self._config_dialog.update_servo_config(value)
                else:
                    self._config_dialog.update_stepper_config(value)

    def _update_config(self):
        self._config_dialog = MotorConfigDialog(self)
        self._config_dialog.motor_selected.connect(self._on_motor_selected)
        self._config_dialog.accepted.connect(self._on_config_accepted)
        self._config_dialog.rejected.connect(lambda: setattr(self, '_config_dialog', None))

        self._config_dialog.setModal(True)
        self._config_dialog.show()

    def _on_config_accepted(self):
        if self._config_dialog.config is not None:
            self._model.set_config((self._config_dialog.config.motor,
                                    self._config_dialog.config))

        self._config_dialog = None

    def _on_motor_selected(self, motor: Motor):
        self._model.get_config(motor)

    def _analysis_property_changed(self, name, value, _):
        if name == "is_force_detector_engaged":
            if value:
                self._headbar_pressure_plot.getPlotItem().getViewBox().setBackgroundColor(
                    (0, 250, 154))
            else:
                self._headbar_pressure_plot.getPlotItem().getViewBox().setBackgroundColor(
                    (220, 220, 220))

    def _measurements_received(self, measurements):
        # switch, pressure, temperature, humidity
        self._head_contact_plot.cache_data(measurements[0])
        self._headbar_pressure_plot.cache_data(measurements[1])
        self._temperature_plot.cache_data(measurements[2])
        self._humidity_plot.cache_data(measurements[3])

        self._perf_monitor.add_cycles(len(measurements[0]))

    def _audio_spectrum_received(self, spectrum):
        self._audio_spectrum_plot.replace_cache(spectrum)

    def _set_magnet_position(self):
        self._model.set_magnet_position(self._magnet_position.value())

    def _set_gate_position(self):
        self._model.set_gate_position(self._gate_position.value())

    def _update_record_enabled(self, b: bool):
        self._model.user_settings.record_enabled = b

    def _enable_data_stream(self, b: bool):
        self._model.set_stream_enabled(b)

    def _set_tone(self):
        self._model.set_tone(self._frequency.value(), self._duration.value())

    def _connected(self):
        if self._record.isChecked():
            self._model.analysis.project_info = ProjectInfo(
                root=self._record_location.text(),
                device_id="TunnelUI",
                ensure_exists=True)
        else:
            self._model.analysis.project_info = None

        self.enable_widgets(True)
        for plot in self._plots_by_name.values():
            plot.reset()

    def _disconnected(self):
        self.enable_widgets(False)
        self._model.analysis.project_info = None

        self._perf_monitor.reset()
        self._curr_magnet_position.setText("(-)")
        self._curr_gate_position.setText("(-)")

    def enable_widgets(self, enable: bool):
        self._magnet_position.setEnabled(enable)
        self._gate_position.setEnabled(enable)
        self._tare_button.setEnabled(enable)
        self._open_gate.setEnabled(enable)
        self._close_gate.setEnabled(enable)
        self._update_magnet_button.setEnabled(enable)
        self._update_gate_button.setEnabled(enable)
        for name, plot in self._plots_by_name.items():
            plot.setEnabled(enable)
        self._record.setEnabled(not enable)
        self._record_location.setEnabled(not enable)
        self._browse_button.setEnabled(not enable)
        self._head_control.setEnabled(enable)
        self._tone_control.setEnabled(enable)

    def _browse_for_location(self):
        dirname = QFileDialog.getExistingDirectory(self, "Select Directory",
                                                   self._record_location.text())

        if len(dirname) > 0:
            self._record_location.setText(dirname)
            self._model.user_settings.record_location = dirname
