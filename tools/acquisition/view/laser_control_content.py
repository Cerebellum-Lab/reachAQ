from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import LaserCalibrationRamp, LaserChannelConfiguration, LaserPulseTrain
from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.laser_model import LaserModel


logger = get_verbose_logger(__name__)


class _LaserOperationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Callable[[], object]):
        super().__init__()
        self._operation = operation

    @Slot()
    def run(self):
        try:
            self.finished.emit(self._operation())
        except Exception as exc:
            logger.exception("Laser operation failed")
            message = str(exc) or exc.__class__.__name__
            self.failed.emit(message)


class LaserControlContent(ContentWidget):
    """Thin operator controls for the laser model."""

    def __init__(self, app_model: AppModel):
        super().__init__()

        self._app_model = app_model
        self._is_editable = True
        self._is_capture_active = False
        self._operation_thread: Optional[QThread] = None
        self._operation_worker: Optional[_LaserOperationWorker] = None

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(QLabel("Backend:"))
        self._backend_label = QLabel("disabled")
        header_layout.addWidget(self._backend_label)
        header_layout.addWidget(QLabel("Rate:"))
        self._sample_rate_label = QLabel("manual")
        header_layout.addWidget(self._sample_rate_label)

        self._card_widget = CardWidget(title="Laser Control", header_right_layout=header_layout)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        selector_layout = QFormLayout()
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setHorizontalSpacing(8)
        self._channel_combo = QComboBox()
        selector_layout.addRow("Channel:", self._channel_combo)
        layout.addLayout(selector_layout)

        pulse_group = QGroupBox("Pulse Train")
        pulse_layout = QGridLayout(pulse_group)
        pulse_layout.setContentsMargins(8, 6, 8, 8)
        pulse_layout.setHorizontalSpacing(8)
        pulse_layout.setVerticalSpacing(4)

        self._amplitude = self._make_voltage_spinbox()
        self._duration_ms = self._make_ms_spinbox(10.0)
        self._baseline_ms = self._make_ms_spinbox(0.0)
        self._post_stim_ms = self._make_ms_spinbox(0.0)
        self._pulse_count = QSpinBox()
        self._pulse_count.setRange(1, 100000)
        self._pulse_count.setValue(1)
        self._frequency_hz = QDoubleSpinBox()
        self._frequency_hz.setRange(0.1, 100000.0)
        self._frequency_hz.setDecimals(3)
        self._frequency_hz.setValue(10.0)
        self._frequency_hz.setSuffix(" Hz")
        self._trigger_source = QLineEdit()
        self._trigger_source.setPlaceholderText("optional NI-DAQ trigger route")
        self._trigger_edge = QComboBox()
        self._trigger_edge.addItems(("rising", "falling"))

        self._open_shutter = QCheckBox("Open shutter")
        self._open_shutter.setChecked(True)
        self._close_shutter = QCheckBox("Close shutter")
        self._close_shutter.setChecked(True)
        self._enable_pmt = QCheckBox("PMT shutter")
        self._emit_trigger = QCheckBox("Trigger DO")
        self._emit_timing_trigger = QCheckBox("Timing DO")

        pulse_layout.addWidget(QLabel("Amplitude:"), 0, 0)
        pulse_layout.addWidget(self._amplitude, 0, 1)
        pulse_layout.addWidget(QLabel("Duration:"), 0, 2)
        pulse_layout.addWidget(self._duration_ms, 0, 3)
        pulse_layout.addWidget(QLabel("Baseline:"), 1, 0)
        pulse_layout.addWidget(self._baseline_ms, 1, 1)
        pulse_layout.addWidget(QLabel("Post-stim:"), 1, 2)
        pulse_layout.addWidget(self._post_stim_ms, 1, 3)
        pulse_layout.addWidget(QLabel("Count:"), 2, 0)
        pulse_layout.addWidget(self._pulse_count, 2, 1)
        pulse_layout.addWidget(QLabel("Frequency:"), 2, 2)
        pulse_layout.addWidget(self._frequency_hz, 2, 3)
        pulse_layout.addWidget(QLabel("Trigger:"), 3, 0)
        pulse_layout.addWidget(self._trigger_source, 3, 1, 1, 2)
        pulse_layout.addWidget(self._trigger_edge, 3, 3)
        pulse_layout.addWidget(self._open_shutter, 4, 0)
        pulse_layout.addWidget(self._close_shutter, 4, 1)
        pulse_layout.addWidget(self._enable_pmt, 4, 2)
        pulse_layout.addWidget(self._emit_trigger, 5, 0)
        pulse_layout.addWidget(self._emit_timing_trigger, 5, 1)
        self._run_pulse_button = QPushButton("Run Pulse")
        pulse_layout.addWidget(self._run_pulse_button, 5, 3)
        layout.addWidget(pulse_group)

        ramp_group = QGroupBox("Calibration Ramp")
        ramp_layout = QGridLayout(ramp_group)
        ramp_layout.setContentsMargins(8, 6, 8, 8)
        ramp_layout.setHorizontalSpacing(8)
        ramp_layout.setVerticalSpacing(4)

        self._ramp_start = self._make_voltage_spinbox()
        self._ramp_stop = self._make_voltage_spinbox()
        self._ramp_steps = QSpinBox()
        self._ramp_steps.setRange(2, 10000)
        self._ramp_steps.setValue(11)
        self._ramp_samples_per_step = QSpinBox()
        self._ramp_samples_per_step.setRange(1, 1000000)
        self._ramp_samples_per_step.setValue(100)
        self._ramp_pmt = QCheckBox("PMT shutter")
        self._run_ramp_button = QPushButton("Run Ramp")

        ramp_layout.addWidget(QLabel("Start:"), 0, 0)
        ramp_layout.addWidget(self._ramp_start, 0, 1)
        ramp_layout.addWidget(QLabel("Stop:"), 0, 2)
        ramp_layout.addWidget(self._ramp_stop, 0, 3)
        ramp_layout.addWidget(QLabel("Steps:"), 1, 0)
        ramp_layout.addWidget(self._ramp_steps, 1, 1)
        ramp_layout.addWidget(QLabel("Samples/step:"), 1, 2)
        ramp_layout.addWidget(self._ramp_samples_per_step, 1, 3)
        ramp_layout.addWidget(self._ramp_pmt, 2, 0)
        ramp_layout.addWidget(self._run_ramp_button, 2, 3)
        layout.addWidget(ramp_group)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(8)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._status_label = QLabel("Laser controller not configured")
        footer_layout.addWidget(self._progress)
        footer_layout.addWidget(self._status_label, stretch=1)

        self._card_widget.setContentWidget(content)
        self._card_widget.footer.setContent(footer)

        root_layout = QVBoxLayout()
        root_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._card_widget)
        self.setLayout(root_layout)

        self._controls = (
            self._channel_combo,
            self._amplitude,
            self._duration_ms,
            self._baseline_ms,
            self._post_stim_ms,
            self._pulse_count,
            self._frequency_hz,
            self._trigger_source,
            self._trigger_edge,
            self._open_shutter,
            self._close_shutter,
            self._enable_pmt,
            self._emit_trigger,
            self._emit_timing_trigger,
            self._ramp_start,
            self._ramp_stop,
            self._ramp_steps,
            self._ramp_samples_per_step,
            self._ramp_pmt,
        )

        self._channel_combo.currentIndexChanged.connect(self._selected_channel_changed)
        self._run_pulse_button.clicked.connect(self._run_pulse)
        self._run_ramp_button.clicked.connect(self._run_calibration_ramp)
        app_model.laser.property_changed += self._on_laser_property_changed

        self._refresh_from_model()

    @staticmethod
    def _make_voltage_spinbox() -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(3)
        spinbox.setSingleStep(0.050)
        spinbox.setSuffix(" V")
        spinbox.setRange(0.0, 10.0)
        return spinbox

    @staticmethod
    def _make_ms_spinbox(value: float) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(3)
        spinbox.setSingleStep(1.0)
        spinbox.setSuffix(" ms")
        spinbox.setRange(0.0, 600000.0)
        spinbox.setValue(value)
        return spinbox

    @invoke_method
    def _on_laser_property_changed(self, property_name: str, _value, _old_value):
        if property_name in (LaserModel.CONFIGURATION, LaserModel.IS_CONNECTED):
            self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        configuration = self._app_model.laser.configuration
        current = self._selected_channel_value()
        self._backend_label.setText(configuration.backend)
        if configuration.sample_rate_hz is None:
            self._sample_rate_label.setText("manual")
        else:
            self._sample_rate_label.setText(f"{configuration.sample_rate_hz:g} Hz")

        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for channel in configuration.channels:
            self._channel_combo.addItem(f"Laser {channel.channel_id.value}", int(channel.channel_id))
        if current is not None:
            index = self._channel_combo.findData(current)
            if index >= 0:
                self._channel_combo.setCurrentIndex(index)
        self._channel_combo.blockSignals(False)
        self._selected_channel_changed()
        self._update_enabled_state()

    def _selected_channel_value(self) -> Optional[int]:
        data = self._channel_combo.currentData()
        return None if data is None else int(data)

    def _selected_channel(self) -> Optional[LaserChannelConfiguration]:
        channel_id = self._selected_channel_value()
        if channel_id is None:
            return None
        for channel in self._app_model.laser.configuration.channels:
            if int(channel.channel_id) == channel_id:
                return channel
        return None

    def _selected_channel_changed(self, *_args) -> None:
        channel = self._selected_channel()
        if channel is None:
            self._set_status("Laser controller not configured", is_error=False)
            return
        minimum = channel.minimum_command_volts
        maximum = channel.maximum_command_volts
        for spinbox in (self._amplitude, self._ramp_start, self._ramp_stop):
            spinbox.setRange(minimum, maximum)
        self._amplitude.setValue(min(max(self._amplitude.value(), minimum), maximum))
        self._ramp_start.setValue(minimum)
        self._ramp_stop.setValue(maximum)
        self._trigger_source.setText(channel.trigger_source or "")
        self._set_status(f"Ready: laser {channel.channel_id.value}", is_error=False)

    def _run_pulse(self) -> None:
        channel = self._selected_channel()
        if channel is None:
            self._set_status("No laser channel is configured", is_error=True)
            return
        pulse_count = self._pulse_count.value()
        frequency_hz = self._frequency_hz.value() if pulse_count > 1 else None
        trigger_source = self._trigger_source.text().strip() or None
        pulse_train = LaserPulseTrain(
            channel_id=channel.channel_id,
            amplitude_volts=self._amplitude.value(),
            duration_ms=self._duration_ms.value(),
            baseline_ms=self._baseline_ms.value(),
            post_stim_ms=self._post_stim_ms.value(),
            pulse_count=pulse_count,
            frequency_hz=frequency_hz,
            trigger_source=trigger_source,
            trigger_edge=self._trigger_edge.currentText(),
            open_shutter=self._open_shutter.isChecked(),
            close_shutter=self._close_shutter.isChecked(),
            enable_pmt_shutter=self._enable_pmt.isChecked(),
            emit_trigger_output=self._emit_trigger.isChecked(),
            emit_timing_trigger_output=self._emit_timing_trigger.isChecked(),
        )

        def operation():
            self._app_model.laser.run_pulse_train(pulse_train)
            return f"Pulse complete: laser {channel.channel_id.value}"

        self._start_operation("Running laser pulse train", operation)

    def _run_calibration_ramp(self) -> None:
        channel = self._selected_channel()
        if channel is None:
            self._set_status("No laser channel is configured", is_error=True)
            return
        ramp = LaserCalibrationRamp(
            channel_id=channel.channel_id,
            start_volts=self._ramp_start.value(),
            stop_volts=self._ramp_stop.value(),
            steps=self._ramp_steps.value(),
            samples_per_step=self._ramp_samples_per_step.value(),
            enable_pmt_shutter=self._ramp_pmt.isChecked(),
        )

        def operation():
            points = self._app_model.laser.run_calibration_ramp(ramp)
            self._app_model.laser.make_diode_power_curve(points)
            last = points[-1]
            return (
                f"Ramp complete: {len(points)} points, last diode {last.diode_volts:.3f} V, "
                "monotonic curve validated"
            )

        self._start_operation("Running laser calibration ramp", operation)

    def _start_operation(self, status: str, operation: Callable[[], object]) -> None:
        if self._operation_thread is not None:
            self._set_status("Laser operation already in progress", is_error=True)
            return
        logger.verbose(status)
        self._set_status(status, is_error=False)
        self._progress.setVisible(True)
        self._set_running(True)
        thread = QThread(self)
        worker = _LaserOperationWorker(operation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._operation_finished)
        worker.failed.connect(self._operation_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._operation_thread_finished)
        self._operation_thread = thread
        self._operation_worker = worker
        thread.start()

    @Slot(object)
    def _operation_finished(self, result) -> None:
        self._set_status(str(result), is_error=False)

    @Slot(str)
    def _operation_failed(self, message: str) -> None:
        self._set_status(message, is_error=True)

    @Slot()
    def _operation_thread_finished(self) -> None:
        self._operation_thread = None
        self._operation_worker = None
        self._progress.setVisible(False)
        self._set_running(False)

    def _set_status(self, message: str, *, is_error: bool) -> None:
        self._status_label.setText(message)
        if is_error:
            self._status_label.setStyleSheet("color: #b00020;")
        else:
            self._status_label.setStyleSheet("")

    def _set_running(self, is_running: bool) -> None:
        self._update_enabled_state(is_running=is_running)

    def _update_enabled_state(self, *, is_running: Optional[bool] = None) -> None:
        if is_running is None:
            is_running = self._operation_thread is not None
        has_channel = self._selected_channel() is not None
        ready = self._is_editable and self._app_model.laser.is_connected and has_channel and not is_running
        for control in self._controls:
            control.setEnabled(ready)
        self._run_pulse_button.setEnabled(ready)
        self._run_ramp_button.setEnabled(ready and not self._is_capture_active)

    @invoke_method
    def set_is_editable(self, is_editable: bool):
        self._is_editable = is_editable
        self._update_enabled_state()

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        self._is_capture_active = is_active
        self._update_enabled_state()
