from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import (
    LaserCalibrationRamp,
    LaserChannelConfiguration,
    LaserChannelId,
    LaserPulseTrain,
    NidaqSignalSampleBlock,
)
from autotrainer.pyside import CardWidget, PGWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.laser_model import LaserModel, LaserTraceBlock


logger = get_verbose_logger(__name__)

_LASER_PULSE_TRAIN_COUNT = 4
_DEFAULT_MINIMUM_COMMAND_VOLTS = 0.0
_DEFAULT_MAXIMUM_COMMAND_VOLTS = 5.0
_MAX_LASER_TRACE_POINTS = 100000


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


class _LaserChannelTab(QWidget):
    """Single-laser controls and pulse preview."""

    def __init__(
        self,
        app_model: AppModel,
        channel: LaserChannelConfiguration,
        is_configured: bool,
        sample_rate_hz: Optional[float],
        start_operation: Callable[[str, Callable[[], object]], None],
        set_status: Callable[[str, bool], None],
    ):
        super().__init__()

        self._app_model = app_model
        self._channel = channel
        self._is_configured = is_configured
        self._sample_rate_hz = sample_rate_hz
        self._start_operation = start_operation
        self._set_parent_status = set_status
        self._controls_can_edit = True
        self._trace_streaming = True
        self._trace_data: Dict[str, Tuple[List[float], List[float]]] = {}

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("LaserChannelTab")
        self.setStyleSheet(
            "#LaserChannelTab QLabel {color: #2f343a;}"
            "#LaserChannelTab QLabel#LaserChannelSummary {color: #5b6470;}"
            "#LaserChannelTab QLabel#LaserPreviewStatus {color: #5b6470;}"
            "#LaserChannelTab QGroupBox {"
            "color: #20242a; font-weight: 600; border: 1px solid #d1d5db; "
            "margin-top: 8px; padding-top: 8px;"
            "}"
            "#LaserChannelTab QGroupBox::title {subcontrol-origin: margin; left: 8px; padding: 0px 3px;}"
            "#LaserChannelTab QCheckBox {color: #2f343a; spacing: 4px;}"
            "#LaserChannelTab QCheckBox:disabled {color: #68717d;}"
            "#LaserChannelTab QLineEdit:disabled,"
            "#LaserChannelTab QSpinBox:disabled,"
            "#LaserChannelTab QDoubleSpinBox:disabled,"
            "#LaserChannelTab QComboBox:disabled {color: #4f5965; background-color: #edf0f3;}"
            "#LaserChannelTab QPushButton {min-height: 22px; padding: 2px 8px;}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        sample_rate = "manual" if sample_rate_hz is None else f"{sample_rate_hz:g} Hz"
        if is_configured:
            channel_text = (
                f"Rate {sample_rate} | AO {channel.analog_output} | "
                f"Diode {channel.diode_input} | Shutter {channel.shutter_output}"
            )
        else:
            channel_text = f"Rate {sample_rate} | Hardware channel not mapped"
        channel_label = QLabel(channel_text)
        channel_label.setObjectName("LaserChannelSummary")
        channel_label.setWordWrap(True)
        layout.addWidget(channel_label)

        pulse_group = QGroupBox("Pulse Train")
        pulse_layout = QGridLayout(pulse_group)
        pulse_layout.setContentsMargins(8, 4, 8, 6)
        pulse_layout.setHorizontalSpacing(6)
        pulse_layout.setVerticalSpacing(3)
        pulse_layout.setColumnMinimumWidth(0, 82)
        pulse_layout.setColumnMinimumWidth(2, 76)

        self._amplitude = self._make_voltage_spinbox(channel)
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
        self._trigger_mode = QComboBox()
        self._trigger_mode.addItems(("internal", "external"))
        if channel.trigger_source:
            self._trigger_mode.setCurrentText("external")
        self._trigger_source = QLineEdit(channel.trigger_source or "")
        self._trigger_source.setPlaceholderText("NI-DAQ trigger route")
        self._trigger_edge = QComboBox()
        self._trigger_edge.addItems(("rising", "falling"))

        self._open_shutter = self._make_checkbox("Open shutter")
        self._open_shutter.setChecked(True)
        self._close_shutter = self._make_checkbox("Close shutter")
        self._close_shutter.setChecked(True)
        self._enable_pmt = self._make_checkbox("PMT shutter")
        self._emit_trigger = self._make_checkbox("Trigger DO")
        self._emit_timing_trigger = self._make_checkbox("Timing DO")
        self._run_pulse_button = QPushButton("Run Pulse")

        pulse_layout.addWidget(self._form_label("Amplitude:"), 0, 0)
        pulse_layout.addWidget(self._amplitude, 0, 1)
        pulse_layout.addWidget(self._form_label("Duration:"), 0, 2)
        pulse_layout.addWidget(self._duration_ms, 0, 3)
        pulse_layout.addWidget(self._form_label("Baseline:"), 1, 0)
        pulse_layout.addWidget(self._baseline_ms, 1, 1)
        pulse_layout.addWidget(self._form_label("Post-stim:"), 1, 2)
        pulse_layout.addWidget(self._post_stim_ms, 1, 3)
        pulse_layout.addWidget(self._form_label("Count:"), 2, 0)
        pulse_layout.addWidget(self._pulse_count, 2, 1)
        pulse_layout.addWidget(self._form_label("Frequency:"), 2, 2)
        pulse_layout.addWidget(self._frequency_hz, 2, 3)
        pulse_layout.addWidget(self._form_label("Trigger Mode:"), 3, 0)
        pulse_layout.addWidget(self._trigger_mode, 3, 1)
        pulse_layout.addWidget(self._form_label("Trigger Type:"), 3, 2)
        pulse_layout.addWidget(self._trigger_edge, 3, 3)
        pulse_layout.addWidget(self._form_label("Trigger Source:"), 4, 0)
        pulse_layout.addWidget(self._trigger_source, 4, 1, 1, 3)
        shutter_options = QWidget()
        shutter_options_layout = QHBoxLayout(shutter_options)
        shutter_options_layout.setContentsMargins(0, 0, 0, 0)
        shutter_options_layout.setSpacing(8)
        shutter_options_layout.addWidget(self._open_shutter)
        shutter_options_layout.addWidget(self._close_shutter)
        shutter_options_layout.addWidget(self._enable_pmt)
        shutter_options_layout.addStretch(1)
        pulse_layout.addWidget(shutter_options, 5, 0, 1, 4)

        trigger_options = QWidget()
        trigger_options_layout = QHBoxLayout(trigger_options)
        trigger_options_layout.setContentsMargins(0, 0, 0, 0)
        trigger_options_layout.setSpacing(8)
        trigger_options_layout.addWidget(self._emit_trigger)
        trigger_options_layout.addWidget(self._emit_timing_trigger)
        trigger_options_layout.addStretch(1)
        pulse_layout.addWidget(trigger_options, 6, 0, 1, 3)
        pulse_layout.addWidget(self._run_pulse_button, 6, 3)
        layout.addWidget(pulse_group)

        self._preview_plot = PGWidget()
        self._preview_plot.setMinimumHeight(120)
        self._preview_plot.setMaximumHeight(180)
        self._preview_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._preview_plot.clear()
        self._preview_plot.setBackground("w")
        self._preview_plot.getAxis("bottom").setLabel("Time", units="s")
        self._preview_plot.getAxis("left").setLabel("Command", units="V")
        self._preview_plot.setMouseEnabled(x=False, y=False)
        self._preview_curve = self._preview_plot.plot([], [], pen=pg.mkPen(color=(30, 90, 180), width=2))
        self._preview_status = QLabel("")
        self._preview_status.setObjectName("LaserPreviewStatus")
        self._preview_status.setWordWrap(True)
        layout.addWidget(self._preview_plot)
        layout.addWidget(self._preview_status)

        ramp_group = QGroupBox("Calibration Ramp")
        ramp_layout = QGridLayout(ramp_group)
        ramp_layout.setContentsMargins(8, 4, 8, 6)
        ramp_layout.setHorizontalSpacing(6)
        ramp_layout.setVerticalSpacing(3)
        ramp_layout.setColumnMinimumWidth(0, 82)
        ramp_layout.setColumnMinimumWidth(2, 76)

        self._ramp_start = self._make_voltage_spinbox(channel)
        self._ramp_start.setValue(channel.minimum_command_volts)
        self._ramp_stop = self._make_voltage_spinbox(channel)
        self._ramp_stop.setValue(channel.maximum_command_volts)
        self._ramp_steps = QSpinBox()
        self._ramp_steps.setRange(2, 10000)
        self._ramp_steps.setValue(11)
        self._ramp_samples_per_step = QSpinBox()
        self._ramp_samples_per_step.setRange(1, 1000000)
        self._ramp_samples_per_step.setValue(100)
        self._ramp_pmt = self._make_checkbox("PMT shutter")
        self._run_ramp_button = QPushButton("Run Ramp")

        ramp_layout.addWidget(self._form_label("Start:"), 0, 0)
        ramp_layout.addWidget(self._ramp_start, 0, 1)
        ramp_layout.addWidget(self._form_label("Stop:"), 0, 2)
        ramp_layout.addWidget(self._ramp_stop, 0, 3)
        ramp_layout.addWidget(self._form_label("Steps:"), 1, 0)
        ramp_layout.addWidget(self._ramp_steps, 1, 1)
        ramp_layout.addWidget(self._form_label("Samples/step:"), 1, 2)
        ramp_layout.addWidget(self._ramp_samples_per_step, 1, 3)
        ramp_actions = QWidget()
        ramp_actions_layout = QHBoxLayout(ramp_actions)
        ramp_actions_layout.setContentsMargins(0, 0, 0, 0)
        ramp_actions_layout.setSpacing(8)
        ramp_actions_layout.addWidget(self._ramp_pmt)
        ramp_actions_layout.addStretch(1)
        ramp_actions_layout.addWidget(self._run_ramp_button)
        ramp_layout.addWidget(ramp_actions, 2, 0, 1, 4)
        layout.addWidget(ramp_group)

        trace_group = QGroupBox("Output Stream")
        trace_layout = QVBoxLayout(trace_group)
        trace_layout.setContentsMargins(8, 4, 8, 6)
        trace_layout.setSpacing(4)
        self._trace_plot = PGWidget()
        self._trace_plot.setMinimumHeight(140)
        self._trace_plot.setMaximumHeight(220)
        self._trace_plot.setBackground("w")
        self._trace_plot.getAxis("bottom").setLabel("Time", units="s")
        self._trace_plot.getAxis("left").setLabel("Voltage", units="V")
        self._trace_plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self._trace_plot.getPlotItem().setClipToView(True)
        self._trace_plot.addLegend(offset=(-8, 8))
        self._trace_curves = {
            "command": self._trace_plot.plot(
                [], [], pen=pg.mkPen(color=(30, 90, 180), width=2), name="Command"
            ),
            "diode": self._trace_plot.plot(
                [], [], pen=pg.mkPen(color=(210, 80, 70), width=1.5), name="Diode feedback"
            ),
            "copy": self._trace_plot.plot(
                [], [], pen=pg.mkPen(color=(50, 150, 90), width=1.5), name="Command copy"
            ),
        }
        self._trace_data = {
            curve_name: ([], [])
            for curve_name in self._trace_curves
        }
        trace_layout.addWidget(self._trace_plot)
        trace_actions = QHBoxLayout()
        self._trace_toggle_button = QPushButton("Pause Stream")
        self._trace_clear_button = QPushButton("Clear")
        self._trace_status = QLabel("Streaming")
        self._trace_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        trace_actions.addWidget(self._trace_toggle_button)
        trace_actions.addWidget(self._trace_clear_button)
        trace_actions.addWidget(self._trace_status, stretch=1)
        trace_layout.addLayout(trace_actions)
        layout.addWidget(trace_group)

        self._pulse_controls = (
            self._amplitude,
            self._duration_ms,
            self._baseline_ms,
            self._post_stim_ms,
            self._pulse_count,
            self._frequency_hz,
            self._trigger_mode,
            self._trigger_source,
            self._trigger_edge,
            self._open_shutter,
            self._close_shutter,
            self._enable_pmt,
            self._emit_trigger,
            self._emit_timing_trigger,
        )
        self._ramp_controls = (
            self._ramp_start,
            self._ramp_stop,
            self._ramp_steps,
            self._ramp_samples_per_step,
            self._ramp_pmt,
        )

        self._run_pulse_button.clicked.connect(self._run_pulse)
        self._run_ramp_button.clicked.connect(self._run_calibration_ramp)
        self._trace_toggle_button.clicked.connect(self._toggle_trace_stream)
        self._trace_clear_button.clicked.connect(self._clear_trace)
        self._connect_preview_signals()
        self._refresh_trigger_mode_enabled()
        self._refresh_preview()

    @property
    def channel_id_value(self) -> int:
        return int(self._channel.channel_id)

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    @staticmethod
    def _form_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setMinimumWidth(74)
        return label

    @staticmethod
    def _make_voltage_spinbox(channel: LaserChannelConfiguration) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(3)
        spinbox.setSingleStep(0.050)
        spinbox.setSuffix(" V")
        spinbox.setRange(channel.minimum_command_volts, channel.maximum_command_volts)
        spinbox.setValue(channel.minimum_command_volts)
        return spinbox

    @staticmethod
    def _make_checkbox(text: str) -> QCheckBox:
        checkbox = QCheckBox(text)
        checkbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        return checkbox

    @staticmethod
    def _make_ms_spinbox(value: float) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(3)
        spinbox.setSingleStep(1.0)
        spinbox.setSuffix(" ms")
        spinbox.setRange(0.0, 600000.0)
        spinbox.setValue(value)
        return spinbox

    def _connect_preview_signals(self) -> None:
        for spinbox in (
            self._amplitude,
            self._duration_ms,
            self._baseline_ms,
            self._post_stim_ms,
            self._pulse_count,
            self._frequency_hz,
        ):
            spinbox.valueChanged.connect(self._refresh_preview)
        self._trigger_source.textChanged.connect(self._refresh_preview)
        self._trigger_mode.currentTextChanged.connect(self._on_trigger_mode_changed)
        self._trigger_edge.currentTextChanged.connect(self._refresh_preview)
        for checkbox in (
            self._open_shutter,
            self._close_shutter,
            self._enable_pmt,
            self._emit_trigger,
            self._emit_timing_trigger,
        ):
            checkbox.toggled.connect(self._refresh_preview)

    def set_controls_enabled(self, can_edit: bool, can_run_pulse: bool, can_run_ramp: bool) -> None:
        self._controls_can_edit = can_edit
        for control in self._pulse_controls:
            control.setEnabled(can_edit)
        self._refresh_trigger_mode_enabled()
        self._run_pulse_button.setEnabled(can_run_pulse)
        for control in self._ramp_controls:
            control.setEnabled(can_edit)
        self._run_ramp_button.setEnabled(can_run_ramp)

    def append_trace(self, trace: LaserTraceBlock) -> None:
        if int(trace.channel_id) != self.channel_id_value:
            return
        is_calibration = trace.source == "calibration"
        if is_calibration:
            self._set_trace_streaming(True)
        if not self._trace_streaming:
            return
        if trace.replace:
            self._clear_trace()
        if not trace.x_values:
            self._trace_status.setText("Calibration running — collecting diode feedback...")
            return

        if trace.replace:
            x_values = list(trace.x_values)
        else:
            latest_values = [values[-1] for values, _samples in self._trace_data.values() if values]
            append_offset = max(latest_values) + self._trace_gap(trace.x_values) if latest_values else 0.0
            first_x = trace.x_values[0]
            x_values = [append_offset + value - first_x for value in trace.x_values]
        for curve_name, values in (
            ("command", trace.command_volts),
            ("diode", trace.diode_volts),
            ("copy", trace.command_copy_volts),
        ):
            if not values:
                continue
            curve_x, curve_y = self._trace_data[curve_name]
            curve_x.extend(x_values[: len(values)])
            curve_y.extend(values)
            if len(curve_x) > _MAX_LASER_TRACE_POINTS:
                trim_count = len(curve_x) - _MAX_LASER_TRACE_POINTS
                del curve_x[:trim_count]
                del curve_y[:trim_count]
            self._trace_curves[curve_name].setData(curve_x, curve_y)
        self._trace_plot.enableAutoRange(axis="y")
        populated_x_values = [
            curve_x
            for curve_x, _curve_y in self._trace_data.values()
            if curve_x
        ]
        x_min = min(curve_x[0] for curve_x in populated_x_values)
        x_max = max(curve_x[-1] for curve_x in populated_x_values)
        if x_max <= x_min:
            x_max = x_min + 0.001
        self._trace_plot.setXRange(x_min, x_max, padding=0.02)
        if is_calibration:
            self._trace_status.setText(
                f"Calibration complete — {len(trace.x_values)} ramp points displayed"
            )
        else:
            self._trace_status.setText(f"Streaming — latest: {trace.source}")

    def append_signal_block(self, block: NidaqSignalSampleBlock) -> None:
        if not self._is_configured or not self._trace_streaming:
            return
        names_by_physical_channel = {
            channel.physical_channel: channel.name
            for channel in block.channels
        }
        diode_name = names_by_physical_channel.get(self._channel.diode_input)
        copy_name = names_by_physical_channel.get(self._channel.command_copy_input)
        diode_values = tuple(block.values.get(diode_name, tuple())) if diode_name else tuple()
        copy_values = tuple(block.values.get(copy_name, tuple())) if copy_name else tuple()
        sample_count = max(len(diode_values), len(copy_values))
        if sample_count == 0:
            return
        self.append_trace(
            LaserTraceBlock(
                channel_id=self._channel.channel_id,
                source="NI-DAQ input stream",
                x_values=tuple(
                    (block.sample_index + index) / block.sample_rate_hz
                    for index in range(sample_count)
                ),
                diode_volts=diode_values,
                command_copy_volts=copy_values,
            )
        )

    def _toggle_trace_stream(self) -> None:
        self._set_trace_streaming(not self._trace_streaming)

    def _set_trace_streaming(self, is_streaming: bool) -> None:
        self._trace_streaming = is_streaming
        self._trace_toggle_button.setText("Pause Stream" if is_streaming else "Start Stream")
        self._trace_status.setText("Streaming" if is_streaming else "Paused")

    def _clear_trace(self) -> None:
        for curve_name, curve in self._trace_curves.items():
            self._trace_data[curve_name] = ([], [])
            curve.setData([], [])
        self._trace_status.setText("Streaming — cleared" if self._trace_streaming else "Paused — cleared")

    @staticmethod
    def _trace_gap(x_values: Tuple[float, ...]) -> float:
        positive_steps = [
            second - first
            for first, second in zip(x_values, x_values[1:])
            if second > first
        ]
        return min(positive_steps) if positive_steps else 0.001

    def _run_pulse(self) -> None:
        if not self._is_configured:
            self._set_parent_status(
                f"Laser {self._channel.channel_id.value} has no hardware channel mapping",
                True,
            )
            return
        try:
            pulse_train = self._build_pulse_train()
            self._validate_pulse_train(pulse_train)
        except Exception as exc:
            self._set_parent_status(str(exc) or exc.__class__.__name__, True)
            return

        def operation():
            self._app_model.laser.run_pulse_train(pulse_train)
            return f"Pulse complete: laser {self._channel.channel_id.value}"

        self._start_operation(f"Running laser {self._channel.channel_id.value} pulse train", operation)

    def _run_calibration_ramp(self) -> None:
        if not self._is_configured:
            self._set_parent_status(
                f"Laser {self._channel.channel_id.value} has no hardware channel mapping",
                True,
            )
            return
        try:
            ramp = LaserCalibrationRamp(
                channel_id=self._channel.channel_id,
                start_volts=self._ramp_start.value(),
                stop_volts=self._ramp_stop.value(),
                steps=self._ramp_steps.value(),
                samples_per_step=self._ramp_samples_per_step.value(),
                enable_pmt_shutter=self._ramp_pmt.isChecked(),
            )
        except Exception as exc:
            self._set_parent_status(str(exc) or exc.__class__.__name__, True)
            return

        def operation():
            signal_monitor = self._app_model.nidaq_signal_monitor
            restart_signal_stream = signal_monitor.is_running
            if restart_signal_stream:
                signal_monitor.stop()
            try:
                points = self._app_model.laser.run_calibration_ramp(ramp)
                self._app_model.laser.make_diode_power_curve(points)
                last = points[-1]
                return (
                    f"Ramp complete: {len(points)} points, last diode {last.diode_volts:.3f} V, "
                    "monotonic curve validated"
                )
            finally:
                if restart_signal_stream:
                    signal_monitor.start()

        self._start_operation(f"Running laser {self._channel.channel_id.value} calibration ramp", operation)

    def _build_pulse_train(self) -> LaserPulseTrain:
        pulse_count = self._pulse_count.value()
        frequency_hz = self._frequency_hz.value() if pulse_count > 1 else None
        trigger_source = None
        if self._trigger_mode.currentText() == "external":
            trigger_source = self._trigger_source.text().strip()
            if not trigger_source:
                raise ValueError("External trigger mode requires a trigger source")
        return LaserPulseTrain(
            channel_id=self._channel.channel_id,
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

    def _on_trigger_mode_changed(self) -> None:
        self._refresh_trigger_mode_enabled()
        self._refresh_preview()

    def _refresh_trigger_mode_enabled(self) -> None:
        is_external = self._trigger_mode.currentText() == "external"
        self._trigger_source.setEnabled(self._controls_can_edit and is_external)
        self._trigger_edge.setEnabled(self._controls_can_edit and is_external)

    def _validate_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        minimum = self._channel.minimum_command_volts
        maximum = self._channel.maximum_command_volts
        if not minimum <= pulse_train.amplitude_volts <= maximum:
            raise ValueError(
                f"laser channel {self._channel.channel_id.value} command "
                f"{pulse_train.amplitude_volts} V is outside {minimum}..{maximum} V"
            )
        if pulse_train.pulse_count > 1:
            period_ms = 1000.0 / pulse_train.frequency_hz
            if pulse_train.duration_ms > period_ms:
                raise ValueError(
                    f"laser pulse duration {pulse_train.duration_ms:g} ms exceeds pulse period "
                    f"{period_ms:g} ms at {pulse_train.frequency_hz:g} Hz"
                )

    def _refresh_preview(self, *_args) -> None:
        try:
            pulse_train = self._build_pulse_train()
            self._validate_pulse_train(pulse_train)
            x_values, y_values = self._build_preview_points(pulse_train)
        except Exception as exc:
            self._preview_curve.setData([], [])
            self._preview_plot.setVisible(False)
            self._preview_status.setText(str(exc) or exc.__class__.__name__)
            self._preview_status.setStyleSheet("color: #b00020;")
            return
        self._preview_curve.setData(x_values, y_values)
        self._preview_plot.setVisible(True)
        self._preview_status.setText("")
        self._preview_status.setStyleSheet("")
        max_x = max(x_values[-1], 0.001)
        span = max(self._channel.maximum_command_volts - self._channel.minimum_command_volts, 1.0)
        self._preview_plot.setXRange(0.0, max_x, padding=0.02)
        self._preview_plot.setYRange(
            self._channel.minimum_command_volts - span * 0.05,
            self._channel.maximum_command_volts + span * 0.05,
            padding=0.0,
        )

    def _build_preview_points(self, pulse_train: LaserPulseTrain) -> Tuple[list, list]:
        minimum = self._channel.minimum_command_volts
        amplitude = pulse_train.amplitude_volts
        duration_s = pulse_train.duration_ms / 1000.0
        baseline_s = pulse_train.baseline_ms / 1000.0
        post_stim_s = pulse_train.post_stim_ms / 1000.0
        period_s = (1.0 / pulse_train.frequency_hz) if pulse_train.frequency_hz is not None else duration_s

        x_values = [0.0]
        y_values = [minimum]
        current_t = 0.0

        def horizontal(to_t: float) -> None:
            nonlocal current_t
            if to_t <= current_t:
                return
            x_values.append(to_t)
            y_values.append(y_values[-1])
            current_t = to_t

        def transition(value: float) -> None:
            x_values.append(current_t)
            y_values.append(y_values[-1])
            x_values.append(current_t)
            y_values.append(value)

        horizontal(baseline_s)
        for pulse_index in range(pulse_train.pulse_count):
            pulse_start = baseline_s + pulse_index * period_s
            horizontal(pulse_start)
            transition(amplitude)
            horizontal(pulse_start + duration_s)
            transition(minimum)
        horizontal(current_t + post_stim_s)
        return x_values, y_values


class LaserControlContent(ContentWidget):
    """Thin operator controls for the laser model."""

    def __init__(self, app_model: AppModel):
        super().__init__()

        self._app_model = app_model
        self._is_editable = True
        self._is_capture_active = False
        self._operation_thread: Optional[QThread] = None
        self._operation_worker: Optional[_LaserOperationWorker] = None
        self._channel_tabs: Tuple[_LaserChannelTab, ...] = tuple()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("LaserControlContent")
        self.setStyleSheet(
            "#LaserControlContent QLabel {color: #2f343a;}"
            "#LaserControlContent QLabel#LaserMetaLabel {color: #5b6470;}"
            "#LaserControlContent QLabel#LaserMetaValue {color: #20242a; font-weight: 600;}"
            "#LaserControlContent QTabWidget::pane {border: 0px; background: #ffffff;}"
            "#LaserControlContent QTabBar::tab {"
            "background: #e7eaee; color: #20242a; border: 1px solid #c9cdd3; "
            "padding: 4px 10px;"
            "}"
            "#LaserControlContent QTabBar::tab:selected {background: #ffffff; border-bottom-color: #ffffff;}"
            "#LaserControlContent QTabBar::tab:!selected {margin-top: 2px;}"
        )

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        backend_text = QLabel("Backend:")
        backend_text.setObjectName("LaserMetaLabel")
        header_layout.addWidget(backend_text)
        self._backend_label = QLabel("disabled")
        self._backend_label.setObjectName("LaserMetaValue")
        header_layout.addWidget(self._backend_label)
        rate_text = QLabel("Rate:")
        rate_text.setObjectName("LaserMetaLabel")
        header_layout.addWidget(rate_text)
        self._sample_rate_label = QLabel("manual")
        self._sample_rate_label.setObjectName("LaserMetaValue")
        header_layout.addWidget(self._sample_rate_label)

        self._card_widget = CardWidget(title="Laser Control", header_right_layout=header_layout)
        self._card_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setMinimumWidth(0)
        self._tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._card_widget.setContentWidget(self._tabs)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._status_label = QLabel("Laser controller not configured")
        self._status_label.setObjectName("LaserStatus")
        footer_layout.addWidget(self._progress)
        footer_layout.addWidget(self._status_label, stretch=1)
        self._card_widget.footer.setContent(footer)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._card_widget, stretch=1)
        self.setLayout(root_layout)

        app_model.laser.property_changed += self._on_laser_property_changed
        app_model.laser.trace_received += self._on_laser_trace_received
        app_model.nidaq_signal_monitor.sample_block_received += self._on_nidaq_sample_block
        self._refresh_from_model()

    def on_close(self):
        self._app_model.laser.property_changed -= self._on_laser_property_changed
        self._app_model.laser.trace_received -= self._on_laser_trace_received
        self._app_model.nidaq_signal_monitor.sample_block_received -= self._on_nidaq_sample_block

    @invoke_method
    def _on_laser_property_changed(self, property_name: str, _value, _old_value):
        if property_name in (LaserModel.CONFIGURATION, LaserModel.IS_CONNECTED):
            self._refresh_from_model()

    @invoke_method
    def _on_laser_trace_received(self, trace: LaserTraceBlock) -> None:
        for tab in self._channel_tabs:
            if tab.channel_id_value == int(trace.channel_id):
                tab.append_trace(trace)
                break

    @invoke_method
    def _on_nidaq_sample_block(self, block: NidaqSignalSampleBlock) -> None:
        for tab in self._channel_tabs:
            tab.append_signal_block(block)

    def _refresh_from_model(self) -> None:
        configuration = self._app_model.laser.configuration
        current = self._current_channel_id()
        self._backend_label.setText(configuration.backend)
        if configuration.sample_rate_hz is None:
            self._sample_rate_label.setText("manual")
        else:
            self._sample_rate_label.setText(f"{configuration.sample_rate_hz:g} Hz")

        self._clear_tabs()
        tabs = []
        configured_channels = {
            int(channel.channel_id): channel
            for channel in configuration.channels
        }
        for channel_index in range(1, _LASER_PULSE_TRAIN_COUNT + 1):
            channel = configured_channels.get(channel_index)
            is_configured = channel is not None
            if channel is None:
                channel = self._make_placeholder_channel(channel_index)
            tab = _LaserChannelTab(
                self._app_model,
                channel,
                is_configured,
                configuration.sample_rate_hz,
                self._start_operation,
                self._set_status_from_tab,
            )
            self._tabs.addTab(tab, f"Laser {channel_index}")
            tabs.append(tab)
        self._channel_tabs = tuple(tabs)
        if current is not None:
            for index, tab in enumerate(self._channel_tabs):
                if tab.channel_id_value == current:
                    self._tabs.setCurrentIndex(index)
                    break
        configured_count = sum(tab.is_configured for tab in self._channel_tabs)
        if configured_count:
            self._set_status(
                f"Ready: {configured_count}/{_LASER_PULSE_TRAIN_COUNT} laser channel(s) mapped",
                is_error=False,
            )
        else:
            self._set_status(
                f"Pulse train editor ready; 0/{_LASER_PULSE_TRAIN_COUNT} hardware channel(s) mapped",
                is_error=False,
            )
        self._update_enabled_state()

    @staticmethod
    def _make_placeholder_channel(channel_index: int) -> LaserChannelConfiguration:
        return LaserChannelConfiguration(
            channel_id=LaserChannelId(channel_index),
            analog_output="unconfigured",
            diode_input="unconfigured",
            shutter_output="unconfigured",
            auxiliary_output="unconfigured",
            minimum_command_volts=_DEFAULT_MINIMUM_COMMAND_VOLTS,
            maximum_command_volts=_DEFAULT_MAXIMUM_COMMAND_VOLTS,
        )

    def _clear_tabs(self) -> None:
        while self._tabs.count():
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            widget.deleteLater()
        self._channel_tabs = tuple()

    def _current_channel_id(self) -> Optional[int]:
        widget = self._tabs.currentWidget()
        if isinstance(widget, _LaserChannelTab):
            return widget.channel_id_value
        return None

    def _set_status_from_tab(self, message: str, is_error: bool) -> None:
        self._set_status(message, is_error=is_error)

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
        can_edit = self._is_editable and not is_running
        can_run = can_edit and self._app_model.laser.is_connected
        for tab in self._channel_tabs:
            can_run_pulse = can_run and tab.is_configured
            can_run_ramp = can_run_pulse and not self._is_capture_active
            tab.set_controls_enabled(can_edit, can_run_pulse, can_run_ramp)

    @invoke_method
    def set_is_editable(self, is_editable: bool):
        self._is_editable = is_editable
        self._update_enabled_state()

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        self._is_capture_active = is_active
        self._update_enabled_state()
