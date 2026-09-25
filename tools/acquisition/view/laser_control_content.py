from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core import NidaqSignalChannelConfiguration
from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import (
    LaserCalibrationRamp,
    LaserChannelConfiguration,
    LaserChannelId,
    LaserPulseTrain,
)
from autotrainer.pyside import CardWidget, PGWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.trial_protocol_schedule import (
    LaserTriggerRoute,
)
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.laser_model import LaserModel, LaserTraceBlock
from tools.acquisition.model.laser_plot_process import LaserPlotFrame, LaserPlotProcess
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.view.pulse_builder_tab import DRAFT_PROFILE_ID, PulseBuilderTab
from tools.acquisition.view.stream_graph_style import (
    StreamGraphLegend,
    color_code_checkbox,
    stream_signal_color,
)


logger = get_verbose_logger(__name__)

_LASER_PULSE_TRAIN_COUNT = 4
_DEFAULT_MINIMUM_COMMAND_VOLTS = 0.0
_DEFAULT_MAXIMUM_COMMAND_VOLTS = 5.0
_COMMAND_TRACE_COLOR = stream_signal_color(0)
_DIODE_TRACE_COLOR = stream_signal_color(1)
_COMMAND_COPY_TRACE_COLOR = stream_signal_color(2)
_TRIGGER_TRACE_COLOR = stream_signal_color(3)
_STIM_TEST_TOOLTIP = (
    "Run the selected profile on this laser as a trial would, started by the "
    "route chosen beside it"
)


def _nidaq_channel_kind(physical_channel: str) -> str:
    """Analog or digital, from the NI channel name.

    NI names a digital line by its port and line, as in Dev1/port0/line3, and
    an analog input as Dev1/ai3. The stimulus line can be wired back into
    either, so the kind follows the name rather than another setting to keep
    in step with it.
    """
    lowered = str(physical_channel).lower()
    return "digital" if "port" in lowered or "line" in lowered else "analog"


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
    """Single-laser controls: fire a profile, calibrate, and watch the output.

    The pulse train itself is shaped in the Pulse Builder; this tab picks a
    saved profile or the builder draft and fires it on its own laser.
    """

    def __init__(
        self,
        app_model: AppModel,
        channel: LaserChannelConfiguration,
        is_configured: bool,
        sample_rate_hz: Optional[float],
        start_operation: Callable[[str, Callable[[], object]], None],
        set_status: Callable[[str, bool], None],
        plot_controller=None,
        draft_provider: Optional[Callable[[], Optional[LaserPulseProfile]]] = None,
    ):
        super().__init__()

        self._app_model = app_model
        self._channel = channel
        self._is_configured = is_configured
        self._sample_rate_hz = sample_rate_hz
        self._start_operation = start_operation
        self._set_parent_status = set_status
        self._plot_controller = plot_controller
        self._draft_provider = draft_provider
        self._controls_can_edit = True
        self._trace_streaming = False
        #: What the latest pulse, ramp or Clear did, shown after the stream state.
        self._trace_note = ""
        self._trace_data = {}

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
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

        self._mode_tabs = QTabWidget(self)
        self._mode_tabs.setDocumentMode(True)
        self._mode_tabs.setMinimumWidth(0)
        self._mode_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._mode_tabs, stretch=1)

        pulse_page = QWidget(self._mode_tabs)
        pulse_page_layout = QVBoxLayout(pulse_page)
        pulse_page_layout.setContentsMargins(2, 4, 2, 2)
        pulse_page_layout.setSpacing(5)
        calibration_page = QWidget(self._mode_tabs)
        calibration_page_layout = QVBoxLayout(calibration_page)
        calibration_page_layout.setContentsMargins(2, 4, 2, 2)
        calibration_page_layout.setSpacing(5)
        output_page = QWidget(self._mode_tabs)
        output_page_layout = QVBoxLayout(output_page)
        output_page_layout.setContentsMargins(2, 4, 2, 2)
        output_page_layout.setSpacing(5)
        # The pulse page stacks the controls, the live output and the board
        # trigger. That is taller than the panel at most sizes, so it scrolls
        # rather than pushing the lower graphs out of reach.
        pulse_scroll = QScrollArea(self._mode_tabs)
        pulse_scroll.setWidgetResizable(True)
        pulse_scroll.setFrameShape(QFrame.Shape.NoFrame)
        pulse_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        pulse_scroll.setWidget(pulse_page)
        self._mode_tabs.addTab(pulse_scroll, "Pulse")
        self._mode_tabs.addTab(calibration_page, "Calibration")
        self._mode_tabs.addTab(output_page, "Output")

        pulse_group = QGroupBox("Pulse Train")
        pulse_layout = QGridLayout(pulse_group)
        pulse_layout.setContentsMargins(8, 4, 8, 6)
        pulse_layout.setHorizontalSpacing(6)
        pulse_layout.setVerticalSpacing(3)
        pulse_layout.setColumnStretch(1, 1)

        # Run Pulse and Test stim both fire the profile picked here, on this
        # laser: the Pulse Builder's unsaved draft or any saved profile. A
        # profile is the waveform alone; Run Pulse starts it with the trigger
        # and shutter options below, Test stim by the route chosen beside it.
        # A new tab picks "(none)", so a laser only fires what someone chose
        # for it, never whatever happens to be on the builder.
        self.stim_profile_selector = QComboBox()
        self.stim_profile_selector.setToolTip(
            "The Pulse Builder draft and every saved laser profile; any of "
            "them can fire on this laser"
        )
        self._profile_summary = QLabel("")
        self._profile_summary.setObjectName("LaserPreviewStatus")
        self._profile_summary.setWordWrap(True)
        self._trigger_mode = QComboBox()
        self._trigger_mode.addItems(("internal", "external"))
        # Internal by default, even when the channel has a trigger route. This
        # used to switch to external whenever one was configured, so Run pulse
        # armed the output for a board STIM pulse that nothing on this tab
        # sends, and failed with a DAQmx timeout (christielab10, 2026-09-24).
        # The route stays filled in for choosing external deliberately.
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

        pulse_layout.addWidget(self._form_label("Profile:"), 0, 0)
        pulse_layout.addWidget(self.stim_profile_selector, 0, 1)
        pulse_layout.addWidget(self._profile_summary, 1, 0, 1, 2)
        pulse_layout.addWidget(self._form_label("Trigger Mode:"), 2, 0)
        pulse_layout.addWidget(self._trigger_mode, 2, 1)
        pulse_layout.addWidget(self._form_label("Trigger Type:"), 3, 0)
        pulse_layout.addWidget(self._trigger_edge, 3, 1)
        pulse_layout.addWidget(self._form_label("Trigger Source:"), 4, 0)
        pulse_layout.addWidget(self._trigger_source, 4, 1)
        shutter_options = QWidget()
        shutter_options_layout = QGridLayout(shutter_options)
        shutter_options_layout.setContentsMargins(0, 0, 0, 0)
        shutter_options_layout.setSpacing(8)
        shutter_options_layout.addWidget(self._open_shutter, 0, 0)
        shutter_options_layout.addWidget(self._close_shutter, 0, 1)
        shutter_options_layout.addWidget(self._enable_pmt, 1, 0)
        pulse_layout.addWidget(shutter_options, 5, 0, 1, 2)

        trigger_options = QWidget()
        trigger_options_layout = QHBoxLayout(trigger_options)
        trigger_options_layout.setContentsMargins(0, 0, 0, 0)
        trigger_options_layout.setSpacing(8)
        trigger_options_layout.addWidget(self._emit_trigger)
        trigger_options_layout.addWidget(self._emit_timing_trigger)
        trigger_options_layout.addStretch(1)
        pulse_layout.addWidget(trigger_options, 6, 0, 1, 2)
        pulse_layout.addWidget(self._run_pulse_button, 7, 1)

        # Run Pulse above drives the analog output straight from the host. Test
        # stim fires the same profile the way a trial does: arm the output,
        # then start it by the route chosen here - the board's timed STIM
        # pulse into this laser's trigger terminal, or a software start.
        self._stim_route = QComboBox()
        self._stim_route.setToolTip(
            "How Test stim starts the profile on this laser")
        self._refresh_stim_route_options()
        self.stim_test_button = QPushButton("Test stim")
        self.stim_test_button.setToolTip(_STIM_TEST_TOOLTIP)
        self.stim_test_button.clicked.connect(self._run_stim_test)
        self.stim_test_result = QLabel()
        self.stim_test_result.setWordWrap(True)
        self.stim_test_result.setObjectName("LaserPreviewStatus")
        pulse_layout.addWidget(self._form_label("Route:"), 8, 0)
        pulse_layout.addWidget(self._stim_route, 8, 1)
        pulse_layout.addWidget(self.stim_test_button, 9, 1)
        pulse_layout.addWidget(self.stim_test_result, 10, 0, 1, 2)

        pulse_page_layout.addWidget(pulse_group)

        ramp_group = QGroupBox("Calibration Ramp")
        ramp_layout = QGridLayout(ramp_group)
        ramp_layout.setContentsMargins(8, 4, 8, 6)
        ramp_layout.setHorizontalSpacing(6)
        ramp_layout.setVerticalSpacing(3)
        ramp_layout.setColumnStretch(1, 1)

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
        ramp_layout.addWidget(self._form_label("Stop:"), 1, 0)
        ramp_layout.addWidget(self._ramp_stop, 1, 1)
        ramp_layout.addWidget(self._form_label("Steps:"), 2, 0)
        ramp_layout.addWidget(self._ramp_steps, 2, 1)
        ramp_layout.addWidget(self._form_label("Samples/step:"), 3, 0)
        ramp_layout.addWidget(self._ramp_samples_per_step, 3, 1)
        ramp_actions = QWidget()
        ramp_actions_layout = QHBoxLayout(ramp_actions)
        ramp_actions_layout.setContentsMargins(0, 0, 0, 0)
        ramp_actions_layout.setSpacing(8)
        ramp_actions_layout.addWidget(self._ramp_pmt)
        ramp_actions_layout.addStretch(1)
        ramp_actions_layout.addWidget(self._run_ramp_button)
        ramp_layout.addWidget(ramp_actions, 4, 0, 1, 2)
        calibration_page_layout.addWidget(ramp_group)
        calibration_page_layout.addStretch(1)

        trace_group = QGroupBox("Output Stream")
        trace_layout = QVBoxLayout(trace_group)
        trace_layout.setContentsMargins(8, 4, 8, 6)
        trace_layout.setSpacing(4)
        self._trace_plot = PGWidget()
        self._trace_plot.setBackground("w")
        self._trace_plot.getAxis("bottom").setLabel("Time from latest sample", units="s")
        self._trace_plot.getAxis("left").setLabel("Voltage", units="V")
        self._trace_plot.getPlotItem().setClipToView(True)
        self._trace_curves = {
            "command": self._trace_plot.plot(
                [], [], pen=pg.mkPen(color=_COMMAND_TRACE_COLOR, width=2.4)
            ),
            "diode": self._trace_plot.plot(
                [], [], pen=pg.mkPen(color=_DIODE_TRACE_COLOR, width=2.0)
            ),
            "copy": self._trace_plot.plot(
                [], [], pen=pg.mkPen(color=_COMMAND_COPY_TRACE_COLOR, width=2.0)
            ),
        }

        # The board's stimulus line, read back on its own axis. It is a TTL
        # edge rather than a command voltage, so sharing the laser plot's Y
        # range would flatten it against the waveform.
        self.trigger_plot = PGWidget()
        self.trigger_plot.setBackground("w")
        self.trigger_plot.getAxis("bottom").setLabel("Time from latest sample", units="s")
        self.trigger_plot.getAxis("left").setLabel("Trigger", units="V")
        self.trigger_plot.getPlotItem().setClipToView(True)
        self.trigger_plot.setMouseEnabled(x=False, y=False)
        # Bounded both ways: small, because it is a single TTL edge beside a
        # waveform, but never squeezed to nothing when the page is crowded.
        self.trigger_plot.setMinimumHeight(80)
        self.trigger_plot.setMaximumHeight(140)
        self.trigger_plot.setYRange(-0.5, 5.5, padding=0)
        self._trace_curves["trigger"] = self.trigger_plot.plot(
            [], [], pen=pg.mkPen(color=_TRIGGER_TRACE_COLOR, width=2.0)
        )
        self._trace_data = {
            curve_name: ([], [])
            for curve_name in self._trace_curves
        }
        self._trace_display_x = {
            curve_name: np.empty(LaserPlotProcess.MAX_POINTS, dtype=np.float32)
            for curve_name in self._trace_curves
        }
        self._trace_display_y = {
            curve_name: np.empty(LaserPlotProcess.MAX_POINTS, dtype=np.float32)
            for curve_name in self._trace_curves
        }
        self._last_plot_frame: Optional[LaserPlotFrame] = None
        stream_configuration = self._app_model.nidaq_signal_monitor.configuration
        self._trace_window_seconds = stream_configuration.rolling_window_seconds

        self._trace_tabs = QTabWidget(trace_group)
        self._trace_tabs.setDocumentMode(True)
        self._trace_tabs.setMinimumWidth(0)
        self._trace_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        self._trace_stream_page = QWidget(self._trace_tabs)
        trace_stream_layout = QVBoxLayout(self._trace_stream_page)
        trace_stream_layout.setContentsMargins(0, 4, 0, 0)
        trace_stream_layout.setSpacing(4)
        # The live output is the reason this page is stacked, so it keeps room
        # even when the pulse controls above it are fully expanded.
        self._trace_plot.setMinimumHeight(140)
        trace_stream_layout.addWidget(self._trace_plot, stretch=1)
        self._trace_legend = StreamGraphLegend(columns=1, parent=self._trace_stream_page)
        self._trace_legend.set_entries(
            (
                ("Command output", _COMMAND_TRACE_COLOR, False),
                ("Diode feedback", _DIODE_TRACE_COLOR, False),
                ("Command copy", _COMMAND_COPY_TRACE_COLOR, False),
                ("Board trigger", _TRIGGER_TRACE_COLOR, False),
            )
        )
        trace_stream_layout.addWidget(self._trace_legend)
        trace_actions = QGridLayout()
        trace_actions.setContentsMargins(0, 0, 0, 0)
        trace_actions.setHorizontalSpacing(5)
        trace_actions.setVerticalSpacing(3)
        # No Start Stream or Start DAQ Inputs: the graph follows the shared
        # NI-DAQ input stream, which runs by itself, and the status beside
        # Clear says what that stream is doing.
        self._trace_clear_button = QPushButton("Clear")
        self._trace_seconds = QDoubleSpinBox()
        self._trace_seconds.setDecimals(1)
        self._trace_seconds.setRange(0.1, 60.0)
        self._trace_seconds.setSingleStep(0.5)
        self._trace_seconds.setValue(self._trace_window_seconds)
        self._trace_seconds.setSuffix(" s")
        self._trace_min_volts = QDoubleSpinBox()
        self._trace_min_volts.setDecimals(2)
        self._trace_min_volts.setRange(-1000.0, 1000.0)
        self._trace_min_volts.setValue(channel.minimum_command_volts)
        self._trace_min_volts.setSuffix(" V")
        self._trace_max_volts = QDoubleSpinBox()
        self._trace_max_volts.setDecimals(2)
        self._trace_max_volts.setRange(-1000.0, 1000.0)
        self._trace_max_volts.setValue(channel.maximum_command_volts)
        self._trace_max_volts.setSuffix(" V")
        self._trace_status = QLabel("")
        self._trace_status.setWordWrap(True)
        self._trace_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        for spinbox in (
            self._trace_seconds,
            self._trace_min_volts,
            self._trace_max_volts,
        ):
            spinbox.setMinimumWidth(0)
            spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        trace_status_row = QHBoxLayout()
        trace_status_row.setContentsMargins(0, 0, 0, 0)
        trace_status_row.setSpacing(5)
        trace_status_row.addWidget(self._trace_status, stretch=1)
        trace_status_row.addWidget(self._trace_clear_button)
        trace_actions.addLayout(trace_status_row, 0, 0, 1, 2)
        trace_actions.addWidget(QLabel("Window:"), 1, 0)
        trace_actions.addWidget(self._trace_seconds, 1, 1)
        trace_actions.addWidget(QLabel("Y min:"), 2, 0)
        trace_actions.addWidget(self._trace_min_volts, 2, 1)
        trace_actions.addWidget(QLabel("Y max:"), 3, 0)
        trace_actions.addWidget(self._trace_max_volts, 3, 1)
        trace_actions.setColumnStretch(1, 1)
        trace_stream_layout.addLayout(trace_actions)

        self._trace_signals_page = QWidget(self._trace_tabs)
        trace_signals_layout = QVBoxLayout(self._trace_signals_page)
        trace_signals_layout.setContentsMargins(8, 8, 8, 8)
        trace_signals_layout.setSpacing(8)
        trace_signals_explanation = QLabel(
            "Choose the signals displayed in this laser's output stream. NI-DAQ inputs are "
            "available only after their ports are assigned in Edit → Edit DAQ Ports."
        )
        trace_signals_explanation.setWordWrap(True)
        trace_signals_layout.addWidget(trace_signals_explanation)

        trace_options = QWidget(self._trace_signals_page)
        trace_options_layout = QVBoxLayout(trace_options)
        trace_options_layout.setContentsMargins(0, 0, 0, 0)
        trace_options_layout.setSpacing(8)
        self._trace_command_checkbox = QCheckBox("Command output (always shown)")
        color_code_checkbox(self._trace_command_checkbox, _COMMAND_TRACE_COLOR)
        self._trace_command_checkbox.setChecked(True)
        self._trace_command_checkbox.setEnabled(False)
        trace_options_layout.addWidget(self._trace_command_checkbox)

        self._trace_signal_candidates = {
            "diode": self._make_trace_signal_candidate("diode"),
            "copy": self._make_trace_signal_candidate("copy"),
            "trigger": self._make_trace_signal_candidate("trigger"),
        }
        self._trace_signal_checkboxes = {}
        for key, label, color in (
            ("diode", "Diode feedback", _DIODE_TRACE_COLOR),
            ("copy", "Command copy", _COMMAND_COPY_TRACE_COLOR),
            ("trigger", "Board trigger readback", _TRIGGER_TRACE_COLOR),
        ):
            candidate = self._trace_signal_candidates[key]
            physical_channel = "not configured" if candidate is None else candidate.physical_channel
            checkbox = QCheckBox(label)
            checkbox.setToolTip(physical_channel)
            color_code_checkbox(checkbox, color)
            self._trace_signal_checkboxes[key] = checkbox
            trace_options_layout.addWidget(checkbox)
        trace_options_layout.addStretch(1)
        trace_signals_layout.addWidget(trace_options)
        trace_signals_layout.addStretch(1)

        self._trace_tabs.addTab(self._trace_stream_page, "Stream")
        self._trace_tabs.addTab(self._trace_signals_page, "Signals")
        trace_layout.addWidget(self._trace_tabs)

        # The live output sits under the pulse controls that produce it, and the
        # board trigger that starts it sits under that, so a press, its waveform
        # and the edge that launched it are all in one view.
        pulse_page_layout.addWidget(trace_group, stretch=2)

        trigger_group = QGroupBox("Board Trigger")
        trigger_layout = QVBoxLayout(trigger_group)
        trigger_layout.setContentsMargins(8, 4, 8, 6)
        trigger_layout.setSpacing(2)
        trigger_layout.addWidget(self.trigger_plot)
        self.trigger_status = QLabel()
        self.trigger_status.setObjectName("LaserPreviewStatus")
        self.trigger_status.setWordWrap(True)
        trigger_layout.addWidget(self.trigger_status)
        pulse_page_layout.addWidget(trigger_group)
        self.refresh_trigger_status()

        output_page_layout.addWidget(
            QLabel(
                "The laser output stream and the board trigger readback now sit "
                "under the pulse controls on the Pulse page."
            )
        )
        output_page_layout.addStretch(1)

        self._pulse_controls = (
            self._trigger_mode,
            self._trigger_source,
            self._trigger_edge,
            self._open_shutter,
            self._close_shutter,
            self._enable_pmt,
            self._emit_trigger,
            self._emit_timing_trigger,
            self._stim_route,
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
        self._trace_clear_button.clicked.connect(self._clear_trace)
        self._trace_seconds.valueChanged.connect(self._apply_trace_view)
        self._trace_min_volts.valueChanged.connect(self._apply_trace_view)
        self._trace_max_volts.valueChanged.connect(self._apply_trace_view)
        self._apply_trace_view()
        for key, checkbox in self._trace_signal_checkboxes.items():
            checkbox.toggled.connect(
                lambda checked, signal_key=key: self._trace_signal_selection_changed(
                    signal_key,
                    checked,
                )
            )
        self.refresh_signal_selections()
        self.refresh_stream_status()
        self.refresh_stim_profiles()
        self._connect_control_signals()
        self._refresh_trigger_mode_enabled()

    @property
    def channel_id_value(self) -> int:
        return int(self._channel.channel_id)

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    def _make_trace_signal_candidate(
        self,
        signal_key: str,
    ) -> Optional[NidaqSignalChannelConfiguration]:
        if not self._is_configured:
            return None
        laser_number = self.channel_id_value
        if signal_key == "diode":
            physical_channel = self._channel.diode_input
            name = f"laser{laser_number}_diode"
            scale = self._channel.feedback_scale
        elif signal_key == "copy":
            physical_channel = self._channel.command_copy_input
            name = f"laser{laser_number}_command_copy"
            scale = self._channel.command_copy_scale
        elif signal_key == "trigger":
            physical_channel = self._channel.trigger_monitor_input
            name = f"laser{laser_number}_trigger"
            scale = 1.0
        else:
            raise ValueError(f"Unknown laser trace signal: {signal_key}")
        if not physical_channel:
            return None
        return NidaqSignalChannelConfiguration(
            name=name,
            physical_channel=physical_channel,
            kind=_nidaq_channel_kind(physical_channel),
            scale=scale,
        )

    def refresh_signal_selections(self) -> None:
        monitor = self._app_model.nidaq_signal_monitor
        configured_by_name = {
            channel.name: channel
            for channel in monitor.configuration.channels
        }
        displayed_names = set(monitor.configuration.display_channels)
        for key, checkbox in self._trace_signal_checkboxes.items():
            candidate = self._trace_signal_candidates[key]
            selected = candidate is not None and candidate.name in displayed_names
            checkbox.blockSignals(True)
            checkbox.setChecked(selected)
            checkbox.blockSignals(False)
            checkbox.setEnabled(
                candidate is not None
                and monitor.hardware_enabled
                and not monitor.is_starting
            )
            channel_tooltip = "" if candidate is None else f"{candidate.physical_channel}\n"
            if candidate is None:
                checkbox.setToolTip("Assign this input in Edit → Edit DAQ Ports first.")
            elif not monitor.hardware_enabled:
                checkbox.setToolTip(
                    channel_tooltip + "NI-DAQ hardware is disabled in the system configuration."
                )
            elif monitor.is_starting:
                checkbox.setToolTip(
                    channel_tooltip + "Wait for the shared NI-DAQ stream to finish starting."
                )
            elif selected and configured_by_name[candidate.name].physical_channel != candidate.physical_channel:
                checkbox.setToolTip(
                    channel_tooltip
                    + "The saved input uses an older port mapping. Toggle this option to apply the current mapping."
                )
            elif monitor.is_running:
                checkbox.setToolTip(
                    channel_tooltip + "Changing this option restarts the shared NI-DAQ input worker."
                )
            else:
                checkbox.setToolTip(
                    channel_tooltip + "Include this input in this laser's streaming graph."
                )

    def refresh_stream_status(self) -> None:
        """The shared stream's state, then what the latest pulse or ramp did."""
        monitor = self._app_model.nidaq_signal_monitor
        text = f"NI-DAQ inputs {monitor.stream_state}"
        if self._trace_note:
            text += f" — {self._trace_note}"
        self._trace_status.setText(text)
        self._trace_status.setToolTip(monitor.error_message or monitor.status_message)

    def _trace_signal_selection_changed(self, signal_key: str, checked: bool) -> None:
        candidate = self._trace_signal_candidates.get(signal_key)
        if candidate is None:
            return
        configuration = self._app_model.nidaq_signal_monitor.configuration
        channel_names = [
            name
            for name in configuration.display_channels
            if name != candidate.name
        ]
        if checked:
            channel_names.append(candidate.name)
        self._app_model.update_nidaq_signal_stream_channels(channel_names)
        self.refresh_signal_selections()

    @staticmethod
    def _form_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
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

    def run_pulse_refusal(self) -> str:
        """Why Run Pulse is unavailable, or an empty string.

        A greyed button with no explanation reads as a fault; both reasons
        here are ordinary states the operator can act on.
        """
        if not self._is_configured:
            return (
                f"Laser {self._channel.channel_id.value} has no hardware "
                "channel in the system configuration"
            )
        if not self._app_model.laser.is_connected:
            return (
                "The laser controller is not open: it claims its NI-DAQ "
                "channels when the system starts, so press Run first"
            )
        return ""

    def _connect_control_signals(self) -> None:
        self._trigger_mode.currentTextChanged.connect(self._on_trigger_mode_changed)
        self.stim_profile_selector.currentIndexChanged.connect(self.refresh_draft)

    def set_controls_enabled(self, can_edit: bool, can_run_pulse: bool, can_run_ramp: bool) -> None:
        self._controls_can_edit = can_edit
        for control in self._pulse_controls:
            control.setEnabled(can_edit)
        self._refresh_trigger_mode_enabled()
        self._run_pulse_button.setEnabled(can_run_pulse)
        self._run_pulse_button.setToolTip(self.run_pulse_refusal())
        # A stim test needs the same open controller. Left out of this method
        # it stayed enabled before the system started, and failed with "Laser
        # controller is not configured".
        self.stim_test_button.setEnabled(can_run_pulse)
        self.stim_test_button.setToolTip(
            self.run_pulse_refusal() or _STIM_TEST_TOOLTIP)
        for control in self._ramp_controls:
            control.setEnabled(can_edit)
        self._run_ramp_button.setEnabled(can_run_ramp)

    def append_trace(self, trace: LaserTraceBlock, *, redraw: bool = True) -> None:
        if int(trace.channel_id) != self.channel_id_value:
            return
        is_calibration = trace.source == "calibration"
        if is_calibration:
            self._set_trace_streaming(True)
        if not self._trace_streaming:
            return
        if not trace.x_values:
            self._trace_note = "calibration running, collecting diode feedback..."
        elif is_calibration:
            self._trace_note = (
                f"calibration complete, {len(trace.x_values)} ramp points displayed"
            )
        else:
            self._trace_note = f"latest: {trace.source}"
        self.refresh_stream_status()
        if self._plot_controller is not None:
            self._plot_controller.submit_laser_trace(trace)

    def redraw_trace(self) -> None:
        frame = self._last_plot_frame
        if frame is None:
            return
        for curve_name, curve in self._trace_curves.items():
            slot = LaserPlotProcess.curve_slot(self.channel_id_value, curve_name)
            point_count = frame.point_counts[slot]
            curve_x = self._trace_display_x[curve_name][:point_count]
            curve_y = self._trace_display_y[curve_name][:point_count]
            self._trace_data[curve_name] = (curve_x, curve_y)
            curve.setData(curve_x, curve_y, skipFiniteCheck=True)

    def accept_plot_frame(self, frame: LaserPlotFrame, *, redraw: bool) -> None:
        self._last_plot_frame = frame
        if redraw:
            self.redraw_trace()

    def physical_plot_width(self) -> int:
        viewport = self._trace_plot.viewport()
        return max(1, round(viewport.width() * viewport.devicePixelRatioF()))

    def _apply_trace_view(self, *_args) -> None:
        seconds = float(self._trace_seconds.value())
        if seconds != self._trace_window_seconds:
            self._trace_window_seconds = seconds

        minimum = self._trace_min_volts.value()
        maximum = self._trace_max_volts.value()
        if maximum <= minimum:
            changed = self.sender()
            if changed is self._trace_min_volts:
                self._trace_max_volts.setValue(minimum + 0.01)
            else:
                self._trace_min_volts.setValue(maximum - 0.01)
            minimum = self._trace_min_volts.value()
            maximum = self._trace_max_volts.value()
        self._trace_plot.setYRange(minimum, maximum, padding=0)
        self._trace_plot.setXRange(-self._trace_window_seconds, 0.0, padding=0)
        # Same time base, so an edge below lines up with the waveform above.
        self.trigger_plot.setXRange(-self._trace_window_seconds, 0.0, padding=0)
        if self._plot_controller is not None:
            self._plot_controller.configure_laser_plot(self)
        self.redraw_trace()

    def _set_trace_streaming(self, is_streaming: bool) -> None:
        """Whether this graph takes samples and traces; set by LaserControlContent."""
        self._trace_streaming = is_streaming
        if self._plot_controller is not None:
            self._plot_controller.set_laser_plot_streaming(
                self.channel_id_value,
                is_streaming,
            )

    def _clear_trace(self) -> None:
        if self._plot_controller is not None:
            self._plot_controller.clear_laser_plot(self.channel_id_value)
        self._last_plot_frame = None
        for curve_name, curve in self._trace_curves.items():
            self._trace_data[curve_name] = ([], [])
            curve.setData([], [])
        self._trace_note = "cleared"
        self.refresh_stream_status()

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

    def refresh_trigger_status(self) -> None:
        """Say whether the board trigger can be read back at all."""
        candidate = self._trace_signal_candidates.get("trigger")
        if candidate is None:
            self.trigger_status.setText(
                "No trigger readback input is configured for this laser. Wire the "
                "board stimulus line into an NI input and set it in Edit DAQ Ports "
                "to see the edge that starts the waveform."
            )
            return
        self.trigger_status.setText(
            "Reading {} as {}. Enable it under Signals.".format(
                candidate.physical_channel, candidate.kind
            )
        )

    def refresh_stim_profiles(self) -> None:
        """(none), the builder draft and every saved profile; any fits any laser."""
        previous = self.stim_profile_selector.currentData()
        self.stim_profile_selector.blockSignals(True)
        self.stim_profile_selector.clear()
        self.stim_profile_selector.addItem("(none)", None)
        self.stim_profile_selector.addItem("(builder draft)", DRAFT_PROFILE_ID)
        state = getattr(self._app_model, "trial_protocol_state", {}) or {}
        for item in state.get("laser_profiles", ()):
            self.stim_profile_selector.addItem(item["profile_id"], item["profile_id"])
        index = self.stim_profile_selector.findData(previous) if previous else -1
        self.stim_profile_selector.setCurrentIndex(max(0, index))
        self.stim_profile_selector.blockSignals(False)
        if previous and index < 0:
            self._report_vanished_profile(previous)
        self.refresh_draft()

    def select_profile(self, profile_id) -> None:
        """Pick this profile; one no longer listed leaves "(none)" and says so."""
        if not profile_id:
            return
        index = self.stim_profile_selector.findData(profile_id)
        self.stim_profile_selector.blockSignals(True)
        self.stim_profile_selector.setCurrentIndex(max(0, index))
        self.stim_profile_selector.blockSignals(False)
        if index < 0:
            self._report_vanished_profile(profile_id)
        self.refresh_draft()

    def _report_vanished_profile(self, profile_id) -> None:
        # Falling back to the builder draft here made a laser whose profile
        # was deleted fire whatever was on the builder, with no word of it.
        self._set_parent_status(
            f"Laser {self._channel.channel_id.value}: profile {profile_id!r} is "
            "no longer saved; pick a profile",
            True,
        )

    def refresh_draft(self) -> None:
        profile_id = self.stim_profile_selector.currentData()
        profile = self._selected_profile()
        if profile is not None:
            text = profile.summary()
        elif not profile_id:
            text = "No profile selected"
        elif profile_id == DRAFT_PROFILE_ID:
            text = "The builder draft is not a valid pulse train"
        else:
            text = f"Profile {profile_id!r} is no longer saved"
        self._profile_summary.setText(text)

    def _profile_refusal(self) -> str:
        """Why the current pick cannot fire; call only when it gave no profile."""
        laser = self._channel.channel_id.value
        profile_id = self.stim_profile_selector.currentData()
        if not profile_id:
            return f"Laser {laser}: pick a saved profile or the builder draft"
        if profile_id == DRAFT_PROFILE_ID:
            return (
                f"Laser {laser}: the builder draft is not a valid pulse train; "
                "fix it in the Pulse Builder"
            )
        return f"Laser {laser}: profile {profile_id!r} is no longer saved; pick a profile"

    def _refresh_stim_route_options(self) -> None:
        self._stim_route.clear()
        line = self._channel.board_stim_line
        terminal = (self._channel.trigger_source or "").strip()
        self._stim_route.addItem(
            f"Board STIM (STIM{line} → {terminal})" if line and terminal
            else "Board STIM", LaserTriggerRoute.HARDWARE_STIM3.value)
        self._stim_route.addItem("Software start", LaserTriggerRoute.DIRECT_NI_SOFTWARE.value)
        if not (line and terminal):
            self._stim_route.model().item(0).setEnabled(False)
            self._stim_route.setItemData(
                0, "This laser has no trigger terminal or board STIM line configured",
                Qt.ItemDataRole.ToolTipRole)
            self._stim_route.setCurrentIndex(1)

    def _selected_profile(self):
        profile_id = self.stim_profile_selector.currentData()
        if profile_id == DRAFT_PROFILE_ID:
            return None if self._draft_provider is None else self._draft_provider()
        return self._app_model.laser_profile(profile_id) if profile_id else None

    def _run_stim_test(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._set_parent_status(self._profile_refusal(), True)
            return
        channel_id = int(self._channel.channel_id.value)
        route = self._stim_route.currentData()

        def operation():
            result = self._app_model.run_stim_bench_test(profile, channel_id, route)
            invoke_method(lambda: self.stim_test_result.setText(str(result)))()
            return str(result)

        self._start_operation(
            "Running stim test {} on laser {}".format(profile.profile_id, channel_id),
            operation,
        )

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
        profile = self._selected_profile()
        if profile is None:
            raise ValueError(self._profile_refusal())
        trigger_source = None
        if self._trigger_mode.currentText() == "external":
            trigger_source = self._trigger_source.text().strip()
            if not trigger_source:
                raise ValueError("External trigger mode requires a trigger source")
        return LaserPulseTrain(
            channel_id=self._channel.channel_id,
            amplitude_volts=profile.amplitude_volts,
            duration_ms=profile.pulse_duration_ms,
            baseline_ms=profile.baseline_ms,
            post_stim_ms=profile.post_stim_ms,
            pulse_count=profile.pulse_count,
            frequency_hz=profile.frequency_hz if profile.pulse_count > 1 else None,
            trigger_source=trigger_source,
            trigger_edge=self._trigger_edge.currentText(),
            open_shutter=self._open_shutter.isChecked(),
            close_shutter=self._close_shutter.isChecked(),
            enable_pmt_shutter=self._enable_pmt.isChecked(),
            pmt_shutter_open_delay_ms=profile.pmt_open_lead_ms,
            pmt_shutter_close_delay_ms=profile.pmt_close_lag_ms,
            emit_trigger_output=self._emit_trigger.isChecked(),
            emit_timing_trigger_output=self._emit_timing_trigger.isChecked(),
        )

    def _on_trigger_mode_changed(self) -> None:
        self._refresh_trigger_mode_enabled()

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
        self._plot_process = LaserPlotProcess(app_model.nidaq_signal_monitor.sample_ring)
        self._plot_configuration_signatures = {}
        self._plot_x_destinations = {}
        self._plot_y_destinations = {}

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
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
        self._stream_telemetry_label = QLabel("Seq 0 | Overruns 0 | Latency n/a")
        self._stream_telemetry_label.setObjectName("LaserMetaValue")
        header_layout.addWidget(self._stream_telemetry_label)

        self._card_widget = CardWidget(title="Laser Control", header_right_layout=header_layout)
        self._card_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setMinimumWidth(0)
        self._tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._tabs.currentChanged.connect(self._redraw_current_trace)
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
        app_model.nidaq_signal_monitor.property_changed += self._on_nidaq_monitor_property_changed
        self._stream_plot_timer = QTimer(self)
        self._stream_plot_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._stream_plot_timer.setInterval(max(
            1,
            int(round(1000.0 / app_model.nidaq_signal_monitor.display_refresh_rate_hz)),
        ))
        self._stream_plot_timer.timeout.connect(self._flush_laser_plots)
        self._stream_plot_timer.start()
        self._builder = PulseBuilderTab(app_model, self._set_status_from_tab)
        self._builder.profiles_changed.connect(self._refresh_channel_profiles)
        self._builder.draft_changed.connect(self._refresh_channel_drafts)
        self._refresh_from_model()

    def on_close(self):
        self._stream_plot_timer.stop()
        self._app_model.laser.property_changed -= self._on_laser_property_changed
        self._app_model.laser.trace_received -= self._on_laser_trace_received
        self._app_model.nidaq_signal_monitor.property_changed -= self._on_nidaq_monitor_property_changed
        self._plot_process.close()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        current_tab = self._tabs.currentWidget()
        if isinstance(current_tab, _LaserChannelTab):
            current_tab.redraw_trace()

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
    def _on_nidaq_monitor_property_changed(self, property_name: str, _value, _old_value) -> None:
        if property_name in (
            NidaqSignalMonitorModel.CONFIGURATION,
            NidaqSignalMonitorModel.HARDWARE_ENABLED,
            NidaqSignalMonitorModel.IS_STARTING,
            NidaqSignalMonitorModel.IS_RUNNING,
        ):
            for tab in self._channel_tabs:
                tab.refresh_signal_selections()
        if property_name in (
            NidaqSignalMonitorModel.CONFIGURATION,
            NidaqSignalMonitorModel.HARDWARE_ENABLED,
            NidaqSignalMonitorModel.IS_STARTING,
            NidaqSignalMonitorModel.IS_RUNNING,
            NidaqSignalMonitorModel.STATUS_MESSAGE,
            NidaqSignalMonitorModel.ERROR_MESSAGE,
        ):
            for tab in self._channel_tabs:
                tab.refresh_stream_status()

    def _flush_laser_plots(self) -> None:
        ring = self._app_model.nidaq_signal_monitor.sample_ring
        if ring is not self._plot_process.raw_ring:
            self._plot_process.close()
            self._plot_process = LaserPlotProcess(ring)
            self._plot_configuration_signatures.clear()
            for tab in self._channel_tabs:
                self.configure_laser_plot(tab, reset=True)
                self.set_laser_plot_streaming(
                    tab.channel_id_value,
                    tab._trace_streaming,
                )
        interval = max(
            1,
            int(round(
                1000.0 / self._app_model.nidaq_signal_monitor.display_refresh_rate_hz,
            )),
        )
        if interval != self._stream_plot_timer.interval():
            self._stream_plot_timer.setInterval(interval)
        for tab in self._channel_tabs:
            self.configure_laser_plot(tab)
        frame = self._plot_process.copy_latest_into(
            self._plot_x_destinations,
            self._plot_y_destinations,
        )
        if frame is None:
            return
        current_tab = self._tabs.currentWidget()
        for tab in self._channel_tabs:
            tab.accept_plot_frame(
                frame,
                redraw=self.isVisible() and tab is current_tab,
            )
        self._stream_telemetry_label.setText(
            f"Seq {frame.raw_generation} | Overruns {frame.overrun_count} | "
            f"Gaps {frame.gap_count} | Latency {frame.source_latency_ms:.1f} ms"
        )

    def _redraw_current_trace(self, _index: int) -> None:
        current_tab = self._tabs.currentWidget()
        if isinstance(current_tab, _LaserChannelTab):
            current_tab.redraw_trace()

    def configure_laser_plot(self, tab: _LaserChannelTab, *, reset: bool = False) -> None:
        names_by_physical_channel = {
            channel.physical_channel: channel.name
            for channel in self._app_model.nidaq_signal_monitor.configuration.channels
        }
        diode_name = names_by_physical_channel.get(tab._channel.diode_input)
        copy_name = names_by_physical_channel.get(tab._channel.command_copy_input)
        trigger_name = names_by_physical_channel.get(
            tab._channel.trigger_monitor_input
        )
        signature = (
            tab._trace_window_seconds,
            tab.physical_plot_width(),
            diode_name,
            copy_name,
            trigger_name,
        )
        if not reset and self._plot_configuration_signatures.get(tab.channel_id_value) == signature:
            return
        self._plot_configuration_signatures[tab.channel_id_value] = signature
        self._plot_process.configure_channel(
            tab.channel_id_value,
            window_seconds=signature[0],
            pixel_width=signature[1],
            diode_name=diode_name,
            copy_name=copy_name,
            trigger_name=trigger_name,
        )
        if reset:
            self._plot_process.clear(tab.channel_id_value)
            self._plot_process.set_streaming(tab.channel_id_value, False)

    def set_laser_plot_streaming(self, channel_id: int, is_streaming: bool) -> None:
        self._plot_process.set_streaming(channel_id, is_streaming)

    def submit_laser_trace(self, trace: LaserTraceBlock) -> None:
        self._plot_process.submit_trace(trace)

    def clear_laser_plot(self, channel_id: int) -> None:
        self._plot_process.clear(channel_id)

    def _refresh_channel_profiles(self) -> None:
        for tab in self._channel_tabs:
            tab.refresh_stim_profiles()

    def _refresh_channel_drafts(self) -> None:
        for tab in self._channel_tabs:
            tab.refresh_draft()

    @staticmethod
    def _amplitude_range(configuration) -> Tuple[float, float]:
        channels = tuple(configuration.channels)
        if not channels:
            return _DEFAULT_MINIMUM_COMMAND_VOLTS, _DEFAULT_MAXIMUM_COMMAND_VOLTS
        return (min(c.minimum_command_volts for c in channels),
                max(c.maximum_command_volts for c in channels))

    def _refresh_from_model(self) -> None:
        configuration = self._app_model.laser.configuration
        current = self._current_channel_id()
        self._backend_label.setText(configuration.backend)
        if configuration.sample_rate_hz is None:
            self._sample_rate_label.setText("manual")
        else:
            self._sample_rate_label.setText(f"{configuration.sample_rate_hz:g} Hz")

        # Every system Run/Stop and DAQ Ports save rebuilds the laser tabs. A
        # new tab starts on "(none)", so without this every laser lost its
        # pick; a pick that is no longer saved stays "(none)" and is reported.
        picked_profiles = {
            tab.channel_id_value: tab.stim_profile_selector.currentData()
            for tab in self._channel_tabs
        }
        self._clear_tabs()
        self._builder.set_amplitude_range(*self._amplitude_range(configuration))
        if self._tabs.indexOf(self._builder) < 0:
            self._tabs.insertTab(0, self._builder, "Pulse Builder")
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
                self,
                draft_provider=self._builder.draft_profile,
            )
            tab.select_profile(picked_profiles.get(channel_index))
            self._tabs.addTab(tab, f"Laser {channel_index}")
            tabs.append(tab)
        self._channel_tabs = tuple(tabs)
        self._plot_x_destinations = {
            (tab.channel_id_value, curve_name): values
            for tab in self._channel_tabs
            for curve_name, values in tab._trace_display_x.items()
        }
        self._plot_y_destinations = {
            (tab.channel_id_value, curve_name): values
            for tab in self._channel_tabs
            for curve_name, values in tab._trace_display_y.items()
        }
        self._plot_configuration_signatures.clear()
        for tab in self._channel_tabs:
            self.configure_laser_plot(tab, reset=True)
        # Every mapped laser's graph takes the shared stream's samples from
        # the start, in Idle too; they used to only while System Mode ran,
        # or after its own Start Stream. The stream itself starts by itself.
        for tab in self._channel_tabs:
            if tab.is_configured:
                tab._set_trace_streaming(True)
        if current is not None:
            for index, tab in enumerate(self._channel_tabs):
                if tab.channel_id_value == current:
                    self._tabs.setCurrentIndex(index + 1)
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
        # The builder stays: its unsaved draft must survive a configuration reload.
        for tab in self._channel_tabs:
            self._tabs.removeTab(self._tabs.indexOf(tab))
            tab.deleteLater()
        self._channel_tabs = tuple()
        self._plot_x_destinations = {}
        self._plot_y_destinations = {}

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
    def _operation_failed(self, _message: str) -> None:
        self._set_status("Laser operation stopped", is_error=False)

    @Slot()
    def _operation_thread_finished(self) -> None:
        self._operation_thread = None
        self._operation_worker = None
        self._progress.setVisible(False)
        self._set_running(False)

    def _set_status(self, message: str, *, is_error: bool) -> None:
        if is_error:
            logger.error("Laser control operation rejected: %s", message)
            return
        self._status_label.setText(message)
        self._status_label.setStyleSheet("")

    def _set_running(self, is_running: bool) -> None:
        self._update_enabled_state(is_running=is_running)

    def _update_enabled_state(self, *, is_running: Optional[bool] = None) -> None:
        if is_running is None:
            is_running = self._operation_thread is not None
        can_edit = self._is_editable and not is_running
        self._builder.set_controls_enabled(can_edit)
        can_run = can_edit and self._app_model.laser.is_connected
        refusals = []
        for tab in self._channel_tabs:
            can_run_pulse = can_run and tab.is_configured
            can_run_ramp = can_run_pulse and not self._is_capture_active
            tab.set_controls_enabled(can_edit, can_run_pulse, can_run_ramp)
            refusal = tab.run_pulse_refusal()
            if refusal and refusal not in refusals:
                refusals.append(refusal)
        # Nothing can be fired and the buttons alone do not say why, so the
        # shared status line carries the reason.
        if refusals and not any(
            tab.is_configured and can_run for tab in self._channel_tabs
        ):
            self._set_status(refusals[0], is_error=False)

    @invoke_method
    def set_is_editable(self, is_editable: bool):
        self._is_editable = is_editable
        self._update_enabled_state()

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        # The graphs no longer follow System Mode; they follow the stream.
        self._is_capture_active = is_active
        self._update_enabled_state()
