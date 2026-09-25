from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
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
from tools.acquisition.view.compact_panel import (
    FIELD_MAXIMUM_WIDTH,
    WIDE_FIELD_MAXIMUM_WIDTH,
    CollapsibleSection,
    ElidedLabel,
    compact_button_style_sheet,
    compact_combo_box,
    compact_font_style_sheet,
    compact_plot_axes,
    field_grid,
    form_label,
)
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
_TRACE_SIGNALS_EXPLANATION = (
    "Choose the signals displayed in this laser's output stream; a change "
    "shows at once, while the stream runs. NI-DAQ inputs are available only "
    "after their ports are assigned in Edit → Edit DAQ Ports."
)
# Sized so the whole Pulse page, every section open, fits the docked panel
# (440 x 860 px on christielab10's 1920x1080 screen) with no scroll bar.
_TRACE_PLOT_MINIMUM_HEIGHT = 140
_TRIGGER_PLOT_HEIGHT = 76


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
            "#LaserChannelTab QCheckBox {color: #2f343a; spacing: 3px;}"
            "#LaserChannelTab QCheckBox:disabled {color: #68717d;}"
            "#LaserChannelTab QLineEdit:disabled,"
            "#LaserChannelTab QSpinBox:disabled,"
            "#LaserChannelTab QDoubleSpinBox:disabled,"
            "#LaserChannelTab QComboBox:disabled {color: #4f5965; background-color: #edf0f3;}"
            + compact_button_style_sheet("LaserChannelTab")
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        sample_rate = "manual" if sample_rate_hz is None else f"{sample_rate_hz:g} Hz"
        if is_configured:
            channel_text = (
                f"Rate {sample_rate} | AO {channel.analog_output} | "
                f"Diode {channel.diode_input} | Shutter {channel.shutter_output}"
            )
        else:
            channel_text = f"Rate {sample_rate} | Hardware channel not mapped"
        channel_label = ElidedLabel(channel_text)
        channel_label.setObjectName("LaserChannelSummary")
        layout.addWidget(channel_label)

        #: Every folding section on this tab, by name; LaserControlContent
        #: keeps their open/closed state across the tab rebuilds.
        self.sections = {}

        self._mode_tabs = QTabWidget(self)
        self._mode_tabs.setDocumentMode(True)
        self._mode_tabs.setMinimumWidth(0)
        self._mode_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._mode_tabs, stretch=1)

        pulse_page = QWidget(self._mode_tabs)
        pulse_page_layout = QVBoxLayout(pulse_page)
        pulse_page_layout.setContentsMargins(2, 3, 2, 2)
        pulse_page_layout.setSpacing(2)
        calibration_page = QWidget(self._mode_tabs)
        calibration_page_layout = QVBoxLayout(calibration_page)
        calibration_page_layout.setContentsMargins(2, 3, 2, 2)
        calibration_page_layout.setSpacing(3)
        # Everything on the pulse page fits the docked panel with every
        # section open; the scroll area is only a safety net for a panel
        # made shorter than that, rather than pushing the graphs out of reach.
        pulse_scroll = QScrollArea(self._mode_tabs)
        pulse_scroll.setWidgetResizable(True)
        pulse_scroll.setFrameShape(QFrame.Shape.NoFrame)
        pulse_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        pulse_scroll.setWidget(pulse_page)
        # Transparent, so the page is white like the Calibration page beside
        # it rather than the scroll area's grey.
        pulse_scroll.setObjectName("LaserPulseScroll")
        pulse_page.setObjectName("LaserPulsePage")
        pulse_scroll.setStyleSheet(
            "#LaserPulseScroll, #LaserPulseScroll > #qt_scrollarea_viewport, "
            "#LaserPulsePage {background: transparent;}"
        )
        # No Output page any more: the output stream and the board trigger
        # sit under the pulse controls, and it only said so.
        self._mode_tabs.addTab(pulse_scroll, "Pulse")
        self._mode_tabs.addTab(calibration_page, "Calibration")

        # Run Pulse and Test stim both fire the profile picked here, on this
        # laser: the Pulse Builder's unsaved draft or any saved profile. A
        # profile is the waveform alone; Run Pulse starts it with the trigger
        # and shutter options below, Test stim by the route chosen beside it.
        # A new tab picks "(none)", so a laser only fires what someone chose
        # for it, never whatever happens to be on the builder.
        self.stim_profile_selector = compact_combo_box()
        self.stim_profile_selector.setToolTip(
            "The Pulse Builder draft and every saved laser profile; any of "
            "them can fire on this laser"
        )
        self._profile_summary = ElidedLabel("")
        self._profile_summary.setObjectName("LaserPreviewStatus")
        self._trigger_mode = compact_combo_box()
        self._trigger_mode.addItems(("internal", "external"))
        self._trigger_mode.setToolTip(
            "internal: start on the NI clock when Run Pulse is pressed; "
            "external: arm and wait for an edge on the trigger source"
        )
        # Internal by default, even when the channel has a trigger route. This
        # used to switch to external whenever one was configured, so Run pulse
        # armed the output for a board STIM pulse that nothing on this tab
        # sends, and failed with a DAQmx timeout (christielab10, 2026-09-24).
        # The route stays filled in for choosing external deliberately.
        self._trigger_source = QLineEdit(channel.trigger_source or "")
        self._trigger_source.setPlaceholderText("NI-DAQ trigger route")
        self._trigger_source.setToolTip("The terminal an external trigger edge arrives on")
        self._trigger_edge = compact_combo_box()
        self._trigger_edge.addItems(("rising", "falling"))
        self._trigger_edge.setToolTip("Which edge of an external trigger starts the pulse")

        self._open_shutter = self._make_checkbox("Open shutter")
        self._open_shutter.setChecked(True)
        self._close_shutter = self._make_checkbox("Close shutter")
        self._close_shutter.setChecked(True)
        self._enable_pmt = self._make_checkbox("PMT shutter")
        self._emit_trigger = self._make_checkbox("Trigger DO")
        self._emit_timing_trigger = self._make_checkbox("Timing DO")
        self._run_pulse_button = QPushButton("Run Pulse")

        profile_row = QHBoxLayout()
        profile_row.setContentsMargins(0, 0, 0, 0)
        profile_row.setSpacing(4)
        profile_row.addWidget(self._form_label("Profile:"))
        self.stim_profile_selector.setMaximumWidth(WIDE_FIELD_MAXIMUM_WIDTH)
        profile_row.addWidget(self.stim_profile_selector, stretch=1)
        profile_row.addStretch(0)
        pulse_page_layout.addLayout(profile_row)
        pulse_page_layout.addWidget(self._profile_summary)

        run_section = self._add_section("run_pulse", "Run Pulse")
        run_layout = QVBoxLayout(run_section.content)
        run_layout.setContentsMargins(4, 0, 2, 2)
        run_layout.setSpacing(2)
        trigger_row = QHBoxLayout()
        trigger_row.setContentsMargins(0, 0, 0, 0)
        trigger_row.setSpacing(4)
        trigger_row.addWidget(self._form_label("Trigger:"))
        trigger_row.addWidget(self._trigger_mode)
        trigger_row.addWidget(self._form_label("Edge:"))
        trigger_row.addWidget(self._trigger_edge)
        trigger_row.addWidget(self._form_label("Source:"))
        self._trigger_source.setMaximumWidth(WIDE_FIELD_MAXIMUM_WIDTH)
        trigger_row.addWidget(self._trigger_source, stretch=1)
        trigger_row.addStretch(0)
        run_layout.addLayout(trigger_row)
        run_options = QGridLayout()
        run_options.setContentsMargins(0, 0, 0, 0)
        run_options.setHorizontalSpacing(10)
        run_options.setVerticalSpacing(1)
        run_options.addWidget(self._open_shutter, 0, 0)
        run_options.addWidget(self._close_shutter, 0, 1)
        run_options.addWidget(self._enable_pmt, 0, 2)
        run_options.addWidget(self._emit_trigger, 1, 0)
        run_options.addWidget(self._emit_timing_trigger, 1, 1)
        run_options.setColumnStretch(3, 1)
        run_options.addWidget(
            self._run_pulse_button, 1, 4, alignment=Qt.AlignmentFlag.AlignRight)
        run_layout.addLayout(run_options)
        pulse_page_layout.addWidget(run_section)

        # Run Pulse above drives the analog output straight from the host. Test
        # stim fires the same profile the way a trial does: arm the output,
        # then start it by the route chosen here - the board's timed STIM
        # pulse into this laser's trigger terminal, or a software start.
        self._stim_route = compact_combo_box()
        self._stim_route.setToolTip(
            "How Test stim starts the profile on this laser")
        self._refresh_stim_route_options()
        self.stim_test_button = QPushButton("Test stim")
        self.stim_test_button.setToolTip(_STIM_TEST_TOOLTIP)
        self.stim_test_button.clicked.connect(self._run_stim_test)
        # Wrapped rather than elided: it is the outcome the test is run for.
        self.stim_test_result = QLabel()
        self.stim_test_result.setWordWrap(True)
        self.stim_test_result.setObjectName("LaserPreviewStatus")
        stim_section = self._add_section("test_stim", "Test stim")
        stim_layout = QVBoxLayout(stim_section.content)
        stim_layout.setContentsMargins(4, 0, 2, 2)
        stim_layout.setSpacing(2)
        stim_row = QHBoxLayout()
        stim_row.setContentsMargins(0, 0, 0, 0)
        stim_row.setSpacing(4)
        stim_row.addWidget(self._form_label("Route:"))
        self._stim_route.setMaximumWidth(WIDE_FIELD_MAXIMUM_WIDTH)
        stim_row.addWidget(self._stim_route, stretch=1)
        stim_row.addStretch(0)
        stim_row.addWidget(self.stim_test_button)
        stim_layout.addLayout(stim_row)
        stim_layout.addWidget(self.stim_test_result)
        pulse_page_layout.addWidget(stim_section)

        ramp_section = self._add_section("calibration_ramp", "Calibration ramp")

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

        ramp_layout = field_grid(ramp_section.content, (
            ("Start:", self._ramp_start), ("Stop:", self._ramp_stop),
            ("Steps:", self._ramp_steps), ("Samples/step:", self._ramp_samples_per_step),
        ))
        ramp_layout.addWidget(self._ramp_pmt, 2, 0, 1, 2)
        ramp_layout.addWidget(
            self._run_ramp_button, 2, 3, 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
        calibration_page_layout.addWidget(ramp_section)
        calibration_page_layout.addStretch(1)

        self._trace_plot = PGWidget()
        self._trace_plot.setBackground("w")
        self._trace_plot.getAxis("bottom").setLabel("Time from latest sample", units="s")
        self._trace_plot.getAxis("left").setLabel("Voltage", units="V")
        compact_plot_axes(self._trace_plot)
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
        # No time label: it shares the output stream's time base, labelled
        # just above, and the height is better spent on the edge itself.
        self.trigger_plot.getAxis("left").setLabel("Trigger", units="V")
        compact_plot_axes(self.trigger_plot)
        self.trigger_plot.getPlotItem().setClipToView(True)
        self.trigger_plot.setMouseEnabled(x=False, y=False)
        # Small and fixed: a single TTL edge beside a waveform, never
        # squeezed to nothing when the page is crowded.
        self.trigger_plot.setFixedHeight(_TRIGGER_PLOT_HEIGHT)
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

        # The live output sits under the pulse controls that produce it, and the
        # board trigger that starts it sits under that, so a press, its waveform
        # and the edge that launched it are all in one view.
        trace_section = self._add_section("output_stream", "Output stream", stretch=True)
        trace_layout = QVBoxLayout(trace_section.content)
        trace_layout.setContentsMargins(0, 0, 0, 1)
        trace_layout.setSpacing(2)
        # The live output is the reason this page is stacked, so it keeps room
        # even when every section above it is open.
        self._trace_plot.setMinimumHeight(_TRACE_PLOT_MINIMUM_HEIGHT)
        trace_layout.addWidget(self._trace_plot, stretch=1)
        self._trace_legend = StreamGraphLegend(columns=2, parent=trace_section.content)
        # Filled by _apply_curve_visibility with the curves that are shown.
        self._trace_legend_entries = {
            "command": ("Command output", _COMMAND_TRACE_COLOR, False),
            "diode": ("Diode feedback", _DIODE_TRACE_COLOR, False),
            "copy": ("Command copy", _COMMAND_COPY_TRACE_COLOR, False),
            "trigger": ("Board trigger", _TRIGGER_TRACE_COLOR, False),
        }
        trace_layout.addWidget(self._trace_legend)
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
        self._trace_seconds.setToolTip("How many seconds of the stream the graph shows")
        self._trace_min_volts = QDoubleSpinBox()
        self._trace_min_volts.setDecimals(2)
        self._trace_min_volts.setRange(-1000.0, 1000.0)
        self._trace_min_volts.setValue(channel.minimum_command_volts)
        self._trace_min_volts.setSuffix(" V")
        self._trace_min_volts.setToolTip("Bottom of the graph's voltage axis")
        self._trace_max_volts = QDoubleSpinBox()
        self._trace_max_volts.setDecimals(2)
        self._trace_max_volts.setRange(-1000.0, 1000.0)
        self._trace_max_volts.setValue(channel.maximum_command_volts)
        self._trace_max_volts.setSuffix(" V")
        self._trace_max_volts.setToolTip("Top of the graph's voltage axis")
        self._trace_status = ElidedLabel("")
        self._trace_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        trace_status_row = QHBoxLayout()
        trace_status_row.setContentsMargins(0, 0, 0, 0)
        trace_status_row.setSpacing(4)
        trace_status_row.addWidget(self._trace_status, stretch=1)
        trace_status_row.addWidget(self._trace_clear_button)
        trace_layout.addLayout(trace_status_row)
        trace_view_row = QHBoxLayout()
        trace_view_row.setContentsMargins(0, 0, 0, 0)
        trace_view_row.setSpacing(4)
        for text, spinbox in (
            ("Window:", self._trace_seconds),
            ("Y min:", self._trace_min_volts),
            ("Y max:", self._trace_max_volts),
        ):
            spinbox.setMaximumWidth(FIELD_MAXIMUM_WIDTH)
            spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            trace_view_row.addWidget(self._form_label(text))
            trace_view_row.addWidget(spinbox, stretch=1)
        trace_view_row.addStretch(0)
        trace_layout.addLayout(trace_view_row)

        # Which signals the graph draws. Folded away by default: it is set
        # once per rig, and the graph needs the height more.
        signals_section = self._add_section("signals", "Signals", expanded=False)
        signals_section.setToolTip(_TRACE_SIGNALS_EXPLANATION)
        trace_layout.addWidget(signals_section)
        signals_layout = QVBoxLayout(signals_section.content)
        signals_layout.setContentsMargins(4, 0, 2, 1)
        signals_layout.setSpacing(1)
        signals_note = ElidedLabel(
            "Each shows or hides at once. Inputs need ports in Edit → Edit DAQ Ports.")
        signals_note.setObjectName("LaserPreviewStatus")
        signals_note.setToolTip(_TRACE_SIGNALS_EXPLANATION)
        signals_layout.addWidget(signals_note)
        trace_options_layout = QGridLayout()
        trace_options_layout.setContentsMargins(0, 0, 0, 0)
        trace_options_layout.setHorizontalSpacing(10)
        trace_options_layout.setVerticalSpacing(1)
        # Optional like the rest. It was ticked and greyed out, "always
        # shown", and Ben asked on 2026-09-24 for the laser command traces to
        # be uncheckable. Not an NI-DAQ input, so its choice is kept by
        # LaserControlContent across tab rebuilds rather than in
        # nidaqStream.displayChannels.
        self._trace_command_checkbox = QCheckBox("Command output")
        color_code_checkbox(self._trace_command_checkbox, _COMMAND_TRACE_COLOR)
        self._trace_command_checkbox.setChecked(True)
        self._trace_command_checkbox.setToolTip(
            "The command waveform a pulse or ramp sends to this laser"
        )
        trace_options_layout.addWidget(self._trace_command_checkbox, 0, 0)

        self._trace_signal_candidates = {
            "diode": self._make_trace_signal_candidate("diode"),
            "copy": self._make_trace_signal_candidate("copy"),
            "trigger": self._make_trace_signal_candidate("trigger"),
        }
        self._trace_signal_checkboxes = {}
        for position, (key, label, color) in enumerate((
            ("diode", "Diode feedback", _DIODE_TRACE_COLOR),
            ("copy", "Command copy", _COMMAND_COPY_TRACE_COLOR),
            ("trigger", "Board trigger readback", _TRIGGER_TRACE_COLOR),
        ), start=1):
            candidate = self._trace_signal_candidates[key]
            physical_channel = "not configured" if candidate is None else candidate.physical_channel
            checkbox = QCheckBox(label)
            checkbox.setToolTip(physical_channel)
            color_code_checkbox(checkbox, color)
            self._trace_signal_checkboxes[key] = checkbox
            trace_options_layout.addWidget(checkbox, position // 2, position % 2)
        trace_options_layout.setColumnStretch(2, 1)
        signals_layout.addLayout(trace_options_layout)
        pulse_page_layout.addWidget(trace_section, stretch=1)

        trigger_section = self._add_section("board_trigger", "Board trigger")
        trigger_layout = QVBoxLayout(trigger_section.content)
        trigger_layout.setContentsMargins(0, 0, 0, 1)
        trigger_layout.setSpacing(1)
        trigger_layout.addWidget(self.trigger_plot)
        self.trigger_status = ElidedLabel()
        self.trigger_status.setObjectName("LaserPreviewStatus")
        trigger_layout.addWidget(self.trigger_status)
        pulse_page_layout.addWidget(trigger_section)
        # Takes the spare height only when the output stream is folded away
        # (its stretch no longer counts then), so the sections stay together
        # at the top instead of spreading down the page.
        pulse_page_layout.addStretch(0)
        self.refresh_trigger_status()

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
        self._trace_command_checkbox.toggled.connect(self._apply_curve_visibility)
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

    def _acquired_channel(self, signal_key: str) -> Optional[NidaqSignalChannelConfiguration]:
        """The acquired stream channel behind this input, found by its pin.

        By pin rather than by the candidate's name, as configure_laser_plot
        finds what to plot. The board trigger readback is not in the plan the
        ports build, and ticking it named a channel the stream did not have,
        which the selection refused with an exception.
        """
        candidate = self._trace_signal_candidates.get(signal_key)
        if candidate is None:
            return None
        return next(
            (
                channel
                for channel in self._app_model.nidaq_signal_monitor.configuration.channels
                if channel.physical_channel == candidate.physical_channel
            ),
            None,
        )

    def refresh_signal_selections(self) -> None:
        monitor = self._app_model.nidaq_signal_monitor
        displayed_names = set(monitor.configuration.display_channels)
        for key, checkbox in self._trace_signal_checkboxes.items():
            candidate = self._trace_signal_candidates[key]
            acquired = self._acquired_channel(key)
            selected = acquired is not None and acquired.name in displayed_names
            checkbox.blockSignals(True)
            checkbox.setChecked(selected)
            checkbox.blockSignals(False)
            # Editable while the stream runs or starts: the plot process
            # buffers every acquired input, so a tick only shows or hides it.
            checkbox.setEnabled(acquired is not None and monitor.hardware_enabled)
            channel_tooltip = "" if candidate is None else f"{candidate.physical_channel}\n"
            if candidate is None:
                checkbox.setToolTip("Assign this input in Edit → Edit DAQ Ports first.")
            elif acquired is None:
                checkbox.setToolTip(
                    channel_tooltip
                    + "This input is not in the NI-DAQ acquisition plan, so there is "
                    "nothing to plot."
                )
            elif not monitor.hardware_enabled:
                checkbox.setToolTip(
                    channel_tooltip + "NI-DAQ hardware is disabled in the system configuration."
                )
            else:
                checkbox.setToolTip(
                    channel_tooltip
                    + "Show or hide this input on this laser's graph. It is recorded either way."
                )
        self._apply_curve_visibility()

    @property
    def command_trace_visible(self) -> bool:
        return self._trace_command_checkbox.isChecked()

    def set_command_trace_visible(self, visible: bool) -> None:
        self._trace_command_checkbox.setChecked(bool(visible))

    def _curve_visible(self, curve_name: str) -> bool:
        if curve_name == "command":
            return self._trace_command_checkbox.isChecked()
        return self._trace_signal_checkboxes[curve_name].isChecked()

    def _apply_curve_visibility(self, *_args) -> None:
        """Show the ticked curves. Their data is kept either way, so one
        ticked again shows its history at once."""
        for curve_name, curve in self._trace_curves.items():
            curve.setVisible(self._curve_visible(curve_name))
        self._trace_legend.set_entries(
            entry
            for curve_name, entry in self._trace_legend_entries.items()
            if self._curve_visible(curve_name)
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
        acquired = self._acquired_channel(signal_key)
        if acquired is None:
            self.refresh_signal_selections()
            return
        configuration = self._app_model.nidaq_signal_monitor.configuration
        channel_names = [
            name
            for name in configuration.display_channels
            if name != acquired.name
        ]
        if checked:
            channel_names.append(acquired.name)
        try:
            self._app_model.update_nidaq_signal_stream_channels(channel_names)
        except Exception as exc:
            # Refused, for one before any configuration is loaded; the box
            # goes back to what is saved rather than claiming a change.
            self._set_parent_status(str(exc) or exc.__class__.__name__, True)
        self.refresh_signal_selections()

    _form_label = staticmethod(form_label)

    def _add_section(
        self, name: str, title: str, *, expanded: bool = True, stretch: bool = False,
    ) -> CollapsibleSection:
        section = CollapsibleSection(title, name, expanded=expanded, stretch=stretch)
        self.sections[name] = section
        return section

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
        if self._acquired_channel("trigger") is None:
            self.trigger_status.setText(
                "{} is set as this laser's trigger readback but is not in the "
                "NI-DAQ acquisition plan, so it cannot be shown.".format(
                    candidate.physical_channel)
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
        #: Open or closed, by section name, for every laser tab. The tabs are
        #: rebuilt on every Run/Stop, and would otherwise open everything again.
        self._section_expanded: Dict[str, bool] = {}

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.setObjectName("LaserControlContent")
        # One smaller font for the whole panel, set here rather than per
        # widget: at the rig's 9 pt default the laser page needed a scroll
        # bar in the docked panel (christielab10, 2026-09-24).
        self.setStyleSheet(
            compact_font_style_sheet("LaserControlContent")
            + "#LaserControlContent QLabel {color: #2f343a;}"
            "#LaserControlContent QLabel#LaserMetaLabel {color: #5b6470;}"
            "#LaserControlContent QLabel#LaserMetaValue {color: #20242a; font-weight: 600;}"
            "#LaserControlContent QTabWidget::pane {border: 0px; background: #ffffff;}"
            "#LaserControlContent QTabBar::tab {"
            "background: #e7eaee; color: #20242a; border: 1px solid #c9cdd3; "
            "padding: 2px 8px;"
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
        # Elided: the whole line is wider than the docked panel.
        self._stream_telemetry_label = ElidedLabel("Seq 0 | Overruns 0 | Latency n/a")
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
        self._status_label = ElidedLabel("Laser controller not configured")
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
        # Likewise whether each laser's command trace is shown; the inputs'
        # choices survive in nidaqStream.displayChannels.
        command_shown = {
            tab.channel_id_value: tab.command_trace_visible
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
            tab.set_command_trace_visible(command_shown.get(channel_index, True))
            self._adopt_sections(tab)
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

    def _adopt_sections(self, tab: _LaserChannelTab) -> None:
        """Open or close a new tab's sections as the operator left them."""
        for name, section in tab.sections.items():
            if name in self._section_expanded:
                section.set_expanded(self._section_expanded[name])
            section.expanded_changed.connect(
                lambda expanded, section_name=name: self._on_section_expanded(
                    section_name, expanded))

    def _on_section_expanded(self, name: str, expanded: bool) -> None:
        # One layout for every laser, so switching lasers does not move the
        # graphs: opening a section on one tab opens it on all of them.
        self._section_expanded[name] = expanded
        for tab in self._channel_tabs:
            section = tab.sections.get(name)
            if section is not None and section.is_expanded != expanded:
                section.set_expanded(expanded)

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
