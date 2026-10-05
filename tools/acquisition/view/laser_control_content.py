from __future__ import annotations

import re
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QTimer, Signal, Slot, Qt
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
from autotrainer.device.laser import CALIBRATION_SETTLE_SECONDS, LaserPulseCancelled
from autotrainer.pyside import CardWidget, PGWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.trial_protocol_schedule import (
    LaserTriggerRoute,
)
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.laser_model import LaserModel, LaserTraceBlock
from tools.acquisition.model.laser_plot_process import LaserPlotFrame, LaserPlotProcess
from tools.acquisition.model.helpers import first_line
from tools.acquisition.model.nidaq_channel_plan import nidaq_channel_kind
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
_RUN_RAMP_TOOLTIP = (
    "Step this laser's command from Start to Stop and record the diode at "
    "each step. Runs in Idle; the NI-DAQ input stream pauses meanwhile"
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
# Short enough for the Board trigger status line's two lines at the docked
# width; the checkbox tooltip adds which inputs qualify.
_TRIGGER_INPUT_INSTRUCTION = (
    "Wire the board STIM line into an NI input and set it as trigger readback "
    "input in Edit DAQ Ports."
)
_TRIGGER_INPUT_KINDS = (
    "Use an analog input or a port0 line, not a PFI terminal."
)
#: reachAQ's error red, as on the main window's "Startup failed"; an error
#: on the footer is drawn in it.
_ERROR_STATUS_COLOR = "#b00020"
_ERROR_STATUS_STYLE = f"color: {_ERROR_STATUS_COLOR};"
#: How long a press of Run Pulse keeps its pulse armed, its shutter open,
#: before it is let go unfired (Ben, 2026-10-02): the arm's own start wait,
#: so the controller lets it go whatever happens to the window.
_RUN_PULSE_ARM_CAP_S = 2.0


#: nidaqmx's DaqError ends its message with the status code on a line of its
#: own, after the driver's text and the task name.
_DAQMX_STATUS_CODE = re.compile(r"^\s*Status Code:\s*(-?\d+)\s*$", re.MULTILINE)


def _failure_line(text: str, maximum_length: int) -> str:
    """A failure's first line, cut to fit, with its DAQmx status code.

    The footer and the status bar show one line, and a DaqError's first
    line is the driver's text alone: the code, -89125 for one, was cut off.
    """
    match = _DAQMX_STATUS_CODE.search(str(text))
    line = first_line(text, maximum_length)
    if match is None or match.group(1) in line:
        return line
    code = f" (DAQmx {match.group(1)})"
    return first_line(text, maximum_length - len(code)) + code


class _LaserOperationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    #: After finished or failed, from the operation's own thread.
    done = Signal()

    def __init__(self, operation: Callable[[], object]):
        super().__init__()
        self._operation = operation

    def run(self):
        try:
            self.finished.emit(self._operation())
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            # With the reason in the message itself: the status bar shows a
            # log record's first line, and "Laser operation failed" alone left
            # a DAQmx error such as -89125 in the log's traceback only.
            logger.exception("Laser operation failed: %s", _failure_line(message, 240))
            self.failed.emit(message)
        finally:
            self.done.emit()

    def detach(self) -> None:
        """Tell nobody when it ends: the panel it would tell is going."""
        for signal in (self.finished, self.failed, self.done):
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass  # Nothing connected.


class _RunPulsePress:
    """One press of Run Pulse in internal mode, from its arm to its end.

    The press arms the pulse and the click fires it (Ben, 2026-10-02,
    decision 1 = D): the click is then one start, its edge about 0.6 ms
    later, where arming, writing and starting it after the click took 12-18
    ms (session004; christielab10, M3 B). The Qt thread has the press, the
    click and the release; the laser_operation thread arms the pulse and
    waits for its end. Under one lock, exactly one of them fires it, and only
    after its click:
    - armed before its click, the click fires it, on the Qt thread;
    - clicked first, as QPushButton.click() and a quick click are, the
      arming thread fires it once armed, later than a press held through it;
    - let go of with no click, dragged off the button say, it is disarmed.
    One neither clicked nor let go of is let go by its arm's start wait,
    _RUN_PULSE_ARM_CAP_S.
    """

    def __init__(self, pulse_train=None, manual_context=None, refusal: str = ""):
        self.pulse_train = pulse_train
        self.manual_context = manual_context
        #: Why no pulse could be made of the pick; shown at the click, as
        #: before Run Pulse armed on its press.
        self.refusal = refusal
        self._lock = threading.Lock()
        #: The ArmedManualPulse, once armed.
        self._handle = None
        self._clicked_at: Optional[float] = None
        self._let_go = False
        #: Set once the arming thread is done with it, or none was started.
        self._ended = False
        #: What its arm raised.
        self._arm_error: Optional[Exception] = None
        #: How its pulse ended unfired, for a click that comes after.
        self.unfired = ""

    def is_live(self) -> bool:
        """Arming or armed, and neither clicked nor let go of."""
        with self._lock:
            return not (self.refusal or self._ended or self._let_go
                        or self._clicked_at is not None)

    def click(self):
        """Its click, on the Qt thread: what is to be done, and with what.

        ("fire", handle): armed, so the click fires it. ("armed later",
        None): the arming thread fires it once armed. ("ended", error): it
        ended first, by its arm's refusal or failure, `error`, which is told
        as the click's, or with none, unfired, and `unfired` says how.
        ("let go", None): its press had ended with no click.
        """
        with self._lock:
            if self._let_go or self._clicked_at is not None:
                return "let go", None
            self._clicked_at = time.perf_counter()
            if self._ended:
                return "ended", self._arm_error
            if self._handle is None:
                return "armed later", None
            return "fire", self._handle

    @property
    def clicked_at(self) -> Optional[float]:
        """When its click came, by perf_counter, or None."""
        with self._lock:
            return self._clicked_at

    def let_go(self):
        """Its press ended with no click; the armed pulse to disarm, if any."""
        with self._lock:
            if self._let_go or self._clicked_at is not None:
                return None
            self._let_go = True
            return None if self._ended else self._handle

    def not_armed(self) -> None:
        """No arm was started for it: another laser operation was running."""
        with self._lock:
            self._ended = True

    def run(self, laser, laser_number: int) -> str:
        """Arm it, then fire it or let it go, and wait for its end.

        On the laser_operation thread, as Run Pulse's operation. What it
        returns, or raises, is what the panel's status line shows.
        """
        try:
            handle = laser.arm_manual_pulse(
                self.pulse_train, manual_context=self.manual_context,
                start_wait_seconds=_RUN_PULSE_ARM_CAP_S)
        except LaserPulseCancelled:
            # A Stop or a close as it armed: unfired, as one cancelled once
            # armed is, with nothing told.
            return self._ended_unfired(laser_number, "was cancelled before its click")
        except Exception as error:
            with self._lock:
                self._ended = True
                self._arm_error = error
                clicked = self._clicked_at is not None
            if clicked:
                # Clicked as it armed: told here, as its click tells one
                # whose arm was refused before it.
                laser.tell_manual_pulse_not_armed(
                    self.pulse_train, manual_context=self.manual_context, error=error)
            raise
        with self._lock:
            self._handle = handle
            clicked_at, let_go = self._clicked_at, self._let_go
        # How long after its click a fallback's start was asked for, said
        # once its outcome is known: "fired once armed" was said whenever
        # trigger() returned, a start the driver refused included (the review
        # of task 9, Minor 8).
        fallback_ms = None
        try:
            if clicked_at is not None:
                fallback_ms = (time.perf_counter() - clicked_at) * 1e3
                if not handle.fire(clicked_at=clicked_at, fallback=True):
                    fallback_ms = None
                    logger.info(
                        "Laser %s: Run Pulse was clicked before its arm was ready, "
                        "and was no longer armed once it was: nothing started",
                        laser_number)
            elif let_go:
                handle.disarm()
            completed = handle.finish()
            operation = handle.operation
            timed_out = isinstance(operation.error, TimeoutError)
            if not completed and operation.error is not None and not timed_out:
                # Failed while armed, by something other than its start wait.
                raise operation.error
        except BaseException as error:
            with self._lock:
                self._ended = True
            if fallback_ms is not None:
                logger.info(
                    "Laser %s: Run Pulse was clicked before its arm was ready; its "
                    "start, asked for %.1f ms after the click, did not deliver the "
                    "pulse: %s", laser_number, fallback_ms, error)
            raise
        if completed:
            with self._lock:
                self._ended = True
            if fallback_ms is not None:
                logger.info(
                    "Laser %s: Run Pulse was clicked before its arm was ready; "
                    "fired once armed, its start asked for %.1f ms after the click",
                    laser_number, fallback_ms)
            return f"Pulse complete: laser {laser_number}"
        return self._ended_unfired(
            laser_number,
            f"was held over {_RUN_PULSE_ARM_CAP_S:g} s" if timed_out
            else "was cancelled before its click")

    def _ended_unfired(self, laser_number: int, how: str) -> str:
        """It ended unfired, `how` unless it was let go of; what to say of it.

        How, and whether a click came, are decided in the one hold that marks
        it ended: a click either finds it ended, and says so itself
        (_LaserChannelTab._on_run_pulse_clicked), or is found here.
        """
        with self._lock:
            if self._let_go:
                how = "ended without a click"
            self.unfired = how
            self._ended = True
            clicked = self._clicked_at is not None
        if clicked:
            # Its click found it no longer armed: said as a failure, since
            # the click asked for a pulse.
            raise RuntimeError(self.not_fired_text(laser_number))
        return f"Run Pulse disarmed, nothing fired: laser {laser_number}'s press {how}"

    def not_fired_text(self, laser_number: int) -> str:
        return (f"Laser {laser_number}: Run Pulse did not fire, its press "
                f"{self.unfired}; click it again")


class _LaserChannelTab(QWidget):
    """Single-laser controls: fire a profile, calibrate, and watch the output.

    The pulse train itself is shaped in the Pulse Builder; this tab picks a
    saved profile or the builder draft and fires it on its own laser.
    """

    #: The operator ticked or unticked Command output.
    command_trace_visible_changed = Signal(bool)

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
        refresh_controls: Optional[Callable[[], None]] = None,
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
        #: Has the panel set every control's enabled state again: once a
        #: press of Run Pulse is over, its button follows the operation.
        self._refresh_controls = refresh_controls
        #: Run Pulse's press in internal mode, until its click or release.
        self._press: Optional[_RunPulsePress] = None
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
        #: keeps their open/closed state across the tab rebuilds and restarts.
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
        # No PMT shutter box: the profile's margins decide, as for trials
        # (LaserModel.pmt_shutter_for_profile).
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
        # 5 ms at christielab10's 100 kHz: its slower diode was still rising
        # through the second half of a 1 ms step (H2b).
        self._ramp_samples_per_step.setValue(500)
        # Taken when the operator finishes typing, or leaves the field, not at
        # each keystroke.
        self._ramp_samples_per_step.setKeyboardTracking(False)
        # Left out of each step's point, while the laser, the diode and the
        # input follow the step: a time, whatever the step's length. The
        # ramp refuses one that leaves no sample of a step (settle_sample_count).
        self._ramp_settle = QSpinBox()
        self._ramp_settle.setRange(0, 100000)
        self._ramp_settle.setSuffix(" µs")
        self._ramp_settle.setValue(int(round(CALIBRATION_SETTLE_SECONDS * 1e6)))
        self._ramp_settle.setToolTip(
            "How long the start of each step is left out of its point, while "
            "the laser, the diode and the input settle to the new command: "
            "600 µs by default, 60 samples at 100 kHz, whatever Samples/step "
            "is. It must leave at least one sample of each step. The ramp's "
            "fields go back to their defaults whenever Laser Control is "
            "rebuilt: on every Run/Stop, on a DAQ Ports save or a "
            "configuration load that changes the lasers, and when a late "
            "close disconnects the laser.")
        self._ramp_pmt = self._make_checkbox("PMT shutter")
        self._run_ramp_button = QPushButton("Run Ramp")

        # Settle beside the Samples/step it is a part of.
        ramp_layout = field_grid(ramp_section.content, (
            ("Start:", self._ramp_start), ("Stop:", self._ramp_stop),
            ("Samples/step:", self._ramp_samples_per_step), ("Settle:", self._ramp_settle),
            ("Steps:", self._ramp_steps),
        ))
        ramp_layout.addWidget(self._ramp_pmt, 2, 2, 1, 2)
        ramp_layout.addWidget(
            self._run_ramp_button, 3, 3, 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
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
        signals_section.set_header_tooltip(_TRACE_SIGNALS_EXPLANATION)
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
        # LaserControlContent, across tab rebuilds and in the user's
        # preferences, rather than in nidaqStream.displayChannels.
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
        # Two lines: on one, the missing-input message lost what to do
        # about it at the docked width.
        self.trigger_status = ElidedLabel(max_lines=2)
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
            self._emit_trigger,
            self._emit_timing_trigger,
            self._stim_route,
        )
        self._ramp_controls = (
            self._ramp_start,
            self._ramp_stop,
            self._ramp_steps,
            self._ramp_samples_per_step,
            self._ramp_settle,
            self._ramp_pmt,
        )

        # Internal: armed on the press, fired by the click (_RunPulsePress).
        self._run_pulse_button.pressed.connect(self._on_run_pulse_pressed)
        self._run_pulse_button.released.connect(self._on_run_pulse_released)
        self._run_pulse_button.clicked.connect(self._on_run_pulse_clicked)
        self._mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self._run_ramp_button.clicked.connect(self._run_calibration_ramp)
        self._trace_clear_button.clicked.connect(self._clear_trace)
        self._trace_seconds.valueChanged.connect(self._apply_trace_view)
        self._trace_min_volts.valueChanged.connect(self._apply_trace_view)
        self._trace_max_volts.valueChanged.connect(self._apply_trace_view)
        self._apply_trace_view()
        self._trace_command_checkbox.toggled.connect(self._apply_curve_visibility)
        self._trace_command_checkbox.toggled.connect(self.command_trace_visible_changed)
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
            # The acquisition plan's own rule, so the kind shown here is the
            # kind that is acquired.
            kind=nidaq_channel_kind(physical_channel),
            scale=scale,
        )

    def _acquired_channel(self, signal_key: str) -> Optional[NidaqSignalChannelConfiguration]:
        """The acquired stream channel behind this input, found by its pin.

        By pin rather than by the candidate's name, as configure_laser_plot
        finds what to plot. An input the stream does not acquire yet - the
        board trigger readback was never in the plan, and a new plan loads
        only after the tabs are rebuilt - named a channel the stream did not
        have when ticked, which the selection refused with an exception.
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
            if candidate is None and key == "trigger":
                checkbox.setToolTip(
                    _TRIGGER_INPUT_INSTRUCTION + " " + _TRIGGER_INPUT_KINDS)
            elif candidate is None:
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
        # It says whether the readback is shown, so it follows every tick.
        self.refresh_trigger_status()

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
        # Still connected, to a controller whose close hung in the driver.
        close_refusal = self._app_model.laser_controller_close_refusal()
        if close_refusal:
            return close_refusal[:1].upper() + close_refusal[1:]
        if not self._app_model.laser.is_connected:
            return (
                "The laser controller is not open: it claims its NI-DAQ "
                "channels when the system starts, so press Run first"
            )
        return ""

    def run_ramp_refusal(self, panel_refusal: str) -> str:
        """Why Run Ramp is unavailable on this laser, or an empty string.

        `panel_refusal` is the reason that applies to every laser, from
        LaserControlContent: another laser operation, System Mode, or what
        the application refuses a ramp for.
        """
        if not self._is_configured:
            return (
                f"Laser {self._channel.channel_id.value} has no hardware "
                "channel in the system configuration"
            )
        return panel_refusal

    def _connect_control_signals(self) -> None:
        self._trigger_mode.currentTextChanged.connect(self._on_trigger_mode_changed)
        self.stim_profile_selector.currentIndexChanged.connect(self.refresh_draft)

    def set_controls_enabled(
        self,
        can_edit: bool,
        can_run_pulse: bool,
        can_run_ramp: bool,
        ramp_refusal: str = "",
        *,
        can_hold_run_pulse: bool = False,
    ) -> None:
        """`can_hold_run_pulse`: Run Pulse could fire, but for the operation running."""
        self._controls_can_edit = can_edit
        for control in self._pulse_controls:
            control.setEnabled(can_edit)
        self._refresh_trigger_mode_enabled()
        # A press arming its pulse is that operation, and keeps its button
        # enabled until its click or release: a QPushButton disabled while
        # it is down never emits its click (QAbstractButton::changeEvent).
        held = (can_hold_run_pulse and self._press is not None
                and self._run_pulse_button.isDown())
        self._run_pulse_button.setEnabled(can_run_pulse or held)
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
        self._run_ramp_button.setToolTip(ramp_refusal or _RUN_RAMP_TOOLTIP)

    def append_trace(self, trace: LaserTraceBlock, *, redraw: bool = True) -> None:
        if int(trace.channel_id) != self.channel_id_value:
            return
        if trace.event != "trace":
            # An event row, a protocol operation's or a manual Run Pulse's
            # request and outcome, has nothing to draw. Taken as an empty
            # trace, it set the note to "calibration running", and every Run
            # Pulse ended reading so.
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

    def _validated_pulse(self):
        """The picked profile's train, checked, and what a recording keeps of it."""
        if not self._is_configured:
            raise ValueError(
                f"Laser {self._channel.channel_id.value} has no hardware channel mapping")
        # Looked up once: the train and what a recording keeps of it
        # name the same profile, at the same revision.
        profile = self._selected_profile()
        if profile is None:
            raise ValueError(self._profile_refusal())
        pulse_train = self._build_pulse_train(profile)
        self._validate_pulse_train(pulse_train)
        return pulse_train, self._manual_pulse_context(profile)

    def _on_run_pulse_pressed(self) -> None:
        if self._press is not None and self._press.is_live():
            # Pressed again before its release was looked at: dragged off the
            # button and back on. Still the one press, and the one arm.
            return
        self._press = None
        if self._trigger_mode.currentText() == "external":
            # Its edge starts it, not the click: armed at the click, as
            # before (_run_pulse).
            return
        try:
            pulse_train, manual_context = self._validated_pulse()
        except Exception as exc:
            self._press = _RunPulsePress(refusal=str(exc) or exc.__class__.__name__)
            return
        press = _RunPulsePress(pulse_train, manual_context)
        self._press = press
        laser = self._app_model.laser
        laser_number = self._channel.channel_id.value
        # The panel keeps this button enabled while it is down
        # (set_controls_enabled), and disables the rest, as for any operation.
        # Armed is all it is until the click, which says it runs (the review
        # of task 9, Minor 4).
        started = self._start_operation(
            f"Laser {laser_number} armed: release to fire",
            lambda: press.run(laser, laser_number))
        if started is False:
            press.not_armed()

    def _on_run_pulse_released(self) -> None:
        # Qt emits released before clicked. A press that ends with no click
        # emits released alone: dragged off the button, the button disabled,
        # or its focus lost, a dialog opening mid-press say (Qt 6.6). Looked
        # at once both could have been.
        if self._press is not None:
            QTimer.singleShot(0, self, self._let_go_unless_down)

    def _on_run_pulse_clicked(self) -> None:
        press, self._press = self._press, None
        laser_number = self._channel.channel_id.value
        if press is None:
            if self._trigger_mode.currentText() == "external":
                self._run_pulse()
            # Internal, its press was let go of first: nothing to fire.
            return
        if press.refusal:
            self._set_parent_status(press.refusal, True)
            return
        action, value = press.click()
        if action == "fire":
            # One start, on this thread; a fire refused, its arm having just
            # ended, is said by the arming thread (_RunPulsePress.run).
            value.fire(clicked_at=press.clicked_at)
        if action in ("fire", "armed later"):
            # After the start: nothing is put before it on the click's path.
            self._set_parent_status(f"Running laser {laser_number} pulse train", False)
        elif action == "ended" and value is not None:
            # Its arm was refused or failed during the press, and the status
            # line says why: told at the click, as run_pulse_train told it.
            self._app_model.laser.tell_manual_pulse_not_armed(
                press.pulse_train, manual_context=press.manual_context, error=value)
        elif action == "ended" and press.unfired:
            self._set_parent_status(press.not_fired_text(laser_number), True)
        if self._refresh_controls is not None:
            # Released: the button follows the running operation from here.
            self._refresh_controls()

    def _let_go_unless_down(self) -> None:
        """A press released with no click, and not pressed again, is let go of."""
        if self._press is not None and not self._run_pulse_button.isDown():
            self.let_go_of_press()

    def let_go_of_press(self) -> None:
        """Let Run Pulse's press go unfired, unless it was clicked.

        Its press ended with no click: dragged off the button, the focus
        gone, its tab switched away from, or the panel closing. Its pulse
        is disarmed: the thread that armed it closes its shutter and aborts
        its output, and nothing is asked of the driver on this one. A
        clicked press is no longer the tab's, and fires.
        """
        press, self._press = self._press, None
        if press is None:
            return
        handle = press.let_go()
        if handle is not None:
            handle.disarm()
        if self._refresh_controls is not None:
            self._refresh_controls()

    def _on_mode_tab_changed(self, _index: int) -> None:
        # The Pulse page, and Run Pulse with it, hidden mid-press: the
        # button can stay down, with no release to come.
        self.let_go_of_press()

    def _run_pulse(self) -> None:
        """Run the picked profile through run_pulse_train, armed and started at once.

        External mode's, from its click: an edge on its trigger source
        starts it. Internal mode arms on the press (_on_run_pulse_pressed).
        """
        try:
            pulse_train, manual_context = self._validated_pulse()
        except Exception as exc:
            self._set_parent_status(str(exc) or exc.__class__.__name__, True)
            return

        def operation():
            self._app_model.laser.run_pulse_train(
                pulse_train, manual_context=manual_context)
            return f"Pulse complete: laser {self._channel.channel_id.value}"

        self._start_operation(f"Running laser {self._channel.channel_id.value} pulse train", operation)

    def refresh_trigger_status(self) -> None:
        """What the Board trigger graph reads, and what it takes to see it.

        One text per state. It said "Reading X. Enable it under Signals."
        with the stream stopped or NI-DAQ disabled, and with the box already
        ticked, and nothing rewrote it when the stream started or stopped.
        """
        candidate = self._trace_signal_candidates.get("trigger")
        if candidate is None:
            self.trigger_status.setText(
                "No trigger readback input is configured. " + _TRIGGER_INPUT_INSTRUCTION
            )
            return
        acquired = self._acquired_channel("trigger")
        if acquired is None:
            self.trigger_status.setText(
                "{} is set as this laser's trigger readback but is not in the "
                "NI-DAQ acquisition plan, so it cannot be shown.".format(
                    candidate.physical_channel)
            )
            return
        source = "{} ({}, recorded as {})".format(
            acquired.physical_channel, acquired.kind, acquired.name)
        monitor = self._app_model.nidaq_signal_monitor
        error = (monitor.error_message or "").strip()
        # The whole error on hover; the line itself has room for its start.
        self.trigger_status.setToolTip(error)
        if not monitor.hardware_enabled:
            text = f"Reads {source}, but NI-DAQ hardware is disabled."
        elif not monitor.is_running and monitor.stream_state == "error":
            # It read "it is error.", and said nothing of what failed.
            text = (
                f"Reads {source} while the NI-DAQ input stream runs; "
                f"the stream failed: {first_line(error, 60)}"
            )
        elif not monitor.is_running:
            text = (
                f"Reads {source} while the NI-DAQ input stream runs; "
                f"it is {monitor.stream_state}."
            )
        elif self._trace_signal_checkboxes["trigger"].isChecked():
            text = f"Showing {source}."
        else:
            text = (
                f"Reading {source}. Tick Board trigger readback under "
                "Signals to show it."
            )
        self.trigger_status.setText(text)

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
        # The protocol check reads the pick: a protocol row fires its own
        # profile, and on 2026-10-02 laser 2's empty pick here was taken for
        # the reason its trials did not fire.
        note = getattr(self._app_model, "note_laser_tab_profile", None)
        if self._is_configured and note is not None:
            note(self.channel_id_value, profile_id)

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
        # What is typed and not yet taken, as when Run Ramp is pressed with
        # Samples/step still focused, is taken now: Samples/step takes a
        # value only as typing ends.
        self._ramp_samples_per_step.interpretText()
        self._ramp_settle.interpretText()
        try:
            ramp = LaserCalibrationRamp(
                channel_id=self._channel.channel_id,
                start_volts=self._ramp_start.value(),
                stop_volts=self._ramp_stop.value(),
                steps=self._ramp_steps.value(),
                samples_per_step=self._ramp_samples_per_step.value(),
                settle_seconds=self._ramp_settle.value() * 1e-6,
                enable_pmt_shutter=self._ramp_pmt.isChecked(),
            )
            # Refused here, as the controller refuses it, before the stream
            # is held: a settle that leaves no sample of a step at the
            # laser's rate.
            sample_rate_hz = self._app_model.laser.configuration.sample_rate_hz
            if sample_rate_hz:
                ramp.settle_sample_count(sample_rate_hz)
        except Exception as exc:
            self._set_parent_status(str(exc) or exc.__class__.__name__, True)
            return

        def operation():
            # The application holds the NI-DAQ stream for the ramp and opens
            # a controller for it. This stopped the stream itself and started
            # it again after, and nothing kept the auto-start from starting it
            # over the ramp's own tasks in between.
            points = self._app_model.run_laser_calibration_ramp(ramp)
            self._app_model.laser.make_diode_power_curve(points)
            last = points[-1]
            return (
                f"Ramp complete: {len(points)} points, last diode {last.diode_volts:.3f} V, "
                "monotonic curve validated"
            )

        self._start_operation(f"Running laser {self._channel.channel_id.value} calibration ramp", operation)

    def _build_pulse_train(self, profile=None) -> LaserPulseTrain:
        """The train of `profile`, or of the picked profile when none is given."""
        if profile is None:
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
            enable_pmt_shutter=self._app_model.laser.pmt_shutter_for_profile(
                profile, self._channel.channel_id),
            pmt_shutter_open_delay_ms=profile.pmt_open_lead_ms,
            pmt_shutter_close_delay_ms=profile.pmt_close_lag_ms,
            emit_trigger_output=self._emit_trigger.isChecked(),
            emit_timing_trigger_output=self._emit_timing_trigger.isChecked(),
        )

    def _manual_pulse_context(self, profile) -> dict:
        """What a recording session keeps of this Run Pulse beyond its train.

        `profile` is the one the train was built from, by its saved id and
        revision or as the builder draft, and then the trigger mode; the
        model takes the laser, the amplitude and the route from the train
        (LaserModel.run_pulse_train).
        """
        if profile.profile_id == DRAFT_PROFILE_ID:
            profile_id, revision = "builder draft", None
        else:
            profile_id, revision = profile.profile_id, profile.revision
        return {
            "profile_id": profile_id,
            "profile_revision": revision,
            "trigger_mode": self._trigger_mode.currentText(),
        }

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
        #: The refusal this panel last put on its status line, if any.
        self._announced_refusal: Optional[str] = None
        self._is_capture_active = False
        self._operation_thread: Optional[threading.Thread] = None
        self._operation_worker: Optional[_LaserOperationWorker] = None
        self._channel_tabs: Tuple[_LaserChannelTab, ...] = tuple()
        self._plot_process = LaserPlotProcess(app_model.nidaq_signal_monitor.sample_ring)
        self._plot_configuration_signatures = {}
        self._plot_x_destinations = {}
        self._plot_y_destinations = {}
        #: Open or closed, by section name, for every laser tab and the Pulse
        #: Builder. The tabs are rebuilt on every Run/Stop, and would
        #: otherwise open everything again.
        self._section_expanded: Dict[str, bool] = {}
        #: Where the folds and each laser's Command output are saved for the
        #: next start: the user's preferences, beside the main splitter. With
        #: none, as in some test stubs, they last until the panel closes; the
        #: panel never makes a QSettings of its own, which would be the
        #: user's real file.
        self._preferences = getattr(app_model, "preferences", None)

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.setObjectName("LaserControlContent")
        # One smaller font for the whole panel, set here rather than per
        # widget: at the application font the laser page needed a scroll
        # bar in the docked panel (christielab10, 2026-09-24). Dialogs it
        # opens keep the application font.
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
        self._tabs.currentChanged.connect(self._let_go_of_hidden_presses)
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
        self._adopt_sections(self._builder)
        # Once the builder exists: the handler updates every control.
        app_model.property_changed += self._on_app_model_property_changed
        self._refresh_from_model()

    def on_close(self):
        # A Run Pulse press still arming or armed fires nothing: only a click
        # fires one.
        for tab in self._channel_tabs:
            tab.let_go_of_press()
        worker = self._operation_worker
        if worker is not None:
            # Still running, as a ramp force-closed by AppModel's close path
            # can be: it ends on its own thread, with nothing left to tell.
            worker.detach()
            self._operation_thread = None
            self._operation_worker = None
        self._stream_plot_timer.stop()
        self._app_model.laser.property_changed -= self._on_laser_property_changed
        self._app_model.laser.trace_received -= self._on_laser_trace_received
        self._app_model.nidaq_signal_monitor.property_changed -= self._on_nidaq_monitor_property_changed
        self._app_model.property_changed -= self._on_app_model_property_changed
        self._plot_process.close()

    @invoke_method
    def _on_app_model_property_changed(self, property_name: str, _value, _old_value) -> None:
        # Run Ramp follows the application's own refusal. A Run start says so
        # through the subsystem statuses long before System Mode reads
        # Running, and the laser connects in between: the button was enabled
        # there, and a ramp pressed then stopped System Mode's stream.
        if property_name in (
            AppModel.Props.SUBSYSTEM_STATUSES,
            AppModel.Props.STATUS,
            AppModel.Props.SESSION_RECORDING_STATUS,
            AppModel.Props.ACQUISITION_RUNNING,
            AppModel.Props.LASER_CALIBRATION_ACTIVE,
        ):
            self._update_enabled_state(announce=False)

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
                # A load or a DAQ ports save rebuilds the tabs against the
                # old plan and loads the new one after, and the readback's
                # state follows the stream's; written only at the rebuild,
                # this said the readback was not acquired, or was being read
                # while the stream was stopped.
                tab.refresh_trigger_status()
            # The DAQ Monitor, a load or a DAQ ports save taking the stream
            # shows here first, and each makes a ramp wait. No tabs yet
            # means the panel is still being built.
            if self._channel_tabs:
                self._update_enabled_state(announce=False)

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

    def _let_go_of_hidden_presses(self, _index: int) -> None:
        # A tab switched away from mid-press keeps its button down, with no
        # release to come.
        current_tab = self._tabs.currentWidget()
        for tab in self._channel_tabs:
            if tab is not current_tab:
                tab.let_go_of_press()

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
        # Likewise whether each laser's command trace is shown, and at the
        # first build what the preferences saved; the inputs' choices
        # survive in nidaqStream.displayChannels.
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
                refresh_controls=lambda: self._update_enabled_state(announce=False),
            )
            shown = command_shown.get(channel_index)
            if shown is None:
                shown = self._saved_command_output_shown(channel_index)
            tab.set_command_trace_visible(shown)
            # Connected once it is set, so the loading saves nothing.
            tab.command_trace_visible_changed.connect(
                lambda visible, laser=channel_index: self._save_command_output_shown(
                    laser, visible))
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
        self._set_status(self._ready_status_text(), is_error=False)
        self._update_enabled_state()
        # Picked after the Ready line, so a profile no longer saved leaves
        # its refusal on the footer; picked as each tab was made, the Ready
        # line replaced it at once.
        for tab in self._channel_tabs:
            tab.select_profile(picked_profiles.get(tab.channel_id_value))

    def _ready_status_text(self) -> str:
        configured_count = sum(tab.is_configured for tab in self._channel_tabs)
        if configured_count:
            return f"Ready: {configured_count}/{_LASER_PULSE_TRAIN_COUNT} laser channel(s) mapped"
        return f"Pulse train editor ready; 0/{_LASER_PULSE_TRAIN_COUNT} hardware channel(s) mapped"

    def _adopt_sections(self, tab) -> None:
        """Open or close a new tab's sections as the operator left them.

        A laser tab's or the Pulse Builder's. The first time a section is
        seen its saved state is read, so a start opens the panel as the last
        one was left; with nothing saved, or anything but true or false, it
        keeps its own default.
        """
        for name, section in tab.sections.items():
            if name not in self._section_expanded:
                saved = (
                    None if self._preferences is None
                    else self._preferences.laser_section_expanded(name))
                self._section_expanded[name] = (
                    section.is_expanded if saved is None else saved)
            section.set_expanded(self._section_expanded[name])
            # Connected once it is set, so the loading saves nothing.
            section.expanded_changed.connect(
                lambda expanded, section_name=name: self._on_section_expanded(
                    section_name, expanded))

    def _on_section_expanded(self, name: str, expanded: bool) -> None:
        # Saved once, by the section the operator clicked; the others follow
        # it below and find it already recorded.
        if self._section_expanded.get(name) != expanded and self._preferences is not None:
            self._preferences.set_laser_section_expanded(name, expanded)
        # One layout for every laser, so switching lasers does not move the
        # graphs: opening a section on one tab opens it on all of them.
        self._section_expanded[name] = expanded
        for tab in self._channel_tabs:
            section = tab.sections.get(name)
            if section is not None and section.is_expanded != expanded:
                section.set_expanded(expanded)

    def _saved_command_output_shown(self, channel_id: int) -> bool:
        """A laser's saved Command output choice; shown when none is saved."""
        saved = (
            None if self._preferences is None
            else self._preferences.laser_command_output_shown(channel_id))
        return True if saved is None else saved

    def _save_command_output_shown(self, channel_id: int, shown: bool) -> None:
        if self._preferences is not None:
            self._preferences.set_laser_command_output_shown(channel_id, shown)

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
            # A press going with its tab fires nothing.
            tab.let_go_of_press()
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

    def _start_operation(self, status: str, operation: Callable[[], object]) -> bool:
        """Run `operation` on a laser_operation thread; whether it was started."""
        if self._operation_thread is not None:
            self._set_status("Laser operation already in progress", is_error=True)
            return False
        logger.verbose(status)
        self._set_status(status, is_error=False)
        self._progress.setVisible(True)
        self._set_running(True)
        # The worker stays on this thread; its signals reach the panel queued.
        worker = _LaserOperationWorker(operation)
        worker.finished.connect(self._operation_finished)
        worker.failed.connect(self._operation_failed)
        worker.done.connect(self._operation_thread_finished)
        # A Python thread, not QThread(self): the panel destroyed while a
        # force-closed ramp still ran destroyed a running QThread with it,
        # and Qt aborts the process for that. Daemon, so a ramp that never
        # returns does not hold reachAQ open; AppModel bounds its close.
        thread = threading.Thread(target=worker.run, name="laser_operation", daemon=True)
        self._operation_thread = thread
        self._operation_worker = worker
        thread.start()
        return True

    @Slot(object)
    def _operation_finished(self, result) -> None:
        self._set_status(str(result), is_error=False)

    @Slot(str)
    def _operation_failed(self, message: str) -> None:
        # It said "Laser operation stopped", and nothing of why. Shown, not
        # logged: the worker logged it, with its traceback. The line is the
        # error's first line; a DAQmx error's task and status code follow it,
        # so the whole error is on hover.
        self._show_status(
            f"Laser operation failed: {_failure_line(message, 160)}", is_error=True,
            detail=message.strip())

    @Slot()
    def _operation_thread_finished(self) -> None:
        self._operation_thread = None
        self._operation_worker = None
        self._progress.setVisible(False)
        self._set_running(False)

    def _set_status(self, message: str, *, is_error: bool) -> None:
        if is_error:
            # The record StatusLogHandler puts on the main window's status bar.
            logger.error("Laser control operation rejected: %s", message)
        self._show_status(message, is_error=is_error)

    def _show_status(self, message: str, *, is_error: bool, detail: str = "") -> None:
        """Put a status or an error on the footer, in place of what it said.

        Errors went to the log and the main window's status bar only, which
        is in the other window when this panel is detached (Ben,
        2026-09-30). An error stays until the panel's next status or error
        replaces it, as every status here does: no timeout, so one seen late
        in a detached panel is still there. Elided to its line, whole on
        hover, followed there by `detail`, which the next status clears.
        """
        self._status_label.setText(message)
        self._status_label.setToolTip(detail)
        self._status_label.setStyleSheet(_ERROR_STATUS_STYLE if is_error else "")

    def _set_running(self, is_running: bool) -> None:
        self._update_enabled_state(is_running=is_running)

    def _ramp_refusal(self, *, is_running: bool) -> str:
        """Why no laser can run a calibration ramp now, or an empty string.

        Idle only (Ben, 2026-09-25). This needed the controller System Mode
        opens and was also disabled whenever System Mode ran, so it was never
        enabled; and between the laser connecting during a Run start and
        System Mode reading Running, it was, and a ramp pressed there stopped
        System Mode's stream.
        """
        if is_running:
            return "Another laser operation is running; wait for it to finish"
        if not self._is_editable:
            return "Laser controls are locked"
        refusal = self._app_model.laser_calibration_refusal()
        if refusal:
            return f"Calibration is unavailable: {refusal}"
        if self._is_capture_active:
            return "System Mode is running; set it to Idle to calibrate"
        return ""

    def _update_enabled_state(
        self, *, is_running: Optional[bool] = None, announce: bool = True,
    ) -> None:
        if is_running is None:
            is_running = self._operation_thread is not None
        can_edit = self._is_editable and not is_running
        self._builder.set_controls_enabled(can_edit)
        # Whether a pulse could fire but for the operation running: a press
        # of Run Pulse arming its pulse is that operation, and keeps its own
        # button enabled.
        can_fire = (
            self._is_editable and self._app_model.laser.is_connected
            and not self._app_model.laser_controller_close_refusal())
        can_run = can_fire and not is_running
        ramp_refusal = self._ramp_refusal(is_running=is_running)
        refusals = []
        can_do_something = False
        for tab in self._channel_tabs:
            can_run_pulse = can_run and tab.is_configured
            tab_ramp_refusal = tab.run_ramp_refusal(ramp_refusal)
            tab.set_controls_enabled(
                can_edit, can_run_pulse, not tab_ramp_refusal, tab_ramp_refusal,
                can_hold_run_pulse=can_fire and tab.is_configured)
            can_do_something = can_do_something or can_run_pulse or not tab_ramp_refusal
            refusal = tab.run_pulse_refusal()
            if refusal and refusal not in refusals:
                refusals.append(refusal)
        if is_running:
            # The operation is why nothing can be fired, and the footer says
            # "Running ..." until its outcome. The first refusal found here
            # was an unmapped laser's, "Laser 3 has no hardware channel ..."
            # on christielab10, and it replaced "Running ..." for the whole
            # operation.
            return
        # Nothing can be fired and the buttons alone do not say why, so the
        # shared status line carries the reason. Not when a ramp can run, as
        # it can in Idle: after a ramp that replaced its own outcome with
        # "press Run first". The refreshes that follow the stream and System
        # Mode do not announce, which would replace whatever the line says,
        # but they do update a refusal of this panel's still on the line: a
        # rebuild during a load's or a DAQ ports save's hold said "press Run
        # first", and nothing took it back once a ramp could run.
        own_refusal_shown = (
            self._announced_refusal is not None
            and self._status_label.text() == self._announced_refusal
        )
        if refusals and not can_do_something:
            if announce or own_refusal_shown:
                self._set_status(refusals[0], is_error=False)
                self._announced_refusal = refusals[0]
        elif own_refusal_shown:
            self._set_status(self._ready_status_text(), is_error=False)
            self._announced_refusal = None

    @invoke_method
    def set_is_editable(self, is_editable: bool):
        self._is_editable = is_editable
        self._update_enabled_state()

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        # The graphs no longer follow System Mode; they follow the stream.
        self._is_capture_active = is_active
        self._update_enabled_state()
