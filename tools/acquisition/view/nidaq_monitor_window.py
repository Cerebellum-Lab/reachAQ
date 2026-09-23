"""A window for reading a rig, rather than for running one.

The application's panels draw the ten channels somebody has already named.
This draws everything the cards have, because the question it exists to
answer is the one you have before the names are right: which line does this
cable arrive on. So it streams every analog input and every clockable digital
line, polls the PFI pins for a level, and puts the stimulus beside them -
tone, laser, board line - so a signal can be made to move while being
watched.

One tab per card, because a card is what a cable is plugged into. Analog and
digital are drawn apart: a volt and a bit share no axis, and stacking bits on
a voltage scale is how a digital line becomes invisible.

Everything is labelled twice, with the terminal the software uses and the
connector printed on the block, so that what is on the screen and what is in
the hand can be matched without a third document.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autotrainer.device.device_interface import (
    TONE_CONFIRMATION_FREQUENCIES_HZ,
)
from tools.acquisition.model.nidaq_monitor_session import NidaqMonitorSession
from tools.acquisition.model.nidaq_wiring_report import format_report
from tools.acquisition.model.nidaq_wiring_verification import (
    CONFIRMED,
    OPAQUE,
    SILENT,
    UNEXPECTED,
    UNTESTED,
)
from tools.acquisition.view.stream_graph_style import color_hex, stream_signal_color

logger = logging.getLogger(__name__)

#: How often the traces are redrawn. Fast enough to watch a hand-triggered
#: pulse, slow enough that forty curves cost nothing.
PLOT_INTERVAL_MS = 50
#: How often the PFI levels are re-read. They are levels, not waveforms.
STATIC_INTERVAL_MS = 200
#: Seconds of history on screen.
WINDOW_SECONDS = 4.0
#: The tone frequencies the board confirms on a TTL line, from the one
#: place that mapping is written down.
TONE_CONFIRMATION_HZ = tuple(
    frequency for frequency, _line in TONE_CONFIRMATION_FREQUENCIES_HZ)

_STATUS_COLORS = {
    CONFIRMED: "#1b7f3b",
    SILENT: "#b3261e",
    UNEXPECTED: "#b3261e",
    UNTESTED: "#9a6700",
    OPAQUE: "#5f6368",
}


class _Worker(QThread):
    """Anything that blocks, kept off the window's thread.

    The stimulus holds a level for a fraction of a second and the wiring test
    runs for the better part of a minute. Doing either inline freezes the
    traces, which is precisely when somebody is watching them.
    """

    finished_with = Signal(bool, str)

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self._work = work

    def run(self) -> None:
        try:
            result = self._work()
        except Exception as error:  # pragma: no cover - reported, not raised
            logger.exception("monitor task failed")
            self.finished_with.emit(False, str(error))
            return
        if isinstance(result, tuple) and len(result) == 2:
            self.finished_with.emit(bool(result[0]), str(result[1]))
        else:
            self.finished_with.emit(bool(result), "")


class _CardTab(QWidget):
    """One card: its analog traces, its digital traces, its PFI levels."""

    def __init__(self, session: NidaqMonitorSession, device: str, parent=None):
        super().__init__(parent)
        self._session = session
        self._device = device
        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._level_labels: Dict[str, QLabel] = {}
        self._checkboxes: Dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        lines = session.survey.for_device(device)
        streamed = [l for l in lines if l.acquisition == "stream"]
        analog = [l for l in streamed if l.kind == "analog"]
        digital = [l for l in streamed if l.kind == "digital"]
        static = [l for l in lines if l.acquisition == "static"]
        unreadable = [l for l in lines if l.acquisition == "unreadable"]

        for note in session.survey.notes_for(device):
            label = QLabel(note)
            label.setWordWrap(True)
            label.setStyleSheet("color: #9a6700;")
            layout.addWidget(label)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(splitter, 1)

        if analog:
            splitter.addWidget(self._plot_group(
                "Analog inputs (V)", analog, digital=False))
        if digital:
            splitter.addWidget(self._plot_group(
                "Digital lines (0/1, drawn on their own scale)", digital,
                digital=True))
        if static:
            splitter.addWidget(self._level_group(static))
        if unreadable:
            splitter.addWidget(self._unreadable_group(unreadable))
        if not lines:
            layout.addWidget(QLabel(f"{device} reports no readable lines."))

    # ------------------------------------------------------------------ building

    def _plot_group(self, title: str, lines, *, digital: bool) -> QWidget:
        group = QGroupBox(title)
        box = QHBoxLayout(group)
        box.setContentsMargins(6, 4, 6, 4)

        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "seconds")
        plot.setLabel("left", "line" if digital else "volts")
        plot.setMouseEnabled(x=False, y=not digital)
        if digital:
            # Each line gets its own lane. Stacked on one axis they overlap
            # exactly and the window shows one trace where there are eight.
            plot.setYRange(-0.5, len(lines) - 0.5, padding=0)
            axis = plot.getAxis("left")
            axis.setTicks([[(index, line.terminal.split("/")[-1])
                            for index, line in enumerate(lines)]])
        box.addWidget(plot, 1)

        selectors = QVBoxLayout()
        selectors.setSpacing(1)
        for index, line in enumerate(lines):
            color = stream_signal_color(index)
            curve = plot.plot([], [], pen=pg.mkPen(color=color, width=1.4))
            self._curves[line.name] = curve
            box_item = QCheckBox(self._describe(line))
            box_item.setChecked(True)
            box_item.setStyleSheet(
                f"QCheckBox {{ color: {color_hex(color)}; }}")
            box_item.toggled.connect(
                lambda shown, name=line.name: self._set_shown(name, shown))
            self._checkboxes[line.name] = box_item
            selectors.addWidget(box_item)
        selectors.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        holder.setLayout(selectors)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(430)
        box.addWidget(scroll)
        return group

    def _unreadable_group(self, lines) -> QWidget:
        """Lines the card has and cannot watch, with why.

        Listed rather than left out. "ao3 is here, and this card cannot read
        its own output" is an answer; a channel missing from the window is
        indistinguishable from one the software forgot about.
        """
        group = QGroupBox(
            f"On this card and not watchable from it ({len(lines)})")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(1)
        for line in lines:
            label = QLabel(f"{self._describe(line)}  —  {line.reason}")
            label.setWordWrap(True)
            label.setStyleSheet("color: #5f6368;")
            layout.addWidget(label)
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(group)
        outer.addStretch(1)
        return wrapper

    def _level_group(self, lines) -> QWidget:
        counters = sum(1 for line in lines if line.kind == "counter")
        title = "PFI and static lines - a level, polled; these have no sample clock"
        if counters:
            title += f"; and {counters} counter(s), as an edge count"
        group = QGroupBox(title)
        grid = QGridLayout(group)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        for index, line in enumerate(lines):
            row, column = index % 8, (index // 8) * 2
            grid.addWidget(QLabel(self._describe(line)), row, column)
            value = QLabel("-")
            value.setFont(QFont("monospace"))
            value.setMinimumWidth(28)
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._level_labels[line.name] = value
            grid.addWidget(value, row, column + 1)
        # Spare row and column take the slack. Without them the rows space
        # themselves across whatever height the splitter gives the group,
        # and eight readouts fill a screen.
        grid.setColumnStretch(grid.columnCount(), 1)
        grid.setRowStretch(grid.rowCount(), 1)
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(group)
        outer.addStretch(1)
        return wrapper

    @staticmethod
    def _describe(line) -> str:
        """The terminal, what claims it, and what it is called on the block."""
        text = line.terminal
        if getattr(line, "watched_through", ""):
            # Named after the pin a cable goes to, not after the internal
            # channel it is actually read through, which is on no panel.
            text += " (read back)"
        if line.assigned_to:
            text += f"  [{line.assigned_to}]"
        if line.label:
            text += f"  {line.label.split(' ', 1)[-1]}"
        return text

    # ------------------------------------------------------------------ updating

    def _set_shown(self, name: str, shown: bool) -> None:
        curve = self._curves.get(name)
        if curve is not None:
            curve.setVisible(shown)

    def accept_frame(self, seconds: np.ndarray, values: Dict[str, np.ndarray],
                     offsets: Dict[str, float]) -> None:
        for name, curve in self._curves.items():
            row = values.get(name)
            if row is None or not len(row):
                continue
            if not self._checkboxes[name].isChecked():
                continue
            offset = offsets.get(name)
            curve.setData(seconds, row + offset if offset is not None else row,
                          skipFiniteCheck=True)

    def accept_levels(self, levels: Dict[str, int]) -> None:
        for name, label in self._level_labels.items():
            level = levels.get(name)
            if level is None:
                label.setText("-")
                label.setStyleSheet("color: #9aa0a6;")
                continue
            label.setText("1" if level else "0")
            label.setStyleSheet(
                "color: #1b7f3b; font-weight: 700;" if level
                else "color: #5f6368;")

    def digital_offsets(self) -> Dict[str, float]:
        """Which lane each digital line is drawn in."""
        offsets = {}
        lanes = [l for l in self._session.survey.for_device(self._device)
                 if l.acquisition == "stream" and l.kind == "digital"]
        for index, line in enumerate(lanes):
            offsets[line.name] = float(index) - 0.4
        return offsets

    def clear(self) -> None:
        for curve in self._curves.values():
            curve.setData([], [])


class NidaqMonitorWindow(QMainWindow):
    """The DAQ monitor: every line on every card, and something to poke."""

    #: Session events arrive on whichever thread produced them - the stream
    #: reaches running on the model's own reader thread - and touching a
    #: widget from there is undefined. Everything is re-emitted through this
    #: and handled on the GUI thread.
    _session_event = Signal(str)

    def __init__(self, app_model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DAQ Monitor")
        self.resize(1280, 860)
        self._app_model = app_model
        self._session = NidaqMonitorSession(app_model)
        self._tabs_by_device: Dict[str, _CardTab] = {}
        self._report_rows: Optional[QGridLayout] = None
        self._destination: Optional[np.ndarray] = None
        self._last_index: Optional[int] = None
        self._history_buffer: Optional[np.ndarray] = None
        self._history_filled = 0
        self._worker: Optional[_Worker] = None
        self._scales: Dict[str, float] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addLayout(self._control_row())
        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._stimulus = self._stimulus_group()
        root.addWidget(self._stimulus)

        self._plot_timer = self.startTimer(PLOT_INTERVAL_MS)
        self._static_timer = self.startTimer(STATIC_INTERVAL_MS)

        self._session_event.connect(self._apply_session_event,
                                    Qt.ConnectionType.QueuedConnection)
        self._session.property_changed += self._on_session_changed
        self._refresh()

    # ------------------------------------------------------------------ the chrome

    def _control_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._start_button = QPushButton("Start monitoring")
        self._start_button.clicked.connect(self._toggle_running)
        row.addWidget(self._start_button)

        refresh = QPushButton("Rescan hardware")
        refresh.clicked.connect(self._refresh)
        row.addWidget(refresh)

        row.addSpacing(16)
        row.addWidget(QLabel("Rate"))
        self._rate = QSpinBox()
        self._rate.setRange(100, 50_000)
        self._rate.setSingleStep(500)
        self._rate.setValue(2_000)
        self._rate.setSuffix(" Hz")
        row.addWidget(self._rate)
        row.addStretch(1)
        return row

    def _stimulus_group(self) -> QWidget:
        group = QGroupBox(
            "Drive something and watch which line moves")
        row = QHBoxLayout(group)
        row.setContentsMargins(8, 4, 8, 4)

        row.addWidget(QLabel("Tone"))
        self._tone_hz = QSpinBox()
        self._tone_hz.setRange(100, 20_000)
        # A mapped frequency by default. The board confirms a tone on a TTL
        # line only where the frequency matches one the devicetree assigns -
        # 5 kHz raises tone1, 6 kHz raises tone2 - so a default of anything
        # else would sound a tone, move no line, and look like a fault.
        self._tone_hz.setValue(TONE_CONFIRMATION_HZ[0])
        self._tone_hz.setSuffix(" Hz")
        self._tone_hz.setToolTip(
            "Only "
            + " and ".join(f"{hz} Hz" for hz in TONE_CONFIRMATION_HZ)
            + " raise a confirmation line (tone1 and tone2). Any other "
              "frequency sounds a tone that no line records.")
        row.addWidget(self._tone_hz)
        tone = QPushButton("Play 0.5 s")
        tone.clicked.connect(self._play_tone)
        row.addWidget(tone)

        row.addSpacing(14)
        for label, stim_line in self._session.board_stimulus_lines():
            button = QPushButton(f"Pulse {label}")
            button.clicked.connect(
                lambda _checked=False, line=stim_line: self._pulse_board(line))
            row.addWidget(button)

        row.addSpacing(14)
        row.addWidget(QLabel("Laser"))
        self._laser_choice = QComboBox()
        for number in self._session.laser_channels():
            self._laser_choice.addItem(f"laser {number}", number)
        row.addWidget(self._laser_choice)
        self._laser_volts = QDoubleSpinBox()
        self._laser_volts.setRange(0.0, 5.0)
        self._laser_volts.setSingleStep(0.1)
        self._laser_volts.setValue(1.0)
        self._laser_volts.setSuffix(" V")
        row.addWidget(self._laser_volts)
        self._open_shutter = QCheckBox("open shutter")
        row.addWidget(self._open_shutter)
        hold = QPushButton("Hold 0.5 s")
        hold.clicked.connect(self._hold_laser)
        row.addWidget(hold)
        row.addStretch(1)
        return group

    def _report_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)

        row = QHBoxLayout()
        self._run_test = QPushButton("Run wiring test")
        self._run_test.setToolTip(
            "Drives every board output and laser command in turn and records "
            "which line followed. Stops the monitor stream first.")
        self._run_test.clicked.connect(self._run_wiring_test)
        row.addWidget(self._run_test)
        self._test_shutters = QCheckBox("open shutters during the test")
        row.addWidget(self._test_shutters)
        save = QPushButton("Save report...")
        save.clicked.connect(self._save_report)
        row.addWidget(save)
        row.addStretch(1)
        layout.addLayout(row)

        self._report_summary = QLabel("")
        self._report_summary.setWordWrap(True)
        layout.addWidget(self._report_summary)

        self._report_rows = QGridLayout()
        self._report_rows.setHorizontalSpacing(12)
        self._report_rows.setVerticalSpacing(2)
        holder = QWidget()
        holder.setLayout(self._report_rows)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        layout.addWidget(scroll, 1)

        self._test_output = QPlainTextEdit()
        self._test_output.setReadOnly(True)
        self._test_output.setMaximumHeight(150)
        self._test_output.setPlaceholderText(
            "Output from the wiring test appears here.")
        layout.addWidget(self._test_output)
        return page

    # ------------------------------------------------------------------ wiring up

    def _refresh(self) -> None:
        self._session.refresh()
        # clear() removes the pages without destroying them, and a rescan
        # would then leak a full set of plots each time.
        while self._tabs.count():
            page = self._tabs.widget(0)
            self._tabs.removeTab(0)
            page.deleteLater()
        self._tabs_by_device.clear()
        for device in self._session.survey.devices:
            tab = _CardTab(self._session, device)
            self._tabs_by_device[device] = tab
            self._tabs.addTab(tab, device)
        self._tabs.addTab(self._report_tab(), "Wiring test")
        self._destination = None
        self._last_index = None
        self._scales = {
            tab_device: 0.0 for tab_device in self._tabs_by_device
        }
        self._update_status()
        self._draw_report()

    def _on_session_changed(self, name, _value, _previous) -> None:
        """Called on any thread; hands the name to the GUI thread."""
        try:
            self._session_event.emit(str(name))
        except RuntimeError:
            # The window went away between the event and its delivery.
            pass

    def _apply_session_event(self, name: str) -> None:
        if name in (NidaqMonitorSession.STATUS_MESSAGE,
                    NidaqMonitorSession.ERROR_MESSAGE,
                    NidaqMonitorSession.IS_RUNNING):
            self._update_status()
        if name == NidaqMonitorSession.REPORT:
            self._draw_report()

    def _update_status(self) -> None:
        running = self._session.is_running
        self._start_button.setText(
            "Stop monitoring" if running else "Start monitoring")
        self._rate.setEnabled(not running)
        parts = [self._session.status_message]
        for issue in self._session.issues:
            parts.append("! " + issue)
        for note in self._session.notes:
            parts.append("note: " + note)
        if self._session.error_message:
            parts.append(self._session.error_message)
        self._status.setText("\n".join(p for p in parts if p))
        self._status.setStyleSheet(
            "color: #b3261e;" if self._session.error_message else
            "color: #9a6700;" if self._session.issues else "")

    def _toggle_running(self) -> None:
        if self._session.is_running:
            self._session.stop()
            for tab in self._tabs_by_device.values():
                tab.clear()
            return
        self._last_index = None
        self._destination = None
        self._history_buffer = None
        self._history_filled = 0
        self._session.start(sample_rate_hz=float(self._rate.value()))

    # ------------------------------------------------------------------ the draw

    def timerEvent(self, event) -> None:
        if event.timerId() == self._static_timer:
            if self._session.is_running or self._session.survey.static():
                levels = self._session.poll_static_levels()
                for tab in self._tabs_by_device.values():
                    tab.accept_levels(levels)
            return
        if event.timerId() != self._plot_timer:
            return
        if not self._session.is_running or not self.isVisible():
            return
        self._draw_traces()

    def _draw_traces(self) -> None:
        ring = self._session.sample_ring
        if ring is None:
            return
        names = list(ring.channel_names)
        if not names:
            return
        capacity = int(max(1, ring.capacity))
        if (self._destination is None
                or self._destination.shape[0] < len(names)):
            self._destination = np.zeros((len(names), capacity), dtype=np.float64)
        read = ring.copy_since(self._last_index, self._destination)
        if read is None:
            # A write was in flight. There will be another frame in 50 ms.
            return
        self._last_index = read.end_sample_index
        count = read.sample_count
        if not count:
            return

        rate = float(getattr(ring, "sample_rate_hz", 0.0) or self._rate.value())
        window = int(max(2, min(capacity, WINDOW_SECONDS * rate)))
        history, filled = self._history(names, window, count)
        # Only what has actually arrived. Drawing the whole buffer while it
        # fills puts a cliff on every trace where the zeros end, which reads
        # as a signal rather than as an empty buffer.
        history = history[:, -filled:]
        seconds = np.arange(-filled, 0, dtype=np.float64) / rate

        rows = {name: history[index] for index, name in enumerate(names)}
        for device, tab in self._tabs_by_device.items():
            tab.accept_frame(seconds, rows, tab.digital_offsets())

    def _history(self, names, window: int, count: int
                 ) -> Tuple[np.ndarray, int]:
        """The last `window` samples, kept across frames.

        copy_since returns only what is new, which at twenty frames a second
        is a hundred samples. Drawing that alone would show a window that is
        always one frame wide, so the frames are shifted into a buffer here.
        """
        if (getattr(self, "_history_buffer", None) is None
                or self._history_buffer.shape != (len(names), window)):
            self._history_buffer = np.zeros((len(names), window),
                                            dtype=np.float64)
            self._history_filled = 0
        # Only the samples copy_since actually wrote. The destination is
        # allocated for the whole ring, and the rest of it is last frame's.
        fresh = self._destination[:len(names), :count]
        taken = min(count, window)
        if taken:
            self._history_buffer[:, :-taken] = self._history_buffer[:, taken:]
            self._history_buffer[:, -taken:] = fresh[:, -taken:]
            self._history_filled = min(window, self._history_filled + taken)
        return self._history_buffer, max(1, self._history_filled)

    # ------------------------------------------------------------------ stimulus

    def _run_in_background(self, work, done) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("another action is still running")
            return
        self._worker = _Worker(work, self)
        self._worker.finished_with.connect(done)
        self._worker.start()

    def _play_tone(self) -> None:
        frequency = int(self._tone_hz.value())
        self._run_in_background(
            lambda: self._session.play_tone(frequency, 0.5),
            lambda ok, message: self._report_action("tone", ok, message))

    def _pulse_board(self, stim_line: int) -> None:
        self._run_in_background(
            lambda: self._session.pulse_board_line(stim_line, 0.25),
            lambda ok, message: self._report_action(
                f"board STIM{stim_line}", ok, message))

    def _hold_laser(self) -> None:
        number = self._laser_choice.currentData()
        if number is None:
            self._status.setText("no laser channels are configured")
            return
        volts = float(self._laser_volts.value())
        shutter = self._open_shutter.isChecked()
        self._run_in_background(
            lambda: self._session.hold_laser_command(
                number, volts, 0.5, open_shutter=shutter),
            lambda ok, message: self._report_action(
                f"laser {number} at {volts:g} V", ok, message))

    def _report_action(self, what: str, ok: bool, message: str) -> None:
        if ok:
            self._status.setText(f"{what}: driven")
            self._status.setStyleSheet("")
        else:
            self._status.setText(
                f"{what} did not run: {message or self._session.error_message}")
            self._status.setStyleSheet("color: #b3261e;")

    # -------------------------------------------------------------- wiring test

    def _run_wiring_test(self) -> None:
        shutters = self._test_shutters.isChecked()
        self._run_test.setEnabled(False)
        self._test_output.setPlainText(
            "Running " + " ".join(self._session.wiring_test_command(
                open_shutters=shutters)) + "\n")
        self._run_in_background(
            lambda: self._session.run_wiring_test(open_shutters=shutters),
            self._wiring_test_finished)

    def _wiring_test_finished(self, ok: bool, output: str) -> None:
        self._run_test.setEnabled(True)
        self._test_output.setPlainText(output)
        self._test_output.verticalScrollBar().setValue(
            self._test_output.verticalScrollBar().maximum())
        self._update_status()
        if not ok:
            self._status.setText("the wiring test did not complete; see below")
            self._status.setStyleSheet("color: #b3261e;")

    def _draw_report(self) -> None:
        # The session emits its first report while refreshing, which happens
        # before the tab holding these rows has been built.
        if getattr(self, "_report_rows", None) is None:
            return
        report = self._session.report
        while self._report_rows.count():
            item = self._report_rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if report is None:
            self._report_summary.setText("No configuration is loaded.")
            return
        self._report_summary.setText(
            f"{report.headline()} — "
            + (f"last checked {report.checked_on}" if report.checked_on
               else "no hardware check has been run"))
        for row, line in enumerate(report.lines):
            status = QLabel(line.status.upper())
            status.setStyleSheet(
                f"color: {_STATUS_COLORS.get(line.status, '#000')}; "
                "font-weight: 600;")
            self._report_rows.addWidget(status, row, 0)
            self._report_rows.addWidget(QLabel(line.name), row, 1)
            self._report_rows.addWidget(QLabel(line.physical_channel), row, 2)
            where = QLabel(line.label or "-")
            where.setToolTip(line.advice() or line.detail)
            self._report_rows.addWidget(where, row, 3)
            detail = QLabel(line.advice() or line.detail)
            detail.setStyleSheet("color: #5f6368;")
            self._report_rows.addWidget(detail, row, 4)
        self._report_rows.setColumnStretch(4, 1)

    def _save_report(self) -> None:
        report = self._session.report
        if report is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save wiring report", "nidaq_wiring_report.txt",
            "Text files (*.txt);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(format_report(report) + "\n")
        except OSError as error:
            QMessageBox.warning(self, "Save failed", str(error))
            return
        self._status.setText(f"report written to {path}")

    # ------------------------------------------------------------------ teardown

    def closeEvent(self, event) -> None:
        # Stop the timers first. A timer that fires after the session has
        # closed reopens the very tasks close() just released, and the
        # process then exits holding them.
        for timer in (self._plot_timer, self._static_timer):
            try:
                self.killTimer(timer)
            except Exception:
                pass
        try:
            self._session.property_changed -= self._on_session_changed
        except Exception:
            pass
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5_000)
        self._session.close()
        super().closeEvent(event)
