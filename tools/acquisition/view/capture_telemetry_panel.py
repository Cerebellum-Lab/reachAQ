import math

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFontDatabase, QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core.logging import get_verbose_logger

from tools.acquisition.model.session_telemetry import SessionTelemetry

logger = get_verbose_logger(__name__)

#: How often the readouts refresh. Elapsed has to tick on its own because it is
#: derived from a clock rather than pushed by a message, and twice a second is
#: the slowest rate at which a seconds counter still looks live.
REFRESH_INTERVAL_MS = 500


class CaptureTelemetryPanel(QWidget):
    """A collapsed strip under the video that opens onto the live session counters.

    Starts collapsed and stays out of the way: the video is what an operator
    watches, and this only earns space when they ask for it. Expanded, it takes
    a fixed small height rather than a share of the layout, so opening it never
    resizes the video by an unpredictable amount.

    The warning indicator is the exception to staying out of the way. Dropped
    frames mean the recording has a hole in it, and that is worth knowing while
    the animal is still in the box rather than during analysis - so the icon
    shows on the collapsed strip too, and clears when the session ends.
    """

    _COLLAPSED_ARROW = "▸"     # right-pointing: opens downward
    _EXPANDED_ARROW = "▾"      # down-pointing: already open
    _WARNING = "⚠"

    #: Carries a counter change onto the GUI thread. The telemetry object is
    #: plain Python and notifies its observers on whichever thread moved the
    #: counter - the pose process's message reader, or the capture message
    #: reader - and touching a widget from there is a segmentation fault, not
    #: an exception. A signal is the cheapest correct hand-off: emitting across
    #: threads is safe, and the queued connection below runs the repaint where
    #: Qt requires it.
    counters_changed = Signal()

    def __init__(self, telemetry: SessionTelemetry, parent=None):
        super().__init__(parent)
        self._telemetry = telemetry

        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # -- the always-visible strip ------------------------------------
        header = QFrame()
        header.setObjectName("captureTelemetryHeader")
        header.setFrameShape(QFrame.Shape.NoFrame)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(6, 2, 6, 2)
        header_layout.setSpacing(8)

        toggle = self._toggle = QToolButton()
        toggle.setText(self._COLLAPSED_ARROW)
        toggle.setToolTip("Show session counters")
        toggle.setAutoRaise(True)
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        toggle.clicked.connect(self._toggle_clicked)
        header_layout.addWidget(toggle)

        title = QLabel("Session")
        title.setStyleSheet("color: palette(mid);")
        header_layout.addWidget(title)

        # A summary that stays readable while collapsed, so opening the panel
        # is a choice rather than the only way to see anything.
        self._collapsed_summary = QLabel("")
        self._collapsed_summary.setStyleSheet("color: palette(mid);")
        # Same reservation as the readouts: the strip is on the header
        # row, so its width growing by a digit moved the header too.
        self._collapsed_summary.setFont(
            QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._collapsed_summary.setMinimumWidth(
            QFontMetrics(self._collapsed_summary.font()).horizontalAdvance(
                "0:00:00   000% inferenced   000.0 ms sensor→result"))
        self._collapsed_summary.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self._collapsed_summary)

        header_layout.addStretch(1)

        warning = self._warning = QLabel(self._WARNING)
        warning.setStyleSheet("color: #d08a00; font-weight: bold;")
        warning.setVisible(False)
        header_layout.addWidget(warning)

        outer.addWidget(header)

        # -- the body, hidden until asked for ----------------------------
        body = self._body = QFrame()
        body.setObjectName("captureTelemetryBody")
        body.setFrameShape(QFrame.Shape.NoFrame)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(10, 4, 10, 6)
        body_layout.setSpacing(18)

        # Widest reading each readout can show, so none of them resizes the
        # panel when a number grows a digit.
        self._elapsed = self._add_readout(body_layout, "Elapsed", "0:00:00")
        self._dropped = self._add_readout(body_layout, "Dropped frames",
                                          "0000000")
        self._inferenced = self._add_readout(body_layout, "Inferenced",
                                             "000%  0000000/0000000")
        # Two separate figures. The call is what the model costs; sensor to
        # result adds the queue wait and is what the sub-5 ms target is about.
        # Showing only one of them would let a model look fast while missing
        # the deadline that matters.
        self._inference_ms = self._add_readout(body_layout, "Inference call",
                                               "000.0 ms  max 0000.0")
        self._e2e_ms = self._add_readout(body_layout, "Sensor → result",
                                          "000.0 ms  max 0000.0")
        body_layout.addStretch(1)

        body.setVisible(False)
        # Tracked explicitly rather than read back from isVisible(): a widget
        # reports invisible until every ancestor is shown, so on an inactive
        # tab - or before the window appears - the panel would think it was
        # collapsed and draw the summary line on top of the open body.
        self._expanded = False
        outer.addWidget(body)

        telemetry.property_changed += self._on_telemetry_changed
        self.counters_changed.connect(self._refresh,
                                      Qt.ConnectionType.QueuedConnection)

        timer = self._timer = QTimer(self)
        timer.setInterval(REFRESH_INTERVAL_MS)
        timer.timeout.connect(self._refresh)
        timer.start()

        self._refresh()

    # -- construction helpers --------------------------------------------

    @staticmethod
    def _add_readout(layout, caption: str, widest: str) -> QLabel:
        """One caption-over-value pair. Returns the value label to update.

        The value reserves the width of the widest reading it can ever
        show. Without that the panel resized whenever a number gained a
        digit - elapsed passing a minute, a percentage reaching three
        figures - and the whole strip twitched several times a second.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        label = QLabel(caption)
        label.setStyleSheet("color: palette(mid); font-size: 10px;")
        column.addWidget(label)

        value = QLabel("-")
        # Set as a font rather than only in the stylesheet, so the width
        # below is measured in the font the label will actually use.
        value.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        value.setMinimumWidth(
            QFontMetrics(value.font()).horizontalAdvance(widest))
        value.setAlignment(Qt.AlignmentFlag.AlignLeft
                           | Qt.AlignmentFlag.AlignVCenter)
        column.addWidget(value)

        layout.addWidget(holder)
        return value

    # -- behaviour --------------------------------------------------------

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._toggle.setText(self._EXPANDED_ARROW if expanded
                             else self._COLLAPSED_ARROW)
        self._toggle.setToolTip("Hide session counters" if expanded
                                else "Show session counters")
        if expanded:
            self._refresh()

    @Slot()
    def _toggle_clicked(self):
        self.set_expanded(not self.is_expanded)

    def _on_telemetry_changed(self, _name, _new, _old):
        """Ask for a repaint on a counter change.

        Called from whichever thread moved the counter, so it must not touch a
        widget itself - it only emits, and the queued connection delivers the
        repaint on the GUI thread.

        The whole panel is repainted rather than the one label that changed,
        because the readouts are derived from each other - the percentage needs
        both the pose count and the frame count - and refreshing one at a time
        would briefly show a percentage computed from a stale denominator.
        """
        self.counters_changed.emit()

    @Slot()
    def _refresh(self):
        telemetry = self._telemetry

        self._elapsed.setText(self._format_elapsed(telemetry.elapsed_seconds))

        dropped = telemetry.dropped_frames
        self._dropped.setText(f"{dropped}")
        self._dropped.setStyleSheet(
            "font-family: monospace; color: #d08a00; font-weight: bold;"
            if dropped else "font-family: monospace;")

        acquired = telemetry.frames_acquired
        posed = telemetry.frames_inferenced
        self._inferenced.setText(
            f"{telemetry.inferenced_percent:.0f}%  {posed}/{acquired}"
            if acquired else "-")

        mean_ms = telemetry.inference_mean_ms
        max_ms = telemetry.inference_max_ms
        self._inference_ms.setText(
            f"{mean_ms:.1f} ms  max {max_ms:.1f}"
            if math.isfinite(mean_ms) else "-")

        e2e_mean = telemetry.sensor_to_result_mean_ms
        e2e_max = telemetry.sensor_to_result_max_ms
        self._e2e_ms.setText(
            f"{e2e_mean:.1f} ms  max {e2e_max:.1f}"
            if math.isfinite(e2e_mean) else "-")

        # The warning survives collapsing, and clears when the session ends.
        self._warning.setVisible(telemetry.has_dropped_frames
                                 and telemetry.is_active)
        self._warning.setToolTip(
            f"{dropped} frame(s) dropped this session" if dropped else "")

        self._collapsed_summary.setText(
            "" if self.is_expanded or not acquired
            else self._collapsed_text(telemetry))

    @staticmethod
    def _collapsed_text(telemetry) -> str:
        """What the strip says while shut.

        Sensor-to-result rather than the call time: if only one number is
        visible without opening the panel, it should be the one the deadline
        is set against.
        """
        parts = [CaptureTelemetryPanel._format_elapsed(telemetry.elapsed_seconds),
                 f"{telemetry.inferenced_percent:.0f}% inferenced"]
        e2e = telemetry.sensor_to_result_mean_ms
        if math.isfinite(e2e):
            parts.append(f"{e2e:.1f} ms sensor→result")
        return "   ".join(parts)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = int(seconds)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def close(self):
        self._timer.stop()
        try:
            self._telemetry.property_changed -= self._on_telemetry_changed
        except Exception:  # pragma: no cover - teardown ordering
            logger.debug("telemetry observer already detached")
