from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core import NidaqSignalChannelConfiguration, NidaqSignalStreamConfiguration
from autotrainer.core.logging import get_verbose_logger
from autotrainer.pyside import CardWidget, PGWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.analysis_plot_process import AnalysisPlotProcess
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.model.pressure_stream_model import (
    ADC_FULL_SCALE,
    REFERENCE_VOLTS,
    PressureStreamModel,
    counts_to_volts,
)
from tools.acquisition.view.stream_graph_style import (
    StreamGraphLegend,
    color_code_checkbox,
    color_hex,
    stream_signal_color,
)


logger = get_verbose_logger(__name__)

_GRAY_COLOR_TUPLE = (240, 240, 240)
_DIGITAL_WINDOW_SECONDS = 10.0
_DIGITAL_Y_MINIMUM = -0.2
_DIGITAL_Y_MAXIMUM = 1.2

_PRESSURE_WINDOW_SECONDS = PressureStreamModel.WINDOW_SECONDS
# Instance to connector/pin, as wired on the pellet board by firmware v2.1.0.
_PRESSURE_SENSOR_LABELS = {
    0: "Sensor 1 - J11 (PA0)",
    1: "Sensor 2 - J21 (PA6)",
}


class _NidaqRollingPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = PGWidget(self)
        self._plot.clear()
        self._plot.setBackground("w")
        self._plot.getPlotItem().getViewBox().setBackgroundColor(_GRAY_COLOR_TUPLE)
        self._plot.getAxis("bottom").setLabel("Time from latest sample (s)")
        self._plot.getAxis("left").setLabel("Digital state")
        view_box = self._plot.getPlotItem().getViewBox()
        view_box.setMouseEnabled(x=True, y=False)
        view_box.setMenuEnabled(False)
        view_box.setLimits(
            xMin=-_DIGITAL_WINDOW_SECONDS,
            xMax=0.0,
            maxXRange=_DIGITAL_WINDOW_SECONDS,
            yMin=_DIGITAL_Y_MINIMUM,
            yMax=_DIGITAL_Y_MAXIMUM,
        )
        self._plot.setXRange(-_DIGITAL_WINDOW_SECONDS, 0.0, padding=0)
        self._plot.setYRange(_DIGITAL_Y_MINIMUM, _DIGITAL_Y_MAXIMUM, padding=0)
        layout.addWidget(self._plot, stretch=1)
        self._legend = StreamGraphLegend(columns=3, parent=self)
        layout.addWidget(self._legend)

        self._configuration = NidaqSignalStreamConfiguration()
        self._style_signature = tuple()
        self._curves: Dict[str, object] = {}
        self._legend_entries: Dict[str, Tuple[str, Tuple[int, int, int], bool]] = {}
        #: Curves shown; the others are still buffered and kept current.
        self._visible: Set[str] = set()
        self._display_x: Dict[str, np.ndarray] = {}
        self._display_y: Dict[str, np.ndarray] = {}
        self._pixel_width = 1
        self._last_frame = None
        self._latest_x = 0.0
        self._window_seconds = _DIGITAL_WINDOW_SECONDS

    def configure(
        self,
        configuration: NidaqSignalStreamConfiguration,
        colors_by_name: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> bool:
        colors_by_name = colors_by_name or {}
        style_signature = tuple(
            (channel.name, colors_by_name.get(channel.name, stream_signal_color(index)))
            for index, channel in enumerate(configuration.channels)
        )
        if configuration == self._configuration and style_signature == self._style_signature:
            return False
        self._configuration = configuration
        self._style_signature = style_signature
        self._plot.clear()
        self._curves.clear()
        self._legend_entries.clear()
        self._display_x.clear()
        self._display_y.clear()
        self._latest_x = 0.0
        self._last_frame = None
        for index, channel in enumerate(configuration.channels):
            color = colors_by_name.get(channel.name, stream_signal_color(index))
            pen = pg.mkPen(color=color, width=2.2, style=Qt.PenStyle.SolidLine)
            display_name = f"{channel.name} ({channel.unit})"
            self._display_x[channel.name] = np.full(
                AnalysisPlotProcess.MAX_POINTS, np.nan, dtype=np.float32,
            )
            self._display_y[channel.name] = np.full(
                AnalysisPlotProcess.MAX_POINTS, np.nan, dtype=np.float32,
            )
            curve = self._plot.plot([], [], pen=pen)
            curve.setVisible(channel.name in self._visible)
            self._curves[channel.name] = curve
            self._legend_entries[channel.name] = (display_name, color, False)
        self._visible &= set(self._curves)
        self._show_legend()
        self._plot.getPlotItem().setClipToView(True)
        return True

    def set_visible_channels(self, channel_names) -> None:
        """Show these curves and hide the rest, keeping every curve's history.

        Every plotted signal is buffered whether or not it is shown, so one
        shown again draws its history at once, from the last frame, without
        waiting for new samples or disturbing the others.
        """
        visible = {name for name in channel_names if name in self._curves}
        if visible == self._visible:
            return
        shown = visible - self._visible
        self._visible = visible
        for channel_name, curve in self._curves.items():
            curve.setVisible(channel_name in visible)
        for index, channel_name in enumerate(self._curves):
            if channel_name in shown:
                self._draw_curve(index, channel_name)
        self._show_legend()

    def _show_legend(self) -> None:
        self._legend.set_entries(
            entry
            for channel_name, entry in self._legend_entries.items()
            if channel_name in self._visible
        )

    def _draw_curve(self, index: int, channel_name: str) -> None:
        frame = self._last_frame
        curve = self._curves[channel_name]
        if frame is None:
            curve.setData([], [])
            return
        point_count = frame.point_counts[index]
        curve.setData(
            self._display_x[channel_name][:point_count],
            self._display_y[channel_name][:point_count],
            skipFiniteCheck=True,
        )

    def clear(self) -> None:
        for curve in self._curves.values():
            curve.setData([], [])
        self._latest_x = 0.0
        self._last_frame = None

    def display_latest(self, plot_process: AnalysisPlotProcess) -> bool:
        frame = plot_process.copy_latest_into(self._display_x, self._display_y)
        if frame is None:
            return False
        self._pixel_width = frame.pixel_width
        self._window_seconds = frame.window_seconds
        self._latest_x = frame.latest_x
        self._last_frame = frame
        # Hidden curves are copied above but not handed to Qt: drawing is the
        # cost worth saving, and _draw_curve catches one up when it is shown.
        for index, channel_name in enumerate(self._curves):
            if channel_name in self._visible:
                self._draw_curve(index, channel_name)
        return True

    def physical_pixel_width(self) -> int:
        viewport = self._plot.viewport()
        width = round(viewport.width() * viewport.devicePixelRatioF())
        return max(1, min(AnalysisPlotProcess.MAX_PIXEL_WIDTH, width))

    def set_pixel_width(self, pixel_width: int) -> bool:
        pixel_width = max(1, min(AnalysisPlotProcess.MAX_PIXEL_WIDTH, int(pixel_width)))
        if pixel_width == self._pixel_width:
            return False
        self._pixel_width = pixel_width
        return True

    def go_live(self) -> None:
        """Move the current horizontal viewport to the newest sample time."""
        view_box = self._plot.getPlotItem().getViewBox()
        x_minimum, x_maximum = view_box.viewRange()[0]
        visible_seconds = min(
            _DIGITAL_WINDOW_SECONDS,
            max(0.001, float(x_maximum - x_minimum)),
        )
        view_box.setXRange(-visible_seconds, 0.0, padding=0)


class _PressurePlot(QWidget):
    """One FSR sensor's rolling trace."""

    def __init__(self, title: str, color, show_x_axis: bool, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        heading = QLabel(title, self)
        heading.setStyleSheet(
            f"color: {color_hex(color)}; font-size: 10px; font-weight: 600;"
        )
        layout.addWidget(heading)

        self._plot = PGWidget(self)
        self._plot.clear()
        self._plot.setBackground("w")
        plot_item = self._plot.getPlotItem()
        view_box = plot_item.getViewBox()
        view_box.setBackgroundColor(_GRAY_COLOR_TUPLE)
        view_box.setMouseEnabled(x=True, y=False)
        view_box.setMenuEnabled(False)
        view_box.setLimits(
            xMin=-_PRESSURE_WINDOW_SECONDS,
            xMax=0.0,
            maxXRange=_PRESSURE_WINDOW_SECONDS,
        )
        self._plot.setXRange(-_PRESSURE_WINDOW_SECONDS, 0.0, padding=0)
        bottom_axis = self._plot.getAxis("bottom")
        if show_x_axis:
            bottom_axis.setLabel("Time from latest sample (s)")
        else:
            # Both graphs share one time origin, so only the lower one is labelled.
            bottom_axis.setStyle(showValues=False)
        plot_item.setClipToView(True)
        layout.addWidget(self._plot, stretch=1)

        self.curve = self._plot.plot(
            [], [], pen=pg.mkPen(color=color, width=2.2, style=Qt.PenStyle.SolidLine),
        )

    def set_y_axis(self, label: str, maximum: float) -> None:
        self._plot.getAxis("left").setLabel(label)
        # A fixed range keeps a railed or floating channel from being autoscaled
        # until its own noise fills the graph and reads as signal.
        padding = maximum * 0.02
        self._plot.getPlotItem().getViewBox().setLimits(
            yMin=-padding, yMax=maximum + padding,
        )
        self._plot.setYRange(-padding, maximum + padding, padding=0)

    def go_live(self) -> None:
        view_box = self._plot.getPlotItem().getViewBox()
        x_minimum, x_maximum = view_box.viewRange()[0]
        visible_seconds = min(
            _PRESSURE_WINDOW_SECONDS,
            max(0.001, float(x_maximum - x_minimum)),
        )
        view_box.setXRange(-visible_seconds, 0.0, padding=0)


class _PressurePlots(QWidget):
    """Stacked per-sensor pellet-board FSR graphs, with a raw/volts toggle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)

        self._raw_counts_checkbox = QCheckBox("Raw ADC counts", self)
        self._raw_counts_checkbox.setToolTip(
            "Plot the unconverted 12-bit ADC sample instead of volts"
        )
        self._raw_counts_checkbox.toggled.connect(self.set_show_raw_counts)
        layout.addWidget(self._raw_counts_checkbox)

        self._show_raw_counts = False
        self._plots = {}
        self.curves = {}
        instances = tuple(PressureStreamModel.INSTANCES)
        for index, instance in enumerate(instances):
            plot = _PressurePlot(
                _PRESSURE_SENSOR_LABELS.get(instance, f"Sensor {instance}"),
                stream_signal_color(index),
                show_x_axis=index == len(instances) - 1,
                parent=self,
            )
            self._plots[instance] = plot
            self.curves[instance] = plot.curve
            layout.addWidget(plot, stretch=1)
        self._apply_y_axis()

    def y_axis_label(self) -> str:
        return "ADC counts" if self._show_raw_counts else "Volts"

    def set_show_raw_counts(self, show_raw_counts: bool) -> None:
        show_raw_counts = bool(show_raw_counts)
        if show_raw_counts == self._show_raw_counts:
            return
        self._show_raw_counts = show_raw_counts
        if self._raw_counts_checkbox.isChecked() != show_raw_counts:
            self._raw_counts_checkbox.setChecked(show_raw_counts)
        self._apply_y_axis()

    def _apply_y_axis(self) -> None:
        maximum = ADC_FULL_SCALE if self._show_raw_counts else REFERENCE_VOLTS
        for plot in self._plots.values():
            plot.set_y_axis(self.y_axis_label(), float(maximum))

    def display_latest(self, frame) -> bool:
        if frame is None:
            return False
        for instance, curve in self.curves.items():
            seconds, counts = frame.series.get(instance, (None, None))
            if seconds is None:
                continue
            values = counts if self._show_raw_counts else counts_to_volts(counts)
            curve.setData(seconds, values, skipFiniteCheck=True)
        return True

    def clear(self) -> None:
        for curve in self.curves.values():
            curve.setData([], [])

    def go_live(self) -> None:
        for plot in self._plots.values():
            plot.go_live()


class AnalysisContent(ContentWidget):
    """Rolling NI-DAQ input stream display for acquisition hardware checks."""

    def __init__(self, app_model):
        super().__init__()

        self._app_model = app_model
        self._nidaq_signal_monitor = app_model.nidaq_signal_monitor
        self._signal_checkboxes: Dict[str, QCheckBox] = {}
        self._signal_candidates: Dict[str, Optional[NidaqSignalChannelConfiguration]] = {}
        self._signal_colors: Dict[str, Tuple[int, int, int]] = {}
        self._channel_colors_by_name: Dict[str, Tuple[int, int, int]] = {}
        self._selector_signature = None
        self._plot_process = AnalysisPlotProcess(self._nidaq_signal_monitor.sample_ring)
        self._pressure_model = PressureStreamModel(app_model.message_handler)
        self._display_refresh_rate_hz = 0.0
        self._next_screen_check = 0.0
        self._window_handle = None

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(QLabel("NI-DAQ:"))
        self._stream_state_label = QLabel("disabled")
        header_layout.addWidget(self._stream_state_label)
        header_layout.addWidget(QLabel("Rate:"))
        self._sample_rate_label = QLabel("n/a")
        header_layout.addWidget(self._sample_rate_label)
        header_layout.addWidget(QLabel("Channels:"))
        self._channel_count_label = QLabel("0")
        header_layout.addWidget(self._channel_count_label)
        header_layout.addWidget(QLabel("Seq:"))
        self._sequence_label = QLabel("0")
        header_layout.addWidget(self._sequence_label)
        header_layout.addWidget(QLabel("Overruns:"))
        self._overrun_label = QLabel("0")
        header_layout.addWidget(self._overrun_label)
        self._telemetry_label = QLabel("Latency: n/a")
        header_layout.addWidget(self._telemetry_label)

        self._card_widget = CardWidget(title="Analysis", header_right_layout=header_layout)
        self._rolling_plot = _NidaqRollingPlot()
        self._content_tabs = QTabWidget()
        self._content_tabs.setDocumentMode(True)
        self._content_tabs.setMinimumSize(0, 0)
        self._content_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._content_tabs.addTab(self._rolling_plot, "Stream")

        self._signal_scroll = QScrollArea()
        self._signal_scroll.setWidgetResizable(True)
        self._signal_widget = QWidget()
        self._signal_layout = QVBoxLayout(self._signal_widget)
        self._signal_layout.setContentsMargins(8, 8, 8, 8)
        self._signal_layout.setSpacing(6)
        self._signal_scroll.setWidget(self._signal_widget)
        self._content_tabs.addTab(self._signal_scroll, "Signals")

        self._pressure_plots = _PressurePlots()
        self._content_tabs.addTab(self._pressure_plots, "Pressure")
        self._content_tabs.currentChanged.connect(self._redraw_visible_stream)
        self._card_widget.setContentWidget(self._content_tabs)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(4, 0, 0, 0)
        footer_layout.setSpacing(8)
        # No Start/Stop: the stream runs by itself whenever NI-DAQ is enabled
        # (see AppModel._request_nidaq_stream), and its state is in the header.
        self._clear_button = QPushButton("Clear")
        self._live_button = QPushButton("Live")
        self._live_button.setToolTip(
            "Move the right edge to live time while preserving horizontal zoom"
        )
        self._status_label = QLabel("NI-DAQ signal stream disabled")
        self._status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        footer_layout.addWidget(self._clear_button)
        footer_layout.addWidget(self._live_button)
        footer_layout.addWidget(self._status_label)
        self._card_widget.footer.setContent(footer)

        layout = QVBoxLayout()
        layout.addWidget(self._card_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self._nidaq_signal_monitor.property_changed += self._model_property_changed
        app_model.configuration_loaded_event += self._configuration_loaded
        app_model.laser.property_changed += self._laser_property_changed
        self._clear_button.clicked.connect(self._clear_plot)
        self._live_button.clicked.connect(self._go_live)
        self._plot_timer = QTimer(self)
        self._plot_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._plot_timer.timeout.connect(self._flush_pending_blocks)
        self._update_display_refresh_rate()
        self._plot_timer.start()
        self._refresh_from_model()

    def on_close(self):
        self._plot_timer.stop()
        self._nidaq_signal_monitor.property_changed -= self._model_property_changed
        self._app_model.configuration_loaded_event -= self._configuration_loaded
        self._app_model.laser.property_changed -= self._laser_property_changed
        self._plot_process.close()
        self._pressure_model.close()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        window_handle = self.window().windowHandle()
        if window_handle is not None and window_handle is not self._window_handle:
            if self._window_handle is not None:
                try:
                    self._window_handle.screenChanged.disconnect(self._screen_changed)
                except (RuntimeError, TypeError):
                    pass
            self._window_handle = window_handle
            window_handle.screenChanged.connect(self._screen_changed)
        self._update_display_refresh_rate()

    def _screen_changed(self, screen) -> None:
        if screen is None:
            self._update_display_refresh_rate()
        else:
            self._apply_display_refresh_rate(float(screen.refreshRate()))

    def _update_display_refresh_rate(self) -> None:
        screen = None
        window_handle = self.window().windowHandle()
        if window_handle is not None:
            screen = window_handle.screen()
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        refresh_rate = 60.0 if screen is None else float(screen.refreshRate())
        self._apply_display_refresh_rate(refresh_rate)

    def _apply_display_refresh_rate(self, refresh_rate: float) -> None:
        if not math.isfinite(refresh_rate) or refresh_rate < 1.0:
            refresh_rate = 60.0
        if math.isclose(refresh_rate, self._display_refresh_rate_hz, rel_tol=0.001):
            return
        self._display_refresh_rate_hz = refresh_rate
        self._plot_timer.setInterval(max(1, int(round(1000.0 / refresh_rate))))
        self._nidaq_signal_monitor.set_display_refresh_rate(refresh_rate)

    def _sync_plot_pixel_width(self) -> None:
        pixel_width = self._rolling_plot.physical_pixel_width()
        if self._rolling_plot.set_pixel_width(pixel_width):
            self._plot_process.set_pixel_width(pixel_width)

    def _flush_pending_blocks(self) -> None:
        self._flush_pressure()
        now = time.monotonic()
        if now >= self._next_screen_check:
            self._next_screen_check = now + 1.0
            self._update_display_refresh_rate()
        self._sync_plot_pixel_width()
        if self._rolling_plot.display_latest(self._plot_process):
            frame = self._rolling_plot._last_frame
            self._sequence_label.setText(str(frame.raw_generation))
            self._overrun_label.setText(str(frame.overrun_count))
            self._telemetry_label.setText(
                f"Latency: {frame.source_latency_ms:.1f} ms | Gaps: {frame.gap_count}"
            )

    def _go_live(self) -> None:
        if self._content_tabs.currentWidget() is self._pressure_plots:
            self._pressure_plots.go_live()
        else:
            self._rolling_plot.go_live()

    def _flush_pressure(self) -> None:
        """Repaint the pressure graphs from CAN telemetry.

        This is independent of the NI-DAQ stream: pressure arrives over CAN, so
        the graphs stay live whether or not that stream is running.  The model
        keeps buffering while the tab is hidden; only the repaint is skipped.
        """
        if self._content_tabs.currentWidget() is not self._pressure_plots:
            return
        self._pressure_plots.display_latest(self._pressure_model.snapshot())

    def _clear_plot(self) -> None:
        self._plot_process.clear()
        self._rolling_plot.clear()
        self._pressure_model.clear()
        self._pressure_plots.clear()

    def _redraw_visible_stream(self, *_args) -> None:
        self._flush_pending_blocks()

    @invoke_method
    def _model_property_changed(self, _name: str, _value, _old_value) -> None:
        self._refresh_from_model()

    @invoke_method
    def _configuration_loaded(self, _configuration) -> None:
        self._selector_signature = None
        self._refresh_from_model()

    @invoke_method
    def _laser_property_changed(self, _name: str, _value, _old_value) -> None:
        self._selector_signature = None
        self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        model = self._nidaq_signal_monitor
        sample_ring = model.sample_ring
        ring_changed = self._plot_process.raw_ring is not sample_ring
        if ring_changed:
            self._plot_process.close()
            self._plot_process = AnalysisPlotProcess(sample_ring)
        configuration = model.configuration
        # Not the stream's state: the checkboxes no longer depend on it.
        selector_signature = (
            configuration.channels,
            tuple(sorted(self._mapped_physical_channels())),
            model.hardware_enabled,
        )
        if selector_signature != self._selector_signature:
            self._selector_signature = selector_signature
            self._rebuild_signal_selector()
        # The plot process is given every plottable signal, shown or not, so
        # ticking one changes only what is drawn. Given only the ticked ones,
        # as it was, each tick rebuilt its buffers and emptied every graph.
        plot_configuration = self._plot_configuration()
        if (
            self._rolling_plot.configure(plot_configuration, self._channel_colors_by_name)
            or ring_changed
        ):
            pixel_width = self._rolling_plot.physical_pixel_width()
            self._rolling_plot.set_pixel_width(pixel_width)
            self._plot_process.configure(
                dataclasses.replace(
                    plot_configuration,
                    rolling_window_seconds=_DIGITAL_WINDOW_SECONDS,
                ),
                pixel_width,
            )
        display_configuration = self._display_configuration()
        self._rolling_plot.set_visible_channels(
            channel.name for channel in display_configuration.channels
        )
        self._stream_state_label.setText(model.stream_state)
        # The details stay out of the footer, which is one line; they are in
        # the Hardware panel, the log, and here on hover.
        self._stream_state_label.setToolTip(model.error_message or model.status_message)
        self._sample_rate_label.setText(f"{configuration.sample_rate_hz:g} Hz")
        self._channel_count_label.setText(str(len(display_configuration.channels)))
        self._clear_button.setEnabled(model.hardware_enabled and bool(plot_configuration.channels))
        self._live_button.setEnabled(bool(display_configuration.channels))
        if not model.hardware_enabled:
            self._status_label.setText("NI-DAQ hardware is disabled")
        else:
            self._status_label.setText(model.status_message)
        self._status_label.setStyleSheet("")

    def _rebuild_signal_selector(self) -> None:
        while self._signal_layout.count():
            item = self._signal_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._signal_checkboxes.clear()
        self._signal_candidates.clear()
        self._signal_colors.clear()
        self._channel_colors_by_name.clear()

        configuration = self._nidaq_signal_monitor.configuration
        mapped_channels = self._mapped_physical_channels()
        configured_by_physical = {
            channel.physical_channel: channel
            for channel in configuration.channels
        }

        explanation = QLabel(
            "Choose which acquired DAQ signals are plotted. Every mapped input is "
            "always acquired and recorded, whether or not it is selected here."
        )
        explanation.setWordWrap(True)
        self._signal_layout.addWidget(explanation)

        candidates = []

        def add_candidate(
            key: str,
            label: str,
            physical_channel: Optional[str],
            kind: str,
        ) -> None:
            channel = None
            if physical_channel:
                channel = configured_by_physical.get(physical_channel)
                if channel is None:
                    channel = NidaqSignalChannelConfiguration(
                        name=key,
                        physical_channel=physical_channel,
                        kind=kind,
                    )
            candidates.append((key, label, channel))

        ports = self._app_model.nidaq_ports
        add_candidate("cam_frames", "Camera frames", ports.cam_frames, "digital")
        add_candidate("barcode", "Barcode", ports.barcode, "digital")
        add_candidate("tone1", "Tone 1", ports.tone1, "digital")
        add_candidate("tone2", "Tone 2", ports.tone2, "digital")
        add_candidate("tone3_r", "Tone 3 right", ports.tone3_r, "digital")
        add_candidate("tone3_l", "Tone 3 left", ports.tone3_l, "digital")

        candidate_physical_channels = {
            channel.physical_channel
            for _key, _label, channel in candidates
            if channel is not None
        }
        for channel in configuration.channels:
            if (
                channel.physical_channel not in candidate_physical_channels
                and not self._is_laser_stream_channel(channel)
            ):
                candidates.append((f"custom:{channel.name}", channel.name, channel))

        selected_channel_names = set(configuration.display_channels)
        selected_colors = {
            channel.physical_channel: stream_signal_color(index)
            for index, channel in enumerate(
                channel
                for channel in configuration.channels
                if channel.physical_channel in mapped_channels
                and not self._is_laser_stream_channel(channel)
            )
        }
        for color_index, (key, label, channel) in enumerate(candidates):
            physical_channel = None if channel is None else channel.physical_channel
            color = selected_colors.get(physical_channel, stream_signal_color(color_index))
            is_mapped = physical_channel in mapped_channels if physical_channel else False
            is_selected = channel.name in selected_channel_names if channel is not None else False
            channel_text = physical_channel or "not configured"
            checkbox = QCheckBox(f"{label} — {channel_text}")
            color_code_checkbox(checkbox, color)
            checkbox.setChecked(is_selected)
            # Editable while the stream runs or starts: a tick only shows or
            # hides a curve the plot process is already buffering.
            checkbox.setEnabled(
                self._nidaq_signal_monitor.hardware_enabled
                and (is_mapped or is_selected)
            )
            if not physical_channel:
                checkbox.setToolTip("Assign this signal in Edit → Edit DAQ Ports first.")
            elif not is_mapped:
                checkbox.setToolTip(
                    "This saved stream channel is no longer mapped. Uncheck it or restore its DAQ-port assignment."
                )
            elif not self._nidaq_signal_monitor.hardware_enabled:
                checkbox.setToolTip("NI-DAQ hardware is disabled in the system configuration.")
            else:
                checkbox.setToolTip(
                    "Show or hide this signal on the graph. It is recorded either way."
                )
            checkbox.toggled.connect(
                lambda checked, candidate_key=key: self._signal_selection_changed(
                    candidate_key,
                    checked,
                )
            )
            self._signal_candidates[key] = channel
            self._signal_colors[key] = color
            if channel is not None:
                self._channel_colors_by_name[channel.name] = color
            self._signal_checkboxes[key] = checkbox
            self._signal_layout.addWidget(checkbox)
        self._signal_layout.addStretch(1)

    def _signal_selection_changed(self, candidate_key: str, checked: bool) -> None:
        candidate = self._signal_candidates.get(candidate_key)
        if candidate is None:
            return
        configuration = self._nidaq_signal_monitor.configuration
        selected_names = list(configuration.display_channels)
        if checked:
            selected_names = [
                name for name in selected_names if name != candidate.name
            ]
            selected_names.append(candidate.name)
        else:
            selected_names = [
                name for name in selected_names if name != candidate.name
            ]
        self._app_model.update_nidaq_signal_stream_channels(selected_names)

    def _plot_configuration(self) -> NidaqSignalStreamConfiguration:
        """Every signal this card can plot, whether or not it is ticked.

        Independent of the selection on purpose, down to display_channels,
        so that ticking a box never looks like a new plot configuration.
        """
        configuration = self._nidaq_signal_monitor.configuration
        mapped_channels = self._mapped_physical_channels()
        channels = tuple(
            channel
            for channel in configuration.channels
            if channel.physical_channel in mapped_channels
            and not self._is_laser_stream_channel(channel)
        )
        return dataclasses.replace(
            configuration,
            channels=channels,
            display_channels=tuple(channel.name for channel in channels),
            is_enabled=configuration.is_enabled and bool(channels),
        )

    def _display_configuration(self) -> NidaqSignalStreamConfiguration:
        """The plottable signals that are ticked, which are the ones drawn."""
        configuration = self._nidaq_signal_monitor.configuration
        mapped_channels = self._mapped_physical_channels()
        channels = tuple(
            channel
            for channel in configuration.channels
            if channel.physical_channel in mapped_channels
            and not self._is_laser_stream_channel(channel)
            and channel.name in configuration.display_channels
        )
        return dataclasses.replace(
            configuration,
            channels=channels,
            display_channels=tuple(channel.name for channel in channels),
            is_enabled=configuration.is_enabled and bool(channels),
        )

    def _mapped_physical_channels(self) -> Set[str]:
        mapped = set()
        ports = self._app_model.nidaq_ports
        for field_name in (
            "tone1",
            "tone2",
            "tone3_r",
            "tone3_l",
            "cam_frames",
            "barcode",
        ):
            value = getattr(ports, field_name)
            if value:
                mapped.add(value)
        for channel in self._app_model.laser.configuration.channels:
            for value in (
                channel.analog_output,
                channel.diode_input,
                channel.shutter_output,
                channel.command_copy_input,
            ):
                if value:
                    mapped.add(value)
        return mapped

    @staticmethod
    def _is_laser_stream_channel(channel: NidaqSignalChannelConfiguration) -> bool:
        name = channel.name.lower()
        return name.startswith("laser") and (
            name.endswith("_diode") or name.endswith("_command_copy")
        )
