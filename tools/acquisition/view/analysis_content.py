from __future__ import annotations

import dataclasses
import math
import threading
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
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
from autotrainer.device import NidaqSignalSampleBlock
from autotrainer.pyside import CardWidget, PGWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.view.stream_graph_style import (
    StreamGraphLegend,
    color_code_checkbox,
    stream_signal_color,
)
from tools.acquisition.view.rolling_stream_buffer import RollingStreamBuffer


logger = get_verbose_logger(__name__)

_GRAY_COLOR_TUPLE = (240, 240, 240)


class _NidaqRollingPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = PGWidget(self)
        self._plot.clear()
        self._plot.setBackground("w")
        self._plot.getPlotItem().getViewBox().setBackgroundColor(_GRAY_COLOR_TUPLE)
        self._plot.getAxis("bottom").setLabel("Time (s)")
        self._plot.getAxis("left").setLabel("Signal")
        layout.addWidget(self._plot, stretch=1)
        self._legend = StreamGraphLegend(columns=3, parent=self)
        layout.addWidget(self._legend)

        self._configuration = NidaqSignalStreamConfiguration()
        self._style_signature = tuple()
        self._curves: Dict[str, object] = {}
        self._buffers: Dict[str, RollingStreamBuffer] = {}
        self._latest_x = 0.0
        self._window_seconds = self._configuration.rolling_window_seconds
        self._voltage_range: Optional[Tuple[float, float]] = None

    def configure(
        self,
        configuration: NidaqSignalStreamConfiguration,
        colors_by_name: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> None:
        colors_by_name = colors_by_name or {}
        style_signature = tuple(
            (channel.name, colors_by_name.get(channel.name, stream_signal_color(index)))
            for index, channel in enumerate(configuration.channels)
        )
        if configuration == self._configuration and style_signature == self._style_signature:
            return
        self._configuration = configuration
        self._style_signature = style_signature
        self._plot.clear()
        self._curves.clear()
        self._buffers.clear()
        self._latest_x = 0.0
        self._window_seconds = min(self._window_seconds, 60.0)
        capacity = self._buffer_capacity()
        legend_entries = []
        for index, channel in enumerate(configuration.channels):
            color = colors_by_name.get(channel.name, stream_signal_color(index))
            pen = pg.mkPen(color=color, width=2.2, style=Qt.PenStyle.SolidLine)
            display_name = f"{channel.name} ({channel.unit})"
            self._curves[channel.name] = self._plot.plot([], [], pen=pen)
            self._buffers[channel.name] = RollingStreamBuffer(capacity)
            legend_entries.append((display_name, color, False))
        self._legend.set_entries(legend_entries)
        if self._voltage_range is None:
            self._apply_y_range(configuration.channels)
        else:
            self._plot.setYRange(*self._voltage_range, padding=0)
        self._plot.getPlotItem().setClipToView(True)
        self._plot.setXRange(0, self._window_seconds, padding=0)

    def clear(self) -> None:
        for channel_name, curve in self._curves.items():
            self._buffers[channel_name].clear()
            curve.setData([], [])
        self._latest_x = 0.0
        self._plot.setXRange(0, self._window_seconds, padding=0)

    def append(self, block: NidaqSignalSampleBlock) -> None:
        first_sample = block.sample_index / block.sample_rate_hz
        sample_count = max((len(values) for values in block.values.values()), default=0)
        x_values = first_sample + np.arange(sample_count, dtype=np.float64) / block.sample_rate_hz
        for channel in self._configuration.channels:
            values = block.values.get(channel.name)
            if not values:
                continue
            self._buffers[channel.name].append(x_values[: len(values)], values)
            self._latest_x = max(self._latest_x, float(x_values[len(values) - 1]))

    def redraw(self) -> None:
        total_point_limit = max(1000, min(3000, self._plot.width() * 3 // 2))
        point_limit = max(256, total_point_limit // max(1, len(self._curves)))
        for channel_name, curve in self._curves.items():
            x_values, y_values = self._buffers[channel_name].ordered_for_plot(point_limit)
            curve.setData(x_values, y_values, skipFiniteCheck=True)

        x_min = max(0.0, self._latest_x - self._window_seconds)
        x_max = max(self._window_seconds, self._latest_x)
        self._plot.setXRange(x_min, x_max, padding=0)

    def set_time_window(self, seconds: float) -> None:
        seconds = max(0.1, min(60.0, float(seconds)))
        if math.isclose(seconds, self._window_seconds):
            return
        self._window_seconds = seconds
        capacity = self._buffer_capacity()
        for buffer in self._buffers.values():
            buffer.resize(capacity)
        self.redraw()

    def set_voltage_range(self, minimum: float, maximum: float) -> None:
        minimum, maximum = float(minimum), float(maximum)
        if maximum <= minimum:
            return
        self._voltage_range = (minimum, maximum)
        self._plot.setYRange(minimum, maximum, padding=0)

    def _buffer_capacity(self) -> int:
        return max(
            self._configuration.read_chunk_size,
            int(math.ceil(self._configuration.sample_rate_hz * self._window_seconds)),
        )

    def _apply_y_range(self, channels: Tuple[NidaqSignalChannelConfiguration, ...]) -> None:
        minimums = [channel.minimum for channel in channels if channel.minimum is not None]
        maximums = [channel.maximum for channel in channels if channel.maximum is not None]
        if minimums and maximums:
            self._plot.setYRange(min(minimums), max(maximums), padding=0.05)
        else:
            self._plot.enableAutoRange(axis="y")


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
        self._pending_blocks: Deque[NidaqSignalSampleBlock] = deque(maxlen=64)
        self._pending_blocks_lock = threading.Lock()

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
        header_layout.addWidget(QLabel("Recording:"))
        self._recording_label = QLabel("off")
        header_layout.addWidget(self._recording_label)

        self._card_widget = CardWidget(title="Analysis", header_right_layout=header_layout)
        self._rolling_plot = _NidaqRollingPlot()
        self._content_tabs = QTabWidget()
        self._content_tabs.setDocumentMode(True)
        self._content_tabs.addTab(self._rolling_plot, "Stream")

        self._signal_scroll = QScrollArea()
        self._signal_scroll.setWidgetResizable(True)
        self._signal_widget = QWidget()
        self._signal_layout = QVBoxLayout(self._signal_widget)
        self._signal_layout.setContentsMargins(8, 8, 8, 8)
        self._signal_layout.setSpacing(6)
        self._signal_scroll.setWidget(self._signal_widget)
        self._content_tabs.addTab(self._signal_scroll, "Signals")
        self._content_tabs.currentChanged.connect(self._redraw_visible_stream)
        self._card_widget.setContentWidget(self._content_tabs)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(4, 0, 0, 0)
        footer_layout.setSpacing(8)
        self._start_stop_button = QPushButton("Start Stream")
        self._clear_button = QPushButton("Clear")
        footer_layout.addWidget(QLabel("Window:"))
        self._graph_seconds = QDoubleSpinBox()
        self._graph_seconds.setDecimals(1)
        self._graph_seconds.setRange(0.1, 60.0)
        self._graph_seconds.setSingleStep(0.5)
        self._graph_seconds.setValue(self._nidaq_signal_monitor.configuration.rolling_window_seconds)
        self._graph_seconds.setSuffix(" s")
        footer_layout.addWidget(QLabel("Y min:"))
        self._graph_min_volts = QDoubleSpinBox()
        self._graph_min_volts.setDecimals(2)
        self._graph_min_volts.setRange(-1000.0, 1000.0)
        self._graph_min_volts.setValue(0.0)
        self._graph_min_volts.setSuffix(" V")
        footer_layout.addWidget(QLabel("Y max:"))
        self._graph_max_volts = QDoubleSpinBox()
        self._graph_max_volts.setDecimals(2)
        self._graph_max_volts.setRange(-1000.0, 1000.0)
        self._graph_max_volts.setValue(5.0)
        self._graph_max_volts.setSuffix(" V")
        self._status_label = QLabel("NI-DAQ signal stream disabled")
        self._status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        footer_layout.addWidget(self._start_stop_button)
        footer_layout.addWidget(self._clear_button)
        footer_layout.addWidget(self._graph_seconds)
        footer_layout.addWidget(self._graph_min_volts)
        footer_layout.addWidget(self._graph_max_volts)
        footer_layout.addWidget(self._status_label)
        self._card_widget.footer.setContent(footer)

        layout = QVBoxLayout()
        layout.addWidget(self._card_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self._nidaq_signal_monitor.sample_block_received += self._sample_block_received
        self._nidaq_signal_monitor.property_changed += self._model_property_changed
        app_model.configuration_loaded_event += self._configuration_loaded
        app_model.laser.property_changed += self._laser_property_changed
        self._start_stop_button.clicked.connect(self._toggle_stream)
        self._clear_button.clicked.connect(self._rolling_plot.clear)
        self._graph_seconds.valueChanged.connect(self._apply_graph_view)
        self._graph_min_volts.valueChanged.connect(self._apply_graph_view)
        self._graph_max_volts.valueChanged.connect(self._apply_graph_view)
        self._apply_graph_view()
        self._plot_timer = QTimer(self)
        self._plot_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._plot_timer.setInterval(33)
        self._plot_timer.timeout.connect(self._flush_pending_blocks)
        self._plot_timer.start()
        self._refresh_from_model()

    def on_close(self):
        self._plot_timer.stop()
        self._nidaq_signal_monitor.sample_block_received -= self._sample_block_received
        self._nidaq_signal_monitor.property_changed -= self._model_property_changed
        self._app_model.configuration_loaded_event -= self._configuration_loaded
        self._app_model.laser.property_changed -= self._laser_property_changed

    def _toggle_stream(self) -> None:
        if not self._nidaq_signal_monitor.hardware_enabled:
            return
        if self._nidaq_signal_monitor.is_running:
            self._nidaq_signal_monitor.stop()
        else:
            self._nidaq_signal_monitor.start()
        self._refresh_from_model()

    def _apply_graph_view(self, *_args) -> None:
        self._rolling_plot.set_time_window(self._graph_seconds.value())
        minimum = self._graph_min_volts.value()
        maximum = self._graph_max_volts.value()
        if maximum <= minimum:
            changed = self.sender()
            if changed is self._graph_min_volts:
                self._graph_max_volts.setValue(minimum + 0.01)
            else:
                self._graph_min_volts.setValue(maximum - 0.01)
            minimum = self._graph_min_volts.value()
            maximum = self._graph_max_volts.value()
        self._rolling_plot.set_voltage_range(minimum, maximum)

    def use_cache(self) -> None:
        """Drain buffered stream samples on the application's display cadence."""
        self._flush_pending_blocks()

    def _sample_block_received(self, block: NidaqSignalSampleBlock) -> None:
        with self._pending_blocks_lock:
            self._pending_blocks.append(block)

    def _flush_pending_blocks(self) -> None:
        with self._pending_blocks_lock:
            blocks = tuple(self._pending_blocks)
            self._pending_blocks.clear()
        if not blocks:
            return
        for block in blocks:
            self._rolling_plot.append(block)
        self._redraw_visible_stream()

    def _redraw_visible_stream(self, *_args) -> None:
        if self.isVisible() and self._content_tabs.currentWidget() is self._rolling_plot:
            self._rolling_plot.redraw()

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
        configuration = model.configuration
        selector_signature = (
            configuration.channels,
            tuple(sorted(self._mapped_physical_channels())),
            model.hardware_enabled,
            model.is_starting,
            model.is_running,
        )
        if selector_signature != self._selector_signature:
            self._selector_signature = selector_signature
            self._rebuild_signal_selector()
        display_configuration = self._display_configuration()
        self._rolling_plot.configure(display_configuration, self._channel_colors_by_name)
        self._stream_state_label.setText("running" if model.is_running else "stopped")
        if not model.hardware_enabled:
            self._stream_state_label.setText("disabled")
        elif not configuration.is_enabled:
            self._stream_state_label.setText("disabled")
        elif model.is_starting:
            self._stream_state_label.setText("starting")
        self._sample_rate_label.setText(f"{configuration.sample_rate_hz:g} Hz")
        self._channel_count_label.setText(str(len(display_configuration.channels)))
        if model.is_starting:
            self._start_stop_button.setText("Starting...")
        else:
            self._start_stop_button.setText("Stop Stream" if model.is_running else "Start Stream")
        can_stream = (
            model.hardware_enabled
            and configuration.is_enabled
            and bool(display_configuration.channels)
            and not model.is_starting
        )
        self._start_stop_button.setEnabled(can_stream)
        self._clear_button.setEnabled(model.hardware_enabled and bool(display_configuration.channels))
        recording_path = model.recording_path
        if not configuration.is_enabled or not model.is_running:
            self._recording_label.setText("off")
        elif recording_path is None:
            self._recording_label.setText("off" if not configuration.record_to_acquisition else "waiting")
        else:
            self._recording_label.setText(recording_path.name)
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
            "Choose the DAQ-port signals acquired and plotted by the shared NI-DAQ stream. "
            "Unmapped signals remain disabled until assigned in Edit → Edit DAQ Ports."
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

        selected_physical_channels = {
            channel.physical_channel
            for channel in configuration.channels
        }
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
            is_selected = physical_channel in selected_physical_channels if physical_channel else False
            channel_text = physical_channel or "not configured"
            checkbox = QCheckBox(f"{label} — {channel_text}")
            color_code_checkbox(checkbox, color)
            checkbox.setChecked(is_selected)
            checkbox.setEnabled(
                self._nidaq_signal_monitor.hardware_enabled
                and not self._nidaq_signal_monitor.is_starting
                and not self._nidaq_signal_monitor.is_running
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
            elif self._nidaq_signal_monitor.is_starting or self._nidaq_signal_monitor.is_running:
                checkbox.setToolTip("Stop the NI-DAQ stream before changing signal selections.")
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
        channels = list(self._nidaq_signal_monitor.configuration.channels)
        if checked:
            channels = [
                channel
                for channel in channels
                if channel.name != candidate.name
                and channel.physical_channel != candidate.physical_channel
            ]
            channels.append(candidate)
        else:
            channels = [
                channel
                for channel in channels
                if channel.name != candidate.name
                and channel.physical_channel != candidate.physical_channel
            ]
        self._app_model.update_nidaq_signal_stream_channels(channels)

    def _display_configuration(self) -> NidaqSignalStreamConfiguration:
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
            is_enabled=configuration.is_enabled and bool(channels),
        )

    def _mapped_physical_channels(self) -> Set[str]:
        mapped = set()
        ports = self._app_model.nidaq_ports
        for field in dataclasses.fields(ports):
            if field.name == "device_name":
                continue
            value = getattr(ports, field.name)
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
