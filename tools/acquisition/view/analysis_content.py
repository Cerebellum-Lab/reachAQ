from __future__ import annotations

from typing import Dict, List, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from autotrainer.core import NidaqSignalChannelConfiguration, NidaqSignalStreamConfiguration
from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import NidaqSignalSampleBlock
from autotrainer.pyside import CardWidget, PGWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel


logger = get_verbose_logger(__name__)

_GRAY_COLOR_TUPLE = (240, 240, 240)
_PLOT_COLORS = (
    (30, 90, 180),
    (210, 80, 70),
    (50, 150, 90),
    (180, 120, 30),
    (135, 85, 170),
    (70, 160, 180),
    (80, 80, 80),
    (190, 70, 130),
)


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
        self._legend = self._plot.addLegend(offset=(-8, 8))
        layout.addWidget(self._plot)

        self._configuration = NidaqSignalStreamConfiguration()
        self._curves: Dict[str, object] = {}
        self._x_values: Dict[str, List[float]] = {}
        self._y_values: Dict[str, List[float]] = {}
        self._latest_x = 0.0

    def configure(self, configuration: NidaqSignalStreamConfiguration) -> None:
        self._configuration = configuration
        self._plot.clear()
        self._legend = self._plot.addLegend(offset=(-8, 8))
        self._curves.clear()
        self._x_values.clear()
        self._y_values.clear()
        self._latest_x = 0.0
        for index, channel in enumerate(configuration.channels):
            color = _PLOT_COLORS[index % len(_PLOT_COLORS)]
            style = Qt.PenStyle.DashLine if channel.kind == "digital" else Qt.PenStyle.SolidLine
            pen = pg.mkPen(color=color, width=1.5, style=style)
            display_name = f"{channel.name} ({channel.unit})"
            self._curves[channel.name] = self._plot.plot([], [], pen=pen, name=display_name)
            self._x_values[channel.name] = []
            self._y_values[channel.name] = []
        self._apply_y_range(configuration.channels)
        self._plot.setXRange(0, configuration.rolling_window_seconds, padding=0)

    def clear(self) -> None:
        for channel_name, curve in self._curves.items():
            self._x_values[channel_name] = []
            self._y_values[channel_name] = []
            curve.setData([], [])
        self._latest_x = 0.0
        self._plot.setXRange(0, self._configuration.rolling_window_seconds, padding=0)

    def append(self, block: NidaqSignalSampleBlock) -> None:
        if tuple(block.channels) != tuple(self._configuration.channels):
            self.configure(
                NidaqSignalStreamConfiguration(
                    channels=block.channels,
                    is_enabled=self._configuration.is_enabled,
                    sample_rate_hz=block.sample_rate_hz,
                    read_chunk_size=self._configuration.read_chunk_size,
                    rolling_window_seconds=self._configuration.rolling_window_seconds,
                    record_to_acquisition=self._configuration.record_to_acquisition,
                    output_name=self._configuration.output_name,
                )
            )

        for channel in block.channels:
            values = block.values.get(channel.name)
            if not values:
                continue
            x_values = self._x_values.setdefault(channel.name, [])
            y_values = self._y_values.setdefault(channel.name, [])
            first_sample = block.sample_index / block.sample_rate_hz
            x_values.extend(first_sample + index / block.sample_rate_hz for index in range(len(values)))
            y_values.extend(values)
            self._latest_x = max(self._latest_x, x_values[-1])

        cutoff = max(0.0, self._latest_x - self._configuration.rolling_window_seconds)
        for channel_name, curve in self._curves.items():
            x_values = self._x_values[channel_name]
            y_values = self._y_values[channel_name]
            trim = 0
            while trim < len(x_values) and x_values[trim] < cutoff:
                trim += 1
            if trim:
                del x_values[:trim]
                del y_values[:trim]
            curve.setData(x_values, y_values)

        x_min = max(0.0, self._latest_x - self._configuration.rolling_window_seconds)
        x_max = max(self._configuration.rolling_window_seconds, self._latest_x)
        self._plot.setXRange(x_min, x_max, padding=0)

    def _apply_y_range(self, channels: Tuple[NidaqSignalChannelConfiguration, ...]) -> None:
        minimums = [channel.minimum for channel in channels if channel.minimum is not None]
        maximums = [channel.maximum for channel in channels if channel.maximum is not None]
        if minimums and maximums:
            self._plot.setYRange(min(minimums), max(maximums), padding=0.05)
        else:
            self._plot.enableAutoRange(axis="y")


class AnalysisContent(ContentWidget):
    """Rolling NI-DAQ input stream display for acquisition hardware checks."""

    def __init__(self, nidaq_signal_monitor: NidaqSignalMonitorModel):
        super().__init__()

        self._nidaq_signal_monitor = nidaq_signal_monitor

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
        self._card_widget.setContentWidget(self._rolling_plot)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(4, 0, 0, 0)
        footer_layout.setSpacing(8)
        self._start_stop_button = QPushButton("Start Stream")
        self._clear_button = QPushButton("Clear")
        self._status_label = QLabel("NI-DAQ signal stream disabled")
        self._status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        footer_layout.addWidget(self._start_stop_button)
        footer_layout.addWidget(self._clear_button)
        footer_layout.addWidget(self._status_label)
        self._card_widget.footer.setContent(footer)

        layout = QVBoxLayout()
        layout.addWidget(self._card_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        nidaq_signal_monitor.sample_block_received += self._sample_block_received
        nidaq_signal_monitor.property_changed += self._model_property_changed
        self._start_stop_button.clicked.connect(self._toggle_stream)
        self._clear_button.clicked.connect(self._rolling_plot.clear)
        self._refresh_from_model()

    def on_close(self):
        self._nidaq_signal_monitor.sample_block_received -= self._sample_block_received
        self._nidaq_signal_monitor.property_changed -= self._model_property_changed

    def _toggle_stream(self) -> None:
        if self._nidaq_signal_monitor.is_running:
            self._nidaq_signal_monitor.stop()
        else:
            self._nidaq_signal_monitor.start()
        self._refresh_from_model()

    def use_cache(self) -> None:
        """Retained for MainContent's periodic refresh loop; stream samples arrive via model events."""

    @invoke_method
    def _sample_block_received(self, block: NidaqSignalSampleBlock) -> None:
        self._rolling_plot.append(block)

    @invoke_method
    def _model_property_changed(self, _name: str, _value, _old_value) -> None:
        self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        model = self._nidaq_signal_monitor
        configuration = model.configuration
        self._rolling_plot.configure(configuration)
        self._stream_state_label.setText("running" if model.is_running else "stopped")
        if not configuration.is_enabled:
            self._stream_state_label.setText("disabled")
        self._sample_rate_label.setText(f"{configuration.sample_rate_hz:g} Hz")
        self._channel_count_label.setText(str(len(configuration.channels)))
        self._start_stop_button.setText("Stop Stream" if model.is_running else "Start Stream")
        self._start_stop_button.setEnabled(configuration.is_enabled)
        recording_path = model.recording_path
        if not configuration.is_enabled or not model.is_running:
            self._recording_label.setText("off")
        elif recording_path is None:
            self._recording_label.setText("off" if not configuration.record_to_acquisition else "waiting")
        else:
            self._recording_label.setText(recording_path.name)
        error_message = model.error_message
        if error_message:
            self._status_label.setText(error_message)
            self._status_label.setStyleSheet("color: #b00020;")
        else:
            self._status_label.setText(model.status_message)
            self._status_label.setStyleSheet("")
