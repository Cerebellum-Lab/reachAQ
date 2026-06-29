import dataclasses

from typing import Tuple, Optional, Dict, List

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QStackedLayout, \
    QDoubleSpinBox, QComboBox

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core import PerfMonitor, SensorAnalysis, LoadCellMonitor, Offset3DTuple, SystemMessageHandler
from autotrainer.pyside import PGWidget, CardWidget, QtIndicator
from autotrainer.pyside.StackedContent import StackedLayout
from autotrainer.pyside.content_widget import ContentWidget, invoke_method

from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.inference_model import InferenceModel
from tools.acquisition.model.user_preferences import UserPreferences


logger = get_verbose_logger(__name__)

_GRAY_COLOR_TUPLE = (240, 240, 240)

_ACTIVE_LOAD_CELL_COLOR = (0, 250, 154)
_INACTIVE_LOAD_CELL_COLOR = _GRAY_COLOR_TUPLE


def _render_offset_3d_value(value: Optional[Offset3DTuple]) -> str:
    return "n/a" if value is None else ", ".join(f"{coord:.1f}" for coord in value)


@dataclasses.dataclass
class _GraphItem:
    measure_idx: int  # correspond to index in tuple pass to measurement received callback
    name: str
    display: str
    unit: str
    y_range: Tuple[int, int]
    x_range: Optional[Tuple[int, int]] = None
    ticks: Optional[List[Tuple[int, str]]] = None


_weight_graph = _GraphItem(
    measure_idx=0,
    name="weight",
    display="Weight",
    unit="gr",
    y_range=(-1, 101),
)
_audio_graph = _GraphItem(
    measure_idx=-1,
    name="audio",
    display="Audio",
    unit="dB",
    y_range=(0, 200), x_range=(0, 64),
    ticks=[(i, str(i * 1500)) for i in range(0, 64, 10)],
)

# NB: same order than SensorAnalysis.measurements_received method
AVAILABLE_GRAPHS = (
    _weight_graph,
    _GraphItem(
        measure_idx=1, name="switch", display="Switch", unit="1/0", y_range=(-1, 2)),
    _GraphItem(measure_idx=2, name="pressure", display="Pressure", unit="Cnts", y_range=(-1, 4099)),
    _GraphItem(measure_idx=3, name="temperature", display="Temperature", unit="\u00b0C", y_range=(-1, 40)),
    _GraphItem(measure_idx=4, name="humidity", display="Humidity", unit="%", y_range=(-1, 101)),
    _audio_graph,
)

_graph_by_name = {
    graph.name: graph
    for graph in AVAILABLE_GRAPHS
}


def _make_graph_plot(graph: _GraphItem):
    widget = QWidget()
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    layout = QHBoxLayout()
    plot = PGWidget(widget)
    plot.setBackground("w")
    plot.setMinimumHeight(140)
    plot.scale_x = 100.0
    if graph.ticks is None:
        plot.getAxis("bottom").setLabel("Time (s)")
    else:
        plot.getAxis('bottom').setTicks([graph.ticks])
    plot.getAxis("left").setLabel(f"{graph.display} ({graph.unit})")
    view_box = plot.getViewBox()
    if graph.x_range is not None:
        view_box.setRange(xRange=graph.x_range)
    view_box.setRange(yRange=graph.y_range)
    layout.addWidget(plot, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    widget.setLayout(layout)
    plot.widget = widget
    plot.getPlotItem().getViewBox().setBackgroundColor(_GRAY_COLOR_TUPLE)
    return plot


class AnalysisContent(ContentWidget):

    diamond_triangle_offset_changed = Signal(str, name="diamond_triangle_offset_changed")
    star_triangle_offset_changed = Signal(str, name="star_triangle_offset_changed")
    measurement_graph_changed = Signal(str, name="measurement_graph_changed")

    def __init__(
        self,
        hardware_model: HardwareModel,
        inference_model: InferenceModel,
        analysis: SensorAnalysis,
        msg_handler: SystemMessageHandler,
        user_pref: UserPreferences,
    ):
        super().__init__()

        self._hardware_model = hardware_model
        self._analysis = analysis
        self._user_pref = user_pref

        # Header
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("D-T:"))
        self._triangle_diamond_offset = QLabel("n/a")
        self.diamond_triangle_offset_changed.connect(self._triangle_diamond_offset.setText)
        layout.addWidget(self._triangle_diamond_offset)

        layout.addWidget(QLabel("S-T:"))
        self._star_triangle_offset = QLabel("n/a")
        self.star_triangle_offset_changed.connect(self._star_triangle_offset.setText)
        layout.addWidget(self._star_triangle_offset)

        self._load_cell_monitor_engaged = QtIndicator(text="Load Cell")
        layout.addWidget(self._load_cell_monitor_engaged)

        self._headbar_pressure_monitor_engaged = QtIndicator(text="Headbar Pressure")
        layout.addWidget(self._headbar_pressure_monitor_engaged)

        self._headbar_switch_engaged = QtIndicator(text="Headbar DIO Switch")
        layout.addWidget(self._headbar_switch_engaged)
        self._tunnel_headfix_widgets = (
            self._load_cell_monitor_engaged,
            self._headbar_pressure_monitor_engaged,
            self._headbar_switch_engaged,
        )

        card = self._card_widget = CardWidget(title="Analysis", header_right_layout=layout)

        self._measurement_plots: Dict[str, PGWidget] = {
            graph.name: _make_graph_plot(graph)
            for graph in AVAILABLE_GRAPHS
        }
        weight_plot = self._plot_weight = self._measurement_plots[_weight_graph.name]
        weight_plot.getPlotItem().getViewBox().setBackgroundColor(_INACTIVE_LOAD_CELL_COLOR)

        self._selected_graph: Optional[_GraphItem] = None

        def on_measurement_graph_changed(graph_name: str):
            # logger.verbose("on_measurement_graph_changed: %s", graph_name)
            graph = _graph_by_name.get(graph_name, None)
            selected = self._selected_graph
            if graph is not None and (selected is None or graph.name != selected.name):
                self._selected_graph = graph
                measure_plot = self._measurement_plots[graph.name]
                # measure_plot.centralWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self._card_widget.setContentWidget(measure_plot)
                # logger.debug("set new graph: %s", measure_plot)

        # Footer
        self._footer = QWidget()
        self._footer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._footer.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        label = self._load_cell_engaged_threshold_label = QLabel("Load Cell Threshold (g):")
        label.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        spinbox = self._load_cell_engaged_threshold_spinbox = QDoubleSpinBox()
        spinbox.setContentsMargins(0, 0, 0, 0)
        spinbox.setMaximumHeight(25)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(1)
        spinbox.setRange(0, 100)
        def value_changed(value):
            analysis.load_cell_monitor.load_cell_engaged_threshold = value
        spinbox.valueChanged.connect(value_changed)
        spinbox.setValue(analysis.load_cell_monitor.load_cell_engaged_threshold)
        layout.addWidget(spinbox)

        combo = self._measurement_graph_combo = QComboBox()
        pref_graph_name = self._user_pref.measurement_graph
        for idx, graph in enumerate(AVAILABLE_GRAPHS):
            combo.addItem(graph.display, graph)
            if graph.name == pref_graph_name:
                combo.setCurrentIndex(idx)
        combo.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(combo, stretch=1, alignment=Qt.AlignmentFlag.AlignRight)
        combo.currentIndexChanged.connect(self._on_graph_combox_changed)

        self._footer.setLayout(layout)

        self._card_widget.footer.setContent(self._footer)
        # default CardWidget.footer layout margins are too big
        self._card_widget.footer.layout().setContentsMargins(4, 0, 0, 0)

        # Final layout
        layout = QVBoxLayout()
        layout.addWidget(self._card_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

        self._perf_monitor = PerfMonitor(name="headFixContent", units="mps", report_window=30)

        self.set_is_editable(False)

        inference_model.star_triangle_offset_changed += self._star_triangle_offset_changed
        inference_model.diamond_triangle_offset_changed += self._diamond_triangle_offset_changed

        self._analysis.load_cell_monitor.property_changed += self._load_cell_monitor_property_changed

        msg_handler.measurement_callback = self._measurement_received
        msg_handler.audio_callback = self._audio_received
        user_pref.property_changed += self._on_user_pref_changed
        hardware_model.property_changed += self._hardware_model_property_changed
        self._update_tunnel_headfix_visibility(hardware_model.tunnel_headfix_enabled)
        #
        on_measurement_graph_changed(
            _graph_by_name.get(user_pref.measurement_graph, AVAILABLE_GRAPHS[0]).name
        )
        self.measurement_graph_changed.connect(on_measurement_graph_changed)

    def _update_tunnel_headfix_visibility(self, is_enabled: bool):
        for widget in self._tunnel_headfix_widgets:
            widget.setVisible(is_enabled)
        self._load_cell_engaged_threshold_label.setVisible(is_enabled)
        self._load_cell_engaged_threshold_spinbox.setVisible(is_enabled)

    @invoke_method
    def _hardware_model_property_changed(self, name: str, value, _):
        if name == HardwareModel.TUNNEL_HEADFIX_ENABLED:
            self._update_tunnel_headfix_visibility(value)

    def set_is_capture_active(self, is_active: bool):
        if is_active:
            self._perf_monitor.reset()

    @invoke_method
    def use_cache(self):
        selected = self._selected_graph
        for plot_name, plot in self._measurement_plots.items():
            if selected is not None and plot_name == selected.name:
                plot.use_cache()
            else:
                plot.replace_cache([])
        self._load_cell_monitor_engaged.setState(self._analysis.load_cell_monitor.is_engaged)
        self._headbar_switch_engaged.setState(self._analysis.is_headbar_switch_engaged)
        self._headbar_pressure_monitor_engaged.setState(self._analysis.headbar_pressure_monitor.is_engaged)

    def _on_graph_combox_changed(self, idx: int):
        graph = self._measurement_graph_combo.itemData(idx)
        if graph is not None:
            self._user_pref.measurement_graph = graph.name
        else:
            logger.warning("graph None")

    @invoke_method
    def _measurement_received(self, measurements: Tuple[List[float], List[bool], List[float], List[float], List[float]]):
        # weight_vals, switch_vals, pressure_vals, temperature_vals, humidity_vals
        values = measurements[_weight_graph.measure_idx]
        self._perf_monitor.add_cycles(len(values))
        #
        selected = self._selected_graph
        for plot_name, plot in self._measurement_plots.items():
            graph = _graph_by_name[plot_name]
            if graph.measure_idx >= 0 and selected is not None and graph.name == selected.name:
                plot.cache_data(measurements[graph.measure_idx])

    @invoke_method
    def _audio_received(self, spectrum):
        selected = self._selected_graph
        if selected is not None and selected.name == _audio_graph.name:
            audio_plot = self._measurement_plots[_audio_graph.name]
            audio_plot.replace_cache(spectrum)

    @invoke_method
    def _load_cell_monitor_property_changed(self, name, value, _):
        if name == LoadCellMonitor.IS_ENGAGED_PROPERTY:
            if value:
                self._plot_weight.getPlotItem().getViewBox().setBackgroundColor(_ACTIVE_LOAD_CELL_COLOR)
            else:
                self._plot_weight.getPlotItem().getViewBox().setBackgroundColor(_INACTIVE_LOAD_CELL_COLOR)
        elif name == LoadCellMonitor.LOAD_CELL_ENGAGED_THRESHOLD_PROPERTY:
            self._load_cell_engaged_threshold_spinbox.setValue(value)

    @invoke_method
    def _diamond_triangle_offset_changed(self, offset: Optional[Offset3DTuple]):
        self.diamond_triangle_offset_changed.emit(_render_offset_3d_value(offset))

    @invoke_method
    def _star_triangle_offset_changed(self, offset: Optional[Offset3DTuple]):
        self.star_triangle_offset_changed.emit(
            "n/a" if offset is None else f"{offset.distance:.1f} mm"
        )

    @invoke_method
    def _on_user_pref_changed(self, name: str, value, old_value):
        if name == UserPreferences.MEASUREMENT_GRAPH:
            self.measurement_graph_changed.emit(value)
            pref_graph_name = value
            combo = self._measurement_graph_combo
            for idx, graph in enumerate(AVAILABLE_GRAPHS):
                if graph.name == pref_graph_name:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
                    break
