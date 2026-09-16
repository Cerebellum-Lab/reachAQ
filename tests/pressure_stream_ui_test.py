import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    ObservableObject,
    SystemStatusMessageKind,
)
from autotrainer.device import PressureReading, Target  # noqa: E402
from tools.acquisition.model.nidaq_signal_monitor_model import (  # noqa: E402
    NidaqSignalMonitorModel,
)
from tools.acquisition.view.analysis_content import AnalysisContent  # noqa: E402

from tests.signal_stream_ui_test import _AnalysisAppStub, _stream_configuration  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def content(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    content = AnalysisContent(_AnalysisAppStub(monitor))
    yield content
    content.on_close()
    content.deleteLater()


def _send_pressure(content, instance: int, pressure: int, perf_time: float) -> None:
    content._app_model.message_handler.decoded_message_received(
        SystemStatusMessageKind.PRESSURE_READING,
        PressureReading(
            target=Target.PELLET_DEVICE,
            instance=instance,
            pressure=pressure,
        ),
        perf_time,
        perf_time + 1_000.0,
    )


def _show_pressure_tab(content):
    content._content_tabs.setCurrentWidget(content._pressure_plots)


def _tab_titles(content):
    return [
        content._content_tabs.tabText(index)
        for index in range(content._content_tabs.count())
    ]


def test_analysis_panel_offers_a_pressure_tab(content):
    assert "Pressure" in _tab_titles(content)


def test_each_sensor_gets_its_own_graph(content):
    assert set(content._pressure_plots.curves) == {0, 1}


def test_readings_are_drawn_on_the_graph_for_their_sensor(content):
    _show_pressure_tab(content)
    content._pressure_plots.set_show_raw_counts(True)
    _send_pressure(content, 0, 4095, 10.0)
    _send_pressure(content, 1, 458, 10.0)

    content._flush_pending_blocks()

    assert content._pressure_plots.curves[0].yData.tolist() == [4095.0]
    assert content._pressure_plots.curves[1].yData.tolist() == [458.0]


def test_graphs_show_volts_by_default(content):
    _show_pressure_tab(content)
    _send_pressure(content, 0, 4095, 10.0)

    content._flush_pending_blocks()

    assert content._pressure_plots.curves[0].yData.tolist() == pytest.approx([3.3])


def test_graphs_show_raw_counts_when_the_toggle_is_set(content):
    _show_pressure_tab(content)
    _send_pressure(content, 0, 4095, 10.0)

    content._pressure_plots.set_show_raw_counts(True)
    content._flush_pending_blocks()

    assert content._pressure_plots.curves[0].yData.tolist() == [4095.0]


def test_toggling_units_relabels_the_y_axis(content):
    content._pressure_plots.set_show_raw_counts(False)
    assert "Volts" in content._pressure_plots.y_axis_label()

    content._pressure_plots.set_show_raw_counts(True)
    assert "ADC" in content._pressure_plots.y_axis_label()


def test_clear_discards_the_pressure_graphs(content):
    _show_pressure_tab(content)
    _send_pressure(content, 0, 4095, 10.0)
    content._flush_pending_blocks()

    content._clear_plot()

    assert content._pressure_plots.curves[0].yData is None or (
        len(content._pressure_plots.curves[0].yData) == 0
    )


def test_pressure_graphs_are_live_without_starting_the_nidaq_stream(content):
    _show_pressure_tab(content)
    assert not content._nidaq_signal_monitor.is_running

    content._pressure_plots.set_show_raw_counts(True)
    _send_pressure(content, 0, 1234, 10.0)
    content._flush_pending_blocks()

    assert content._pressure_plots.curves[0].yData.tolist() == [1234.0]


def test_closing_the_panel_unsubscribes_the_pressure_model(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    content = AnalysisContent(_AnalysisAppStub(monitor))
    content.on_close()
    try:
        _send_pressure(content, 0, 4095, 10.0)

        assert content._pressure_model.snapshot() is None
    finally:
        content.deleteLater()


def test_readings_keep_buffering_while_the_pressure_tab_is_hidden(content):
    content._content_tabs.setCurrentIndex(0)

    _send_pressure(content, 0, 4095, 10.0)
    content._flush_pending_blocks()

    assert content._pressure_plots.curves[0].yData is None
    _show_pressure_tab(content)
    content._flush_pending_blocks()
    assert content._pressure_plots.curves[0].yData.tolist() == pytest.approx([3.3])
