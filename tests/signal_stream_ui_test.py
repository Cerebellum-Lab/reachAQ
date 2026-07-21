import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NidaqPortConfiguration,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    ObservableObject,
)
from autotrainer.device import (  # noqa: E402
    LaserCalibrationRamp,
    LaserPulseTrain,
    NidaqSignalSampleBlock,
    NullLaserController,
)
from tools.acquisition.model.laser_model import LaserModel  # noqa: E402
from tools.acquisition.model.app_model import AppModel  # noqa: E402
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel  # noqa: E402
from tools.acquisition.view.analysis_content import AnalysisContent  # noqa: E402
from tools.acquisition.view.laser_control_content import _LaserChannelTab  # noqa: E402
from tools.acquisition.view.rolling_stream_buffer import RollingStreamBuffer  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _AnalysisAppStub(ObservableObject):
    def __init__(self, monitor):
        super().__init__(("configuration_loaded_event",))
        self.nidaq_signal_monitor = monitor
        self.nidaq_ports = NidaqPortConfiguration(
            cam_frames="Dev1/port0/line0",
            tone1="Dev1/port0/line3",
            tone2="Dev1/port0/line4",
        )
        self.laser = LaserModel()
        self.laser.set_configuration_offline(
            LaserSystemConfiguration.from_channels((_laser_channel(),), backend="disabled")
        )
        self.signal_configuration_save_count = 0

    def update_nidaq_signal_stream_channels(self, channels):
        self.nidaq_signal_monitor.set_stream_channels(channels)
        self.signal_configuration_save_count += 1


class _LaserAppStub:
    def __init__(self, laser):
        self.laser = laser
        self.nidaq_signal_monitor = NidaqSignalMonitorModel()
        self.nidaq_signal_monitor._hardware_enabled = True
        self.signal_configuration_save_count = 0

    def update_nidaq_signal_stream_channels(self, channels):
        self.nidaq_signal_monitor.set_stream_channels(channels)
        self.signal_configuration_save_count += 1


def _crashing_signal_worker(_configuration, _message_queue, _stop_event, _log_dict_config):
    os._exit(23)


def _hanging_signal_worker(_configuration, _message_queue, stop_event, _log_dict_config):
    stop_event.wait(60.0)


def _stream_configuration():
    return NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration(
                name="cam_frames",
                physical_channel="Dev1/port0/line0",
                kind="digital",
            ),
        ),
        is_enabled=True,
    )


def _laser_channel():
    return LaserChannelConfiguration(
        channel_id=LaserChannelId.LASER_1,
        analog_output="Dev1/ao0",
        diode_input="Dev1/ai0",
        shutter_output="Dev1/port0/line2",
        command_copy_input="Dev1/ai1",
    )


def test_signal_monitor_refuses_to_start_when_nidaq_hardware_is_disabled():
    monitor = NidaqSignalMonitorModel()
    monitor.load_configuration(_stream_configuration())

    assert not monitor.hardware_enabled
    assert not monitor.is_running
    assert not monitor.start()
    assert "hardware disabled" in monitor.status_message


def test_configured_signal_monitor_remains_stopped_until_explicit_start():
    monitor = NidaqSignalMonitorModel()
    monitor._hardware_enabled = True
    start_calls = []
    monitor.start = lambda: start_calls.append(True)

    monitor.load_configuration(_stream_configuration())

    assert start_calls == []
    assert not monitor.is_running
    assert monitor.status_message == "NI-DAQ signal stream stopped"


def test_numpy_stream_buffer_wraps_without_shifting_existing_window():
    buffer = RollingStreamBuffer(5)
    buffer.append((0, 1, 2), (10, 11, 12))
    buffer.append((3, 4, 5, 6), (13, 14, 15, 16))

    x_values, y_values = buffer.ordered()

    assert x_values.tolist() == [2, 3, 4, 5, 6]
    assert y_values.tolist() == [12, 13, 14, 15, 16]
    assert buffer.latest_x == 6


def test_analysis_signal_selection_requires_mapped_port_and_nidaq_enable(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    app_model = _AnalysisAppStub(monitor)
    content = AnalysisContent(app_model)
    try:
        camera_frames = content._signal_checkboxes["cam_frames"]
        barcode = content._signal_checkboxes["barcode"]
        tone1 = content._signal_checkboxes["tone1"]
        tone2 = content._signal_checkboxes["tone2"]
        assert camera_frames.isEnabled()
        assert camera_frames.isChecked()
        assert not barcode.isEnabled()
        assert not barcode.isChecked()
        assert tone1.isEnabled()
        assert tone2.isEnabled()
        assert set(content._signal_checkboxes) == {
            "cam_frames",
            "barcode",
            "tone1",
            "tone2",
            "tone3_r",
            "tone3_l",
        }
        assert camera_frames.property("signalColor") == "#1769e0"
        assert barcode.property("signalColor") == "#128a43"
        assert content._rolling_plot._legend.entries == (
            ("cam_frames (logic)", (23, 105, 224), False),
        )
        assert all(
            curve.opts["pen"].style().name == "SolidLine"
            for curve in content._rolling_plot._curves.values()
        )
        assert content._channel_count_label.text() == "1"
        assert content._start_stop_button.isEnabled()
        assert content._clear_button.isEnabled()

        tone1.setChecked(True)
        qapp.processEvents()
        assert content._signal_checkboxes["tone1"].property("signalColor") == "#128a43"
        assert content._rolling_plot._legend.entries[1] == (
            "tone1 (logic)",
            (18, 138, 67),
            False,
        )
        content._signal_checkboxes["tone1"].setChecked(False)
        qapp.processEvents()

        content._signal_checkboxes["cam_frames"].setChecked(False)
        qapp.processEvents()
        assert monitor.configuration.channels == tuple()
        assert not content._start_stop_button.isEnabled()
        assert app_model.signal_configuration_save_count == 3

        content._signal_checkboxes["tone1"].setChecked(True)
        qapp.processEvents()
        assert content._signal_checkboxes["tone1"].property("signalColor") == "#1769e0"
        assert content._rolling_plot._legend.entries == (
            ("tone1 (logic)", (23, 105, 224), False),
        )
        content._signal_checkboxes["tone1"].setChecked(False)
        qapp.processEvents()

        content._signal_checkboxes["cam_frames"].setChecked(True)
        qapp.processEvents()
        assert tuple(
            channel.physical_channel
            for channel in monitor.configuration.channels
        ) == ("Dev1/port0/line0",)
        assert content._start_stop_button.isEnabled()
        assert app_model.signal_configuration_save_count == 6

        monitor.set_hardware_enabled(False)
        qapp.processEvents()

        assert not content._start_stop_button.isEnabled()
        assert not content._clear_button.isEnabled()
        assert not content._signal_checkboxes["cam_frames"].isEnabled()

        monitor._set_error("native NI-DAQ crash details")
        qapp.processEvents()
        assert "native NI-DAQ crash details" not in content._status_label.text()
        assert content._status_label.styleSheet() == ""
    finally:
        content.on_close()
        content.deleteLater()


def test_analysis_excludes_laser_owned_inputs_from_selector_and_graph(qapp):
    laser_input = NidaqSignalChannelConfiguration(
        name="laser1_diode",
        physical_channel="Dev1/ai0",
        kind="analog",
    )
    base = _stream_configuration()
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = NidaqSignalStreamConfiguration(
        channels=base.channels + (laser_input,),
        is_enabled=True,
    )
    monitor._hardware_enabled = True
    app_model = _AnalysisAppStub(monitor)
    content = AnalysisContent(app_model)
    try:
        assert not any(key.startswith("laser") for key in content._signal_checkboxes)
        assert tuple(channel.name for channel in content._display_configuration().channels) == (
            "cam_frames",
        )
        assert content._channel_count_label.text() == "1"
        assert tuple(entry[0] for entry in content._rolling_plot._legend.entries) == (
            "cam_frames (logic)",
        )
    finally:
        content.on_close()
        content.deleteLater()


def test_analysis_coalesces_sample_blocks_before_single_rolling_redraw(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    content = AnalysisContent(_AnalysisAppStub(monitor))
    content.show()
    qapp.processEvents()
    redraw_calls = []
    original_redraw = content._rolling_plot.redraw
    content._rolling_plot.redraw = lambda: (redraw_calls.append(True), original_redraw())
    try:
        for sample_index in (0, 100, 200):
            content._sample_block_received(
                NidaqSignalSampleBlock(
                    wall_time=1.0,
                    perf_time=1.0,
                    sample_rate_hz=1000.0,
                    sample_index=sample_index,
                    channels=monitor.configuration.channels,
                    values={"cam_frames": tuple(float(index % 2) for index in range(100))},
                )
            )

        assert content._rolling_plot._buffers["cam_frames"].size == 0
        content._flush_pending_blocks()

        assert redraw_calls == [True]
        assert content._rolling_plot._buffers["cam_frames"].size == 300
    finally:
        content.on_close()
        content.deleteLater()


def test_signal_monitor_contains_native_worker_failure_outside_application_process():
    monitor = NidaqSignalMonitorModel(worker_target=_crashing_signal_worker)
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True

    assert monitor.start()
    deadline = time.monotonic() + 10.0
    while (monitor.is_starting or not monitor.error_message) and time.monotonic() < deadline:
        time.sleep(0.02)

    assert not monitor.is_starting
    assert not monitor.is_running
    assert "exit code 23" in monitor.error_message


def test_signal_monitor_times_out_hung_runtime_without_blocking_caller():
    monitor = NidaqSignalMonitorModel(
        worker_target=_hanging_signal_worker,
        startup_timeout_seconds=0.2,
    )
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True

    call_started = time.monotonic()
    assert monitor.start()
    assert time.monotonic() - call_started < 0.5

    deadline = time.monotonic() + 10.0
    while (monitor.is_starting or not monitor.error_message) and time.monotonic() < deadline:
        time.sleep(0.02)

    assert not monitor.is_starting
    assert not monitor.is_running
    assert "did not become ready within 0.2 seconds" in monitor.error_message


def test_app_model_persists_analysis_signal_selection_to_loaded_configuration():
    monitor = NidaqSignalMonitorModel()
    app_model = object.__new__(AppModel)
    app_model._nidaq_signal_monitor = monitor
    app_model._loaded_configuration = SimpleNamespace(nidaq_stream=None)
    save_calls = []
    app_model.save_configuration = lambda: save_calls.append(True)

    channels = _stream_configuration().channels
    app_model.update_nidaq_signal_stream_channels(channels)

    assert monitor.configuration.channels == channels
    assert monitor.configuration.is_enabled
    assert app_model._loaded_configuration.nidaq_stream == monitor.configuration
    assert save_calls == [True]


def test_laser_trace_auto_resumes_and_displays_entire_calibration_ramp(qapp):
    channel = _laser_channel()
    configuration = LaserSystemConfiguration.from_channels(
        (channel,),
        backend="null",
        sample_rate_hz=1000.0,
    )
    laser = LaserModel(NullLaserController(configuration))
    app_model = _LaserAppStub(laser)
    tab = _LaserChannelTab(
        app_model,
        channel,
        True,
        configuration.sample_rate_hz,
        lambda _status, operation: operation(),
        lambda _message, _is_error: None,
    )
    laser.trace_received += tab.append_trace
    try:
        assert not tab._trace_streaming
        assert tab._trace_toggle_button.text() == "Start Stream"
        tab._set_trace_streaming(False)
        points = laser.run_calibration_ramp(
            LaserCalibrationRamp(
                channel_id=LaserChannelId.LASER_1,
                start_volts=0.0,
                stop_volts=4.0,
                steps=5,
                samples_per_step=100,
            )
        )

        command_x, command_y = tab._trace_data["command"]
        diode_x, diode_y = tab._trace_data["diode"]
        assert tab._trace_streaming
        assert tab._trace_toggle_button.text() == "Stop Stream"
        assert tab._trace_signal_checkboxes["diode"].property("signalColor") == "#128a43"
        assert tab._trace_signal_checkboxes["copy"].property("signalColor") == "#d66b00"
        assert tuple(
            tab._trace_tabs.tabText(index)
            for index in range(tab._trace_tabs.count())
        ) == ("Stream", "Signals")
        assert tab._trace_stream_page.isAncestorOf(tab._trace_plot)
        assert tab._trace_signals_page.isAncestorOf(tab._trace_signal_checkboxes["diode"])
        assert tab._trace_signals_page.isAncestorOf(tab._trace_signal_checkboxes["copy"])
        assert tuple(entry[0] for entry in tab._trace_legend.entries) == (
            "Command output",
            "Diode feedback",
            "Command copy",
        )
        assert all(
            curve.opts["pen"].style().name == "SolidLine"
            for curve in tab._trace_curves.values()
        )
        assert len(points) == len(command_x) == len(command_y) == 5
        assert len(diode_x) == len(diode_y) == 5
        assert command_y == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
        assert diode_y == pytest.approx(command_y)
        assert "5 ramp points displayed" in tab._trace_status.text()

        laser.run_pulse_train(
            LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1,
                amplitude_volts=2.0,
                duration_ms=10.0,
            )
        )
        assert "internal pulse" in tab._trace_status.text()
        assert len(tab._trace_data["command"][0]) > 5
    finally:
        laser.trace_received -= tab.append_trace
        app_model.nidaq_signal_monitor.close()
        laser.close()
        tab.deleteLater()


def test_laser_tab_owns_and_persists_its_input_stream_options(qapp):
    channel = _laser_channel()
    configuration = LaserSystemConfiguration.from_channels(
        (channel,),
        backend="null",
        sample_rate_hz=1000.0,
    )
    laser = LaserModel(NullLaserController(configuration))
    app_model = _LaserAppStub(laser)
    tab = _LaserChannelTab(
        app_model,
        channel,
        True,
        configuration.sample_rate_hz,
        lambda _status, operation: operation(),
        lambda _message, _is_error: None,
    )
    try:
        diode = tab._trace_signal_checkboxes["diode"]
        command_copy = tab._trace_signal_checkboxes["copy"]
        assert diode.isEnabled()
        assert command_copy.isEnabled()
        assert not diode.isChecked()
        assert not command_copy.isChecked()

        diode.setChecked(True)
        qapp.processEvents()
        assert tuple(
            (stream_channel.name, stream_channel.physical_channel)
            for stream_channel in app_model.nidaq_signal_monitor.configuration.channels
        ) == (("laser1_diode", "Dev1/ai0"),)
        assert app_model.signal_configuration_save_count == 1
        assert tab._trace_daq_button.isEnabled()
        assert tab._trace_daq_button.text() == "Start DAQ Inputs"

        command_copy.setChecked(True)
        qapp.processEvents()
        assert tuple(
            stream_channel.name
            for stream_channel in app_model.nidaq_signal_monitor.configuration.channels
        ) == ("laser1_diode", "laser1_command_copy")
        assert app_model.signal_configuration_save_count == 2
    finally:
        app_model.nidaq_signal_monitor.close()
        laser.close()
        tab.deleteLater()
