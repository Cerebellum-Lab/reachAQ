import dataclasses
import os
import time
from types import SimpleNamespace

import pytest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QSizePolicy  # noqa: E402

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
from tools.acquisition.model.laser_plot_process import LaserPlotProcess  # noqa: E402
from tools.acquisition.model.app_model import AppModel  # noqa: E402
from tools.acquisition.model.nidaq_signal_monitor_model import (  # noqa: E402
    NidaqSignalMonitorModel,
)
from tools.acquisition.model.nidaq_sample_ring import SharedNidaqSampleRing  # noqa: E402
from tools.acquisition.view.analysis_content import AnalysisContent  # noqa: E402
from tools.acquisition.view.laser_control_content import (  # noqa: E402
    LaserControlContent,
    _LaserChannelTab,
)
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


def _crashing_signal_worker(
    _configuration, _message_queue, _sample_ring, _stop_event, _log_dict_config,
):
    os._exit(23)


def _hanging_signal_worker(
    _configuration, _message_queue, _sample_ring, stop_event, _log_dict_config,
):
    stop_event.wait(60.0)


def _shared_ring_signal_worker(
    configuration, message_queue, sample_ring, stop_event, _log_dict_config,
):
    sample_ring.write_block(NidaqSignalSampleBlock(
        wall_time=time.time(),
        perf_time=time.perf_counter(),
        sample_rate_hz=configuration.sample_rate_hz,
        sample_index=0,
        channels=configuration.channels,
        values={"cam_frames": (0.0, 1.0, 0.0, 1.0)},
    ))
    message_queue.put(("ready", None))
    stop_event.wait(10.0)
    message_queue.put(("stopped", None))


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


def _wait_for_plot_snapshot(content, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if content._rolling_plot.display_latest(content._plot_process):
            frame = content._rolling_plot._last_frame
            if any(
                np.isfinite(content._rolling_plot._display_y[channel_name][:point_count]).any()
                for channel_name, point_count in zip(
                    content._rolling_plot._curves,
                    frame.point_counts,
                )
            ):
                return frame
        time.sleep(0.02)
    raise AssertionError("analysis plot process did not produce a snapshot")


def _wait_for_laser_plot(content, channel_id=1, timeout=5.0, minimum_points=1):
    deadline = time.monotonic() + timeout
    tab = content._channel_tabs[channel_id - 1]
    while time.monotonic() < deadline:
        content._flush_laser_plots()
        tab.redraw_trace()
        if any(
            len(x_values) >= minimum_points
            for x_values, _y_values in tab._trace_data.values()
        ):
            return tab
        time.sleep(0.02)
    raise AssertionError("laser plot process did not produce a snapshot")


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


def test_signal_monitor_derives_read_chunk_from_display_refresh_rate():
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()

    monitor.set_display_refresh_rate(120.0)

    assert monitor.display_refresh_rate_hz == 120.0
    assert monitor.effective_read_chunk_size == 83
    assert monitor.configuration.read_chunk_size == 500


def test_signal_monitor_shared_ring_preserves_samples_and_reports_overrun():
    configuration = _stream_configuration()
    ring = SharedNidaqSampleRing(configuration, capacity=5)
    ring.write_block(NidaqSignalSampleBlock(
        wall_time=1.0,
        perf_time=1.0,
        sample_rate_hz=10_000.0,
        sample_index=0,
        channels=configuration.channels,
        values={"cam_frames": (0.0, 1.0, 0.0)},
    ))
    ring.write_block(NidaqSignalSampleBlock(
        wall_time=1.1,
        perf_time=1.1,
        sample_rate_hz=10_000.0,
        sample_index=3,
        channels=configuration.channels,
        values={"cam_frames": (1.0, 0.0, 1.0, 0.0)},
    ))
    destination = np.empty((1, 5), dtype=np.float32)

    snapshot = ring.copy_since(0, destination)

    assert snapshot.start_sample_index == 2
    assert snapshot.end_sample_index == 7
    assert snapshot.overrun_samples == 2
    assert snapshot.epoch == 0
    assert snapshot.generation == 2
    assert destination[0, :5].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_signal_monitor_shared_ring_reports_source_sample_gap():
    configuration = _stream_configuration()
    ring = SharedNidaqSampleRing(configuration, capacity=8)
    ring.write_block(NidaqSignalSampleBlock(
        wall_time=1.0,
        perf_time=1.0,
        sample_rate_hz=10_000.0,
        sample_index=0,
        channels=configuration.channels,
        values={"cam_frames": (0.0, 1.0)},
    ))
    ring.write_block(NidaqSignalSampleBlock(
        wall_time=1.1,
        perf_time=1.1,
        sample_rate_hz=10_000.0,
        sample_index=5,
        channels=configuration.channels,
        values={"cam_frames": (1.0, 0.0)},
    ))
    destination = np.empty((1, 8), dtype=np.float32)

    snapshot = ring.copy_since(2, destination)

    assert snapshot.gap_count == 1
    assert snapshot.start_sample_index == 5
    assert snapshot.end_sample_index == 7
    assert snapshot.overrun_samples == 3
    assert destination[0, :2].tolist() == [1.0, 0.0]


def test_laser_plot_process_bounds_raw_daq_history_outside_qt():
    stream_channel = NidaqSignalChannelConfiguration(
        name="laser1_diode",
        physical_channel="Dev1/ai0",
        kind="analog",
        unit="V",
    )
    configuration = NidaqSignalStreamConfiguration(
        channels=(stream_channel,),
        is_enabled=True,
        sample_rate_hz=1000.0,
        rolling_window_seconds=10.0,
    )
    ring = SharedNidaqSampleRing(configuration)
    process = LaserPlotProcess(ring)
    destinations_x = {
        (1, curve_name): np.empty(process.MAX_POINTS, dtype=np.float32)
        for curve_name in process.CURVE_NAMES
    }
    destinations_y = {
        (1, curve_name): np.empty(process.MAX_POINTS, dtype=np.float32)
        for curve_name in process.CURVE_NAMES
    }
    try:
        process.configure_channel(
            1,
            window_seconds=10.0,
            pixel_width=300,
            diode_name="laser1_diode",
            copy_name=None,
        )
        process.set_streaming(1, True)
        ring.write_block(NidaqSignalSampleBlock(
            wall_time=1.0,
            perf_time=1.0,
            sample_rate_hz=1000.0,
            sample_index=0,
            channels=(stream_channel,),
            values={
                "laser1_diode": tuple(
                    float((index // 25) % 2) for index in range(10_000)
                ),
            },
        ))
        deadline = time.monotonic() + 5.0
        frame = None
        diode_slot = process.curve_slot(1, "diode")
        while time.monotonic() < deadline:
            candidate = process.copy_latest_into(destinations_x, destinations_y)
            if candidate is not None and candidate.point_counts[diode_slot]:
                frame = candidate
                break
            time.sleep(0.02)

        assert frame is not None
        assert process.pid != os.getpid()
        assert frame.point_counts[diode_slot] <= 300
        displayed = destinations_y[(1, "diode")][:frame.point_counts[diode_slot]]
        assert displayed.min() == 0.0
        assert displayed.max() == 1.0
    finally:
        process.close()


def test_numpy_stream_buffer_wraps_without_shifting_existing_window():
    buffer = RollingStreamBuffer(5)
    buffer.append((0, 1, 2), (10, 11, 12))
    buffer.append((3, 4, 5, 6), (13, 14, 15, 16))

    x_values, y_values = buffer.ordered()

    assert x_values.tolist() == [2, 3, 4, 5, 6]
    assert y_values.tolist() == [12, 13, 14, 15, 16]
    assert buffer.latest_x == 6
    reusable_y = np.empty(5)
    assert buffer.copy_ordered_y_into(reusable_y) == 5
    assert reusable_y.tolist() == [12, 13, 14, 15, 16]
    reusable_x = np.empty(5)
    assert buffer.copy_ordered_into(reusable_x, reusable_y) == 5
    assert reusable_x.tolist() == [2, 3, 4, 5, 6]
    assert reusable_y.tolist() == [12, 13, 14, 15, 16]


def test_numpy_stream_buffer_peak_reduction_is_bounded_and_preserves_ttl_pulses():
    buffer = RollingStreamBuffer(10_000)
    x_values = tuple(index / 10_000 for index in range(10_000))
    y_values = [0.0] * 10_000
    y_values[123] = 5.0
    y_values[8_765] = 5.0
    buffer.append(x_values, y_values)

    plot_x, plot_y = buffer.ordered_for_plot(400)

    assert len(plot_x) == len(plot_y) <= 400
    assert plot_y.min() == 0.0
    assert plot_y.max() == 5.0
    assert list(plot_x) == sorted(plot_x)


def test_numpy_stream_buffer_resize_keeps_newest_samples():
    buffer = RollingStreamBuffer(10)
    buffer.append(range(10), range(10))

    buffer.resize(4)

    x_values, y_values = buffer.ordered()
    assert x_values.tolist() == [6, 7, 8, 9]
    assert y_values.tolist() == [6, 7, 8, 9]


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
        assert content._live_button.isEnabled()
        assert not hasattr(content, "_graph_seconds")
        assert not hasattr(content, "_graph_min_volts")
        assert not hasattr(content, "_graph_max_volts")
        assert content._rolling_plot.minimumSize().isEmpty()
        assert content._rolling_plot._plot.minimumSize().isEmpty()
        assert content._rolling_plot._plot.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
        assert content._rolling_plot._plot.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Ignored

        view_box = content._rolling_plot._plot.getPlotItem().getViewBox()
        assert view_box.state["mouseEnabled"] == [True, False]
        assert view_box.viewRange()[0] == pytest.approx([-10.0, 0.0])
        assert view_box.viewRange()[1] == pytest.approx([-0.2, 1.2])
        assert view_box.state["limits"]["xLimits"] == [-10.0, 0.0]
        assert view_box.state["limits"]["yLimits"] == [-0.2, 1.2]
        assert view_box.state["limits"]["xRange"][1] == 10.0
        view_box.setXRange(-5.0, 0.0, padding=0)
        assert view_box.viewRange()[0] == pytest.approx([-5.0, 0.0])
        assert view_box.viewRange()[1] == pytest.approx([-0.2, 1.2])
        view_box.setXRange(-20.0, 0.0, padding=0)
        assert view_box.viewRange()[0] == pytest.approx([-10.0, 0.0])
        view_box.setXRange(-8.0, -3.0, padding=0)
        content._live_button.click()
        assert view_box.viewRange()[0] == pytest.approx([-5.0, 0.0])
        assert view_box.viewRange()[1] == pytest.approx([-0.2, 1.2])
        assert content._plot_process.pid != os.getpid()

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
        assert not content._live_button.isEnabled()
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


def test_analysis_prepares_sample_blocks_in_dedicated_process(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = dataclasses.replace(
        _stream_configuration(), sample_rate_hz=1000.0,
    )
    monitor._hardware_enabled = True
    content = AnalysisContent(_AnalysisAppStub(monitor))
    content.show()
    qapp.processEvents()
    try:
        monitor.sample_ring.write_block(
            NidaqSignalSampleBlock(
                wall_time=1.0,
                perf_time=1.0,
                sample_rate_hz=1000.0,
                sample_index=0,
                channels=monitor.configuration.channels,
                values={"cam_frames": tuple(float(index % 2) for index in range(300))},
            )
        )

        snapshot = _wait_for_plot_snapshot(content)
        assert content._plot_process.pid != os.getpid()
        assert not hasattr(content._plot_process, "_snapshot_queue")
        assert not hasattr(content._plot_process, "_block_queue")
        assert "_sample_queue" not in vars(monitor)
        displayed = content._rolling_plot._display_y["cam_frames"][:snapshot.point_counts[0]]
        assert np.count_nonzero(np.isfinite(displayed)) == 600
        assert snapshot.latest_x == pytest.approx(0.299)
        display_array_id = id(content._rolling_plot._display_y["cam_frames"])
        monitor.sample_ring.write_block(
            NidaqSignalSampleBlock(
                wall_time=1.1,
                perf_time=1.1,
                sample_rate_hz=1000.0,
                sample_index=300,
                channels=monitor.configuration.channels,
                values={"cam_frames": (0.0, 1.0)},
            )
        )
        _wait_for_plot_snapshot(content)
        assert id(content._rolling_plot._display_y["cam_frames"]) == display_array_id

        monitor.sample_ring.reset()
        monitor.sample_ring.write_block(
            NidaqSignalSampleBlock(
                wall_time=2.0,
                perf_time=2.0,
                sample_rate_hz=1000.0,
                sample_index=0,
                channels=monitor.configuration.channels,
                values={"cam_frames": (1.0, 1.0)},
            )
        )
        restarted = _wait_for_plot_snapshot(content)
        restarted_values = content._rolling_plot._display_y["cam_frames"][
            :restarted.point_counts[0]
        ]
        assert restarted.latest_x == pytest.approx(0.001)
        assert restarted_values.tolist() == [1.0, 1.0]
    finally:
        content.on_close()
        content.deleteLater()


def test_analysis_redraw_sends_timestamped_square_steps_to_qt(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    content = AnalysisContent(_AnalysisAppStub(monitor))
    try:
        values = [0.0] * 100_000
        values[50_000] = 1.0
        monitor.sample_ring.write_block(
            NidaqSignalSampleBlock(
                wall_time=1.0,
                perf_time=1.0,
                sample_rate_hz=10_000.0,
                sample_index=0,
                channels=monitor.configuration.channels,
                values={"cam_frames": tuple(values)},
            )
        )
        content.show()
        qapp.processEvents()
        _wait_for_plot_snapshot(content)

        curve = content._rolling_plot._curves["cam_frames"]
        assert len(curve.xData) == 6
        assert curve.yData.min() == 0.0
        assert curve.yData.max() == 1.0
        assert curve.xData[1] == curve.xData[2]
        assert curve.xData[3] == curve.xData[4]
        changing_segments = np.flatnonzero(np.diff(curve.yData) != 0)
        assert np.all(np.diff(curve.xData)[changing_segments] == 0)
    finally:
        content.on_close()
        content.deleteLater()


def test_analysis_does_not_stretch_partial_history_across_full_window(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    content = AnalysisContent(_AnalysisAppStub(monitor))
    try:
        monitor.sample_ring.write_block(
            NidaqSignalSampleBlock(
                wall_time=1.0,
                perf_time=1.0,
                sample_rate_hz=10_000.0,
                sample_index=0,
                channels=monitor.configuration.channels,
                values={"cam_frames": tuple(float((index // 100) % 2) for index in range(1000))},
            )
        )
        content.show()
        qapp.processEvents()
        _wait_for_plot_snapshot(content)

        curve = content._rolling_plot._curves["cam_frames"]
        assert curve.xData[0] == pytest.approx(-0.0999, abs=1e-5)
        assert curve.xData[-1] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(curve.xData) >= 0)
    finally:
        content.on_close()
        content.deleteLater()


def test_analysis_uses_active_screen_refresh_rate_for_timer_and_chunk(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    content = AnalysisContent(_AnalysisAppStub(monitor))
    try:
        content._screen_changed(SimpleNamespace(refreshRate=lambda: 120.0))

        assert content._display_refresh_rate_hz == 120.0
        assert content._plot_timer.interval() == 8
        assert monitor.effective_read_chunk_size == 83
    finally:
        content.on_close()
        content.deleteLater()


def test_signal_monitor_has_no_analysis_stream_persistence(tmp_path):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = dataclasses.replace(
        _stream_configuration(),
        record_to_acquisition=True,
    )
    monitor.project = SimpleNamespace(
        get_source_path=lambda _name: (_ for _ in ()).throw(
            AssertionError("visualization stream must not request an output path")
        )
    )

    assert "_open_recording_file" not in vars(type(monitor))
    assert list(tmp_path.iterdir()) == []


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


def test_signal_monitor_worker_publishes_directly_to_shared_ring():
    monitor = NidaqSignalMonitorModel(worker_target=_shared_ring_signal_worker)
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    try:
        assert monitor.start()
        deadline = time.monotonic() + 5.0
        while not monitor.is_running and time.monotonic() < deadline:
            time.sleep(0.01)

        destination = np.empty((1, monitor.sample_ring.capacity), dtype=np.float32)
        snapshot = monitor.sample_ring.copy_since(None, destination)

        assert monitor.is_running
        assert snapshot.sample_count == 4
        assert snapshot.generation == 1
        assert destination[0, :4].tolist() == [0.0, 1.0, 0.0, 1.0]
    finally:
        monitor.close()


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
    content = LaserControlContent(app_model)
    tab = content._channel_tabs[0]
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
        _wait_for_laser_plot(content, minimum_points=5)

        command_x, command_y = tab._trace_data["command"]
        diode_x, diode_y = tab._trace_data["diode"]
        assert tab._trace_streaming
        assert tab._trace_toggle_button.text() == "Stop Stream"
        assert tuple(
            tab._mode_tabs.tabText(index)
            for index in range(tab._mode_tabs.count())
        ) == ("Pulse", "Calibration", "Output")
        assert tab._trace_signal_checkboxes["diode"].property("signalColor") == "#128a43"
        assert tab._trace_signal_checkboxes["copy"].property("signalColor") == "#d66b00"
        assert tuple(
            tab._trace_tabs.tabText(index)
            for index in range(tab._trace_tabs.count())
        ) == ("Stream", "Signals")
        assert tab._trace_stream_page.isAncestorOf(tab._trace_plot)
        assert tab._trace_signals_page.isAncestorOf(tab._trace_signal_checkboxes["diode"])
        assert tab._trace_signals_page.isAncestorOf(tab._trace_signal_checkboxes["copy"])
        assert tab._trace_signal_checkboxes["diode"].text().endswith("Diode feedback")
        assert "Dev1/ai0" not in tab._trace_signal_checkboxes["diode"].text()
        assert tab._trace_signal_checkboxes["diode"].toolTip().startswith("Dev1/ai0\n")
        assert tab._trace_legend._columns == 1
        assert tab._preview_plot.minimumSize().isEmpty()
        assert tab._trace_plot.minimumSize().isEmpty()
        assert tab._preview_plot.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Ignored
        assert tab._trace_plot.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Ignored
        assert tab.minimumSizeHint().width() < 430
        assert tuple(entry[0] for entry in tab._trace_legend.entries) == (
            "Command output",
            "Diode feedback",
            "Command copy",
        )
        assert all(
            curve.opts["pen"].style().name == "SolidLine"
            for curve in tab._trace_curves.values()
        )
        assert tab._trace_seconds.suffix() == " s"
        assert tab._trace_min_volts.suffix() == " V"
        assert tab._trace_max_volts.suffix() == " V"
        assert tab._trace_plot.getPlotItem().getViewBox().viewRange()[0] == pytest.approx(
            [-10.0, 0.0]
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
        deadline = time.monotonic() + 5.0
        while len(tab._trace_data["command"][0]) <= 5 and time.monotonic() < deadline:
            content._flush_laser_plots()
            tab.redraw_trace()
            time.sleep(0.02)
        assert "internal pulse" in tab._trace_status.text()
        assert len(tab._trace_data["command"][0]) > 5
    finally:
        content.on_close()
        content.deleteLater()
        app_model.nidaq_signal_monitor.close()
        laser.close()


def test_laser_control_reads_nidaq_samples_directly_from_shared_ring(qapp):
    channel = _laser_channel()
    configuration = LaserSystemConfiguration.from_channels(
        (channel,), backend="null", sample_rate_hz=1000.0,
    )
    laser = LaserModel(NullLaserController(configuration))
    app_model = _LaserAppStub(laser)
    stream_channel = NidaqSignalChannelConfiguration(
        name="laser1_diode",
        physical_channel="Dev1/ai0",
        kind="analog",
        unit="V",
    )
    app_model.nidaq_signal_monitor._configuration = NidaqSignalStreamConfiguration(
        channels=(stream_channel,),
        is_enabled=True,
        sample_rate_hz=1000.0,
    )
    content = LaserControlContent(app_model)
    try:
        tab = content._channel_tabs[0]
        tab._set_trace_streaming(True)
        app_model.nidaq_signal_monitor.sample_ring.write_block(
            NidaqSignalSampleBlock(
                wall_time=1.0,
                perf_time=1.0,
                sample_rate_hz=1000.0,
                sample_index=0,
                channels=(stream_channel,),
                values={"laser1_diode": (0.25, 0.5, 0.75)},
            )
        )

        _wait_for_laser_plot(content, minimum_points=3)

        diode_x, diode_y = tab._trace_data["diode"]
        assert diode_x == pytest.approx([-0.002, -0.001, 0.0])
        assert diode_y == pytest.approx([0.25, 0.5, 0.75])
        assert content._plot_process.pid != os.getpid()
        assert not hasattr(content._plot_process, "_block_queue")
        assert "_trace_buffers" not in vars(tab)
        assert tab._last_plot_frame.raw_generation == 1
        assert "Seq 1" in content._stream_telemetry_label.text()
        display_array_id = id(tab._trace_display_y["diode"])

        app_model.nidaq_signal_monitor.sample_ring.write_block(
            NidaqSignalSampleBlock(
                wall_time=1.1,
                perf_time=1.1,
                sample_rate_hz=1000.0,
                sample_index=3,
                channels=(stream_channel,),
                values={"laser1_diode": (1.0, 1.25)},
            )
        )
        deadline = time.monotonic() + 5.0
        while (
            tab._last_plot_frame.raw_generation < 2
            and time.monotonic() < deadline
        ):
            content._flush_laser_plots()
            time.sleep(0.02)
        assert id(tab._trace_display_y["diode"]) == display_array_id
        assert tab._last_plot_frame.raw_generation == 2

        app_model.nidaq_signal_monitor.set_display_refresh_rate(120.0)
        content._flush_laser_plots()
        assert content._stream_plot_timer.interval() == 8
    finally:
        content.on_close()
        content.deleteLater()
        app_model.nidaq_signal_monitor.close()
        laser.close()


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
