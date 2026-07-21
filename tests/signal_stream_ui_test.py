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
from autotrainer.device import LaserCalibrationRamp, LaserPulseTrain, NullLaserController  # noqa: E402
from tools.acquisition.model.laser_model import LaserModel  # noqa: E402
from tools.acquisition.model.app_model import AppModel  # noqa: E402
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel  # noqa: E402
from tools.acquisition.view.analysis_content import AnalysisContent  # noqa: E402
from tools.acquisition.view.laser_control_content import _LaserChannelTab  # noqa: E402


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
        self.nidaq_ports = NidaqPortConfiguration(cam_frames="Dev1/port0/line0")
        self.laser = LaserModel()
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


def test_analysis_signal_selection_requires_mapped_port_and_nidaq_enable(qapp):
    monitor = NidaqSignalMonitorModel()
    monitor._configuration = _stream_configuration()
    monitor._hardware_enabled = True
    app_model = _AnalysisAppStub(monitor)
    content = AnalysisContent(app_model)
    try:
        camera_frames = content._signal_checkboxes["cam_frames"]
        barcode = content._signal_checkboxes["barcode"]
        assert camera_frames.isEnabled()
        assert camera_frames.isChecked()
        assert not barcode.isEnabled()
        assert not barcode.isChecked()
        assert content._channel_count_label.text() == "1"
        assert content._start_stop_button.isEnabled()
        assert content._clear_button.isEnabled()

        camera_frames.setChecked(False)
        qapp.processEvents()
        assert monitor.configuration.channels == tuple()
        assert not content._start_stop_button.isEnabled()
        assert app_model.signal_configuration_save_count == 1

        content._signal_checkboxes["cam_frames"].setChecked(True)
        qapp.processEvents()
        assert tuple(
            channel.physical_channel
            for channel in monitor.configuration.channels
        ) == ("Dev1/port0/line0",)
        assert content._start_stop_button.isEnabled()
        assert app_model.signal_configuration_save_count == 2

        monitor.set_hardware_enabled(False)
        qapp.processEvents()

        assert not content._start_stop_button.isEnabled()
        assert not content._clear_button.isEnabled()
        assert not content._signal_checkboxes["cam_frames"].isEnabled()
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
    tab = _LaserChannelTab(
        type("AppStub", (), {"laser": laser})(),
        channel,
        True,
        configuration.sample_rate_hz,
        lambda _status, operation: operation(),
        lambda _message, _is_error: None,
    )
    laser.trace_received += tab.append_trace
    try:
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
        assert tab._trace_toggle_button.text() == "Pause Stream"
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
        laser.close()
        tab.deleteLater()
