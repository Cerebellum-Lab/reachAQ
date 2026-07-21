import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import ObservableObject  # noqa: E402
from autotrainer.core.capture import CaptureProcessStatus  # noqa: E402
from autotrainer.video import CaptureCameraAttrs  # noqa: E402
from tools.acquisition.model.hardware_scan import HardwareScanEntry  # noqa: E402
from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts  # noqa: E402
from tools.acquisition.view.hardware_status_content import HardwareStatusContent  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _ObservableStub(ObservableObject):
    def __init__(self, **values):
        super().__init__()
        for name, value in values.items():
            setattr(self, name, value)


class _AppModelStub(ObservableObject):
    def __init__(self):
        super().__init__(("configuration_loaded_event",))
        self.message_handler = _ObservableStub()
        self.cameras = (
            _ObservableStub(
                name="left",
                is_enabled=True,
                camera_source=CaptureCameraAttrs("Spinnaker 111", "spinnaker://111"),
                capture_process_status=CaptureProcessStatus.UNKNOWN,
            ),
            _ObservableStub(
                name="right",
                is_enabled=False,
                camera_source=CaptureCameraAttrs("Spinnaker 222", "spinnaker://222"),
                capture_process_status=CaptureProcessStatus.UNKNOWN,
            ),
        )
        self.hardware = _ObservableStub(
            nidaq_enabled=False,
            can_enabled=False,
            pellet_controller_enabled=False,
            connected=False,
        )
        self.nidaq_signal_monitor = _ObservableStub(
            configuration=SimpleNamespace(channels=tuple()),
            is_starting=False,
            is_running=False,
        )
        self.laser = _ObservableStub(
            configuration=SimpleNamespace(backend="disabled", channels=tuple()),
            is_connected=False,
        )
        self.nidaq_ports = SimpleNamespace(device_name="DevOutputs")
        self.configured_nidaq_device_names = ("DevOutputs", "DevInputs")
        self.hardware_scan_results = {
            "cameras": HardwareScanEntry(
                "2 source(s) discovered: Spinnaker 111, Spinnaker 222",
                "ok",
            ),
            "nidaq": HardwareScanEntry(
                "2 device(s) discovered: DevInputs, DevOutputs",
                "ok",
            ),
            "can": HardwareScanEntry("CAN/pellet not in use", "disabled"),
            "pellet": HardwareScanEntry("CAN/pellet not in use", "disabled"),
            "laser": HardwareScanEntry("not probed; backend disabled", "disabled"),
        }


def test_status_panel_columns_and_scan_results(qapp):
    content = HardwareStatusContent(_AppModelStub())
    try:
        assert tuple(content._header_labels) == ("enabled", "devices", "info")
        assert content._enabled_labels["nidaq"].text() == "No"
        assert content._device_labels["nidaq"].text() == "NI-DAQ"

        nidaq_info = content._info_labels["nidaq"].text()
        assert "2 device(s) discovered: DevInputs, DevOutputs" in nidaq_info
        assert "configured: DevOutputs, DevInputs" in nidaq_info
        assert nidaq_info.endswith("disabled")

        camera_info = content._info_labels["cameras"].text()
        assert "2 source(s) discovered" in camera_info
        assert "left=Spinnaker 111 idle" in camera_info
    finally:
        content._timer.stop()
        content.deleteLater()


def test_hardware_refresh_publishes_discovered_devices(app_model, monkeypatch):
    from tools.acquisition.model import app_model as app_model_module

    camera_sources = (
        CaptureCameraAttrs("Random Image", "random://0?width=300&height=200"),
        CaptureCameraAttrs("Spinnaker 111", "spinnaker://111"),
    )
    nidaq_devices = (
        NidaqDevicePorts(name="DevOutputs", analog_outputs=("DevOutputs/ao0",)),
        NidaqDevicePorts(name="DevInputs", analog_inputs=("DevInputs/ai0",)),
    )
    monkeypatch.setattr(app_model_module, "create_camera_list", lambda **_kwargs: camera_sources)
    monkeypatch.setattr(app_model_module, "discover_nidaq_devices", lambda: (nidaq_devices, None))
    monkeypatch.setattr(
        app_model,
        "_scan_can_pellet_hardware",
        lambda _warnings: "CAN/pellet found 123",
    )

    app_model.refresh_hardware_bindings()

    results = app_model.hardware_scan_results
    assert set(results) == {"cameras", "nidaq", "can", "pellet", "laser"}
    assert "Spinnaker 111" in results["cameras"].info
    assert results["cameras"].state == "ok"
    assert "DevInputs, DevOutputs" in results["nidaq"].info
    assert results["nidaq"].state == "ok"
    assert results["can"] == HardwareScanEntry("CAN/pellet found 123", "ok")
