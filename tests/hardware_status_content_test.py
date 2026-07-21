import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import CameraId, ObservableObject  # noqa: E402
from autotrainer.core.capture import CaptureProcessStatus  # noqa: E402
from autotrainer.video import CaptureCameraAttrs  # noqa: E402
from tools.acquisition.model.hardware_scan import HardwareScanEntry, scan_can_adapters, scan_gpus  # noqa: E402
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
                camera_id=CameraId.Left,
                is_enabled=True,
                camera_source=CaptureCameraAttrs("Spinnaker 111", "spinnaker://111"),
                capture_process_status=CaptureProcessStatus.UNKNOWN,
            ),
            _ObservableStub(
                name="right",
                camera_id=CameraId.Right,
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
            "can": HardwareScanEntry("PEAK PCIe adapter present; driver peak_pciefd; can0 UP", "ok"),
            "pellet": HardwareScanEntry("Pellet controller not in use", "disabled"),
            "gpu": HardwareScanEntry("✓ 1 NVIDIA GPU\n→ GPU0 Test GPU · 4096 MiB · drv 1.0", "ok"),
            "laser": HardwareScanEntry("not probed; backend disabled", "disabled"),
        }


def test_status_panel_columns_and_scan_results(qapp):
    content = HardwareStatusContent(_AppModelStub())
    try:
        assert tuple(content._header_labels) == ("enabled", "devices", "info")
        assert content._enabled_labels["nidaq"].text() == "No"
        assert content._device_labels["nidaq"].text() == "NI-DAQ"
        assert content._device_labels["can"].text() == "CAN Adapter"
        assert "PEAK PCIe adapter present" in content._info_labels["can"].text()

        nidaq_info = content._info_labels["nidaq"].text()
        assert "2 device(s) discovered: DevInputs, DevOutputs" in nidaq_info
        assert "selected: DevOutputs, DevInputs" in nidaq_info
        assert nidaq_info.endswith("stream: disabled")

        camera_info = content._info_labels["cameras"].text()
        assert "2 source(s) discovered" in camera_info
        assert "left: Spinnaker 111 · idle" in camera_info
        assert "GPU0 Test GPU" in content._info_labels["gpu"].text()
    finally:
        content.deleteLater()


def test_hardware_refresh_publishes_discovered_devices(app_model, monkeypatch):
    from tools.acquisition.model import app_model as app_model_module

    camera_sources = (
        CaptureCameraAttrs("Random Image", "random://0?width=300&height=200"),
        CaptureCameraAttrs("Spinnaker 111", "spinnaker://111"),
    )
    nidaq_devices = (
        NidaqDevicePorts(
            name="DevOutputs",
            product_type="PXI-6713",
            product_number=11136,
            analog_outputs=("DevOutputs/ao0",),
        ),
        NidaqDevicePorts(
            name="DevInputs",
            product_type="PXI-6221",
            product_number=28853,
            analog_inputs=("DevInputs/ai0",),
        ),
    )
    monkeypatch.setattr(app_model_module, "create_camera_list", lambda **_kwargs: camera_sources)
    monkeypatch.setattr(app_model_module, "discover_nidaq_devices", lambda: (nidaq_devices, None))
    monkeypatch.setattr(
        app_model_module,
        "scan_can_adapters",
        lambda **_kwargs: HardwareScanEntry(
            "✓ PEAK PCIe adapter\n→ can0 ↑ · selected\n  ↳ app: socketcan → can0",
            "ok",
        ),
    )
    monkeypatch.setattr(
        app_model_module,
        "scan_gpus",
        lambda: HardwareScanEntry("✓ 1 NVIDIA GPU\n→ GPU0 Test GPU", "ok"),
    )

    app_model.refresh_hardware_bindings()

    results = app_model.hardware_scan_results
    assert set(results) == {"cameras", "nidaq", "can", "pellet", "gpu", "laser"}
    assert "Spinnaker 111" in results["cameras"].info
    assert results["cameras"].state == "ok"
    assert "DevInputs" in results["nidaq"].info
    assert "DevOutputs" in results["nidaq"].info
    assert "DevOutputs · PXI-6713 · #11136" in results["nidaq"].info
    assert "DevInputs · PXI-6221 · #28853" in results["nidaq"].info
    assert results["nidaq"].state == "ok"
    assert "can0 ↑ · selected" in results["can"].info
    assert "connection starts with acquisition" in results["pellet"].info
    assert results["gpu"].state == "ok"


def test_acquisition_state_changes_do_not_replace_scan_snapshot(qapp):
    app_model = _AppModelStub()
    app_model.hardware.can_enabled = True
    content = HardwareStatusContent(app_model)
    try:
        original = content._info_labels["can"].text()
        app_model.hardware.connected = True
        app_model.hardware.property_changed("connected", True, False)
        qapp.processEvents()
        assert content._info_labels["can"].text() == original
    finally:
        content.deleteLater()


def test_can_adapter_scan_reports_peak_driver_and_interface(tmp_path):
    net_root = tmp_path / "net"
    pci_root = tmp_path / "pci"
    can0 = net_root / "can0"
    pci_device = pci_root / "0000:03:00.0"
    driver = tmp_path / "drivers" / "peak_pciefd"
    can0.mkdir(parents=True)
    pci_device.mkdir(parents=True)
    driver.mkdir(parents=True)
    (can0 / "flags").write_text("0x10043\n")
    (pci_device / "vendor").write_text("0x001c\n")
    (can0 / "device").symlink_to(pci_device, target_is_directory=True)
    (pci_device / "driver").symlink_to(driver, target_is_directory=True)

    result = scan_can_adapters(
        selected_interface="can0",
        selected_backend="socketcan",
        net_root=net_root,
        pci_root=pci_root,
    )

    assert result.state == "ok"
    assert "PEAK PCIe adapter" in result.info
    assert "peak_pciefd" in result.info
    assert "can0 ↑ · selected" in result.info
    assert "app: socketcan → can0" in result.info


def test_gpu_scan_formats_compact_identity(monkeypatch):
    from tools.acquisition.model import hardware_scan

    monkeypatch.setattr(
        hardware_scan.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="0, NVIDIA T1000, 595.71.05, 4096\n",
            stderr="",
        ),
    )

    result = scan_gpus()

    assert result.state == "ok"
    assert result.info == "✓ 1 NVIDIA GPU\n→ GPU0 NVIDIA T1000 · 4096 MiB · drv 595.71.05"
