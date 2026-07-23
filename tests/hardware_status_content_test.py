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
            _ObservableStub(
                name="stimCam",
                camera_id=CameraId.Camera3,
                is_enabled=False,
                camera_source=CaptureCameraAttrs("stimCam", "spinnaker://333"),
                capture_process_status=CaptureProcessStatus.UNKNOWN,
            ),
            _ObservableStub(
                name="web",
                camera_id=CameraId.Web,
                is_enabled=True,
                camera_source=CaptureCameraAttrs("USB Camera 0", "opencv://0"),
                capture_process_status=CaptureProcessStatus.UNKNOWN,
            ),
        )
        self.hardware = _ObservableStub(
            nidaq_enabled=False,
            can_enabled=False,
            pellet_controller_enabled=False,
            connected=False,
            pellet_version="",
        )
        self.nidaq_signal_monitor = _ObservableStub(
            configuration=SimpleNamespace(
                channels=(
                    SimpleNamespace(
                        name="cam_frames",
                        physical_channel="DevInputs/port0/line2",
                        kind="digital",
                    ),
                )
            ),
            is_starting=False,
            is_running=False,
        )
        self.laser = _ObservableStub(
            configuration=SimpleNamespace(backend="disabled", channels=tuple()),
            is_connected=False,
        )
        self.nidaq_ports = SimpleNamespace(
            device_name="DevOutputs",
            tone1="DevOutputs/port0/line0",
        )
        self.configured_nidaq_device_names = ("DevOutputs", "DevInputs")
        self.hardware_scan_results = {
            "cameras": HardwareScanEntry(
                "✓ 4 camera source(s)\n"
                "→ Spinnaker 111\n"
                "→ Spinnaker 222\n"
                "→ Spinnaker 333\n"
                "→ Random Image",
                "ok",
            ),
            "nidaq": HardwareScanEntry(
                "✓ 2 NI-DAQ card(s)\n"
                "→ DevInputs · PXI-6221 · #28853\n"
                "→ DevOutputs · PXI-6713 · #11136",
                "ok",
            ),
            "can": HardwareScanEntry(
                "✓ PEAK PCIe adapter · peak_pciefd\n"
                "→ can0 ↑ · selected\n"
                "  ↳ app: socketcan → can0",
                "ok",
            ),
            "pellet": HardwareScanEntry("Pellet controller not in use", "disabled"),
            "gpu": HardwareScanEntry("✓ 1 NVIDIA GPU\n→ GPU0 Test GPU · 4096 MiB · drv 1.0", "ok"),
            "laser": HardwareScanEntry("not probed; backend disabled", "disabled"),
        }


def test_status_panel_columns_and_scan_results(qapp):
    content = HardwareStatusContent(_AppModelStub())
    try:
        content.resize(480, 420)
        content.show()
        qapp.processEvents()
        assert tuple(content._category_panels) == (
            "cameras",
            "nidaq",
            "can",
            "pellet",
            "gpu",
            "laser",
        )
        assert all(not panel.is_expanded for panel in content._category_panels.values())
        assert all(label.isHidden() for label in content._info_labels.values())
        assert all(panel.details_scroll.isHidden() for panel in content._category_panels.values())
        assert content._enabled_labels["nidaq"].text() == "Disabled"
        assert content._device_labels["nidaq"].text() == "NI-DAQ"
        assert content._device_labels["can"].text() == "CAN Adapter"
        can_info = content._category_panels["can"].details_text
        assert "PEAK PCIe adapter | peak_pciefd" in can_info
        assert "| connected" in can_info
        assert "can0" in can_info and "| selected" in can_info and "| up" in can_info
        assert "#f0f2f4" in content._category_panels["can"].header.styleSheet()
        assert "#e8f5ec" in content._category_panels["cameras"].header.styleSheet()

        nidaq_panel = content._category_panels["nidaq"]
        nidaq_info = nidaq_panel.details_text
        nidaq_lines = nidaq_info.splitlines()
        pipe_positions = tuple(index for index, char in enumerate(nidaq_lines[0]) if char == "|")
        assert pipe_positions
        assert all(
            tuple(index for index, char in enumerate(line) if char == "|") == pipe_positions
            for line in nidaq_lines[1:]
            if not line.startswith("!")
        )
        assert "<b><u>" in content._info_labels["nidaq"].text()
        assert "DevInputs" in nidaq_info and "PXI-6221 · #28853" in nidaq_info
        assert "DevOutputs" in nidaq_info and "PXI-6713 · #11136" in nidaq_info
        assert "input stream" in nidaq_info and "DevOutputs, DevInputs" in nidaq_info
        assert "tone1" in nidaq_info and "DevOutputs/port0/line0" in nidaq_info
        assert "cam_frames" in nidaq_info and "digital stream" in nidaq_info

        camera_info = content._category_panels["cameras"].details_text
        assert "left" in camera_info and "111" in camera_info and "idle" in camera_info
        assert "right" in camera_info and "222" in camera_info and "disabled" in camera_info
        assert "stimCam" in camera_info and "333" in camera_info
        assert "web" not in camera_info
        assert "Random Image" not in camera_info
        gpu_info = content._category_panels["gpu"].details_text
        assert "GPU0" in gpu_info and "Test GPU · 4096 MiB · drv 1.0" in gpu_info
        pellet_info = content._category_panels["pellet"].details_text
        assert "firmware" in pellet_info and "unknown" in pellet_info

        content._device_labels["can"].click()
        qapp.processEvents()
        assert content._category_panels["can"].is_expanded
        assert not content._info_labels["can"].isHidden()
        assert not content._category_panels["can"].details_scroll.isHidden()
        assert content._category_panels["can"].details_scroll.maximumHeight() == 170

        nidaq_panel.toggle_button.click()
        nidaq_panel.details_scroll.setFixedSize(240, 72)
        qapp.processEvents()
        assert nidaq_panel.details_scroll.horizontalScrollBar().maximum() > 0
        assert nidaq_panel.details_scroll.verticalScrollBar().maximum() > 0

        content._device_labels["can"].click()
        assert content._info_labels["can"].isHidden()
        assert content._category_panels["can"].details_scroll.isHidden()
    finally:
        content.deleteLater()


def test_pellet_firmware_version_updates_hardware_status(qapp):
    app_model = _AppModelStub()
    app_model.hardware.pellet_controller_enabled = True
    content = HardwareStatusContent(app_model)
    try:
        app_model.hardware.pellet_version = "1.2.5"
        app_model.hardware.property_changed("pellet_version", "1.2.5", "")
        qapp.processEvents()

        pellet_info = content._category_panels["pellet"].details_text
        assert "firmware" in pellet_info
        assert "Pellet: 1.2.5" in pellet_info
        assert "reported" in pellet_info
    finally:
        content.deleteLater()


def test_hardware_model_persists_reported_pellet_version(app_model):
    hardware = app_model.hardware

    hardware._message_handler_property_changed(
        app_model.message_handler.FIRMWARE_VERSION_PROPERTY,
        "Pellet: 1.2.5",
        None,
    )

    assert hardware.pellet_version == "1.2.5"


def test_enabled_category_uses_warning_color_from_scan(qapp):
    app_model = _AppModelStub()
    app_model.hardware.nidaq_enabled = True
    app_model.hardware_scan_results["nidaq"] = HardwareScanEntry("DAQ partially ready", "warning")
    content = HardwareStatusContent(app_model)
    try:
        assert content._enabled_labels["nidaq"].text() == "Enabled"
        assert "#fff4d6" in content._category_panels["nidaq"].header.styleSheet()
        assert content._info_labels["nidaq"].isHidden()
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
