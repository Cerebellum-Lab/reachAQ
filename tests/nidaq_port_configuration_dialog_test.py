import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import NidaqPortConfiguration, SystemConfiguration  # noqa: E402
from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts  # noqa: E402
from tools.acquisition.view.nidaq_port_configuration_dialog import NidaqPortConfigurationDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _combo_values(combo):
    return tuple(combo.itemData(index) for index in range(combo.count()))


def _set_combo_value(combo, value):
    index = combo.findData(value)
    assert index >= 0, f"{value!r} not in {_combo_values(combo)!r}"
    combo.setCurrentIndex(index)


def test_selected_daq_channel_is_removed_from_other_roles(qapp):
    device = NidaqDevicePorts(
        name="Dev1",
        digital_outputs=("Dev1/port0/line0", "Dev1/port0/line1"),
        digital_inputs=("Dev1/port0/line0", "Dev1/port0/line1"),
    )
    dialog = NidaqPortConfigurationDialog(SystemConfiguration(), devices=(device,))

    tone1 = dialog._general_combos["tone1"]
    tone2 = dialog._general_combos["tone2"]
    cam_frames = dialog._general_combos["cam_frames"]

    _set_combo_value(tone1, "Dev1/port0/line0")

    assert "Dev1/port0/line0" in _combo_values(tone1)
    assert "Dev1/port0/line0" not in _combo_values(tone2)
    assert "Dev1/port0/line0" not in _combo_values(cam_frames)


def test_unsupported_channel_kind_combo_is_disabled(qapp):
    device = NidaqDevicePorts(
        name="PXI1Slot4",
        analog_outputs=("PXI1Slot4/ao0",),
        analog_inputs=(),
        digital_outputs=("PXI1Slot4/port0/line0",),
        digital_inputs=("PXI1Slot4/port0/line0",),
    )
    dialog = NidaqPortConfigurationDialog(SystemConfiguration(), devices=(device,))

    diode = dialog._laser_combos[1]["diode"]
    laser_copy = dialog._laser_combos[1]["laser_copy"]

    assert not diode.isEnabled()
    assert not laser_copy.isEnabled()
    assert _combo_values(diode) == (None,)
    assert "AI 0" in dialog._status_label.text()
    assert "Unsupported channel type(s)" in dialog._status_label.text()


def test_duplicate_daq_channel_assignments_are_rejected(qapp):
    config = SystemConfiguration()
    config.nidaq_ports = NidaqPortConfiguration(
        tone1="Dev1/port0/line0",
        tone2="Dev1/port0/line0",
    )
    device = NidaqDevicePorts(
        name="Dev1",
        digital_outputs=("Dev1/port0/line0", "Dev1/port0/line1"),
    )
    dialog = NidaqPortConfigurationDialog(config, devices=(device,))

    with pytest.raises(ValueError, match="Duplicate channel assignment"):
        dialog._validate_selected_channel_assignments(device)
