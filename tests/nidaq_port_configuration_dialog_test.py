import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    NidaqDeviceIdentity,
    NidaqPortConfiguration,
    NidaqTimingConfiguration,
    SystemConfiguration,
)
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


def _set_device(dialog, device_name):
    index = dialog._device_combo.findData(device_name)
    assert index >= 0
    dialog._device_combo.setCurrentIndex(index)


def test_selected_daq_channel_is_removed_from_other_roles(qapp):
    device = NidaqDevicePorts(
        name="Dev1",
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
        digital_inputs=("Dev1/port0/line0", "Dev1/port0/line1"),
    )
    dialog = NidaqPortConfigurationDialog(config, devices=(device,))

    with pytest.raises(ValueError, match="Duplicate channel assignment"):
        dialog._validate_selected_channel_assignments(device)


def test_laser_assignments_remain_visible_when_switching_channel_source(qapp):
    output_device = NidaqDevicePorts(
        name="DevOutputs",
        analog_outputs=tuple(f"DevOutputs/ao{index}" for index in range(4)),
    )
    input_device = NidaqDevicePorts(
        name="DevInputs",
        analog_inputs=tuple(f"DevInputs/ai{index}" for index in range(4)),
    )
    dialog = NidaqPortConfigurationDialog(
        SystemConfiguration(),
        devices=(output_device, input_device),
    )

    _set_device(dialog, "DevOutputs")
    for laser_index in range(1, 5):
        _set_combo_value(
            dialog._laser_combos[laser_index]["laser_out"],
            f"DevOutputs/ao{laser_index - 1}",
        )

    _set_device(dialog, "DevInputs")
    for laser_index in range(1, 5):
        laser_out = dialog._laser_combos[laser_index]["laser_out"]
        expected_output = f"DevOutputs/ao{laser_index - 1}"
        assert laser_out.currentData() == expected_output
        assert expected_output in _combo_values(laser_out)

        _set_combo_value(
            dialog._laser_combos[laser_index]["laser_copy"],
            f"DevInputs/ai{laser_index - 1}",
        )

    assert "Assignments retained from other device(s): DevOutputs" in dialog._status_label.text()
    assert not dialog._unsupported_selected_channels(input_device)

    _set_device(dialog, "DevOutputs")
    for laser_index in range(1, 5):
        laser_copy = dialog._laser_combos[laser_index]["laser_copy"]
        expected_copy = f"DevInputs/ai{laser_index - 1}"
        assert laser_copy.currentData() == expected_copy
        assert expected_copy in _combo_values(laser_copy)


def test_timing_master_is_portable_identity_and_only_enabled_for_multi_device(qapp):
    config = SystemConfiguration()
    config.nidaq_ports = NidaqPortConfiguration(
        cam_frames="Acquire/port0/line0",
        tone1="Confirm/port0/line0",
        timing=NidaqTimingConfiguration(
            timing_master=NidaqDeviceIdentity(
                logical_name="acquisition",
                runtime_name="Acquire",
                product_type="InputModel",
                serial_number=100,
            ),
        ),
    )
    devices = (
        NidaqDevicePorts(
            name="Acquire",
            product_type="InputModel",
            serial_number=100,
            digital_inputs=("Acquire/port0/line0",),
            counter_outputs=("Acquire/ctr0",),
        ),
        NidaqDevicePorts(
            name="Confirm",
            product_type="OtherModel",
            serial_number=200,
            digital_inputs=("Confirm/port0/line0",),
            counter_outputs=("Confirm/ctr0",),
        ),
    )

    dialog = NidaqPortConfigurationDialog(config, devices=devices)

    assert dialog._timing_master_combo.isEnabled()
    selected = dialog._timing_master_combo.currentData()
    assert selected.serial_number == 100
    assert selected.runtime_name == "Acquire"
    built = dialog._build_timing_configuration()
    assert built.timing_master.serial_number == 100
    ports = dialog._build_nidaq_port_configuration("Acquire")
    assert {
        identity.serial_number
        for identity in ports.device_identities
    } == {100, 200}


def test_external_timing_mode_exposes_route_overrides(qapp):
    device = NidaqDevicePorts(
        name="Dev1",
        analog_inputs=("Dev1/ai0",),
    )
    dialog = NidaqPortConfigurationDialog(SystemConfiguration(), devices=(device,))

    dialog._sync_mode_combo.setCurrentIndex(
        dialog._sync_mode_combo.findData("external")
    )

    assert dialog._reference_clock_edit.isEnabled()
    assert dialog._start_trigger_edit.isEnabled()
    assert dialog._sample_clock_edit.isEnabled()
    assert not dialog._timing_master_combo.isEnabled()
