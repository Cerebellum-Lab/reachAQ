import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserSystemConfiguration,
    NidaqDeviceIdentity,
    NidaqPortConfiguration,
    NidaqTimingConfiguration,
    NidaqTimingRoute,
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


def test_unedited_advanced_timing_fields_survive_dialog_save(qapp):
    route = NidaqTimingRoute(
        "sample_clock", "/Dev1/PFI0", ("/Dev2/PFI0",),
    )
    config = SystemConfiguration()
    config.nidaq_ports = NidaqPortConfiguration(
        timing=NidaqTimingConfiguration(
            task_strategy="auto_multidevice",
            sample_clock_export_terminal="/Dev1/PFI1",
            start_trigger_export_terminal="/Dev1/PFI2",
            external_routes=(route,),
            transfer_mechanism_overrides=(("Dev1.di", "interrupt"),),
        )
    )
    device = NidaqDevicePorts(name="Dev1", analog_inputs=("Dev1/ai0",))

    dialog = NidaqPortConfigurationDialog(config, devices=(device,))
    built = dialog._build_timing_configuration()

    assert built.task_strategy == "auto_multidevice"
    assert built.sample_clock_export_terminal == "/Dev1/PFI1"
    assert built.start_trigger_export_terminal == "/Dev1/PFI2"
    assert built.external_routes == (route,)
    assert built.transfer_mechanism_overrides == (("Dev1.di", "interrupt"),)


def test_laser_trigger_selector_only_offers_external_trigger_terminals(qapp):
    device = NidaqDevicePorts(
        name="Dev1",
        terminals=(
            "/Dev1/PFI0",
            "/Dev1/RTSI0",
            "/Dev1/ai/SampleClock",
        ),
    )

    dialog = NidaqPortConfigurationDialog(SystemConfiguration(), devices=(device,))
    values = _combo_values(dialog._laser_combos[1]["trigger_listener"])

    assert "/Dev1/PFI0" in values
    assert "/Dev1/RTSI0" in values
    assert "/Dev1/ai/SampleClock" not in values


def test_a_contradictory_timing_configuration_cannot_be_built(qapp):
    """C5. Most combinations of these four settings mean nothing.

    They used to be expressible, and said nothing until a plan came back
    invalid several layers later for a reason nobody traced back here.
    """
    with pytest.raises(ValueError) as refused:
        NidaqTimingConfiguration(sync_mode="independent",
                                 require_hardware_synchronization=True)
    assert "Independent means each board runs" in str(refused.value)

    with pytest.raises(ValueError) as refused:
        NidaqTimingConfiguration(sync_mode="independent",
                                 require_hardware_synchronization=False,
                                 task_strategy="forced_multidevice")
    assert "opposite of independent" in str(refused.value)

    # The knob nothing ever read now says so instead of accepting silently.
    with pytest.raises(ValueError) as refused:
        NidaqTimingConfiguration(require_distinct_start_trigger=True)
    assert "not implemented" in str(refused.value)


def test_independent_boards_without_requiring_synchronization_is_allowed():
    """The combination that does mean something still does."""
    timing = NidaqTimingConfiguration(
        sync_mode="independent", require_hardware_synchronization=False)

    assert timing.sync_mode == "independent"
    assert timing.task_strategy == "per_device"


def test_saving_laser_ports_keeps_the_fields_the_dialog_does_not_edit(qapp):
    # The dialog rebuilt each channel field by field and dropped the rest, so
    # every save erased trigger_route_source - the PXI_Trig route a board STIM
    # trigger needs on christielab10 - and trigger_monitor_input, which the
    # dialog now edits as the trigger readback input.
    device = NidaqDevicePorts(
        name="Dev1",
        analog_outputs=("Dev1/ao0",),
        analog_inputs=("Dev1/ai0", "Dev1/ai1", "Dev1/ai2"),
        digital_outputs=("Dev1/port0/line4",),
    )
    config = SystemConfiguration()
    config.laser = LaserSystemConfiguration.from_channels((
        LaserChannelConfiguration(
            channel_id=1,
            analog_output="Dev1/ao0",
            diode_input="Dev1/ai0",
            shutter_output="Dev1/port0/line4",
            command_copy_input="Dev1/ai1",
            trigger_source="/Dev1/PXI_Trig0",
            trigger_route_source="/Dev1/PFI0",
            trigger_monitor_input="Dev1/ai2",
            board_stim_line=3,
            board_trigger_pulse_us=1500,
        ),
    ))
    dialog = NidaqPortConfigurationDialog(config, devices=(device,))

    channel = dialog._build_laser_configuration().get_channel(1)

    assert channel.trigger_source == "/Dev1/PXI_Trig0"
    assert channel.trigger_route_source == "/Dev1/PFI0"
    assert channel.trigger_monitor_input == "Dev1/ai2"
    assert channel.board_stim_line == 3
    assert channel.board_trigger_pulse_us == 1500


def _readback_device():
    return NidaqDevicePorts(
        name="Dev1",
        analog_outputs=("Dev1/ao0", "Dev1/ao1"),
        analog_inputs=("Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"),
        digital_outputs=("Dev1/port0/line4", "Dev1/port0/line5"),
        digital_inputs=(
            "Dev1/port0/line0", "Dev1/port0/line1",
            "Dev1/port0/line4", "Dev1/port0/line5",
        ),
        terminals=("/Dev1/PFI0", "/Dev1/PXI_Trig0"),
    )


def _readback_config(**overrides):
    values = dict(
        channel_id=1,
        analog_output="Dev1/ao0",
        diode_input="Dev1/ai0",
        shutter_output="Dev1/port0/line4",
        command_copy_input="Dev1/ai1",
    )
    pmt_shutter_output = overrides.pop("pmt_shutter_output", None)
    values.update(overrides)
    config = SystemConfiguration()
    config.laser = LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(**values),),
        pmt_shutter_output=pmt_shutter_output,
    )
    return config


def _ok_enabled(dialog):
    return dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_each_laser_has_a_trigger_readback_input_of_inputs_only(qapp):
    dialog = NidaqPortConfigurationDialog(
        SystemConfiguration(), devices=(_readback_device(),))

    for laser_index in range(1, 5):
        combo = dialog._laser_combos[laser_index]["trigger_readback"]
        values = _combo_values(combo)
        assert values[0] is None and combo.itemText(0) == "(none)"
        # Analog inputs and digital input lines; a PFI, an output or a
        # backplane line cannot be streamed.
        assert set(values[1:]) == {
            "Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3",
            "Dev1/port0/line0", "Dev1/port0/line1",
            "Dev1/port0/line4", "Dev1/port0/line5",
        }


def test_the_trigger_readback_input_is_loaded_and_saved(qapp):
    config = _readback_config(trigger_monitor_input="Dev1/ai2")
    dialog = NidaqPortConfigurationDialog(config, devices=(_readback_device(),))
    combo = dialog._laser_combos[1]["trigger_readback"]

    assert combo.currentData() == "Dev1/ai2"

    _set_combo_value(combo, "Dev1/port0/line1")
    assert dialog._build_laser_configuration().get_channel(1).trigger_monitor_input == (
        "Dev1/port0/line1")

    combo.setCurrentIndex(0)
    assert dialog._build_laser_configuration().get_channel(1).trigger_monitor_input is None


def test_a_trigger_readback_input_is_not_offered_to_another_input(qapp):
    config = _readback_config(trigger_monitor_input="Dev1/ai2")
    dialog = NidaqPortConfigurationDialog(config, devices=(_readback_device(),))

    assert "Dev1/ai2" not in _combo_values(dialog._laser_combos[1]["diode"])
    assert "Dev1/ai2" not in _combo_values(dialog._laser_combos[2]["laser_copy"])
    readback = _combo_values(dialog._laser_combos[1]["trigger_readback"])
    for taken in ("Dev1/ai0", "Dev1/ai1", "Dev1/port0/line4"):
        assert taken not in readback


def test_a_trigger_readback_on_another_input_is_refused_before_the_dialog_closes(qapp):
    # Refused only by the application after the dialog had closed, the
    # refusal reached the log and every edit in the dialog was lost.
    config = _readback_config(trigger_monitor_input="Dev1/ai1")
    dialog = NidaqPortConfigurationDialog(config, devices=(_readback_device(),))

    status = dialog._status_label.text()
    assert "Duplicate channel assignment" in status
    assert "trigger readback input" in status
    assert not _ok_enabled(dialog)
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.laser_configuration == config.laser


@pytest.mark.parametrize("terminal", ["/Dev1/PFI0", "Dev1/ao1"], ids=["pfi", "output"])
def test_a_trigger_readback_that_is_not_one_input_is_refused_in_the_dialog(qapp, terminal):
    config = _readback_config(trigger_monitor_input=terminal)
    dialog = NidaqPortConfigurationDialog(config, devices=(_readback_device(),))

    status = dialog._status_label.text()
    assert terminal in status
    assert "analog input" in status and "digital input line" in status
    assert not _ok_enabled(dialog)
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_a_trigger_readback_on_the_pmt_shutter_line_is_refused_in_the_dialog(qapp):
    # Not one of the dialog's own fields, so its duplicate check never saw it.
    config = _readback_config(
        trigger_monitor_input="Dev1/port0/line5", pmt_shutter_output="Dev1/port0/line5")
    dialog = NidaqPortConfigurationDialog(config, devices=(_readback_device(),))

    assert "PMT shutter output" in dialog._status_label.text()
    assert not _ok_enabled(dialog)
