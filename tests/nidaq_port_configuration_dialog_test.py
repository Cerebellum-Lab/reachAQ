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
from tools.acquisition.model.nidaq_channel_plan import (  # noqa: E402
    build_nidaq_acquisition_configuration,
)
from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts  # noqa: E402
from tools.acquisition.view.nidaq_port_configuration_dialog import NidaqPortConfigurationDialog  # noqa: E402

from nidaq_stream_lifecycle_test import _settle, nidaq_app  # noqa: E402,F401


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


#: What discovery reports for a board that clocks digital input, a PXI-6221.
_BUFFERED_DI_RATE = 1_000_000.0


def test_selected_daq_channel_is_removed_from_other_roles(qapp):
    device = NidaqDevicePorts(
        name="Dev1",
        digital_inputs=("Dev1/port0/line0", "Dev1/port0/line1"),
        digital_input_max_rate=_BUFFERED_DI_RATE,
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
        digital_input_max_rate=_BUFFERED_DI_RATE,
    )
    dialog = NidaqPortConfigurationDialog(config, devices=(device,))

    with pytest.raises(ValueError, match="Duplicate channel assignment"):
        dialog._validate_selected_channel_assignments(device)


_DIGITAL_PORT_ROLES = ("tone1", "tone2", "tone3_r", "tone3_l", "cam_frames", "barcode")


def _input_card(name="Dev1"):
    # An M Series board: port0 clocks, port1 and port2 are the PFI pins.
    return NidaqDevicePorts(
        name=name,
        analog_inputs=(f"{name}/ai0",),
        digital_inputs=(
            f"{name}/port0/line0", f"{name}/port0/line1",
            f"{name}/port1/line0", f"{name}/port2/line7",
        ),
        digital_input_max_rate=_BUFFERED_DI_RATE,
    )


def _output_card(name="Dev2"):
    # A PXI-6713: lines, but no DI rate, and no clocked digital input.
    return NidaqDevicePorts(
        name=name,
        analog_outputs=(f"{name}/ao0",),
        digital_outputs=(f"{name}/port0/line0",),
        digital_inputs=(f"{name}/port0/line0", f"{name}/port0/line1"),
    )


def test_every_digital_port_role_offers_only_lines_the_stream_can_clock(qapp):
    # It offered every digital input line discovery reported, port1/port2
    # PFI pins and PXI-6713 lines included, and any of them took every NI
    # input down (-200452).
    dialog = NidaqPortConfigurationDialog(
        SystemConfiguration(), devices=(_input_card(), _output_card()))

    _set_device(dialog, "Dev1")
    for role in _DIGITAL_PORT_ROLES:
        assert set(_combo_values(dialog._general_combos[role])) == {
            None, "Dev1/port0/line0", "Dev1/port0/line1"}, role

    _set_device(dialog, "Dev2")
    for role in _DIGITAL_PORT_ROLES:
        assert _combo_values(dialog._general_combos[role]) == (None,), role


def test_a_stored_pfi_pin_tone_is_shown_as_invalid_and_blocks_ok(qapp):
    config = SystemConfiguration()
    config.nidaq_ports = NidaqPortConfiguration(
        device_name="Dev1", tone1="Dev1/port1/line0")
    dialog = NidaqPortConfigurationDialog(config, devices=(_input_card(),))

    # Kept and shown, not dropped, so the operator sees what to change.
    assert dialog._general_combos["tone1"].currentData() == "Dev1/port1/line0"
    status = dialog._status_label.text()
    assert "tone1" in status and "Dev1/port1/line0" in status
    assert "PFI pins" in status
    assert not _ok_enabled(dialog)
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_a_stored_line_on_a_board_that_cannot_clock_it_is_shown_as_invalid(qapp):
    config = SystemConfiguration()
    config.nidaq_ports = NidaqPortConfiguration(
        device_name="Dev1", tone2="Dev2/port0/line0")
    dialog = NidaqPortConfigurationDialog(
        config, devices=(_input_card(), _output_card()))

    assert dialog._general_combos["tone2"].currentData() == "Dev2/port0/line0"
    status = dialog._status_label.text()
    assert "tone2" in status and "Dev2 cannot clock digital input" in status
    assert not _ok_enabled(dialog)


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


def _readback_device(*, digital_input_max_rate=1_000_000.0):
    # A PXI-6221: buffered digital input on port0, and port1 and port2 as
    # the static PFI pins, listed among the digital inputs all the same.
    return NidaqDevicePorts(
        name="Dev1",
        analog_outputs=("Dev1/ao0", "Dev1/ao1"),
        analog_inputs=("Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"),
        digital_outputs=("Dev1/port0/line4", "Dev1/port0/line5"),
        digital_inputs=(
            "Dev1/port0/line0", "Dev1/port0/line1",
            "Dev1/port0/line4", "Dev1/port0/line5",
            "Dev1/port1/line0", "Dev1/port2/line7",
        ),
        digital_input_max_rate=digital_input_max_rate,
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
        # Analog inputs and port0 lines, which the stream can clock; not the
        # port1/port2 PFI pins, a PFI terminal, an output or a backplane line.
        assert set(values[1:]) == {
            "Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3",
            "Dev1/port0/line0", "Dev1/port0/line1",
            "Dev1/port0/line4", "Dev1/port0/line5",
        }


def test_a_board_that_cannot_clock_digital_input_offers_no_readback_lines(qapp):
    # A PXI-6713: the driver gives no DI rate, and a buffered task on its
    # lines fails at -200452.
    dialog = NidaqPortConfigurationDialog(
        SystemConfiguration(), devices=(_readback_device(digital_input_max_rate=None),))

    values = _combo_values(dialog._laser_combos[1]["trigger_readback"])

    assert set(values[1:]) == {"Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"}


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


@pytest.mark.parametrize(
    "terminal", ["/Dev1/PFI0", "Dev1/ao1", "Dev1/port1/line0"],
    ids=["pfi", "output", "pfi_pin_line"])
def test_a_trigger_readback_that_is_not_one_input_is_refused_in_the_dialog(qapp, terminal):
    config = _readback_config(trigger_monitor_input=terminal)
    dialog = NidaqPortConfigurationDialog(config, devices=(_readback_device(),))

    status = dialog._status_label.text()
    assert terminal in status
    assert "analog input" in status and "port0 line" in status
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


def _pins(configuration):
    return {channel.physical_channel for channel in configuration.channels}


def test_choosing_none_for_the_readback_stops_it_being_acquired_for_good(
    qapp, nidaq_app, system_config, trainer_config_dir,
):
    # The dialog saved no readback, and the application rebuilt the plan from
    # the live one, which still held laser1_trigger on Dev1/ai7; the laser
    # claimed neither that pin nor the name, so the channel came back as a
    # hidden custom input, recorded from then on.
    system_config.laser = LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=1,
                analog_output="Dev1/ao0",
                diode_input="Dev1/ai0",
                shutter_output="Dev1/port0/line2",
                command_copy_input="Dev1/ai1",
                trigger_monitor_input="Dev1/ai7",
            ),
        ),
        backend="null",
    )
    system_config.save_default(trainer_config_dir)
    assert nidaq_app.load_configuration() is True
    _settle(nidaq_app)
    assert "Dev1/ai7" in _pins(nidaq_app.nidaq_signal_monitor.configuration)
    device = NidaqDevicePorts(
        name="Dev1",
        analog_outputs=("Dev1/ao0",),
        analog_inputs=("Dev1/ai0", "Dev1/ai1", "Dev1/ai7"),
        digital_outputs=("Dev1/port0/line2",),
        digital_inputs=("Dev1/port0/line0", "Dev1/port0/line2"),
        digital_input_max_rate=1_000_000.0,
    )
    dialog = NidaqPortConfigurationDialog(nidaq_app.loaded_configuration, devices=(device,))
    readback = dialog._laser_combos[1]["trigger_readback"]
    assert readback.currentData() == "Dev1/ai7"

    readback.setCurrentIndex(readback.findData(None))
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    nidaq_app.update_daq_port_configuration(nidaq_app.nidaq_ports, dialog.laser_configuration)
    _settle(nidaq_app)

    assert nidaq_app.laser.configuration.get_channel(1).trigger_monitor_input is None
    assert "Dev1/ai7" not in _pins(nidaq_app.nidaq_signal_monitor.configuration)
    assert "Dev1/ai7" not in _pins(nidaq_app.loaded_configuration.nidaq_stream)

    # And from the saved file, through the plan a load builds from it. (A
    # full reload is not possible here: this fixture's inference model is a
    # stand-in that saves no inference section.)
    saved = nidaq_app.get_config_from_location(nidaq_app.get_config_location())
    assert saved.laser.get_channel(1).trigger_monitor_input is None
    assert "Dev1/ai7" not in _pins(saved.nidaq_stream)
    reloaded = build_nidaq_acquisition_configuration(
        saved.nidaq_stream, saved.nidaq_ports, saved.laser)
    assert "Dev1/ai7" not in _pins(reloaded)
    assert "laser1_trigger" not in {channel.name for channel in reloaded.channels}


def test_clearing_a_port_role_stops_it_being_recorded_for_good(
    qapp, nidaq_app, system_config, trainer_config_dir,
):
    # camFrames cleared in the dialog came back as a custom channel from the
    # live plan: saved, reloaded, recorded, and listed in Analysis as "no
    # longer mapped".
    from autotrainer.core import NidaqSignalChannelConfiguration, NidaqSignalStreamConfiguration
    from tools.acquisition.view.analysis_content import AnalysisContent

    system_config.nidaq_stream = NidaqSignalStreamConfiguration(
        channels=(NidaqSignalChannelConfiguration("stim_readback", "Dev1/ai5"),),
        is_enabled=True,
    )
    system_config.save_default(trainer_config_dir)
    assert nidaq_app.load_configuration() is True
    _settle(nidaq_app)
    names = {channel.name for channel in nidaq_app.nidaq_signal_monitor.configuration.channels}
    assert {"cam_frames", "stim_readback"} <= names
    device = NidaqDevicePorts(
        name="Dev1",
        analog_inputs=("Dev1/ai5",),
        digital_inputs=("Dev1/port0/line0", "Dev1/port0/line1"),
        digital_input_max_rate=_BUFFERED_DI_RATE,
    )
    dialog = NidaqPortConfigurationDialog(nidaq_app.loaded_configuration, devices=(device,))
    cam_frames = dialog._general_combos["cam_frames"]
    assert cam_frames.currentData() == "Dev1/port0/line0"

    cam_frames.setCurrentIndex(cam_frames.findData(None))
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    nidaq_app.update_daq_port_configuration(dialog.nidaq_ports, dialog.laser_configuration)
    _settle(nidaq_app)

    for plan in (
        nidaq_app.nidaq_signal_monitor.configuration,
        nidaq_app.loaded_configuration.nidaq_stream,
    ):
        assert "Dev1/port0/line0" not in _pins(plan)
        assert "stim_readback" in {channel.name for channel in plan.channels}
    # And from the saved file, through the plan a load builds from it.
    saved = nidaq_app.get_config_from_location(nidaq_app.get_config_location())
    assert saved.nidaq_ports.cam_frames is None
    reloaded = build_nidaq_acquisition_configuration(
        saved.nidaq_stream, saved.nidaq_ports, saved.laser)
    assert "Dev1/port0/line0" not in _pins(reloaded)
    assert "stim_readback" in {channel.name for channel in reloaded.channels}

    content = AnalysisContent(nidaq_app)
    try:
        assert "custom:cam_frames" not in content._signal_checkboxes
        assert "custom:stim_readback" in content._signal_checkboxes
    finally:
        content.on_close()
        content.deleteLater()
