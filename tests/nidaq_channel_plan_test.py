from autotrainer.core import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NidaqPortConfiguration,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from tools.acquisition.model.nidaq_channel_plan import (
    build_nidaq_acquisition_configuration,
    with_display_channels,
)


def test_all_mapped_inputs_are_acquired_independently_from_display_selection():
    configured = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration(
                name="barcode",
                physical_channel="InputCard/port0/line1",
                kind="digital",
            ),
        ),
        is_enabled=True,
        display_channels=("barcode",),
    )
    ports = NidaqPortConfiguration(
        cam_frames="InputCard/port0/line0",
        barcode="InputCard/port0/line1",
        tone1="InputCard/port0/line2",
    )
    laser = LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="OutputCard/ao0",
                diode_input="InputCard/ai0",
                shutter_output="OutputCard/port0/line0",
                command_copy_input="InputCard/ai1",
                feedback_scale=2.0,
                command_copy_scale=3.0,
            ),
        ),
        backend="nidaq",
    )

    result = build_nidaq_acquisition_configuration(configured, ports, laser)

    assert tuple(channel.name for channel in result.channels) == (
        "cam_frames",
        "barcode",
        "tone1",
        "laser1_diode",
        "laser1_command_copy",
    )
    assert result.display_channels == ("barcode",)
    assert result.is_enabled
    assert result.channels[-2].scale == 2.0
    assert result.channels[-1].scale == 3.0


def test_display_selection_does_not_change_acquisition_channels():
    configuration = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration("first", "Dev1/ai0"),
            NidaqSignalChannelConfiguration("second", "Dev1/ai1"),
        ),
        is_enabled=True,
    )

    result = with_display_channels(configuration, ("second",))

    assert result.channels == configuration.channels
    assert result.display_channels == ("second",)


def test_existing_unmapped_channel_is_retained_as_custom_input():
    custom = NidaqSignalChannelConfiguration("force", "Dev2/ai3")
    configured = NidaqSignalStreamConfiguration(
        channels=(custom,),
        is_enabled=True,
    )

    result = build_nidaq_acquisition_configuration(
        configured,
        NidaqPortConfiguration(),
        LaserSystemConfiguration(),
    )

    assert result.channels == (custom,)
