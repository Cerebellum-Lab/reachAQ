import pytest

from autotrainer.device import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NullLaserController,
)


def make_channel(channel_id=LaserChannelId.LASER_1):
    return LaserChannelConfiguration(
        channel_id=channel_id,
        analog_output="Dev1/ao0",
        diode_input="Dev1/ai0",
        shutter_output="Dev1/port0/line0",
        auxiliary_output="Dev1/port0/line1",
        command_monitor_input="Dev1/ai1",
        command_copy_output="Dev1/ao1",
    )


def test_laser_channel_configuration_normalizes_channel_id():
    config = make_channel(1)

    assert config.channel_id == LaserChannelId.LASER_1


def test_laser_channel_configuration_clamps_command_voltage():
    config = make_channel()

    assert config.clamp_command_voltage(-1) == 0.0
    assert config.clamp_command_voltage(2.5) == 2.5
    assert config.clamp_command_voltage(10) == 5.0


def test_laser_system_configuration_limits_channels_to_four():
    channels = [
        make_channel(LaserChannelId.LASER_1),
        make_channel(LaserChannelId.LASER_2),
        make_channel(LaserChannelId.LASER_3),
        make_channel(LaserChannelId.LASER_4),
        make_channel(LaserChannelId.LASER_1),
    ]

    with pytest.raises(ValueError):
        LaserSystemConfiguration.from_channels(channels)


def test_laser_system_configuration_rejects_duplicate_channel_ids():
    channels = [make_channel(LaserChannelId.LASER_1), make_channel(LaserChannelId.LASER_1)]

    with pytest.raises(ValueError):
        LaserSystemConfiguration.from_channels(channels)


def test_null_laser_controller_tracks_outputs():
    system = LaserSystemConfiguration.from_channels([make_channel()])
    controller = NullLaserController(system)

    applied = controller.set_command_voltage(LaserChannelId.LASER_1, 2.25)
    controller.set_shutter_open(LaserChannelId.LASER_1, True)
    controller.set_auxiliary_output(LaserChannelId.LASER_1, True)

    assert applied == 2.25
    assert controller.read_diode_voltage(LaserChannelId.LASER_1) == 2.25
    assert controller.read_command_monitor_voltage(LaserChannelId.LASER_1) == 2.25
    assert controller.read_feedback_sample(LaserChannelId.LASER_1).command_monitor_volts == 2.25
    assert controller.is_shutter_open(LaserChannelId.LASER_1)
    assert controller.is_auxiliary_output_enabled(LaserChannelId.LASER_1)

    controller.close_all_shutters()

    assert not controller.is_shutter_open(LaserChannelId.LASER_1)
