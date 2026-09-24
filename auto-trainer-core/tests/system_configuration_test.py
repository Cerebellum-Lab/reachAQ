import io

import pytest
import yaml

from autotrainer.core import (
    SystemConfiguration,
    HardwareConfiguration,
    CameraId,
    CameraConfiguration,
    Offset3DTuple,
    LaserChannelConfiguration,
    LaserSystemConfiguration,
)
from autotrainer.core.configuration import (
    NidaqSignalStreamConfiguration,
    SystemConfigurationDumper,
    SystemConfigurationLoader,
)
from autotrainer.core.configuration.behavior_configuration import (
    PelletDeliveryConfiguration,
    ShiftXYZBufferHandlerConfig,
)


@pytest.mark.parametrize(
    ("constructor", "kwargs", "retired_name"),
    (
        (
            NidaqSignalStreamConfiguration,
            {"record_to_acquisition": True},
            "record_to_acquisition",
        ),
        (
            ShiftXYZBufferHandlerConfig,
            {"target_x": 1.0},
            "target_x",
        ),
    ),
)
def test_current_configuration_rejects_retired_fields(
    constructor,
    kwargs,
    retired_name,
):
    with pytest.raises(TypeError, match=retired_name):
        constructor(**kwargs)

def test_same_version_unknown_attribute_raise():
    config_text = f"""
!SystemConfiguration
version: {SystemConfiguration.version}
unknown_attribute: 42
"""
    with pytest.raises(TypeError, match="unknown_attribute"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_older_version_is_rejected():
    config_text = """
!SystemConfiguration
version: 56
behavior: !BehaviorConfiguration
  pelletDelivery: !PelletDeliveryConfiguration
    isEnabled: true
    maxPelletsPerSession: 20
    maxPelletsPerDay: 100
"""

    with pytest.raises(ValueError, match="requires version"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_higher_version_is_rejected():
    config_text = f"""
!SystemConfiguration
version: {SystemConfiguration.version + 1}
unknown_attribute: 42
persistence: !PersistenceConfiguration
  outputLocation: /output_location_path
  another_unknown_attribute: foobar
"""
    with pytest.raises(ValueError, match="requires version"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_unknown_tags_in_unsupported_version_are_rejected_by_version():
    config_text = f"""
    !SystemConfiguration
    version: {SystemConfiguration.version + 1}
    unknown_attribute: 42
    bar: !unknown_tag
    baz:
    - !second_unknown_tag
      param1: anything
    """
    with pytest.raises(ValueError, match="requires version"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_save_file_without_specify_save_type_fails():
    cfg = SystemConfiguration()
    with pytest.raises(ValueError, match="Missing one of as_json or as_yaml"):
        cfg.save_file("foobar.baz")


def test_offset3d_yaml():
    o = Offset3DTuple(1, 2, 3.5)
    data = yaml.dump(o, Dumper=SystemConfigurationDumper)
    o2 = yaml.load(data, Loader=SystemConfigurationLoader)
    assert isinstance(o2, Offset3DTuple)
    assert o2 == o


def test_rfid_hardware_settings_round_trip_in_system_yaml():
    configuration = SystemConfiguration(
        hardware=HardwareConfiguration(
            can_enabled=False,
            pellet_controller_enabled=False,
            nidaq_enabled=True,
            rfid_reader_enabled=True,
            rfid_device="/dev/serial/by-id/test-rfid",
        )
    )

    loaded = SystemConfiguration.load_yaml(io.StringIO(configuration.dump_yaml()))

    assert loaded.hardware.can_enabled is False
    assert loaded.hardware.pellet_controller_enabled is False
    assert loaded.hardware.nidaq_enabled is True
    assert loaded.hardware.rfid_reader_enabled is True
    assert loaded.hardware.rfid_device == "/dev/serial/by-id/test-rfid"


def test_laser_board_trigger_wiring_round_trips_in_system_yaml():
    configuration = SystemConfiguration(
        laser=LaserSystemConfiguration.from_channels((
            LaserChannelConfiguration(
                channel_id=2,
                analog_output="Dev1/ao1",
                diode_input="Dev1/ai4",
                shutter_output="Dev1/port0/line5",
                trigger_source="/Dev1/PXI_Trig2",
                board_stim_line=2,
                board_trigger_pulse_us=1500,
            ),
        )),
    )

    text = configuration.dump_yaml()
    loaded = SystemConfiguration.load_yaml(io.StringIO(text))

    assert "boardStimLine: 2" in text
    assert "boardTriggerPulseUs: 1500" in text
    channel = loaded.laser.get_channel(2)
    assert channel.board_stim_line == 2
    assert channel.board_trigger_pulse_us == 1500


def test_a_laser_without_board_wiring_has_no_board_line():
    channel = LaserChannelConfiguration(
        channel_id=1, analog_output="a", diode_input="b", shutter_output="c")

    assert channel.board_stim_line is None
    assert channel.board_trigger_pulse_us == 1000


@pytest.mark.parametrize("line", [0, 1, 4])
def test_a_laser_refuses_a_board_line_that_cannot_trigger_it(line):
    with pytest.raises(ValueError, match="board_stim_line"):
        LaserChannelConfiguration(
            channel_id=1, analog_output="a", diode_input="b", shutter_output="c",
            board_stim_line=line)


@pytest.mark.parametrize("width", [99, 5_000_001])
def test_a_laser_refuses_a_board_pulse_outside_the_firmware_range(width):
    with pytest.raises(ValueError, match="board_trigger_pulse_us"):
        LaserChannelConfiguration(
            channel_id=1, analog_output="a", diode_input="b", shutter_output="c",
            board_trigger_pulse_us=width)


def test_the_system_configuration_is_version_58():
    # 58 added the laser channel's board trigger wiring.
    assert SystemConfiguration.version == 58
