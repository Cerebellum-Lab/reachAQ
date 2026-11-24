import math
from pathlib import Path

import pytest

from autotrainer.core import Motor
from autotrainer.device import MotorConfigurationFile, StepperConfig, ServoConfig, stepper_travel_refusal


@pytest.fixture
def motor_file_cfg_instance():
    return MotorConfigurationFile()


@pytest.fixture
def motor_file_from_empty_dict():
    return MotorConfigurationFile.from_yaml_dict({}, source="pytest")


@pytest.mark.parametrize("cfg_source", ["motor_file_cfg_instance", "motor_file_from_empty_dict"])
def test_it_gets_motor_set_on_construction(request, cfg_source):
    file_cfg = request.getfixturevalue(cfg_source)
    assert isinstance(file_cfg, MotorConfigurationFile)
    for m, motor_cfg in (
        file_cfg.x_config, file_cfg.y_config, file_cfg.z_config,
    ):
        assert isinstance(m, Motor)
        assert m != Motor.NONE
        assert m == motor_cfg.motor
        assert isinstance(motor_cfg, StepperConfig)
    #
    for m, motor_cfg in (
        file_cfg.cover_config,
        file_cfg.load_config,
    ):
        assert isinstance(m, Motor)
        assert m != Motor.NONE
        assert m == motor_cfg.motor
        assert isinstance(motor_cfg, ServoConfig)


# Stepper travel limits (min_pos / max_pos in motor_config.yaml, mm). HardwareModel refuses
# a pellet x/y/z move outside them; an axis without them is not limited.

CHRISTIELAB10_MOTOR_CONFIG = Path(__file__).with_name("christielab10_motor_config.yaml")


def test_a_stepper_without_travel_keys_has_no_limits():
    config = StepperConfig.from_dict({"max_vel": 120, "max_acc": 600})
    assert config.minimum_position is None
    assert config.maximum_position is None
    assert StepperConfig().minimum_position is None and StepperConfig().maximum_position is None


def test_a_stepper_parses_both_travel_keys():
    config = StepperConfig.from_dict({"min_pos": 0, "max_pos": 35})
    assert config.minimum_position == 0.0
    assert config.maximum_position == 35.0


@pytest.mark.parametrize("data, expected", [
    ({"min_pos": 2.5}, (2.5, None)),
    ({"max_pos": 35}, (None, 35.0)),
])
def test_a_stepper_with_one_travel_key_is_limited_on_that_side_only(data, expected):
    config = StepperConfig.from_dict(data)
    assert (config.minimum_position, config.maximum_position) == expected


@pytest.mark.parametrize("data", [
    {"min_pos": 36, "max_pos": 35},
    {"min_pos": math.nan, "max_pos": 35},
    {"min_pos": 0, "max_pos": math.inf},
    {"min_pos": -math.inf},
    {"max_pos": None},
    {"max_pos": "35"},
    {"max_pos": True},
])
def test_a_stepper_with_bad_travel_is_refused(data):
    with pytest.raises(ValueError):
        StepperConfig.from_dict(data)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_a_motor_file_with_bad_travel_is_refused_naming_the_axis(axis):
    pellet = {a: {"min_pos": 0, "max_pos": 35} for a in "xyz"}
    pellet[axis] = {"min_pos": 40, "max_pos": 35}
    with pytest.raises(ValueError, match=f"pellet-{axis}"):
        MotorConfigurationFile.from_yaml_dict({"pellet": pellet}, source="pytest")


def test_christielab10_motor_config_loads_with_its_travel():
    configs = MotorConfigurationFile.from_file(CHRISTIELAB10_MOTOR_CONFIG)
    for (motor, config), expected_motor in zip(
        (configs.x_config, configs.y_config, configs.z_config),
        (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR),
    ):
        assert motor == expected_motor == config.motor
        assert (config.minimum_position, config.maximum_position) == (0.0, 35.0)
        assert (config.maximum_velocity, config.maximum_acceleration, config.homing_velocity) == (120, 600, 30)
        assert (config.microsteps, config.steps_per_revolution) == (8, 24.0)
    assert configs.x_config[1].flip_limit_orientation is True
    assert configs.y_config[1].flip_limit_orientation is False
    # The servos keep their own limits, as before.
    assert (configs.load_config[1].minimum_position, configs.load_config[1].maximum_position) == (0, 120)


def test_a_target_a_rounding_error_past_a_limit_is_inside():
    # A DCS conversion of a nominal 35.0 can come out as 35.0000000001.
    config = StepperConfig.from_dict({"min_pos": 0, "max_pos": 35})
    assert stepper_travel_refusal("x", 35.0000000001, config) is None
    assert stepper_travel_refusal("x", -1e-9, config) is None
    assert stepper_travel_refusal("x", 35.00001, config) is not None
    assert stepper_travel_refusal("x", -0.00001, config) is not None
