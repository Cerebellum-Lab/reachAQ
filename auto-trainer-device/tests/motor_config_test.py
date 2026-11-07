import pytest

from autotrainer.core import Motor
from autotrainer.device import MotorConfigurationFile, StepperConfig, ServoConfig


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
