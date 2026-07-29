from pathlib import Path

import pytest

from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core import SystemConfiguration, CameraConfiguration, CameraId
from autotrainer.device import MotorConfigurationFile, CompoundMovements

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.user_preferences import UserPreferences


import top_fixtures


@pytest.fixture
def system_config(trainer_config_dir, tmp_path):
    config = SystemConfiguration()
    # mostly ~all default params are good, but we need:
    config.behavior.pellet_delivery.is_enabled = True
    config.behavior.pellet_delivery.is_intersession_analysis_enabled = True
    # Application tests use the mock device stack and must not depend on the
    # host SocketCAN interface being up.
    config.hardware.can_enabled = False
    for cam_member in (CameraId.Left, CameraId.Right, CameraId.Web):
        params = dict(width=300, height=200)
        cam = CameraConfiguration(name=cam_member.name, params=params)
        cam.record_prebuffer_duration = 0
        cam.scheme = "random"
        cam.id = cam_member
        config.cameras.append(cam)
    config.persistence.output_location = tmp_path.joinpath("Data").as_posix()
    config.save_default(trainer_config_dir)
    return config


@pytest.fixture
def config_file_path(trainer_config_dir):
    return trainer_config_dir.joinpath(SystemConfiguration.make_default_yaml_config_path(trainer_config_dir))


@pytest.fixture
def calib_dir():
    # could be todo: copy it top-level, or generate new temporary one as above for system config.
    return top_fixtures.repo_root_dir.joinpath(f"auto-trainer-inference/tests/{DEFAULT_3D_CALIB_DIR_NAME}")


@pytest.fixture
def app_model(mock_system, user_pref, calib_dir, diamond_config_path, system_config, monkeypatch, fake_system_msg_handler):
    # for now:
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    assert BehaviorAlgorithm._no_handler_thread is True  # to be safe to start with
    #
    app = AppModel(
        user_pref,
        system_machine=mock_system.system_machine,
        sensor_analysis=mock_system.sensor_analysis,
        inference_model=mock_system.inference,
        system_message_handler=fake_system_msg_handler,
        calib_dir=calib_dir,
    )
    return app
