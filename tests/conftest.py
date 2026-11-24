import sys
from pathlib import Path

import pytest

# Tests here (calibration_install_test.py) import the synthetic rig, multires_synthetic, from
# tests/inference. That folder is on sys.path only once pytest has collected a test
# from it, so such a test passed in the full-suite order and failed when run alone or per
# directory. Appended, not inserted: a module of the same name here still wins. Not the pytest
# `pythonpath` ini option, which pytest.ini does not use and which would apply to every directory.
_INFERENCE_TESTS = str(Path(__file__).resolve().parents[1] / "tests" / "inference")
if _INFERENCE_TESTS not in sys.path:
    sys.path.append(_INFERENCE_TESTS)

from reachaq.behavior import BehaviorAlgorithm
from reachaq.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from reachaq.core import SystemConfiguration, CameraConfiguration, CameraId
from reachaq.core.logging import unregister_fatal_exception_callback
from reachaq.device import MotorConfigurationFile, CompoundMovements

from reachaq.app.model.app_model import AppModel
from reachaq.app.model.user_preferences import UserPreferences


import top_fixtures


@pytest.fixture(autouse=True)
def _isolated_runtime_dir(tmp_path_factory, monkeypatch):
    """Give each test its own XDG_RUNTIME_DIR, inherited by what it launches.

    The application refuses to start while another instance holds the per-user
    lock in the runtime directory. Tests that launch the real CLI were refused
    with exit 3 whenever an operator's reachAQ was running on the same account,
    as on christielab10 on 2026-09-24.
    """
    runtime = tmp_path_factory.mktemp("runtime")
    runtime.chmod(0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))


@pytest.fixture
def system_config(trainer_config_dir, tmp_path):
    config = SystemConfiguration()
    # mostly ~all default params are good, but we need:
    config.behavior.pellet_delivery.is_enabled = True
    config.behavior.pellet_delivery.is_intersession_analysis_enabled = True
    # Application tests use the mock device stack and must not depend on the
    # host SocketCAN interface being up.
    config.hardware.can_enabled = False
    for cam_member in (CameraId.Left, CameraId.Right, CameraId.Camera3):
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
    return top_fixtures.repo_root_dir.joinpath(f"tests/inference/{DEFAULT_3D_CALIB_DIR_NAME}")


@pytest.fixture
def app_model(mock_system, user_pref, calib_dir, diamond_config_path, system_config, monkeypatch, fake_system_msg_handler):
    # for now:
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    assert BehaviorAlgorithm._no_handler_thread is True  # to be safe to start with
    #
    app = AppModel(
        user_pref,
        system_machine=mock_system.system_machine,
        analysis=mock_system.sensor_analysis,
        inference_model=mock_system.inference,
        system_message_handler=fake_system_msg_handler,
        calib_dir=calib_dir,
    )
    try:
        yield app
    finally:
        # Most tests use mocked worker lifecycles and cannot safely call the
        # full AppModel.on_close(). Still unregister the process-global fatal
        # callback so one later test exception cannot fan out across every
        # AppModel created earlier in the pytest process.
        unregister_fatal_exception_callback(app._on_fatal_exception)
        top_fixtures.release_multiprocessing_resources(app)
