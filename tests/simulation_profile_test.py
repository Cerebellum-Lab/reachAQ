import os
from pathlib import Path

import pytest

from autotrainer.core import PersistenceConfiguration, SystemConfiguration
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.device.compound_movement_file import CompoundMovements
from autotrainer.device.motor_configuration_file import MotorConfigurationFile
from tools.acquisition.args import make_autotrainer_parser
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.nidaq_channel_plan import build_nidaq_acquisition_configuration
from tools.acquisition.model.nidaq_discovery import simulated_nidaq_devices
from tools.acquisition.model.nidaq_timing import build_nidaq_timing_plan
from tools.acquisition.model.user_preferences import UserPreferences
from tools.simulation import profile
from tools.simulation.profile import SIMULATION_ENVIRONMENT, prepare_simulation
from tools.simulation.sim_configuration import build_simulation_configuration


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """HOME under tmp, with the environment and the defaults prepare_simulation moves restored after."""
    home = tmp_path.joinpath("operator-home")
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for name in SIMULATION_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for owner, attribute in (
        (PersistenceConfiguration, "DEFAULT_OUTPUT_PATH"),
        (AppModel, "status_file_path"),
        (MotorConfigurationFile, "DEFAULT_LOCATION"),
        (CompoundMovements, "DEFAULT_LOCATION"),
        (DiamondTriangleOffsetConfig, "DEFAULT_CONFIG_PATH"),
    ):
        monkeypatch.setattr(owner, attribute, getattr(owner, attribute))
    return home


def _parse(*argv):
    return make_autotrainer_parser().parse_args(["--simulate", *argv])


def _files(root: Path) -> set:
    return {path for path in root.rglob("*")}


def test_prepare_writes_only_under_sim_home(home, tmp_path):
    before = _files(tmp_path)
    run = prepare_simulation(_parse())

    assert run.home == home.joinpath(".reachaq-sim").resolve()
    created = _files(tmp_path) - before
    assert created and all(path.resolve().is_relative_to(run.home) for path in created), sorted(created)
    assert not home.joinpath("Autotrainer").exists()
    assert run.config_path.is_file() and run.preferences_path.is_file()
    assert run.bus_path.parent.is_dir() and run.bus_path.parent.parent == run.home / "runs"
    assert run.lock_path == run.home / "app.lock"

    # What the app reads or writes before, or regardless of, its configuration.
    preferences = UserPreferences(settings_file_path=run.preferences_path)
    for location in (
        preferences.configuration_location,
        preferences.animal_location,
        preferences.softmouse_manifest_path,
        SystemConfiguration.load_yaml_file(run.config_path).persistence.output_location,
        PersistenceConfiguration.get_default_output_path(),
        AppModel.status_file_path,
        MotorConfigurationFile.DEFAULT_LOCATION,
        CompoundMovements.DEFAULT_LOCATION,
        DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH,
    ):
        assert Path(location).resolve().is_relative_to(run.home), location
    assert preferences.softmouse_nightly_refresh is False


def test_second_run_reuses_config_and_reset_regenerates(home, tmp_path):
    sim_home = tmp_path.joinpath("sim")
    first = prepare_simulation(_parse("--sim-home", sim_home.as_posix()))
    edited = first.config_path.read_text().replace("fps: 150", "fps: 100")
    assert edited != first.config_path.read_text()
    first.config_path.write_text(edited)
    preferences = UserPreferences(settings_file_path=first.preferences_path)
    preferences.log_level = 10
    preferences.save()

    args = _parse("--sim-home", sim_home.as_posix())
    second = prepare_simulation(args)
    assert second.config_path.read_text() == edited
    assert UserPreferences(settings_file_path=second.preferences_path).log_level == 10
    # The kept file's primary camera sets the simulated grid.
    assert float(os.environ["REACHAQ_SIM_CAMERA_FPS"]) == 100.0
    assert second.bus_path != first.bus_path

    third = prepare_simulation(_parse("--sim-home", sim_home.as_posix(), "--sim-reset"))
    assert "fps: 100" not in third.config_path.read_text()
    assert UserPreferences(settings_file_path=third.preferences_path).log_level != 10
    assert float(os.environ["REACHAQ_SIM_CAMERA_FPS"]) == 150.0


def test_user_given_files_are_kept(home, tmp_path):
    config, prefs = tmp_path / "mine.yaml", tmp_path / "mine.ini"
    args = _parse("--sim-home", (tmp_path / "sim").as_posix(), "-c", config.as_posix(),
                  "--preferences-file", prefs.as_posix())
    prepare_simulation(args)
    assert args.configuration == config and args.preferences_file == prefs


def test_generated_config_builds_nidaq_plan(tmp_path):
    configuration = build_simulation_configuration(tmp_path)
    configuration.save_default(tmp_path)
    # As the app sees it: through the YAML loader.
    configuration = SystemConfiguration.load_yaml_file(
        SystemConfiguration.make_default_yaml_config_path(tmp_path))

    cameras = {camera.id.name: camera for camera in configuration.cameras}
    assert sorted(cameras) == ["Camera3", "Left", "Right"]
    assert all(camera.scheme == "sim" and camera.is_enabled and camera.is_record_enabled
               for camera in cameras.values())
    assert cameras["Left"].params["primary"] == "yes"
    assert configuration.inference.is_enabled and configuration.inference.pose_model_location == ""
    calibration = configuration.nidaq_stream.delivery_calibration
    rate = configuration.nidaq_stream.sample_rate_hz
    assert calibration.offset_seconds == pytest.approx(0.5 / rate)
    assert calibration.spread_seconds == pytest.approx(0.3 / rate)
    assert (calibration.sessions, calibration.measured) == (3, "simulation")

    laser = configuration.laser
    stream = build_nidaq_acquisition_configuration(
        configuration.nidaq_stream, configuration.nidaq_ports, laser)
    assert {channel.name: channel.physical_channel for channel in stream.channels} == {
        "tone1": "SimDAQ1/port0/line0",
        "tone2": "SimDAQ1/port0/line1",
        "cam_frames": "SimDAQ1/port0/line2",
        "laser1_diode": "SimDAQ1/ai0",
        "laser1_command_copy": "SimDAQ1/ai1",
    }
    assert stream.is_enabled
    plan = build_nidaq_timing_plan(
        stream,
        configuration.nidaq_ports.timing,
        simulated_nidaq_devices(),
        hardware_timed_output_devices=("SimDAQ1",),
        hardware_timed_output_channels=tuple(channel.analog_output for channel in laser.channels),
        laser=laser,
    )
    assert plan.is_valid, plan.reason


def test_environment_set(home, tmp_path):
    args = _parse("--sim-home", (tmp_path / "sim").as_posix())
    assert getattr(args, "allow_can_emulation", False) is False
    run = prepare_simulation(args)

    assert os.environ["REACHAQ_SIMULATION"] == "1"
    assert os.environ["REACHAQ_SIM_BUS"] == run.bus_path.as_posix()
    assert float(os.environ["REACHAQ_SIM_EPOCH"]) > 0
    assert os.environ["AUTOTRAINER_CAN_TRANSPORT"] == "emulation"
    assert float(os.environ["REACHAQ_SIM_CAMERA_FPS"]) == 150.0
    assert float(os.environ["REACHAQ_SIM_CAMERA_EXPOSURE_S"]) == pytest.approx(0.002)
    assert os.environ["REACHAQ_SIM_LINE_ALIASES"] == "{}"
    assert args.allow_can_emulation is True
    assert args.configuration == run.config_path
    assert args.preferences_file == run.preferences_path
    assert set(SIMULATION_ENVIRONMENT) <= set(os.environ)
    assert profile.ENV_DIAMOND_TRIANGLE_CONFIG in SIMULATION_ENVIRONMENT
