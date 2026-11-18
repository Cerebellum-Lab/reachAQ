import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

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
from tools.acquisition import instance_lock
from tools.acquisition.instance_lock import AlreadyRunning, acquire_instance_lock
from tools.simulation import sim_home
from tools.simulation.sim_home import SIMULATION_ENVIRONMENT, prepare_simulation
from tools.simulation.sim_configuration import build_simulation_configuration


needs_flock = pytest.mark.skipif(instance_lock.fcntl is None, reason="flock is POSIX only")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """HOME under tmp, with the environment and the defaults prepare_simulation moves restored after."""
    home = tmp_path.joinpath("operator-home")
    home.mkdir()
    before = {name: os.environ.get(name) for name in SIMULATION_ENVIRONMENT}
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for owner, attribute in (
        (PersistenceConfiguration, "DEFAULT_OUTPUT_PATH"),
        (AppModel, "status_file_path"),
        (MotorConfigurationFile, "DEFAULT_LOCATION"),
        (CompoundMovements, "DEFAULT_LOCATION"),
        (DiamondTriangleOffsetConfig, "DEFAULT_CONFIG_PATH"),
    ):
        monkeypatch.setattr(owner, attribute, getattr(owner, attribute))
    # A whole copy, restored whole: monkeypatch.delenv records nothing for a
    # variable that is absent, so what prepare_simulation set would outlive the test.
    with mock.patch.dict(os.environ):
        os.environ["HOME"] = home.as_posix()
        for name in SIMULATION_ENVIRONMENT:
            os.environ.pop(name, None)
        yield home
    assert {name: os.environ.get(name) for name in SIMULATION_ENVIRONMENT} == before


def _parse(*argv):
    return make_autotrainer_parser().parse_args(["--simulate", *argv])


def _prepare(args):
    """prepare_simulation for a test that does not hold the run: its lock is let go."""
    run = prepare_simulation(args)
    if run.instance_lock is not None:
        run.instance_lock.close()
    return run


def _files(root: Path) -> set:
    return {path for path in root.rglob("*")}


def test_prepare_writes_only_under_sim_home(home, tmp_path):
    before = _files(tmp_path)
    run = _prepare(_parse())

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
    sim = tmp_path.joinpath("sim")
    first = _prepare(_parse("--sim-home", sim.as_posix()))
    edited = first.config_path.read_text().replace("fps: 150", "fps: 100")
    assert edited != first.config_path.read_text()
    first.config_path.write_text(edited)
    preferences = UserPreferences(settings_file_path=first.preferences_path)
    preferences.log_level = 10
    preferences.save()

    args = _parse("--sim-home", sim.as_posix())
    second = _prepare(args)
    assert second.config_path.read_text() == edited
    assert UserPreferences(settings_file_path=second.preferences_path).log_level == 10
    # The kept file's primary camera sets the simulated grid.
    assert float(os.environ["REACHAQ_SIM_CAMERA_FPS"]) == 100.0
    assert second.bus_path != first.bus_path

    third = _prepare(_parse("--sim-home", sim.as_posix(), "--sim-reset"))
    assert "fps: 100" not in third.config_path.read_text()
    assert UserPreferences(settings_file_path=third.preferences_path).log_level != 10
    assert float(os.environ["REACHAQ_SIM_CAMERA_FPS"]) == 150.0


def test_user_given_files_are_kept(home, tmp_path):
    mine = tmp_path / "mine"
    build_simulation_configuration(mine).save_default(mine)
    config = SystemConfiguration.make_default_yaml_config_path(mine)
    config.write_text(config.read_text().replace("fps: 150", "fps: 100"))
    config_text = config.read_text()
    prefs = tmp_path / "mine.ini"
    preferences = UserPreferences(settings_file_path=prefs)
    preferences.log_level = 10
    preferences.save()
    prefs_text = prefs.read_text()

    args = _parse("--sim-home", (tmp_path / "sim").as_posix(), "-c", config.as_posix(),
                  "--preferences-file", prefs.as_posix())
    run = _prepare(args)
    assert args.configuration == config and args.preferences_file == prefs
    assert config.read_text() == config_text and prefs.read_text() == prefs_text
    # The user's file is the one the run reads its camera grid from.
    assert float(os.environ["REACHAQ_SIM_CAMERA_FPS"]) == 100.0
    assert run.config_path != config and run.config_path.is_file()


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
    run = _prepare(args)

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
    assert sim_home.ENV_DIAMOND_TRIANGLE_CONFIG in SIMULATION_ENVIRONMENT


def test_environment_is_gone_after_a_run(home):
    # After test_environment_set, which ran prepare_simulation; the home
    # fixture's teardown checks the same after every test here.
    assert not set(SIMULATION_ENVIRONMENT) & set(os.environ)


@needs_flock
def test_run_holds_the_home_lock_and_a_second_run_writes_nothing(home, tmp_path):
    sim = tmp_path / "sim"
    run = prepare_simulation(_parse("--sim-home", sim.as_posix()))
    try:
        assert run.instance_lock is not None
        assert run.lock_path.read_text().strip() == str(os.getpid())
        run.config_path.write_text("held")
        before = _files(sim)
        with pytest.raises(AlreadyRunning):
            prepare_simulation(_parse("--sim-home", sim.as_posix(), "--sim-reset"))
        assert _files(sim) == before and run.config_path.read_text() == "held"
    finally:
        run.instance_lock.close()
    acquire_instance_lock(run.lock_path).close()  # released with the handle


@needs_flock
def test_a_held_home_refuses_before_creating_it(home, tmp_path):
    sim = tmp_path / "sim"
    held = acquire_instance_lock(sim / "app.lock")
    try:
        with pytest.raises(AlreadyRunning):
            prepare_simulation(_parse("--sim-home", sim.as_posix()))
        assert sorted(path.name for path in sim.iterdir()) == ["app.lock"]
    finally:
        held.close()


@needs_flock
@pytest.mark.parametrize("main", ["tools.acquisition.headless", "tools.acquisition.gui"])
def test_a_main_on_a_held_home_exits_already_running(main, tmp_path, app_launch_env):
    sim = tmp_path / "sim"
    held = acquire_instance_lock(sim / "app.lock")
    try:
        done = subprocess.run(
            [sys.executable, "-m", main, "--simulate", "--sim-home", sim.as_posix()],
            env=app_launch_env, capture_output=True, text=True, timeout=60)
    finally:
        held.close()
    assert done.returncode == instance_lock.ALREADY_RUNNING_EXIT, done.stderr
    assert "already running" in done.stderr and str(os.getpid()) in done.stderr
    assert sorted(path.name for path in sim.iterdir()) == ["app.lock"]


def test_calibration_is_loaded_from_the_home(home, tmp_path, monkeypatch):
    from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
    from autotrainer.inference import calibration_FLIR
    from types import SimpleNamespace

    import autotrainer.core.logging as core_logging
    from tools.acquisition import headless
    from tools.acquisition.model import app_model

    args = _parse("--sim-home", (tmp_path / "sim").as_posix())
    run = _prepare(args)
    assert run.calib_dir == run.home / DEFAULT_3D_CALIB_DIR_NAME
    # What reload_calib reads is there.
    for name in ("calibration_userset.yaml", "camera_matrix/stereo_params.pickle",
                 "camera_offsets.pkl"):
        assert run.calib_dir.joinpath(name).stat().st_size > 0, name
    assert calibration_FLIR.get_video_list(run.calib_dir.as_posix()) == [
        "20241029_agx001_session002_left", "20241029_agx001_session002_right"]

    built = []

    class _Stop(Exception):
        pass

    def fake_app_model(preferences, **kwargs):
        built.append(kwargs)
        raise _Stop

    monkeypatch.setattr(app_model, "AppModel", fake_app_model)
    monkeypatch.setattr(core_logging, "get_console_handler",
                        lambda: SimpleNamespace(setLevel=lambda _level: None))
    with pytest.raises(_Stop):
        headless._exec_main(args, calib_dir=run.calib_dir)
    assert built == [{"calib_dir": run.calib_dir}]
    assert Path(built[0]["calib_dir"]).resolve().is_relative_to(run.home)

    # --sim-reset puts back a calibration that was changed.
    run.calib_dir.joinpath("camera_offsets.pkl").write_bytes(b"edited")
    _prepare(_parse("--sim-home", (tmp_path / "sim").as_posix(), "--sim-reset"))
    assert run.calib_dir.joinpath("camera_offsets.pkl").read_bytes() != b"edited"


class _Stop(Exception):
    pass


def _quiet_console(monkeypatch, module):
    """No console handler is set up in a test; the window and run_acquisition set its level."""
    from types import SimpleNamespace

    monkeypatch.setattr(module, "get_console_handler",
                        lambda: SimpleNamespace(setLevel=lambda _level: None))


def _recording_app_model(built):
    def fake_app_model(*args, **kwargs):
        built.append((args, kwargs))
        raise _Stop
    return fake_app_model


@pytest.mark.parametrize("simulate", [True, False])
def test_main_window_builds_app_model_with_the_home_calibration(simulate, home, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from tools.acquisition.view import main_window

    app = QApplication.instance() or QApplication([])
    args = make_autotrainer_parser().parse_args(
        ["--simulate", "--sim-home", (tmp_path / "sim").as_posix()] if simulate
        else ["--preferences-file", (tmp_path / "rig.ini").as_posix()])
    run = _prepare(args) if simulate else None
    built = []
    monkeypatch.setattr(main_window, "AppModel", _recording_app_model(built))
    _quiet_console(monkeypatch, main_window)
    preferences = UserPreferences(settings_file_path=args.preferences_file)
    with pytest.raises(_Stop):
        main_window.MainWindow(app, preferences, args.configuration,
                               calib_dir=getattr(args, "sim_calib_dir", None))
    if simulate:
        assert args.sim_calib_dir == run.calib_dir
        assert built == [((preferences,), {"calib_dir": run.calib_dir})]
    else:
        # A rig run: the same bare call as before.
        assert getattr(args, "sim_calib_dir", None) is None
        assert built == [((preferences,), {})]


def test_run_acquisition_hands_the_sim_calibration_to_the_window(home, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import PySide6.QtWidgets
    from autotrainer.model import EnvironmentProvider
    from tools.acquisition import run_acquisition as run_module
    from tools.acquisition.view import disabled_style, main_window

    args = _parse("--sim-home", (tmp_path / "sim").as_posix())
    args.live_inference, args.dev = None, False
    run = _prepare(args)
    windows = []

    def fake_window(*window_args, **kwargs):
        windows.append(kwargs)
        raise _Stop

    monkeypatch.setattr(PySide6.QtWidgets, "QApplication", lambda argv: object())
    monkeypatch.setattr(disabled_style, "apply_disabled_style", lambda app: None)
    monkeypatch.setattr(EnvironmentProvider, "enable_can_emulation", lambda enabled: None)
    _quiet_console(monkeypatch, run_module)
    monkeypatch.setattr(run_module, "EventManager",
                        SimpleNamespace(default=lambda: SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(run_module, "try_register_api_event_plugin", lambda: None)
    monkeypatch.setattr(run_module.BehaviorAlgorithm, "close_algorithm_handler", lambda: None)
    monkeypatch.setattr(main_window, "MainWindow", fake_window)
    with pytest.raises(_Stop):
        run_module.run_acquisition(args)
    assert len(windows) == 1 and windows[0]["calib_dir"] == run.calib_dir


def test_calibration_pointer_source_is_refused(home, tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    for name in sim_home._CALIBRATION_FILES:
        target = fixture / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"version https://git-lfs.github.com/spec/v1" + bytes([10]) + b"oid sha256:00" + bytes([10]))
    (fixture / "source_videos").mkdir()
    monkeypatch.setattr(sim_home, "_CALIBRATION_FIXTURE", fixture)
    with pytest.raises(RuntimeError, match="git lfs checkout"):
        sim_home._copy_calibration(tmp_path / "out")
    assert not (tmp_path / "out").exists()
    with pytest.raises(RuntimeError, match="missing"):
        monkeypatch.setattr(sim_home, "_CALIBRATION_FIXTURE", tmp_path / "nowhere")
        sim_home._copy_calibration(tmp_path / "out")


def test_half_copied_calibration_is_repaired(home, tmp_path):
    sim = (tmp_path / "sim").as_posix()
    run = _prepare(_parse("--sim-home", sim))
    run.calib_dir.joinpath("camera_offsets.pkl").unlink()
    assert not sim_home._calibration_complete(run.calib_dir)
    _prepare(_parse("--sim-home", sim))
    assert run.calib_dir.joinpath("camera_offsets.pkl").stat().st_size > 0
    assert sim_home._calibration_complete(run.calib_dir)
