"""Prepare a `--simulate` run: its sim home, its environment and its defaults.

`prepare_simulation` runs in the GUI's and headless' main() right after the
arguments are parsed, before logging and AppModel. It takes the home's
instance lock before it writes anything, in place of the rig's lock. The NI
worker is a forkserver started in AppModel.__init__, and its children see only
the environment the forkserver started with, so every variable is set here.

Everything the run writes lands in the sim home (default ~/.reachaq-sim):

    system_configuration.yaml   generated on first run, kept after, --sim-reset regenerates
    preferences.ini             the same
    4mm_6r_8c_4x/               the 3D calibration, from the repository's test fixture; the same
    Data/                       session data and logs (persistence.outputLocation)
    animals/                    the animal registry
    runs/<utc>/signals.jsonl    this run's signal bus
    app.lock                    the instance lock, apart from a rig instance's

Protocols, stimulus profiles and the wiring record sit beside the
configuration, as they do on a rig.

The class-attribute redirections (_redirect_user_defaults) reach the main
process only; only what is carried in the environment reaches spawned children.
"""

import dataclasses
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional, Tuple

DEFAULT_SIM_HOME = Path("~/.reachaq-sim")

ENV_SIMULATION = "REACHAQ_SIMULATION"
ENV_SIM_BUS = "REACHAQ_SIM_BUS"
ENV_SIM_EPOCH = "REACHAQ_SIM_EPOCH"
ENV_CAN_TRANSPORT = "AUTOTRAINER_CAN_TRANSPORT"
ENV_SIM_CAMERA_FPS = "REACHAQ_SIM_CAMERA_FPS"
ENV_SIM_CAMERA_EXPOSURE_S = "REACHAQ_SIM_CAMERA_EXPOSURE_S"
ENV_SIM_LINE_ALIASES = "REACHAQ_SIM_LINE_ALIASES"
#: Read once, when autotrainer.core.diamond_triangle_config is imported.
ENV_DIAMOND_TRIANGLE_CONFIG = "AUTOTRAINER_DIAMOND_TRIANGLE_CONFIG"

#: Every variable prepare_simulation sets.
SIMULATION_ENVIRONMENT: Tuple[str, ...] = (
    ENV_SIMULATION, ENV_SIM_BUS, ENV_SIM_EPOCH, ENV_CAN_TRANSPORT,
    ENV_SIM_CAMERA_FPS, ENV_SIM_CAMERA_EXPOSURE_S, ENV_SIM_LINE_ALIASES,
    ENV_DIAMOND_TRIANGLE_CONFIG,
)

PREFERENCES_NAME = "preferences.ini"
LOCK_NAME = "app.lock"
#: Beside the operator's preferences on a rig (~/.config/Colorado).
RUNNING_STATUS_NAME = "autotrainer_running_status.env"
#: The motor and move files a rig keeps in ~/Autotrainer, seeded from the
#: repository's copies so the emulated board gets the same motor setup.
_MOTOR_FILES = ("motor_config.yaml", "move_config.yaml")
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: autotrainer.core.configuration.DEFAULT_3D_CALIB_DIR_NAME, which cannot be
#: imported before the diamond-triangle variable is set.
CALIBRATION_NAME = "4mm_6r_8c_4x"
_CALIBRATION_FIXTURE = _REPO_ROOT / "auto-trainer-inference" / "tests" / CALIBRATION_NAME
#: What a checkout without its LFS objects holds in place of each fixture file.
_LFS_POINTER_PREFIX = b"version https://git-lfs"
#: What AppModel.reload_calib reads. Of source_videos it lists only the names.
_CALIBRATION_FILES = ("calibration_userset.yaml", "camera_matrix/stereo_params.pickle", "camera_offsets.pkl")


@dataclasses.dataclass(frozen=True)
class SimulationRun:
    home: Path
    config_path: Path
    preferences_path: Path
    bus_path: Path
    lock_path: Path
    #: The 3D calibration AppModel loads, in place of ~/Autotrainer's.
    calib_dir: Path
    #: The held instance lock; keep it for the process's life. None where
    #: flock does not exist.
    instance_lock: Optional[IO[str]]


def prepare_simulation(args) -> SimulationRun:
    """Lock the sim home, fill it, set the simulation environment, and point args at it.

    `args.configuration` and `args.preferences_file` are set to the home's
    files only when the user left them unset. Raises
    tools.acquisition.instance_lock.AlreadyRunning, having written nothing,
    while another simulation holds the home.
    """
    from tools.acquisition.instance_lock import acquire_instance_lock

    home = Path(getattr(args, "sim_home", None) or DEFAULT_SIM_HOME).expanduser().resolve()
    reset = bool(getattr(args, "sim_reset", False))
    # Before any file is written: a second run must not regenerate the
    # configuration under a running one. The lock makes only the home's parent.
    home.parent.mkdir(parents=True, exist_ok=True)
    instance_lock = acquire_instance_lock(home / LOCK_NAME)
    try:
        return _prepare_locked(args, home, reset, instance_lock)
    except BaseException:
        if instance_lock is not None:
            instance_lock.close()
        raise


def _prepare_locked(args, home: Path, reset: bool, instance_lock) -> SimulationRun:
    for directory in (home, home / "Data", home / "animals"):
        directory.mkdir(parents=True, exist_ok=True)
    # Before autotrainer.core is imported: it imports diamond_triangle_config,
    # which reads this once.
    os.environ[ENV_DIAMOND_TRIANGLE_CONFIG] = home.joinpath("diamond_triangle_offset.yaml").as_posix()

    from autotrainer.core import SystemConfiguration

    config_path = SystemConfiguration.make_default_yaml_config_path(home)
    preferences_path = home / PREFERENCES_NAME
    if reset or not config_path.is_file():
        from tools.simulation.sim_configuration import build_simulation_configuration
        build_simulation_configuration(home).save_default(home)
    if reset or not preferences_path.is_file():
        _write_preferences(home, preferences_path)
    for name in _MOTOR_FILES:
        source, target = _REPO_ROOT / "tests" / name, home / name
        if source.is_file() and (reset or not target.is_file()):
            shutil.copyfile(source, target)
    calib_dir = home / CALIBRATION_NAME
    if reset or not _calibration_complete(calib_dir):
        _copy_calibration(calib_dir)

    if args.configuration is None:
        args.configuration = config_path
    if args.preferences_file is None:
        args.preferences_file = preferences_path
    args.allow_can_emulation = True
    # The GUI builds AppModel in MainWindow, so the calibration rides on args.
    args.sim_calib_dir = calib_dir

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = home / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    bus_path = run_dir / "signals.jsonl"

    fps, exposure_us = _primary_camera_timing(Path(args.configuration))
    os.environ.update({
        ENV_SIMULATION: "1",
        ENV_SIM_BUS: bus_path.as_posix(),
        ENV_SIM_EPOCH: repr(time.perf_counter()),
        ENV_CAN_TRANSPORT: "emulation",
        ENV_SIM_CAMERA_FPS: repr(float(fps)),
        ENV_SIM_CAMERA_EXPOSURE_S: repr(float(exposure_us) * 1e-6),
        # {nidaq channel: bus line}; no channel reads a line under another name yet.
        ENV_SIM_LINE_ALIASES: "{}",
    })
    _redirect_user_defaults(home)
    return SimulationRun(
        home=home,
        config_path=config_path,
        preferences_path=preferences_path,
        bus_path=bus_path,
        lock_path=home / LOCK_NAME,
        calib_dir=calib_dir,
        instance_lock=instance_lock,
    )


def _calibration_complete(target: Path) -> bool:
    """Every file AppModel loads is there, so a half-copied directory is redone."""
    return (target / "source_videos").is_dir() and all(
        (target / name).is_file() for name in _CALIBRATION_FILES)


def _check_calibration_source() -> None:
    for name in _CALIBRATION_FILES:
        source = _CALIBRATION_FIXTURE / name
        if not source.is_file():
            raise RuntimeError(f"the calibration fixture {source} is missing")
        with source.open("rb") as handle:
            if handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                raise RuntimeError(
                    f"{source} is a Git LFS pointer, not the file; run `git lfs checkout` "
                    "in the repository and start the simulation again")


def _copy_calibration(target: Path) -> None:
    """The repository's test calibration, as much of it as AppModel loads.

    The source videos are 13 MB that reload_calib only lists by name, so
    they are empty files here. The copy is made beside the target and renamed
    into place, so an interrupted copy never looks complete.
    """
    _check_calibration_source()
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    for name in _CALIBRATION_FILES:
        destination = staging / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_CALIBRATION_FIXTURE / name, destination)
    videos = staging / "source_videos"
    videos.mkdir()
    for video in sorted((_CALIBRATION_FIXTURE / "source_videos").glob("*.mp4")):
        videos.joinpath(video.name).touch()
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def _write_preferences(home: Path, path: Path) -> None:
    """A fresh preferences file whose every location is in the home."""
    from tools.acquisition.model.user_preferences import UserPreferences

    path.unlink(missing_ok=True)
    preferences = UserPreferences(settings_file_path=path)
    preferences.configuration_location = home.as_posix()
    preferences.animal_location = home.joinpath("animals").as_posix()
    preferences.log_location = home.joinpath("Data").as_posix()
    # The default manifest is on the lab share; a simulation never syncs it.
    preferences.softmouse_manifest_path = home.joinpath(
        "softmouse", "SoftMouse-AnimalList-current.manifest.json").as_posix()
    preferences.softmouse_nightly_refresh = False
    preferences.save()


def _primary_camera_timing(config_path: Path) -> Tuple[float, float]:
    """The primary camera's fps and exposure (µs) in the configuration the run loads.

    The simulated NI stream draws cam_frames on that camera's grid. Falls back
    to the generated configuration's values when the file cannot be read; the
    app's own load then reports why.
    """
    from autotrainer.core import SystemConfiguration
    from tools.simulation.sim_configuration import CAMERA_EXPOSURE_US, CAMERA_FPS

    fps, exposure_us = float(CAMERA_FPS), float(CAMERA_EXPOSURE_US)
    if config_path.is_dir():
        config_path = SystemConfiguration.make_default_yaml_config_path(config_path)
    try:
        configuration: Optional[SystemConfiguration] = SystemConfiguration.load_yaml_file(config_path)
    except Exception:
        return fps, exposure_us
    for camera in configuration.cameras:
        params = camera.params or {}
        if str(params.get("primary", "")).lower() in {"true", "yes", "on", "1"}:
            fps = float(params.get("fps", fps))
            exposure_us = float(params.get("exposure", exposure_us))
            break
    return fps, exposure_us


def _redirect_user_defaults(home: Path) -> None:
    """Move the app's built-in user-data defaults into the home, for this process.

    These are the class attributes tests/top_fixtures.py replaces for the
    same reason: each names a file or folder of the operator's that the app
    reads or writes before, or regardless of, any configuration. The
    configuration and preferences cover the rest.
    """
    from autotrainer.core import PersistenceConfiguration
    from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
    from autotrainer.device.compound_movement_file import CompoundMovements
    from autotrainer.device.motor_configuration_file import MotorConfigurationFile
    from tools.acquisition.model.app_model import AppModel

    # AppModel's output location until a configuration is loaded. The
    # output_location field's default was fixed at import and is unchanged;
    # the generated configuration names its own.
    PersistenceConfiguration.DEFAULT_OUTPUT_PATH = home / "Data"
    AppModel.status_file_path = home / RUNNING_STATUS_NAME
    MotorConfigurationFile.DEFAULT_LOCATION = home / _MOTOR_FILES[0]
    CompoundMovements.DEFAULT_LOCATION = home / _MOTOR_FILES[1]
    # In case it was imported before the variable was set.
    DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH = Path(os.environ[ENV_DIAMOND_TRIANGLE_CONFIG])
