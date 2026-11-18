"""Prepare a `--simulate` run: its sim home, its environment and its defaults.

`prepare_simulation` runs in the GUI's and headless' main() right after the
arguments are parsed, before the instance lock, logging and AppModel. The NI
worker is a forkserver started in AppModel.__init__, and its children see only
the environment the forkserver started with, so every variable is set here.

Everything the run writes lands in the sim home (default ~/.reachaq-sim):

    system_configuration.yaml   generated on first run, kept after, --sim-reset regenerates
    preferences.ini             the same
    Data/                       session data and logs (persistence.outputLocation)
    animals/                    the animal registry
    runs/<utc>/signals.jsonl    this run's signal bus
    app.lock                    the instance lock, apart from a rig instance's

Protocols, stimulus profiles and the wiring record sit beside the
configuration, as they do on a rig.
"""

import dataclasses
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

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


@dataclasses.dataclass(frozen=True)
class SimulationRun:
    home: Path
    config_path: Path
    preferences_path: Path
    bus_path: Path
    lock_path: Path


def prepare_simulation(args) -> SimulationRun:
    """Make the sim home, set the simulation environment, and point args at the home.

    `args.configuration` and `args.preferences_file` are set to the home's
    files only when the user left them unset.
    """
    home = Path(getattr(args, "sim_home", None) or DEFAULT_SIM_HOME).expanduser().resolve()
    reset = bool(getattr(args, "sim_reset", False))
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

    if args.configuration is None:
        args.configuration = config_path
    if args.preferences_file is None:
        args.preferences_file = preferences_path
    args.allow_can_emulation = True

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
    )


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

    # Where AppModel opens its log before a configuration's outputLocation is loaded.
    PersistenceConfiguration.DEFAULT_OUTPUT_PATH = home / "Data"
    AppModel.status_file_path = home / RUNNING_STATUS_NAME
    MotorConfigurationFile.DEFAULT_LOCATION = home / _MOTOR_FILES[0]
    CompoundMovements.DEFAULT_LOCATION = home / _MOTOR_FILES[1]
    # In case it was imported before the variable was set.
    DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH = Path(os.environ[ENV_DIAMOND_TRIANGLE_CONFIG])
