import logging
import os
import time
import sys
import faulthandler
from multiprocessing import set_start_method
from pathlib import Path
# NB: do not put any imports of reachaq* or any module not part from standard python lib.


def _bootstrap_repo_sources():
    """Prefer packages beside this launcher over stale editable installs."""
    repo_root = Path(__file__).resolve().parents[3]
    source_roots = (repo_root, repo_root / "src")
    for source_root in reversed(source_roots):
        if not source_root.exists():
            continue
        source_text = str(source_root)
        if source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)


_bootstrap_repo_sources()


def _exec_main(args, calib_dir=None):

    from reachaq.core.logging import get_verbose_logger, get_console_handler
    from reachaq.core.event import try_register_api_event_plugin
    from reachaq.app.model.user_preferences import UserPreferences
    from reachaq.app.model.helpers import get_config_location

    from reachaq.app.model.app_model import AppModel

    logger = get_verbose_logger("reachaq.headless")

    configuration = args.configuration
    if configuration and not os.path.exists(configuration):
        logger.error("Provided configuration location does not exist: %s",
                     configuration)
        return -1

    preferences = UserPreferences(settings_file_path=args.preferences_file)
    config_file = get_config_location(preferences, configuration)

    get_console_handler().setLevel(preferences.log_level)

    # A simulation's calibration is in its home; a rig run passes none, as before.
    if calib_dir is None:
        app_model = AppModel(preferences)
    else:
        app_model = AppModel(preferences, calib_dir=calib_dir)

    plugin = try_register_api_event_plugin()
    app_model.rpc_service = plugin.service
    # There is no Edit DAQ Ports here: a refused NI-DAQ plan is fixed in the
    # file, at the field each refusal names.
    app_model.nidaq_plan_remedy = f"fix the field it names in {config_file}"

    try:
        app_model.load_configuration(config_file, random_cameras=args.random_cameras)
    except Exception as err:
        logger.exception("Could not load config: %s", err)
        app_model.on_close()
        return 1
    # The GUI loads a line the NI-DAQ stream cannot acquire and holds the
    # stream until Edit DAQ Ports fixes it. There is no dialog here, so this
    # refuses to run on it, as every load did before.
    if app_model.nidaq_plan_error:
        logger.error("Could not load config: NI-DAQ plan refused: %s",
                     app_model.nidaq_plan_error)
        app_model.on_close()
        return 1

    if args.live_inference is not None:
        app_model.set_runtime_live_inference_override(args.live_inference)

    target_status = args.start_mode
    if not app_model.capture_start(target_status=target_status):
        logger.error("failed to start capture")
        app_model.on_close()
        return 1

    logger.success("App is now running")

    app_model.on_activated()

    exit_rc = 1
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting..")
        exit_rc = 0
    except Exception as err:
        logger.exception("Fatal error: %s", err)

    app_model.on_close()

    return exit_rc


def parse_start_mode(value: str):
    from reachaq.app.model.app_model_status import AppModelStatus
    try:
        return AppModelStatus(value.lower())  # values are lower, so force it
    except ValueError:
        pass
    try:
        return getattr(AppModelStatus, value.upper())  # names are upper, so force it
    except AttributeError:
        pass
    raise ValueError(f"Unknown AppModelStatus: {value!r}")


def main():
    faulthandler.enable()
    set_start_method("spawn")

    # must be AFTER set_start_method before:

    from reachaq.app.args import make_reachaq_parser

    # Headless operation still starts acquisition by default; an idle headless
    # process has no UI from which acquisition can subsequently be started.
    from reachaq.app.model.app_model_status import AppModelStatus
    parser = make_reachaq_parser(default_start_mode=AppModelStatus.RUNNING)

    args = parser.parse_args()

    # After argument parsing, so -h still answers while an instance runs, and
    # before logging, so a refused start leaves no log behind.
    from reachaq.app.instance_lock import (
        ALREADY_RUNNING_EXIT, AlreadyRunning, acquire_instance_lock)
    simulation = None
    try:
        if args.simulate:
            # Takes the lock in its own home before writing there, and runs
            # before AppModel starts the forkserver, which keeps the
            # environment it starts with.
            from reachaq.simulation.sim_home import prepare_simulation
            simulation = prepare_simulation(args)
            instance_lock = simulation.instance_lock  # noqa: F841 - held until exit
        else:
            instance_lock = acquire_instance_lock()  # noqa: F841 - held until exit
    except AlreadyRunning as err:
        print(err, file=sys.stderr)
        return ALREADY_RUNNING_EXIT

    from reachaq.core.logging import setup_logging, stop_multiproc_logging

    # BEFORE setup_logging: after it, a level set in this process no longer takes effect.
    from reachaq.app.can_frame_logging import limit_can_frame_logging, log_can_frame_logging
    can_frame_logging = limit_can_frame_logging()

    logger = setup_logging(logger_level=logging.DEBUG, time_precision=6, multiprocess_enabled=True)
    log_can_frame_logging(logger, can_frame_logging)

    try:
        return _exec_main(args, calib_dir=simulation.calib_dir if simulation else None)
    except KeyboardInterrupt:
        logger.notice("KeyboardInterrupt")
        exit_code = 0
    except Exception as err:
        logger.exception("Fatal error: %s", err)
        exit_code = 1
    finally:
        from reachaq.core.event.event_manager import EventManager
        from reachaq.behavior import BehaviorAlgorithm
        BehaviorAlgorithm.close_algorithm_handler()
        EventManager.try_close_default()
        stop_multiproc_logging()
    #
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
