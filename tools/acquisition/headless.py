import logging
import os
import time
import sys
import faulthandler
from multiprocessing import set_start_method
from pathlib import Path
# NB: do not put any imports of autotrainer* or any module not part from standard python lib.


def _bootstrap_repo_sources():
    """Prefer packages beside this launcher over stale editable installs."""
    repo_root = Path(__file__).resolve().parents[2]
    source_roots = (
        repo_root,
        *(repo_root / package / "src" for package in (
            "auto-trainer-behavior",
            "auto-trainer-core",
            "auto-trainer-device",
            "auto-trainer-inference",
            "auto-trainer-model",
            "auto-trainer-pyside",
            "auto-trainer-video",
        )),
    )
    for source_root in reversed(source_roots):
        if not source_root.exists():
            continue
        source_text = str(source_root)
        if source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)


_bootstrap_repo_sources()


def _exec_main(args):

    from autotrainer.core.logging import get_verbose_logger, get_console_handler
    from autotrainer.core.event import try_register_api_event_plugin
    from tools.acquisition.model.user_preferences import UserPreferences
    from tools.acquisition.model.helpers import get_config_location

    from tools.acquisition.model.app_model import AppModel

    logger = get_verbose_logger("autotrainer.headless")

    configuration = args.configuration
    if configuration and not os.path.exists(configuration):
        logger.error("Provided configuration location does not exist: %s",
                     configuration)
        return -1

    preferences = UserPreferences(settings_file_path=args.preferences_file)
    config_file = get_config_location(preferences, configuration)

    get_console_handler().setLevel(preferences.log_level)

    app_model = AppModel(preferences)

    plugin = try_register_api_event_plugin()
    app_model.rpc_service = plugin.service

    try:
        app_model.load_configuration(config_file, random_cameras=args.random_cameras)
    except Exception as err:
        logger.exception("Could not load config: %s", err)
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
    from tools.acquisition.model.app_model_status import AppModelStatus
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

    from tools.acquisition.args import make_autotrainer_parser

    # Headless operation still starts acquisition by default; an idle headless
    # process has no UI from which acquisition can subsequently be started.
    from tools.acquisition.model.app_model_status import AppModelStatus
    parser = make_autotrainer_parser(default_start_mode=AppModelStatus.RUNNING)

    args = parser.parse_args()

    from autotrainer.core.logging import setup_logging, stop_multiproc_logging

    logger = setup_logging(logger_level=logging.DEBUG, time_precision=6, multiprocess_enabled=True)

    try:
        return _exec_main(args)
    except KeyboardInterrupt:
        logger.notice("KeyboardInterrupt")
        exit_code = 0
    except Exception as err:
        logger.exception("Fatal error: %s", err)
        exit_code = 1
    finally:
        from autotrainer.core.event.event_manager import EventManager
        from autotrainer.behavior import BehaviorAlgorithm
        BehaviorAlgorithm.close_algorithm_handler()
        EventManager.try_close_default()
        stop_multiproc_logging()
    #
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
