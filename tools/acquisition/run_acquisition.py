import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6 import QtGui

from autotrainer.core import EventManager, ApiEventKind
from autotrainer.core.event import try_register_api_event_plugin
from autotrainer.core.logging import (get_verbose_logger, get_console_handler, set_log_location)
from autotrainer.pyside import CardHeader

from autotrainer.behavior import BehaviorAlgorithm

logger = get_verbose_logger(__name__)

missing_file = "The configuration file {0} does not exist.  A default configuration will be loaded."

CardHeader.DEFAULT_BACKGROUND_COLOR = "#cfb87c"
CardHeader.DEFAULT_TITLE_COLOR = "black"


def verify_configuration(configuration: Optional[Path]):
    from PySide6.QtWidgets import QMessageBox

    if configuration is not None and not configuration.exists():
        # noinspection PyTypeChecker
        result = QMessageBox.warning(None, "Configuration File not Found", missing_file.format(configuration),
                                     QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Close)
        return result == QMessageBox.StandardButton.Ok

    return True


def run_acquisition(
    args,  # see tools.acquisition.args
) -> int:
    from PySide6.QtWidgets import QApplication

    from autotrainer.model import EnvironmentProvider

    from tools.autotrainer_version import __version__ as app_version
    from tools.acquisition.model.user_preferences import UserPreferences
    from tools.acquisition.args import AutoTrainerParsedArgs
    from tools.acquisition.view.main_window import MainWindow

    args: AutoTrainerParsedArgs

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if not verify_configuration(args.configuration):
        return -1

    EnvironmentProvider.enable_can_emulation(args.allow_can_emulation)

    preferences = UserPreferences(settings_file_path=args.preferences_file)

    logging.info("Set log level to %s", preferences.log_level)
    get_console_handler().setLevel(preferences.log_level)

    event_manager = EventManager.default()
    plugin = try_register_api_event_plugin()

    try:
        window = MainWindow(
            app,
            preferences,
            args.configuration,
            is_dev=args.dev,
            random_cameras=args.random_cameras,
            live_inference=args.live_inference,
        )
    except:
        event_manager.close()
        BehaviorAlgorithm.close_algorithm_handler()
        raise

    if plugin is not None:
        window.app_model.rpc_service = plugin.service

    # conveniently allow close/exit app with SIGINT (ctrl-c) :
    sigint_received = 0
    def handle_sigint(signum, frame):
        nonlocal sigint_received
        sigint_received += 1
        logger.notice("Got signal %s ; closing window..", signum)
        window.close()
        if sigint_received > 2:
            logger.critical("too many sigint, exiting with SIG_TERMINATE ..")
            time.sleep(0.5)
            os.kill(-os.getpid(), signal.SIGTERM)
            # killing negative of pid is killing process group

    signal.signal(signal.SIGINT, handle_sigint)

    window.show()
    # window.showMaximized()
    window.move(QtGui.QGuiApplication.primaryScreen().availableGeometry().center() - window.rect().center())

    try:
        window.on_activated(target_status=args.start_mode)
    except:
        event_manager.close()
        BehaviorAlgorithm.close_algorithm_handler()
        raise

    event_manager.post_event_content(ApiEventKind.applicationLaunched, data=dict(version=app_version))

    logger.info("Executing app now ..")
    try:
        return app.exec()
    finally:
        # could move somewhere in main_window.on_close eventually:
        event_manager.post_event_content(
            ApiEventKind.applicationTerminating, dict(reason="exit-requested")
        )
        logger.verbose("Closing event manager and behavior algo thread handler..")
        # close first the behavior algo handler thread,
        BehaviorAlgorithm.close_algorithm_handler()
        # given it can push some events to the event manager,
        # then only close event manager:
        event_manager.close()
