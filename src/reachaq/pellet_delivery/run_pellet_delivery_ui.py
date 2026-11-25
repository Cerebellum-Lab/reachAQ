import logging
import sys

from PySide6.QtWidgets import QApplication

from reachaq.model import EnvironmentProvider
from reachaq.pellet_delivery.model.app_model import AppModel
from reachaq.pellet_delivery.view.main_window import MainWindow

logger = logging.getLogger(__name__)


def set_log_location():
    from reachaq.core.logging import get_log_file_location
    log_file = get_log_file_location(
        full_format=f"{{log_location}}/pellet_ui/{{date_stamp}}_{{idx:03d}}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s\t [%(name)s] %(message)s"))
    logging.root.addHandler(file_handler)


def run_pellet_delivery_ui(allow_can_emulation: bool = False) -> int:
    set_log_location()

    app = QApplication(sys.argv)

    EnvironmentProvider.enable_can_emulation(allow_can_emulation)

    model = AppModel()

    window = MainWindow(app, model)

    window.show()

    window.on_activated()

    return app.exec()
