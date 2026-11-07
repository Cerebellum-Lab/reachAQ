import logging
import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Slot  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.view.status_log_handler import StatusLogHandler  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.messages = []

    @Slot(str)
    def receive(self, message: str) -> None:
        self.messages.append(message)


def test_error_log_is_forwarded_to_gui_thread_status_callback(qapp):
    receiver = _Receiver()
    handler = StatusLogHandler(receiver.receive)
    test_logger = logging.getLogger("reachaq.test.status_log_handler")
    previous_handlers = list(test_logger.handlers)
    previous_level = test_logger.level
    previous_propagate = test_logger.propagate
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.ERROR)
    test_logger.propagate = False
    try:
        thread = threading.Thread(
            target=test_logger.error,
            args=("NI-DAQ worker failed\nfull details remain in the log",),
        )
        thread.start()
        thread.join()

        deadline = time.monotonic() + 2.0
        while not receiver.messages and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)

        assert receiver.messages == ["Error: NI-DAQ worker failed"]
    finally:
        test_logger.handlers = previous_handlers
        test_logger.setLevel(previous_level)
        test_logger.propagate = previous_propagate
        handler.close()
