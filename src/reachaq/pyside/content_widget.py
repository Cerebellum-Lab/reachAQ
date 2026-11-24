import functools
import threading
from typing import Callable

from PySide6.QtWidgets import QWidget

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QGuiApplication


class InvokeMethod(QObject):

    def __init__(self, method: Callable, *args, **kwargs):
        """
        Invokes a method on the main thread. Taking care of garbage collection "bugs".
        """
        super().__init__()

        if threading.current_thread() is threading.main_thread():
            method(*args, **kwargs)
            return

        app = QGuiApplication.instance()
        main_qthread = app.thread()
        self.moveToThread(main_qthread)
        self.setParent(app)
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.called.connect(self.execute)
        self.called.emit()

    called = Signal()

    @Slot()
    def execute(self):
        try:
            self.method(*self.args, **self.kwargs)
        finally:
            # ensure we don't keep a ref to any of these:
            self.args = self.kwargs = self.method = None
            # then trigger garbage collector
            self.setParent(None)


def invoke_method(func):
    """Allow to decorate/wrap a function to invoke it in main UI thread"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        InvokeMethod(func, *args, **kwargs)

    return wrapped


class ContentWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)

    def set_is_editable(self, is_editable: bool):
        pass

    def set_is_capture_active(self, is_active: bool):
        pass

    def on_activated(self):
        pass

    def on_close(self):
        pass
