import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QPlainTextEdit


class QSignaler(QObject):
    log_message = Signal(str)


class TextBoxHandler(logging.Handler):
    def __init__(self, text_edit: QPlainTextEdit):
        super().__init__()
        self.emitter = QSignaler()
        self.text_edit = text_edit
        self.emitter.log_message.connect(self.text_edit.appendPlainText)

    def emit(self, record):
        self.emitter.log_message.emit(self.format(record))
