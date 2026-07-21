from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QObject, Signal


class _StatusLogEmitter(QObject):
    message_received = Signal(str)


class StatusLogHandler(logging.Handler):
    """Forward error log records to a Qt callback on the GUI thread."""

    def __init__(self, callback: Callable[[str], None], *, maximum_length: int = 320):
        super().__init__(level=logging.ERROR)
        self._maximum_length = maximum_length
        self._emitter = _StatusLogEmitter()
        self._emitter.message_received.connect(callback)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage().strip()
            summary = next((line.strip() for line in message.splitlines() if line.strip()), "Unknown error")
            if len(summary) > self._maximum_length:
                summary = summary[: self._maximum_length - 1].rstrip() + "…"
            self._emitter.message_received.emit(f"Error: {summary}")
        except Exception:
            self.handleError(record)
