from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from autotrainer.core.logging import (
    ALWAYS_CONSOLE_LOG_ATTRIBUTE,
    get_verbose_logger,
)
from autotrainer.device.rfid_reader import RfidReaderService, RfidTagRead

from .animal_registry import AnimalRegistry


logger = get_verbose_logger(__name__)


def _qt_dispatcher():
    try:
        from PySide6.QtCore import QObject, Qt, Signal
    except ImportError:
        return None, None

    class Relay(QObject):
        requested = Signal(object)

        def __init__(self):
            super().__init__()
            self.requested.connect(self._run, Qt.ConnectionType.QueuedConnection)

        @staticmethod
        def _run(callback):
            callback()

    relay = Relay()
    return relay, relay.requested.emit


class RfidMetadataController:
    """Thin scan-path coordinator; performs no file or network access."""

    def __init__(
        self,
        *,
        app_model,
        registry: AnimalRegistry,
        dispatch: Optional[Callable[[Callable[[], None]], None]] = None,
        reader_factory=RfidReaderService,
        cache_stale_after_seconds: float = 48 * 60 * 60,
        **reader_options,
    ):
        self.app_model = app_model
        self.registry = registry
        self._relay = None
        if dispatch is None:
            self._relay, dispatch = _qt_dispatcher()
        self.dispatch = dispatch or (lambda callback: callback())
        self.last_resolution = None
        self.reader_status = None
        self.cache_stale_after_seconds = float(cache_stale_after_seconds)
        self.reader = reader_factory(
            on_tag=self._on_tag,
            on_status=self._on_status,
            **reader_options,
        )

    def _on_status(self, status) -> None:
        self.reader_status = status
        state = getattr(getattr(status, "state", None), "value", "unknown")
        status_logger = (
            logger.error
            if state == "failed"
            else logger.warning
            if state == "reconnecting"
            else logger.info
        )
        status_logger(
            "RFID reader status: state=%s device=%s reason=%s",
            state,
            getattr(status, "device", self.reader.device),
            getattr(status, "reason", "") or "none",
        )
        callback = getattr(self.app_model, "set_rfid_reader_status", None)
        if callback is not None:
            self.dispatch(lambda: callback(status))

    def _on_tag(self, event: RfidTagRead) -> None:
        # The registry is local SQLite and safe to query on the reader thread;
        # all AppModel mutation is handed to the configured GUI dispatcher.
        logger.info(
            "RFID scan received: rfid=%s device=%s",
            event.rfid,
            self.reader.device,
            extra={ALWAYS_CONSOLE_LOG_ATTRIBUTE: True},
        )
        try:
            record = self.registry.resolve_rfid(event.rfid)
            ledger = self.registry.last_complete_import()
        except Exception:
            logger.exception("RFID registry lookup failed: rfid=%s", event.rfid)
            raise
        cache_stale = True
        if ledger is not None:
            try:
                imported = datetime.fromisoformat(
                    ledger["imported_utc"].replace("Z", "+00:00")
                )
                cache_stale = (
                    datetime.now(timezone.utc) - imported
                ).total_seconds() > self.cache_stale_after_seconds
            except (TypeError, ValueError):
                cache_stale = True

        logger.info(
            "RFID registry lookup: rfid=%s matched=%s subject_id=%s "
            "cache_stale=%s import_id=%s",
            event.rfid,
            record is not None,
            None if record is None else record.identity.subject_id,
            cache_stale,
            None if ledger is None else ledger["import_id"],
        )

        def resolve() -> None:
            handler = getattr(self.app_model, "handle_rfid_scan", None)
            if handler is None:
                handler = self.app_model.resolve_external_record
            try:
                self.last_resolution = handler(
                    record,
                    scanned_rfid=event.rfid,
                    registry_import_id=None if ledger is None else ledger["import_id"],
                    source_file_sha256=(
                        None if ledger is None else ledger["source_file_sha256"]
                    ),
                    imported_utc=None if ledger is None else ledger["imported_utc"],
                    cache_stale=cache_stale,
                )
            except Exception:
                logger.exception("RFID scan resolution failed: rfid=%s", event.rfid)
                raise

        self.dispatch(resolve)

    def start(self) -> None:
        logger.info("Starting RFID reader: device=%s", self.reader.device)
        self.reader.start()

    def stop(self) -> None:
        logger.info("Stopping RFID reader: device=%s", self.reader.device)
        self.reader.stop()
