from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from autotrainer.core.logging import get_verbose_logger

from .animal_registry import AnimalRegistry, RegistryImportResult
from .animal_sources.config import ConfiguredSource
from .animal_sources.mapping import SpreadsheetImportPreview
from .animal_sources.publication import (
    AnimalTablePublisher,
    PublicationResult,
    ensure_publication_directory,
)


logger = get_verbose_logger(__name__)


@dataclass(frozen=True)
class MetadataRefreshResult:
    preview: SpreadsheetImportPreview
    registry_result: RegistryImportResult
    publication: Optional[PublicationResult] = None


@dataclass(frozen=True)
class MetadataCacheStatus:
    configured: bool
    last_import_utc: Optional[str] = None
    source_file_sha256: Optional[str] = None
    total_source_rows: int = 0
    tagged_rows: int = 0
    ignored_missing_rfid_rows: int = 0
    ignored_ended_rows: int = 0
    age_seconds: Optional[float] = None


class AnimalMetadataSyncService:
    """Fetch the configured source and replace the local cache without partial reads.

    With a publish target, a validated snapshot is also written as the shared
    publication before the local cache changes.
    """

    def __init__(
        self,
        *,
        configured: ConfiguredSource,
        local_staging_directory: Path,
        registry: AnimalRegistry,
        can_refresh: Callable[[], bool],
        publish_directory: Optional[Path] = None,
        publish_mount: Optional[Path] = None,
    ):
        self.configured = configured
        self.source = configured.source
        self.local_staging_directory = Path(local_staging_directory)
        self.registry = registry
        self.can_refresh = can_refresh
        self.publish_directory = None if publish_directory is None else Path(publish_directory)
        self.publish_mount = None if publish_mount is None else Path(publish_mount)
        self._lock = threading.Lock()

    def describe(self) -> str:
        return self.source.describe()

    def refresh_now(self) -> MetadataRefreshResult:
        if not self.can_refresh():
            logger.warning("Animal metadata refresh blocked: session is active")
            raise RuntimeError("Animal metadata refresh is disabled while a session is active")
        if not self._lock.acquire(blocking=False):
            logger.warning("Animal metadata refresh blocked: refresh already running")
            raise RuntimeError("An animal metadata refresh is already running")
        try:
            logger.info(
                "Animal metadata refresh started: source=%s staging_directory=%s",
                self.describe(),
                self.local_staging_directory,
            )
            work = self.local_staging_directory / "fetch"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True, exist_ok=True)
            try:
                table = self.source.fetch(work)
                mapper = self.configured.mapper_for(table.mapping, table.provider)
                previous = self.registry.last_complete_import()
                previous_rows = (
                    None
                    if previous is None or previous["provider"] != mapper.provider
                    else previous["total_source_rows"]
                )
                preview = mapper.preview_table(
                    table, previous_source_row_count=previous_rows
                )
                batch = preview.batch
                if (
                    table.expected_total_rows is not None
                    and batch.total_source_rows != int(table.expected_total_rows)
                ):
                    raise ValueError(
                        "Parsed source row count differs from publication manifest"
                    )
                if (
                    table.expected_accepted_rows is not None
                    and batch.accepted_rows != int(table.expected_accepted_rows)
                ):
                    raise ValueError(
                        "Parsed accepted row count differs from publication manifest"
                    )
                if not self.can_refresh():
                    raise RuntimeError(
                        "Animal metadata refresh was cancelled because a session became active"
                    )
                publication = None
                if self.publish_directory is not None:
                    destination = ensure_publication_directory(
                        self.publish_directory, mount_root=self.publish_mount
                    )
                    publication = AnimalTablePublisher(destination).publish(table, mapper)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            registry_result = self.registry.replace(batch)
            logger.info(
                "Animal metadata refresh validated and committed: import_id=%s "
                "rows=%d accepted=%d unchanged=%s source_sha256=%s published=%s",
                batch.import_id,
                batch.total_source_rows,
                batch.accepted_rows,
                registry_result.unchanged,
                batch.source_file_sha256,
                None if publication is None else publication.manifest_path,
            )
            return MetadataRefreshResult(preview, registry_result, publication)
        except Exception:
            logger.exception(
                "Animal metadata refresh transaction failed: source=%s", self.describe()
            )
            raise
        finally:
            self._lock.release()

    def refresh_due(self) -> bool:
        version = self.source.current_version()
        previous = self.registry.last_complete_import()
        due = (
            version is None
            or previous is None
            or previous["source_file_sha256"].casefold() != version.casefold()
        )
        logger.info(
            "Animal metadata catch-up check: due=%s source_version=%s local_sha256=%s",
            due,
            version,
            None if previous is None else previous["source_file_sha256"],
        )
        return due

    def cache_status(
        self, *, now: Optional[datetime] = None
    ) -> MetadataCacheStatus:
        previous = self.registry.last_complete_import()
        if previous is None:
            return MetadataCacheStatus(configured=True)
        now = now or datetime.now(timezone.utc)
        imported = datetime.fromisoformat(
            previous["imported_utc"].replace("Z", "+00:00")
        )
        return MetadataCacheStatus(
            configured=True,
            last_import_utc=previous["imported_utc"],
            source_file_sha256=previous["source_file_sha256"],
            total_source_rows=previous["total_source_rows"],
            tagged_rows=previous["accepted_rows"],
            ignored_missing_rfid_rows=previous["ignored_missing_rfid_rows"],
            ignored_ended_rows=previous["ignored_ended_rows"],
            age_seconds=max(0.0, (now - imported).total_seconds()),
        )
