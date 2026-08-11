from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from autotrainer.core.logging import get_verbose_logger

from .animal_registry import AnimalRegistry, RegistryImportResult
from .softmouse_spreadsheet_source import (
    SoftMouseSpreadsheetSource,
    SpreadsheetImportPreview,
)


logger = get_verbose_logger(__name__)


DEFAULT_SOFTMOUSE_PUBLICATION_DIRECTORY = Path(
    "/mnt/isilon/Data/ReachingData/SoftMouse"
)
DEFAULT_SOFTMOUSE_MANIFEST_PATH = (
    DEFAULT_SOFTMOUSE_PUBLICATION_DIRECTORY
    / "SoftMouse-AnimalList-current.manifest.json"
)


@dataclass(frozen=True)
class PublishedExportManifest:
    schema_version: int
    filename: str
    sha256: str
    size: int
    published_utc: str
    source_export_utc: Optional[str] = None
    total_source_rows: Optional[int] = None
    tagged_rows: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.sha256
        ):
            raise ValueError("Publication manifest contains an invalid SHA-256")
        if self.size < 1:
            raise ValueError("Publication manifest size must be positive")

    @classmethod
    def from_file(cls, path: Path) -> "PublishedExportManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("schemaVersion") != 1:
            raise ValueError("Unsupported SoftMouse publication manifest version")
        return cls(
            schema_version=1,
            filename=value["filename"],
            sha256=value["sha256"],
            size=int(value["size"]),
            published_utc=value["publishedUtc"],
            source_export_utc=value.get("sourceExportUtc"),
            total_source_rows=value.get("totalSourceRows"),
            tagged_rows=value.get("taggedRows"),
        )


@dataclass(frozen=True)
class MetadataRefreshResult:
    preview: SpreadsheetImportPreview
    registry_result: RegistryImportResult


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
    """Validate a published export and replace a local cache without partial reads."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        local_staging_directory: Path,
        source: SoftMouseSpreadsheetSource,
        registry: AnimalRegistry,
        can_refresh: Callable[[], bool],
    ):
        self.manifest_path = Path(manifest_path)
        self.local_staging_directory = Path(local_staging_directory)
        self.source = source
        self.registry = registry
        self.can_refresh = can_refresh
        self._lock = threading.Lock()

    def refresh_now(self) -> MetadataRefreshResult:
        if not self.can_refresh():
            logger.warning("SoftMouse cache refresh blocked: session is active")
            raise RuntimeError("SoftMouse refresh is disabled while a session is active")
        if not self._lock.acquire(blocking=False):
            logger.warning("SoftMouse cache refresh blocked: refresh already running")
            raise RuntimeError("A SoftMouse refresh is already running")
        try:
            logger.info(
                "SoftMouse cache refresh started: manifest=%s staging_directory=%s",
                self.manifest_path,
                self.local_staging_directory,
            )
            manifest = PublishedExportManifest.from_file(self.manifest_path)
            if Path(manifest.filename).name != manifest.filename:
                raise ValueError("Publication manifest filename must not contain a path")
            shared_source = self.manifest_path.parent / manifest.filename
            local_copy = self._copy_and_verify(shared_source, manifest)
            previous = self.registry.last_complete_import()
            previous_rows = None if previous is None else previous["total_source_rows"]
            preview = self.source.preview(
                local_copy,
                previous_source_row_count=previous_rows,
            )
            if preview.batch.source_file_sha256 != manifest.sha256:
                raise ValueError("Local export hash differs from publication manifest")
            if (
                manifest.total_source_rows is not None
                and preview.batch.total_source_rows != int(manifest.total_source_rows)
            ):
                raise ValueError("Parsed source row count differs from publication manifest")
            if (
                manifest.tagged_rows is not None
                and preview.batch.accepted_rows != int(manifest.tagged_rows)
            ):
                raise ValueError("Parsed tagged row count differs from publication manifest")
            if not self.can_refresh():
                raise RuntimeError(
                    "SoftMouse refresh was cancelled because a session became active"
                )
            registry_result = self.registry.replace(preview.batch)
            logger.info(
                "SoftMouse cache refresh validated and committed: import_id=%s "
                "rows=%d tagged=%d unchanged=%s source_sha256=%s",
                preview.batch.import_id,
                preview.batch.total_source_rows,
                preview.batch.accepted_rows,
                registry_result.unchanged,
                preview.batch.source_file_sha256,
            )
            return MetadataRefreshResult(preview, registry_result)
        except Exception:
            logger.exception(
                "SoftMouse cache refresh transaction failed: manifest=%s",
                self.manifest_path,
            )
            raise
        finally:
            self._lock.release()

    def _copy_and_verify(
        self, source_path: Path, manifest: PublishedExportManifest
    ) -> Path:
        before = source_path.stat()
        if before.st_size != manifest.size:
            raise ValueError("Published export size differs from manifest")
        self.local_staging_directory.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="softmouse-", suffix=source_path.suffix, dir=self.local_staging_directory
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        destination = self.local_staging_directory / (
            "SoftMouse-AnimalList-local" + source_path.suffix.casefold()
        )
        try:
            shutil.copyfile(source_path, temporary_path)
            after = source_path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Published export changed while it was being copied")
            digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
            if digest != manifest.sha256:
                raise ValueError("Published export hash differs from manifest")
            os.replace(temporary_path, destination)
            logger.info(
                "SoftMouse publication copied locally: source=%s destination=%s "
                "bytes=%d sha256=%s",
                source_path,
                destination,
                manifest.size,
                digest,
            )
            return destination
        finally:
            temporary_path.unlink(missing_ok=True)

    def refresh_due(self) -> bool:
        manifest = PublishedExportManifest.from_file(self.manifest_path)
        previous = self.registry.last_complete_import()
        due = (
            previous is None
            or previous["source_file_sha256"].casefold() != manifest.sha256.casefold()
        )
        logger.info(
            "SoftMouse cache catch-up check: due=%s manifest_sha256=%s "
            "local_sha256=%s",
            due,
            manifest.sha256,
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
