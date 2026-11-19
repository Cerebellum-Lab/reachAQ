"""A shared publication: one computer fetches a source, the others read its copy.

The writer puts a CSV snapshot and a ``.manifest.json`` next to it, the
manifest last, so a reader never sees a half-written snapshot. The manifest
carries the hash, row counts, provider and column mapping, so a reading
computer needs no mapping of its own.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from autotrainer.core.logging import get_verbose_logger

from .base import AnimalMetadataSource, SetupField
from .mapping import (
    AnimalTableMapper,
    ColumnMapping,
    SourceTable,
    read_table_file,
    softmouse_mapping,
)
from .spreadsheet import copy_unchanged


logger = get_verbose_logger(__name__)

MANIFEST_SUFFIX = "-current.manifest.json"


def publication_stem(provider: str) -> str:
    # SoftMouse keeps the name its readers were configured with.
    return "SoftMouse-AnimalList" if provider == "softmouse" else f"{provider}-AnimalList"


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
    provider: str = "softmouse"
    mapping: Optional[ColumnMapping] = None

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.sha256
        ):
            raise ValueError("Publication manifest contains an invalid SHA-256")
        if self.size < 1:
            raise ValueError("Publication manifest size must be positive")
        if Path(self.filename).name != self.filename:
            raise ValueError("Publication manifest filename must not contain a path")

    @classmethod
    def from_file(cls, path: Path) -> "PublishedExportManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("schemaVersion") != 1:
            raise ValueError("Unsupported publication manifest version")
        mapping = value.get("mapping")
        return cls(
            schema_version=1,
            filename=value["filename"],
            sha256=value["sha256"],
            size=int(value["size"]),
            published_utc=value["publishedUtc"],
            source_export_utc=value.get("sourceExportUtc"),
            total_source_rows=value.get("totalSourceRows"),
            tagged_rows=value.get("taggedRows"),
            provider=value.get("provider") or "softmouse",
            mapping=None if mapping is None else ColumnMapping.from_dict(mapping),
        )


class PublicationSource(AnimalMetadataSource):
    """Read the publication another computer keeps on a share."""

    kind = "publication"
    label = "Shared publication (written by another reachAQ computer)"
    provider = "softmouse"

    @classmethod
    def setup_fields(cls) -> Tuple[SetupField, ...]:
        return (
            SetupField(
                "manifest",
                "Publication manifest",
                kind="path",
                help="The publication's .manifest.json on the share.",
            ),
        )

    @classmethod
    def default_mapping(cls) -> ColumnMapping:
        # Used only for a manifest that carries no mapping: one written by
        # the SoftMouse publisher before sources existed.
        return softmouse_mapping()

    @property
    def manifest_path(self) -> Path:
        return Path(self.settings["manifest"]).expanduser()

    def describe(self) -> str:
        return f"Shared publication {self.manifest_path}"

    def current_version(self) -> Optional[str]:
        return PublishedExportManifest.from_file(self.manifest_path).sha256

    def fetch(self, work_directory: Path) -> SourceTable:
        manifest = PublishedExportManifest.from_file(self.manifest_path)
        shared = self.manifest_path.parent / manifest.filename
        if shared.stat().st_size != manifest.size:
            raise ValueError("Published export size differs from manifest")
        local = copy_unchanged(
            shared,
            Path(work_directory) / ("publication-local" + shared.suffix.casefold()),
        )
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest != manifest.sha256.casefold():
            raise ValueError("Published export hash differs from manifest")
        logger.info(
            "Publication copied locally: source=%s destination=%s bytes=%d sha256=%s",
            shared,
            local,
            manifest.size,
            digest,
        )
        table = read_table_file(local)
        return SourceTable(
            headers=table.headers,
            rows=table.rows,
            content_sha256=table.content_sha256,
            source_name=manifest.filename,
            sheet=table.sheet,
            path=local,
            mapping=manifest.mapping,
            provider=manifest.provider,
            expected_total_rows=manifest.total_source_rows,
            expected_accepted_rows=manifest.tagged_rows,
        )


@dataclass(frozen=True)
class PublicationResult:
    export_path: Path
    manifest_path: Path
    archive_path: Path
    sha256: str
    total_source_rows: int
    tagged_rows: int
    published_utc: str


def ensure_publication_directory(
    destination: Path,
    *,
    mount_root: Optional[Path] = None,
    is_mount=os.path.ismount,
) -> Path:
    destination = Path(destination)
    if mount_root is not None and not is_mount(Path(mount_root)):
        raise RuntimeError(
            f"The publication mount {mount_root} is not mounted; publication stopped"
        )
    if not destination.parent.is_dir():
        raise RuntimeError(f"Publication parent does not exist: {destination.parent}")
    destination.mkdir(exist_ok=True)
    return destination


class AnimalTablePublisher:
    """Write a source snapshot as the shared publication, all or nothing."""

    def __init__(self, destination_directory: Path, *, lock_path: Optional[Path] = None):
        self.destination_directory = Path(destination_directory)
        self.lock_path = Path(lock_path or (self.destination_directory / ".publish.lock"))

    def publish(self, table: SourceTable, mapper: AnimalTableMapper) -> PublicationResult:
        logger.info(
            "Publication transaction started: destination=%s lock=%s provider=%s",
            self.destination_directory,
            self.lock_path,
            mapper.provider,
        )
        stem = publication_stem(mapper.provider)
        manifest_path = self.destination_directory / (stem + MANIFEST_SUFFIX)
        self.destination_directory.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            self._lock(lock_file)
            with tempfile.TemporaryDirectory(prefix="animal-publish-") as temporary:
                snapshot = Path(temporary) / (stem + ".csv")
                self._write_snapshot(table, snapshot)
                previous_rows = None
                if manifest_path.is_file():
                    try:
                        previous_rows = int(
                            json.loads(manifest_path.read_text(encoding="utf-8"))[
                                "totalSourceRows"
                            ]
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        raise ValueError(
                            "Existing publication manifest is invalid; refusing to replace it"
                        )
                batch = mapper.preview_table(
                    read_table_file(snapshot),
                    previous_source_row_count=previous_rows,
                ).batch
                if batch.accepted_rows < 1:
                    raise ValueError(
                        "The source has no importable active animals; "
                        "refusing to replace the current publication"
                    )
                digest = batch.source_file_sha256
                published_utc = self._utc_now()
                stable_name = stem + "-current.csv"
                stable_path = self.destination_directory / stable_name
                archive_directory = (
                    self.destination_directory / "archive" / published_utc[:4] / published_utc[5:7]
                )
                compact_time = published_utc.replace("-", "").replace(":", "")
                archive_path = archive_directory / f"{stem}-{compact_time}-{digest[:12]}.csv"
                self._atomic_copy(snapshot, archive_path)
                manifest = {
                    "schemaVersion": 1,
                    "filename": stable_name,
                    "sha256": digest,
                    "size": snapshot.stat().st_size,
                    "sourceExportUtc": None,
                    "publishedUtc": published_utc,
                    "totalSourceRows": batch.total_source_rows,
                    "taggedRows": batch.accepted_rows,
                    "ignoredMissingRfidRows": batch.ignored_missing_rfid_rows,
                    "ignoredEndedRows": batch.ignored_ended_rows,
                    "mappingProfileId": batch.mapping_profile_id,
                    "mappingProfileVersion": batch.mapping_profile_version,
                    "provider": mapper.provider,
                    "mapping": mapper.profile.to_dict(),
                }
                previous_stable = Path(temporary) / "previous.csv"
                had_previous_stable = stable_path.is_file()
                if had_previous_stable:
                    shutil.copy2(stable_path, previous_stable)
                stable_replaced = False
                try:
                    self._atomic_copy(snapshot, stable_path)
                    stable_replaced = True
                    self._atomic_bytes(
                        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
                        manifest_path,
                    )
                except Exception:
                    logger.exception("Publication commit failed; restoring previous export")
                    if stable_replaced:
                        if had_previous_stable:
                            self._atomic_copy(previous_stable, stable_path)
                        else:
                            stable_path.unlink(missing_ok=True)
                    raise
                logger.info(
                    "Publication transaction committed: export=%s manifest=%s "
                    "archive=%s sha256=%s",
                    stable_path,
                    manifest_path,
                    archive_path,
                    digest,
                )
                return PublicationResult(
                    stable_path,
                    manifest_path,
                    archive_path,
                    digest,
                    batch.total_source_rows,
                    batch.accepted_rows,
                    published_utc,
                )

    @staticmethod
    def _write_snapshot(table: SourceTable, destination: Path) -> None:
        if table.path is not None and table.path.suffix.casefold() == ".csv":
            # Keep the source's own bytes, so its hash carries through.
            shutil.copyfile(table.path, destination)
            return
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(table.headers)
            for row in table.rows:
                writer.writerow(
                    [
                        "" if value is None
                        else value.isoformat() if isinstance(value, (dt.date, dt.datetime))
                        else value
                        for value in row
                    ]
                )

    @staticmethod
    def _lock(file_object) -> None:
        try:
            import fcntl
        except ImportError:  # Windows: a single writer is the operator's job
            return
        try:
            fcntl.flock(file_object.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another publisher is already running") from exc

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as handle:
                temporary = Path(handle.name)
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_bytes(content: bytes, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _utc_now() -> str:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
