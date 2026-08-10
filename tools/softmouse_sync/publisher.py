from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from tools.acquisition.model.softmouse_spreadsheet_source import SoftMouseSpreadsheetSource


class ExportSource(Protocol):
    def download(self, destination_directory: Path) -> Path: ...


@dataclass(frozen=True)
class PublicationResult:
    export_path: Path
    manifest_path: Path
    archive_path: Path
    sha256: str
    total_source_rows: int
    tagged_rows: int
    published_utc: str


class SoftMouseExportPublisher:
    def __init__(
        self,
        *,
        export_source: ExportSource,
        destination_directory: Path,
        spreadsheet_source: SoftMouseSpreadsheetSource,
        lock_path: Optional[Path] = None,
    ):
        self.export_source = export_source
        self.destination_directory = Path(destination_directory)
        self.spreadsheet_source = spreadsheet_source
        self.lock_path = Path(lock_path or (self.destination_directory / ".publish.lock"))

    def publish(self) -> PublicationResult:
        self.destination_directory.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            self._lock(lock_file)
            with tempfile.TemporaryDirectory(prefix="softmouse-publish-") as temporary:
                downloaded = self.export_source.download(Path(temporary))
                previous_rows = None
                previous_manifest = (
                    self.destination_directory
                    / "SoftMouse-AnimalList-current.manifest.json"
                )
                if previous_manifest.is_file():
                    try:
                        previous_rows = int(
                            json.loads(previous_manifest.read_text(encoding="utf-8"))[
                                "totalSourceRows"
                            ]
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        raise ValueError(
                            "Existing publication manifest is invalid; refusing to replace it"
                        )
                preview = self.spreadsheet_source.preview(
                    downloaded,
                    previous_source_row_count=previous_rows,
                )
                batch = preview.batch
                digest = batch.source_file_sha256
                published_utc = self._utc_now()
                suffix = downloaded.suffix.casefold()
                stable_name = "SoftMouse-AnimalList-current" + suffix
                stable_path = self.destination_directory / stable_name
                archive_directory = self.destination_directory / "archive" / published_utc[:4] / published_utc[5:7]
                archive_directory.mkdir(parents=True, exist_ok=True)
                compact_time = published_utc.replace("-", "").replace(":", "")
                archive_path = archive_directory / (
                    f"SoftMouse-AnimalList-{compact_time}-{digest[:12]}{suffix}"
                )
                self._atomic_copy(downloaded, archive_path)
                manifest_path = self.destination_directory / "SoftMouse-AnimalList-current.manifest.json"
                manifest = {
                    "schemaVersion": 1,
                    "filename": stable_name,
                    "sha256": digest,
                    "size": downloaded.stat().st_size,
                    "sourceExportUtc": None,
                    "publishedUtc": published_utc,
                    "totalSourceRows": batch.total_source_rows,
                    "taggedRows": batch.accepted_rows,
                    "ignoredMissingRfidRows": batch.ignored_missing_rfid_rows,
                    "ignoredEndedRows": batch.ignored_ended_rows,
                    "mappingProfileId": batch.mapping_profile_id,
                    "mappingProfileVersion": batch.mapping_profile_version,
                }
                previous_stable = Path(temporary) / ("previous" + suffix)
                had_previous_stable = stable_path.is_file()
                if had_previous_stable:
                    shutil.copy2(stable_path, previous_stable)
                stable_replaced = False
                try:
                    self._atomic_copy(downloaded, stable_path)
                    stable_replaced = True
                    self._atomic_bytes(
                        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
                        + b"\n",
                        manifest_path,
                    )
                except Exception:
                    if stable_replaced:
                        if had_previous_stable:
                            self._atomic_copy(previous_stable, stable_path)
                        else:
                            stable_path.unlink(missing_ok=True)
                    raise
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
    def _lock(file_object) -> None:
        try:
            import fcntl

            fcntl.flock(file_object.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another SoftMouse publisher is already running") from exc

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
