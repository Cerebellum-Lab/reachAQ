from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple

from .base import AnimalMetadataSource, SetupField
from .mapping import SourceTable, read_table_file


def copy_unchanged(source_path: Path, destination: Path) -> Path:
    """Copy a file that someone may be saving, refusing a copy taken mid-write."""
    source_path = Path(source_path)
    before = source_path.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    after = source_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        destination.unlink(missing_ok=True)
        raise ValueError(f"{source_path.name} changed while it was being copied")
    return destination


class SpreadsheetSource(AnimalMetadataSource):
    """A .csv or .xlsx file the lab keeps, on this computer or a share."""

    kind = "spreadsheet"
    label = "Spreadsheet (CSV or XLSX)"
    provider = "spreadsheet"

    @classmethod
    def setup_fields(cls) -> Tuple[SetupField, ...]:
        return (
            SetupField("path", "Spreadsheet file", kind="path",
                       help="A .csv or .xlsx file; a network share works."),
            SetupField("sheet", "Worksheet", required=False,
                       help="XLSX only; empty means the first sheet."),
        )

    @property
    def path(self) -> Path:
        return Path(self.settings["path"]).expanduser()

    def describe(self) -> str:
        sheet = self.settings.get("sheet")
        return f"Spreadsheet {self.path}" + (f" ({sheet})" if sheet else "")

    def current_version(self) -> Optional[str]:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def fetch(self, work_directory: Path) -> SourceTable:
        path = self.path
        if path.suffix.casefold() not in {".csv", ".xlsx"}:
            raise ValueError("A spreadsheet source must be an .xlsx or .csv file")
        local = copy_unchanged(
            path, Path(work_directory) / ("spreadsheet-copy" + path.suffix.casefold())
        )
        table = read_table_file(local, self.settings.get("sheet") or None)
        return SourceTable(
            headers=table.headers,
            rows=table.rows,
            content_sha256=table.content_sha256,
            source_name=path.name,
            sheet=table.sheet,
            path=local,
        )
