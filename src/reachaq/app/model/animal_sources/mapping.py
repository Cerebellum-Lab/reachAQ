"""Turn a source's table of rows into the app's normalized animal records.

Every source, built in or a lab's own, hands the app a :class:`SourceTable`.
A :class:`ColumnMapping` says which column holds which field, and
:class:`AnimalTableMapper` applies it, with the same checks for every source.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from reachaq.core.logging import get_verbose_logger
from reachaq.core.animal.external_metadata import (
    PHYSICAL_TAG_ID_KIND,
    ExternalAnimalRecord,
    ExternalIdentity,
    NormalizedAnimalBatch,
    normalize_optional_text,
    normalize_rfid,
    normalize_state,
    split_genotype,
)


logger = get_verbose_logger(__name__)

GENERIC_SUBJECT_ID_KIND = "subject_id"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _header_key(value: Any) -> str:
    text = normalize_optional_text(value)
    return "" if text is None else re.sub(r"[^a-z0-9]", "", text.casefold())


def _json_value(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _identifier_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return normalize_optional_text(value)


def _record_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def table_sha256(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A content hash for a table that did not come from a file."""
    encoded = json.dumps(
        [list(headers), [[_json_value(value) for value in row] for row in rows]],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ColumnMapping:
    """Which source column holds each field the app uses.

    Only the external ID and the new-animal name are required. With
    ``require_rfid`` off, a row with no valid RFID is still imported and can
    be linked by hand; with it on, such rows are counted and skipped.
    """

    external_identity_column: str
    new_animal_name_column: str
    rfid_column: Optional[str] = None
    require_rfid: bool = False
    state_column: Optional[str] = None
    sex_column: Optional[str] = None
    date_of_birth_column: Optional[str] = None
    strain_column: Optional[str] = None
    genotype_column: Optional[str] = None
    cage_column: Optional[str] = None
    protocol_column: Optional[str] = None
    ended_states: Tuple[str, ...] = ("ended",)
    subject_id_kind: str = GENERIC_SUBJECT_ID_KIND
    profile_id: str = "custom"
    version: int = 1

    #: Field name, operator label and whether it may be left empty, in the
    #: order setup shows them.
    COLUMN_FIELDS = (
        ("external_identity_column", "Permanent external ID", False),
        ("new_animal_name_column", "New-animal name", False),
        ("rfid_column", "RFID", True),
        ("state_column", "State", True),
        ("sex_column", "Sex", True),
        ("date_of_birth_column", "Date of birth", True),
        ("strain_column", "Strain", True),
        ("genotype_column", "Genotype", True),
        ("cage_column", "Cage", True),
        ("protocol_column", "Protocol", True),
    )

    def __post_init__(self) -> None:
        for name, label, optional in self.COLUMN_FIELDS:
            value = normalize_optional_text(getattr(self, name))
            if value is None and not optional:
                raise ValueError(f"The {label} column must be named")
            object.__setattr__(self, name, value)
        if self.require_rfid and self.rfid_column is None:
            raise ValueError("Requiring an RFID needs an RFID column")
        object.__setattr__(self, "ended_states", tuple(self.ended_states))

    def required_columns(self) -> Tuple[str, ...]:
        required = [self.external_identity_column, self.new_animal_name_column]
        if self.state_column:
            required.append(self.state_column)
        if self.require_rfid:
            required.append(self.rfid_column)
        return tuple(dict.fromkeys(required))

    def configured_columns(self) -> Tuple[str, ...]:
        optional = (
            self.rfid_column,
            self.sex_column,
            self.date_of_birth_column,
            self.strain_column,
            self.genotype_column,
            self.cage_column,
            self.protocol_column,
        )
        return tuple(
            dict.fromkeys(self.required_columns() + tuple(item for item in optional if item))
        )

    def to_dict(self) -> Dict[str, Any]:
        value = {
            name: getattr(self, name)
            for name in (
                "external_identity_column",
                "new_animal_name_column",
                "rfid_column",
                "require_rfid",
                "state_column",
                "sex_column",
                "date_of_birth_column",
                "strain_column",
                "genotype_column",
                "cage_column",
                "protocol_column",
                "subject_id_kind",
                "profile_id",
                "version",
            )
        }
        value["ended_states"] = list(self.ended_states)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ColumnMapping":
        names = {item.name for item in dataclasses.fields(ColumnMapping)}
        unknown = set(value) - names
        if unknown:
            raise ValueError(f"Unknown column mapping keys: {sorted(unknown)}")
        arguments = dict(value)
        if "ended_states" in arguments:
            arguments["ended_states"] = tuple(arguments["ended_states"] or ())
        return ColumnMapping(**arguments)


def _drop_shared_rfids(
    records: Sequence[ExternalAnimalRecord],
) -> Tuple[List[ExternalAnimalRecord], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    """Remove each identifier that more than one animal has, from all of them.

    A shared value cannot identify one animal, so those animals are kept
    without an identifier (they can still be linked by hand).
    """
    holders: Dict[str, List[str]] = {}
    for record in records:
        if record.physical_rfid is not None:
            holders.setdefault(record.physical_rfid, []).append(record.identity.subject_id)
    shared = {value: tuple(subjects) for value, subjects in holders.items() if len(subjects) > 1}
    if not shared:
        return list(records), ()
    kept = [
        dataclasses.replace(record, physical_rfid=None)
        if record.physical_rfid in shared
        else record
        for record in records
    ]
    return kept, tuple(shared.items())


def softmouse_mapping(new_animal_name_column: str = "Physical Tag") -> ColumnMapping:
    """The mapping for SoftMouse's Animals export, as it has always been read."""
    return ColumnMapping(
        external_identity_column="Physical Tag",
        new_animal_name_column=new_animal_name_column,
        rfid_column="Plate ID",
        require_rfid=True,
        state_column="State",
        sex_column="Sex",
        date_of_birth_column="Date of Birth",
        strain_column="Strain",
        genotype_column="Genotype",
        cage_column="Cage Tag",
        protocol_column="Protocol",
        subject_id_kind=PHYSICAL_TAG_ID_KIND,
        profile_id="softmouse-default",
        version=2,
    )


@dataclass(frozen=True)
class SourceTable:
    """One complete snapshot of a source: a header row and its data rows.

    ``content_sha256`` identifies the snapshot; an unchanged hash means an
    unchanged import. ``exact_numbers`` says numeric cells are exact (a
    database column), not spreadsheet cells that may have lost digits.
    A publication also carries its mapping and the counts it promised.
    """

    headers: Tuple[str, ...]
    rows: Tuple[Tuple[Any, ...], ...]
    content_sha256: str
    source_name: str
    sheet: str = ""
    path: Optional[Path] = None
    exact_numbers: bool = False
    mapping: Optional[ColumnMapping] = None
    provider: Optional[str] = None
    expected_total_rows: Optional[int] = None
    expected_accepted_rows: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", tuple(self.headers))
        object.__setattr__(self, "rows", tuple(tuple(row) for row in self.rows))


def read_table_file(path: Path, sheet_name: Optional[str] = None) -> SourceTable:
    """Read a .csv or .xlsx file; ``sheet_name`` None means the first sheet."""
    source_path = Path(path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    suffix = source_path.suffix.casefold()
    if suffix == ".csv":
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            raise ValueError("Spreadsheet is empty")
        data_rows = [tuple(r) for r in rows[1:] if any(str(value).strip() for value in r)]
        return SourceTable(
            headers=tuple(str(value).strip() for value in rows[0]),
            rows=tuple(data_rows),
            content_sha256=source_hash,
            source_name=source_path.name,
            path=source_path,
        )
    if suffix != ".xlsx":
        raise ValueError("A spreadsheet source must be an .xlsx or .csv file")
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to import .xlsx files") from exc
    workbook = openpyxl.load_workbook(source_path, read_only=True, data_only=True)
    try:
        if sheet_name:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(
                    f"Missing worksheet {sheet_name!r}; found {workbook.sheetnames!r}"
                )
            sheet = workbook[sheet_name]
        else:
            sheet = workbook.worksheets[0]
        iterator = sheet.iter_rows(values_only=True)
        try:
            raw_headers = next(iterator)
        except StopIteration as exc:
            raise ValueError("Spreadsheet is empty") from exc
        headers = tuple(normalize_optional_text(value) or "" for value in raw_headers)
        rows = tuple(tuple(row) for row in iterator if any(value is not None for value in row))
        return SourceTable(
            headers=headers,
            rows=rows,
            content_sha256=source_hash,
            source_name=source_path.name,
            sheet=sheet.title,
            path=source_path,
        )
    finally:
        workbook.close()


@dataclass(frozen=True)
class ImportGuardrails:
    minimum_source_rows: int = 1
    maximum_fractional_row_drop: float = 0.35

    def validate(self, row_count: int, previous_row_count: Optional[int]) -> None:
        if row_count < self.minimum_source_rows:
            raise ValueError(
                f"Source has {row_count} rows; expected at least "
                f"{self.minimum_source_rows} for a complete export"
            )
        if previous_row_count and previous_row_count > 0:
            minimum = math.ceil(
                previous_row_count * (1 - self.maximum_fractional_row_drop)
            )
            if row_count < minimum:
                raise ValueError(
                    f"Source row count dropped from {previous_row_count} to {row_count}; "
                    "refusing a potentially filtered export"
                )


@dataclass(frozen=True)
class SpreadsheetImportPreview:
    batch: NormalizedAnimalBatch
    source_path: Optional[Path]
    source_size: int
    source_modified_ns: int
    source_headers: Tuple[str, ...]
    mapped_columns: Tuple[str, ...]
    ignored_columns: Tuple[str, ...]


class AnimalTableMapper:
    """Apply a :class:`ColumnMapping` to a :class:`SourceTable`."""

    def __init__(
        self,
        mapping: ColumnMapping,
        guardrails: ImportGuardrails = ImportGuardrails(),
        *,
        provider: str,
    ):
        self.profile = mapping
        self.guardrails = guardrails
        self.provider = provider

    def preview_table(
        self,
        table: SourceTable,
        *,
        previous_source_row_count: Optional[int] = None,
    ) -> SpreadsheetImportPreview:
        profile = self.profile
        logger.info(
            "Animal metadata validation started: source=%s provider=%s previous_rows=%s "
            "rfid_column=%s require_rfid=%s name_column=%s",
            table.source_name,
            self.provider,
            previous_source_row_count,
            profile.rfid_column,
            profile.require_rfid,
            profile.new_animal_name_column,
        )
        headers = list(table.headers)
        source_rows = table.rows
        resolved = self._resolve_headers(headers)
        self.guardrails.validate(len(source_rows), previous_source_row_count)

        records: List[ExternalAnimalRecord] = []
        ignored_missing_rfid = 0
        ignored_ended = 0
        ended_states = {normalize_state(value) for value in profile.ended_states}
        for row_number, values in enumerate(source_rows, start=2):
            payload = {
                header: _json_value(values[index]) if index < len(values) else None
                for index, header in enumerate(headers)
                if header
            }
            state = normalize_state(
                self._value(values, resolved[profile.state_column])
                if profile.state_column in resolved
                else None
            )
            if state in ended_states:
                ignored_ended += 1
                continue
            rfid = self._rfid(values, resolved, row_number, table.exact_numbers)
            if rfid is None and profile.require_rfid:
                ignored_missing_rfid += 1
                continue
            external_id = _identifier_value(
                self._value(values, resolved[profile.external_identity_column])
            )
            if external_id is None:
                if profile.require_rfid:
                    raise ValueError(
                        f"RFID-bearing source row {row_number} has no "
                        f"{profile.external_identity_column!r}"
                    )
                raise ValueError(
                    f"Source row {row_number} has no "
                    f"{profile.external_identity_column!r}"
                )
            records.append(
                ExternalAnimalRecord(
                    identity=ExternalIdentity(
                        external_id,
                        provider=self.provider,
                        subject_id_kind=profile.subject_id_kind,
                    ),
                    physical_rfid=rfid,
                    new_animal_name_candidate=normalize_optional_text(
                        self._value(values, resolved[profile.new_animal_name_column])
                    ),
                    state=state,
                    source_payload=payload,
                    source_hash=_record_hash(payload),
                    sex=self._optional(values, resolved, profile.sex_column),
                    date_of_birth=self._optional(
                        values, resolved, profile.date_of_birth_column
                    ),
                    strain=self._optional(values, resolved, profile.strain_column),
                    genotype=split_genotype(
                        self._optional(values, resolved, profile.genotype_column)
                    ),
                    cage=self._optional(values, resolved, profile.cage_column),
                    protocol=self._optional(values, resolved, profile.protocol_column),
                )
            )

        records, duplicates = _drop_shared_rfids(records)
        warnings = tuple(
            f"RFID {value!r} is shared by {', '.join(subjects)}; "
            "it is ignored for all of them"
            for value, subjects in duplicates
        )
        for warning in warnings:
            logger.warning("Animal metadata duplicate identifier: %s", warning)
        header_signature = hashlib.sha256(
            json.dumps(headers, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        batch = NormalizedAnimalBatch(
            import_id=str(uuid.uuid4()),
            provider=self.provider,
            schema_version=1,
            mapping_profile_id=profile.profile_id,
            mapping_profile_version=profile.version,
            source_filename=table.source_name,
            source_sheet=table.sheet,
            source_file_sha256=table.content_sha256,
            header_signature=header_signature,
            imported_utc=_utc_now(),
            total_source_rows=len(source_rows),
            ignored_missing_rfid_rows=ignored_missing_rfid,
            ignored_ended_rows=ignored_ended,
            records=tuple(records),
            warnings=warnings,
            duplicate_rfids=duplicates,
        )
        stat = table.path.stat() if table.path is not None else None
        preview = SpreadsheetImportPreview(
            batch=batch,
            source_path=table.path,
            source_size=0 if stat is None else stat.st_size,
            source_modified_ns=0 if stat is None else stat.st_mtime_ns,
            source_headers=tuple(headers),
            mapped_columns=tuple(
                header for index, header in enumerate(headers) if index in resolved.values()
            ),
            ignored_columns=tuple(
                header
                for index, header in enumerate(headers)
                if header and index not in resolved.values()
            ),
        )
        logger.info(
            "Animal metadata validation complete: source=%s total_rows=%d accepted_rows=%d "
            "with_rfid=%d ignored_missing_rfid=%d ignored_ended=%d source_sha256=%s",
            table.source_name,
            batch.total_source_rows,
            batch.accepted_rows,
            sum(record.physical_rfid is not None for record in batch.records),
            batch.ignored_missing_rfid_rows,
            batch.ignored_ended_rows,
            batch.source_file_sha256,
        )
        return preview

    def _rfid(
        self,
        values: Sequence[Any],
        resolved: Mapping[str, int],
        row_number: int,
        exact_numbers: bool,
    ) -> Optional[str]:
        column = self.profile.rfid_column
        if column is None or column not in resolved:
            return None
        raw_rfid = self._value(values, resolved[column])
        if normalize_optional_text(raw_rfid) is None:
            return None
        if exact_numbers and isinstance(raw_rfid, int) and not isinstance(raw_rfid, bool):
            raw_rfid = str(raw_rfid)
        if not isinstance(raw_rfid, str):
            raise ValueError(
                f"RFID at source row {row_number} is stored as a numeric cell; "
                "format the RFID column as text to prevent spreadsheet "
                "precision loss"
            )
        try:
            # Any non-empty code is an identifier: an ISO 11784 RFID, a
            # barcode, or a lab's own ID (SoftMouse Alt. ID).
            return normalize_rfid(raw_rfid)
        except ValueError as exc:
            logger.warning(
                "Animal metadata RFID at source row %d ignored: %s", row_number, exc
            )
            return None

    def _resolve_headers(self, headers: Sequence[str]) -> Dict[str, int]:
        aliases: Dict[str, List[int]] = {}
        for index, header in enumerate(headers):
            aliases.setdefault(_header_key(header), []).append(index)
        result: Dict[str, int] = {}
        required = set(self.profile.required_columns())
        for configured in self.profile.configured_columns():
            matches = aliases.get(_header_key(configured), [])
            if len(matches) > 1:
                raise ValueError(f"Ambiguous source column {configured!r}")
            if not matches:
                if configured in required:
                    raise ValueError(f"Missing required source column {configured!r}")
                continue
            result[configured] = matches[0]
        return result

    @staticmethod
    def _value(values: Sequence[Any], index: int) -> Any:
        return values[index] if index < len(values) else None

    def _optional(
        self,
        values: Sequence[Any],
        resolved: Mapping[str, int],
        column: Optional[str],
    ) -> Optional[str]:
        if not column or column not in resolved:
            return None
        value = self._value(values, resolved[column])
        if isinstance(value, dt.datetime):
            return value.date().isoformat()
        if isinstance(value, dt.date):
            return value.isoformat()
        return normalize_optional_text(value)
