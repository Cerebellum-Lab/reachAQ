import csv
import sqlite3

import openpyxl
import pytest

from reachaq.app.model.animal_registry import AnimalRegistry, RegistrySchemaError
from reachaq.app.model.animal_sources.mapping import (
    AnimalTableMapper,
    ColumnMapping,
    ImportGuardrails,
    SourceTable,
    read_table_file,
    softmouse_mapping,
)


class SoftMouseSpreadsheetSource:
    """The SoftMouse mapping applied to an export file, as a source does."""

    def __init__(self, guardrails=ImportGuardrails()):
        self.mapper = AnimalTableMapper(softmouse_mapping(), guardrails, provider="softmouse")

    def _read(self, path):
        return read_table_file(path)

    def preview(self, path, *, previous_source_row_count=None):
        return self.mapper.preview_table(
            self._read(path), previous_source_row_count=previous_source_row_count
        )


TAG_A = "360002353933099"
TAG_B = "360002353933101"
HEADERS = [
    "Physical Tag",
    "Plate ID",
    "Sex",
    "Date of Birth",
    "State",
    "Strain",
    "Genotype",
    "Cage Tag",
    "Protocol",
]


def write_export(path, rows, headers=HEADERS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def row(tag, rfid, state="Stock"):
    return [tag, rfid, "M", "2026-01-01", state, "C57", "Cre+;WT", "C-1", "P-1"]


def test_source_keeps_only_tagged_current_animals(tmp_path):
    path = tmp_path / "animals.csv"
    write_export(
        path,
        [row("PT-1", TAG_A), row("PT-2", ""), row("PT-3", TAG_B, "Ended")],
    )

    preview = SoftMouseSpreadsheetSource().preview(path)

    assert preview.batch.total_source_rows == 3
    assert preview.batch.accepted_rows == 1
    assert preview.batch.ignored_missing_rfid_rows == 1
    assert preview.batch.ignored_ended_rows == 1
    record = preview.batch.records[0]
    assert record.identity.subject_id == "PT-1"
    assert record.physical_rfid == TAG_A
    assert record.genotype == ("Cre+", "WT")


def test_source_keeps_non_iso_codes_and_rejects_ambiguous_headers(tmp_path):
    path = tmp_path / "codes.csv"
    write_export(path, [row("PT-1", "BC-0001")])
    preview = SoftMouseSpreadsheetSource().preview(path)
    assert preview.batch.accepted_rows == 1
    assert preview.batch.records[0].physical_rfid == "BC-0001"
    assert preview.batch.ignored_missing_rfid_rows == 0

    write_export(
        path,
        [row("PT-1", TAG_A) + ["duplicate"]],
        HEADERS + ["PlateID"],
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        SoftMouseSpreadsheetSource().preview(path)


def test_source_rejects_numeric_rfid_cells_to_prevent_precision_loss(tmp_path):
    path = tmp_path / "numeric.csv"
    write_export(path, [row("PT-1", 12345678901234567890123456)])
    # CSV values are text by definition; exercise the spreadsheet cell guard
    # without requiring openpyxl in the test environment.
    source = SoftMouseSpreadsheetSource()
    source._read = lambda _path: SourceTable(
        headers=HEADERS,
        rows=[tuple(row("PT-1", 12345678901234567890123456))],
        content_sha256="0" * 64,
        source_name="numeric.xlsx",
    )
    with pytest.raises(ValueError, match="numeric cell"):
        source.preview(path)


def test_source_reads_real_xlsx_cells_and_dates(tmp_path):
    path = tmp_path / "animals.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Animal List"
    sheet.append(HEADERS)
    sheet.append(row("PT-1", TAG_A))
    workbook.save(path)
    workbook.close()

    preview = SoftMouseSpreadsheetSource().preview(path)

    assert preview.batch.source_sheet == "Animal List"
    assert preview.batch.records[0].physical_rfid == TAG_A
    assert preview.source_headers == tuple(HEADERS)


def test_full_export_guard_rejects_suspicious_shrink(tmp_path):
    path = tmp_path / "animals.csv"
    write_export(path, [row("PT-1", TAG_A)])
    source = SoftMouseSpreadsheetSource(
        guardrails=ImportGuardrails(maximum_fractional_row_drop=0.2)
    )
    with pytest.raises(ValueError, match="potentially filtered"):
        source.preview(path, previous_source_row_count=10)


def test_registry_replacement_is_current_and_idempotent(tmp_path):
    export = tmp_path / "animals.csv"
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")
    source = SoftMouseSpreadsheetSource()

    write_export(export, [row("PT-1", TAG_A)])
    first = source.preview(export).batch
    result = registry.replace(first)
    assert not result.unchanged
    assert registry.resolve_rfid(TAG_A).identity.subject_id == "PT-1"

    repeated = source.preview(export).batch
    assert registry.replace(repeated).unchanged

    write_export(export, [row("PT-2", TAG_B)])
    second = source.preview(export).batch
    registry.replace(second)
    assert registry.resolve_rfid(TAG_A) is None
    assert registry.resolve_rfid(TAG_B).identity.subject_id == "PT-2"


def test_shared_rfid_is_dropped_from_every_animal_that_has_it(tmp_path):
    export = tmp_path / "animals.csv"
    write_export(export, [row("PT-2", TAG_B), row("PT-3", TAG_B), row("PT-4", TAG_A)])

    batch = SoftMouseSpreadsheetSource().preview(export).batch

    rfids = {r.identity.subject_id: r.physical_rfid for r in batch.records}
    assert rfids == {"PT-2": None, "PT-3": None, "PT-4": TAG_A}
    assert batch.duplicate_rfids == ((TAG_B, ("PT-2", "PT-3")),)


def test_failed_replacement_preserves_last_good_cache(tmp_path):
    export = tmp_path / "animals.csv"
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")
    source = SoftMouseSpreadsheetSource()
    write_export(export, [row("PT-1", TAG_A)])
    registry.replace(source.preview(export).batch)

    write_export(export, [row("PT-2", TAG_B), row("PT-2", TAG_A)])
    with pytest.raises(ValueError, match="duplicate external subject"):
        source.preview(export)

    assert registry.resolve_rfid(TAG_A).identity.subject_id == "PT-1"


def test_registry_recovers_corrupt_rebuildable_cache(tmp_path):
    path = tmp_path / "cache.sqlite3"
    path.write_bytes(b"not a sqlite database")

    registry = AnimalRegistry(path)

    assert registry.recovered_corrupt_path is not None
    assert registry.recovered_corrupt_path.read_bytes() == b"not a sqlite database"
    assert registry.resolve_rfid(TAG_A) is None


def test_registry_rejects_unknown_newer_schema_without_overwriting(tmp_path):
    path = tmp_path / "cache.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")

    with pytest.raises(RegistrySchemaError, match="newer than supported"):
        AnimalRegistry(path)

    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 99


def test_registry_readers_see_old_or_new_transaction_not_partial_state(tmp_path):
    export = tmp_path / "animals.csv"
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")
    source = SoftMouseSpreadsheetSource()
    write_export(export, [row("PT-1", TAG_A)])
    registry.replace(source.preview(export).batch)

    writer = sqlite3.connect(registry.path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE external_animals SET current=0")
        writer.execute("UPDATE external_identifiers SET active=0")
        assert registry.resolve_rfid(TAG_A).identity.subject_id == "PT-1"
        writer.commit()
    finally:
        writer.close()

    assert registry.resolve_rfid(TAG_A) is None


def test_a_v1_cache_migrates_in_place_and_keeps_its_records(tmp_path):
    path = tmp_path / "cache.sqlite3"
    export = tmp_path / "animals.csv"
    write_export(export, [row("PT-1", TAG_A)])
    registry = AnimalRegistry(path)
    registry.replace(SoftMouseSpreadsheetSource().preview(export).batch)
    with sqlite3.connect(path) as db:
        # Recreate the v1 table, where physical_rfid was NOT NULL.
        db.executescript(
            """
            ALTER TABLE external_animals RENAME TO new_table;
            CREATE TABLE external_animals (
                provider TEXT NOT NULL, subject_id_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL, physical_rfid TEXT NOT NULL,
                name_candidate TEXT, state TEXT NOT NULL, record_json TEXT NOT NULL,
                source_hash TEXT NOT NULL, first_seen_import_id TEXT NOT NULL,
                last_seen_import_id TEXT NOT NULL,
                current INTEGER NOT NULL CHECK(current IN (0, 1)),
                PRIMARY KEY(provider, subject_id_kind, subject_id)
            );
            INSERT INTO external_animals SELECT * FROM new_table;
            DROP TABLE new_table;
            PRAGMA user_version=1;
            """
        )

    migrated = AnimalRegistry(path)

    assert migrated.resolve_rfid(TAG_A).identity.subject_id == "PT-1"
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        notnull = {r[1]: r[3] for r in db.execute("PRAGMA table_info(external_animals)")}
    assert notnull["physical_rfid"] == 0


def _generic_batch(rows, provider="spreadsheet"):
    mapper = AnimalTableMapper(
        ColumnMapping(
            external_identity_column="Physical Tag",
            new_animal_name_column="Physical Tag",
            rfid_column="Plate ID",
            state_column="State",
        ),
        provider=provider,
    )
    return mapper.preview_table(
        SourceTable(
            headers=("Physical Tag", "Plate ID", "State"),
            rows=rows,
            content_sha256=str(len(rows)) * 64,
            source_name="lab.csv",
        )
    ).batch


def test_records_without_rfid_are_cached_and_listed_but_not_resolvable(tmp_path):
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")

    registry.replace(_generic_batch([("M-1", "", "Stock"), ("M-2", TAG_A, "Stock")]))

    records = {r.identity.subject_id: r for r in registry.list_current_records()}
    assert set(records) == {"M-1", "M-2"}
    assert records["M-1"].physical_rfid is None
    assert registry.resolve_rfid(TAG_A).identity.subject_id == "M-2"


def test_switching_sources_retires_the_previous_providers_records(tmp_path):
    export = tmp_path / "animals.csv"
    write_export(export, [row("PT-1", TAG_A)])
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")
    softmouse = SoftMouseSpreadsheetSource().preview(export).batch
    registry.replace(softmouse)

    registry.replace(_generic_batch([("M-9", TAG_B, "Stock")]))

    assert registry.resolve_rfid(TAG_A) is None
    assert [r.identity.subject_id for r in registry.list_current_records()] == ["M-9"]

    # Switching back to the same SoftMouse snapshot imports it again.
    result = registry.replace(SoftMouseSpreadsheetSource().preview(export).batch)
    assert not result.unchanged
    assert registry.resolve_rfid(TAG_A).identity.subject_id == "PT-1"
    assert registry.resolve_rfid(TAG_B) is None
