import csv
import sqlite3

import openpyxl
import pytest

from tools.acquisition.model.animal_registry import AnimalRegistry, RegistrySchemaError
from tools.acquisition.model.softmouse_spreadsheet_source import (
    ImportGuardrails,
    SoftMouseSpreadsheetSource,
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


def test_source_ignores_non_rfid_plate_ids_and_rejects_ambiguous_headers(tmp_path):
    path = tmp_path / "bad.csv"
    write_export(path, [row("PT-1", "not-a-tag")])
    preview = SoftMouseSpreadsheetSource().preview(path)
    assert preview.batch.accepted_rows == 0
    assert preview.batch.ignored_missing_rfid_rows == 1

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
    source._read_rows = lambda _path: (
        HEADERS,
        [tuple(row("PT-1", 12345678901234567890123456))],
        "Animal List",
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


def test_failed_replacement_preserves_last_good_cache(tmp_path):
    export = tmp_path / "animals.csv"
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")
    source = SoftMouseSpreadsheetSource()
    write_export(export, [row("PT-1", TAG_A)])
    registry.replace(source.preview(export).batch)

    write_export(export, [row("PT-2", TAG_B), row("PT-3", TAG_B)])
    with pytest.raises(ValueError, match="duplicate active RFID"):
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
