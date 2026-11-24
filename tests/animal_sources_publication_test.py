"""The shared publication: one computer writes it from any source, others read it."""

import csv
import hashlib
import json

import pytest

from reachaq.app.model.animal_metadata_sync import AnimalMetadataSyncService
from reachaq.app.model.animal_registry import AnimalRegistry
from reachaq.app.model.animal_sources.config import SourceConfiguration, build_source
from reachaq.app.model.animal_sources.credentials import CredentialStore
from reachaq.app.model.animal_sources.mapping import (
    AnimalTableMapper,
    ColumnMapping,
    SourceTable,
    softmouse_mapping,
    table_sha256,
)
from reachaq.app.model.animal_sources.publication import (
    AnimalTablePublisher,
    PublicationSource,
)


TAG = "360002353933099"
SOFTMOUSE = AnimalTableMapper(softmouse_mapping(), provider="softmouse")
LAB_MAPPING = ColumnMapping(
    external_identity_column="Mouse ID",
    new_animal_name_column="Mouse ID",
    rfid_column="RFID",
)


def csv_table(tmp_path, rows, headers=("Physical Tag", "Plate ID", "State")):
    path = tmp_path / "download.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return SourceTable(
        headers=headers,
        rows=tuple(tuple(row) for row in rows),
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_name=path.name,
        path=path,
    )


def credentials(tmp_path):
    return CredentialStore(keyring_backend=False, fallback_path=tmp_path / "secrets.json")


def test_publisher_writes_manifest_last_contract_and_archive(tmp_path):
    table = csv_table(tmp_path, [["PT-1", TAG, "Stock"]])

    result = AnimalTablePublisher(tmp_path / "shared").publish(table, SOFTMOUSE)

    assert result.export_path.read_bytes() == result.archive_path.read_bytes()
    assert result.export_path.read_bytes() == table.path.read_bytes()
    manifest = json.loads(result.manifest_path.read_text())
    assert result.manifest_path.name == "SoftMouse-AnimalList-current.manifest.json"
    assert manifest["filename"] == "SoftMouse-AnimalList-current.csv"
    assert manifest["taggedRows"] == 1
    assert manifest["provider"] == "softmouse"
    assert manifest["mapping"] == softmouse_mapping().to_dict()
    assert manifest["sha256"] == table.content_sha256


def test_invalid_snapshot_never_replaces_previous_publication(tmp_path):
    destination = tmp_path / "shared"
    first = AnimalTablePublisher(destination).publish(
        csv_table(tmp_path, [["PT-1", TAG, "Stock"]]), SOFTMOUSE
    )
    old_export, old_manifest = first.export_path.read_bytes(), first.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="no importable active animals"):
        AnimalTablePublisher(destination).publish(
            csv_table(tmp_path, [["PT-2", "", "Stock"]]), SOFTMOUSE
        )

    assert first.export_path.read_bytes() == old_export
    assert first.manifest_path.read_bytes() == old_manifest


def test_suspicious_source_shrink_never_replaces_publication(tmp_path):
    destination = tmp_path / "shared"
    rows = [[f"PT-{index}", TAG[:-1] + str(index), "Stock"] for index in range(3)]
    initial = AnimalTablePublisher(destination).publish(csv_table(tmp_path, rows), SOFTMOUSE)
    old_manifest = initial.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="potentially filtered"):
        AnimalTablePublisher(destination).publish(
            csv_table(tmp_path, [["PT-1", TAG, "Stock"]]), SOFTMOUSE
        )

    assert initial.manifest_path.read_bytes() == old_manifest


def test_manifest_failure_rolls_back_the_canonical_export(tmp_path, monkeypatch):
    destination = tmp_path / "shared"
    first = AnimalTablePublisher(destination).publish(
        csv_table(tmp_path, [["PT-1", TAG, "Stock"]]), SOFTMOUSE
    )
    old_export, old_manifest = first.export_path.read_bytes(), first.manifest_path.read_bytes()
    replacement = AnimalTablePublisher(destination)
    monkeypatch.setattr(
        replacement,
        "_atomic_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest failed")),
    )

    with pytest.raises(OSError, match="manifest failed"):
        replacement.publish(csv_table(tmp_path, [["PT-2", TAG[:-1] + "1", "Stock"]]), SOFTMOUSE)

    assert first.export_path.read_bytes() == old_export
    assert first.manifest_path.read_bytes() == old_manifest


def test_publisher_refuses_an_overlapping_run(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    destination = tmp_path / "shared"
    destination.mkdir()
    lock_path = destination / ".publish.lock"
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            AnimalTablePublisher(destination, lock_path=lock_path).publish(
                csv_table(tmp_path, [["PT-1", TAG, "Stock"]]), SOFTMOUSE
            )


def test_a_database_table_is_published_with_its_mapping_and_read_back_unmapped(tmp_path):
    headers = ("Mouse ID", "RFID")
    rows = (("M-1", TAG), ("M-2", None))
    table = SourceTable(
        headers=headers,
        rows=rows,
        content_sha256=table_sha256(headers, rows),
        source_name="SQLite database colony.sqlite",
        exact_numbers=True,
    )
    result = AnimalTablePublisher(tmp_path / "shared").publish(
        table, AnimalTableMapper(LAB_MAPPING, provider="database")
    )
    assert result.manifest_path.name == "database-AnimalList-current.manifest.json"

    # The reading computer was set up with only the manifest: no mapping.
    configured = build_source(
        SourceConfiguration("publication", {"manifest": str(result.manifest_path)}, None),
        credentials=credentials(tmp_path),
    )
    service = AnimalMetadataSyncService(
        configured=configured,
        local_staging_directory=tmp_path / "reader",
        registry=AnimalRegistry(tmp_path / "reader.sqlite3"),
        can_refresh=lambda: True,
    )
    batch = service.refresh_now().preview.batch

    assert batch.provider == "database"
    assert [(r.identity.subject_id, r.physical_rfid) for r in batch.records] == [
        ("M-1", TAG),
        ("M-2", None),
    ]
    assert not service.refresh_due()


def test_a_reader_refuses_a_snapshot_that_does_not_match_its_manifest(tmp_path):
    result = AnimalTablePublisher(tmp_path / "shared").publish(
        csv_table(tmp_path, [["PT-1", TAG, "Stock"]]), SOFTMOUSE
    )
    result.export_path.write_bytes(result.export_path.read_bytes().replace(b"PT-1", b"PT-9"))

    with pytest.raises(ValueError, match="hash differs"):
        PublicationSource({"manifest": str(result.manifest_path)}, credentials(tmp_path)).fetch(
            tmp_path / "work"
        )


def test_a_refresh_with_a_publish_target_writes_the_publication_first(tmp_path):
    lab = tmp_path / "animals.csv"
    lab.write_text(f"Mouse ID,RFID\nM-1,{TAG}\nM-2,\n", encoding="utf-8")
    configured = build_source(
        SourceConfiguration("spreadsheet", {"path": str(lab)}, LAB_MAPPING),
        credentials=credentials(tmp_path),
    )
    share = tmp_path / "share"
    share.mkdir()
    service = AnimalMetadataSyncService(
        configured=configured,
        local_staging_directory=tmp_path / "local",
        registry=AnimalRegistry(tmp_path / "registry.sqlite3"),
        can_refresh=lambda: True,
        publish_directory=share / "reachaq",
    )

    result = service.refresh_now()

    assert result.publication.manifest_path == share / "reachaq" / "spreadsheet-AnimalList-current.manifest.json"
    assert result.publication.export_path.read_bytes() == lab.read_bytes()
    assert result.preview.batch.accepted_rows == 2


def test_an_unmounted_publish_target_fails_the_refresh_and_keeps_the_cache(tmp_path):
    lab = tmp_path / "animals.csv"
    lab.write_text(f"Mouse ID,RFID\nM-1,{TAG}\n", encoding="utf-8")
    configured = build_source(
        SourceConfiguration("spreadsheet", {"path": str(lab)}, LAB_MAPPING),
        credentials=credentials(tmp_path),
    )
    service = AnimalMetadataSyncService(
        configured=configured,
        local_staging_directory=tmp_path / "local",
        registry=AnimalRegistry(tmp_path / "registry.sqlite3"),
        can_refresh=lambda: True,
        publish_directory=tmp_path / "share",
        publish_mount=tmp_path / "not-a-mount",
    )

    with pytest.raises(RuntimeError, match="not mounted"):
        service.refresh_now()

    assert service.registry.last_complete_import() is None
