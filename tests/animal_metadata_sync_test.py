import csv
import hashlib
import json

import pytest

from tools.acquisition.model.animal_metadata_sync import AnimalMetadataSyncService
from tools.acquisition.model.animal_registry import AnimalRegistry
from tools.acquisition.model.softmouse_spreadsheet_source import SoftMouseSpreadsheetSource


TAG = "360002353933099"


def publish(directory, *, digest_override=None, total_rows=None, tagged_rows=None):
    source = directory / "SoftMouse-AnimalList-current.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Physical Tag", "Plate ID", "State"])
        writer.writerow(["PT-1", TAG, "Stock"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = directory / "SoftMouse-AnimalList-current.manifest.json"
    value = {
                "schemaVersion": 1,
                "filename": source.name,
                "sha256": digest_override or digest,
                "size": source.stat().st_size,
                "publishedUtc": "2026-08-10T00:00:00Z",
            }
    if total_rows is not None:
        value["totalSourceRows"] = total_rows
    if tagged_rows is not None:
        value["taggedRows"] = tagged_rows
    manifest.write_text(json.dumps(value))
    return manifest


def service(tmp_path, manifest, can_refresh=lambda: True):
    return AnimalMetadataSyncService(
        manifest_path=manifest,
        local_staging_directory=tmp_path / "local",
        source=SoftMouseSpreadsheetSource(),
        registry=AnimalRegistry(tmp_path / "registry.sqlite3"),
        can_refresh=can_refresh,
    )


def test_manifest_is_completion_marker_for_verified_local_refresh(tmp_path):
    manifest = publish(tmp_path)
    sync = service(tmp_path, manifest)

    result = sync.refresh_now()

    assert result.preview.batch.accepted_rows == 1
    assert sync.registry.resolve_rfid(TAG).identity.subject_id == "PT-1"


def test_refresh_is_refused_when_busy(tmp_path):
    manifest = publish(tmp_path)
    with pytest.raises(RuntimeError, match="session is active"):
        service(tmp_path, manifest, can_refresh=lambda: False).refresh_now()


def test_bad_manifest_preserves_last_known_good_cache(tmp_path):
    manifest = publish(tmp_path)
    sync = service(tmp_path, manifest)
    sync.refresh_now()
    publish(tmp_path, digest_override="0" * 64)

    with pytest.raises(ValueError, match="hash differs"):
        sync.refresh_now()

    assert sync.registry.resolve_rfid(TAG).identity.subject_id == "PT-1"


def test_refresh_rechecks_session_state_before_replacing_cache(tmp_path):
    manifest = publish(tmp_path)
    checks = iter((True, False))
    sync = service(tmp_path, manifest, can_refresh=lambda: next(checks))

    with pytest.raises(RuntimeError, match="session became active"):
        sync.refresh_now()

    assert sync.registry.resolve_rfid(TAG) is None


def test_manifest_counts_must_match_parsed_export(tmp_path):
    manifest = publish(tmp_path, total_rows=99, tagged_rows=1)
    sync = service(tmp_path, manifest)

    with pytest.raises(ValueError, match="row count differs"):
        sync.refresh_now()


def test_refresh_due_and_cache_status_use_import_ledger(tmp_path):
    manifest = publish(tmp_path, total_rows=1, tagged_rows=1)
    sync = service(tmp_path, manifest)
    assert sync.refresh_due()

    result = sync.refresh_now()
    status = sync.cache_status()

    assert not sync.refresh_due()
    assert status.source_file_sha256 == result.preview.batch.source_file_sha256
    assert status.tagged_rows == 1
    assert status.age_seconds is not None
