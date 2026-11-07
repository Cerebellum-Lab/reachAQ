import csv
import hashlib
import json

import pytest

from tools.acquisition.model.softmouse_spreadsheet_source import SoftMouseSpreadsheetSource
from tools.softmouse_sync.publisher import SoftMouseExportPublisher


TAG = "360002353933099"


class CsvExportSource:
    def __init__(self, rows):
        self.rows = rows

    def download(self, destination_directory):
        path = destination_directory / "download.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Physical Tag", "Plate ID", "State"])
            writer.writerows(self.rows)
        return path


def test_publisher_writes_manifest_last_contract_and_archive(tmp_path):
    publisher = SoftMouseExportPublisher(
        export_source=CsvExportSource([["PT-1", TAG, "Stock"]]),
        destination_directory=tmp_path / "shared",
        spreadsheet_source=SoftMouseSpreadsheetSource(),
    )

    result = publisher.publish()

    assert result.export_path.read_bytes() == result.archive_path.read_bytes()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["filename"] == "SoftMouse-AnimalList-current.csv"
    assert manifest["taggedRows"] == 1
    assert manifest["sha256"] == hashlib.sha256(result.export_path.read_bytes()).hexdigest()


def test_invalid_download_never_replaces_previous_publication(tmp_path):
    destination = tmp_path / "shared"
    good = SoftMouseExportPublisher(
        export_source=CsvExportSource([["PT-1", TAG, "Stock"]]),
        destination_directory=destination,
        spreadsheet_source=SoftMouseSpreadsheetSource(),
    )
    first = good.publish()
    old_export = first.export_path.read_bytes()
    old_manifest = first.manifest_path.read_bytes()

    bad = SoftMouseExportPublisher(
        export_source=CsvExportSource([["PT-2", "bad", "Stock"]]),
        destination_directory=destination,
        spreadsheet_source=SoftMouseSpreadsheetSource(),
    )
    with pytest.raises(ValueError, match="no valid RFID-tagged"):
        bad.publish()

    assert first.export_path.read_bytes() == old_export
    assert first.manifest_path.read_bytes() == old_manifest


def test_suspicious_source_shrink_never_replaces_publication(tmp_path):
    destination = tmp_path / "shared"
    initial = SoftMouseExportPublisher(
        export_source=CsvExportSource(
            [[f"PT-{index}", TAG[:-1] + str(index), "Stock"] for index in range(3)]
        ),
        destination_directory=destination,
        spreadsheet_source=SoftMouseSpreadsheetSource(),
    ).publish()
    old_manifest = initial.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="potentially filtered"):
        SoftMouseExportPublisher(
            export_source=CsvExportSource([["PT-1", TAG, "Stock"]]),
            destination_directory=destination,
            spreadsheet_source=SoftMouseSpreadsheetSource(),
        ).publish()

    assert initial.manifest_path.read_bytes() == old_manifest


def test_manifest_publication_failure_rolls_back_canonical_export(tmp_path, monkeypatch):
    destination = tmp_path / "shared"
    publisher = SoftMouseExportPublisher(
        export_source=CsvExportSource([["PT-1", TAG, "Stock"]]),
        destination_directory=destination,
        spreadsheet_source=SoftMouseSpreadsheetSource(),
    )
    first = publisher.publish()
    old_export = first.export_path.read_bytes()
    old_manifest = first.manifest_path.read_bytes()

    replacement = SoftMouseExportPublisher(
        export_source=CsvExportSource([["PT-2", TAG[:-1] + "1", "Stock"]]),
        destination_directory=destination,
        spreadsheet_source=SoftMouseSpreadsheetSource(),
    )
    monkeypatch.setattr(
        replacement,
        "_atomic_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest failed")),
    )

    with pytest.raises(OSError, match="manifest failed"):
        replacement.publish()

    assert first.export_path.read_bytes() == old_export
    assert first.manifest_path.read_bytes() == old_manifest


def test_publisher_refuses_overlapping_run(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    destination = tmp_path / "shared"
    destination.mkdir()
    lock_path = destination / ".publish.lock"
    publisher = SoftMouseExportPublisher(
        export_source=CsvExportSource([["PT-1", TAG, "Stock"]]),
        destination_directory=destination,
        spreadsheet_source=SoftMouseSpreadsheetSource(),
        lock_path=lock_path,
    )
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            publisher.publish()
