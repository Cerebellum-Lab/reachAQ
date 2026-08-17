import json

import pytest

from autotrainer.core import AnimalSubject
from autotrainer.core.animal.external_metadata import (
    ExternalIdentity,
    ExternalMetadataSnapshot,
)


def test_save_load(tmp_path):
    animal = AnimalSubject(
        name="animal1",
        is_pellet_dcs=True,
        pellet_x=-1,
        pellet_y=1,
        pellet_z=2,
        target_y_limit=0.88,
    )
    assert isinstance(animal.id, str) and len(animal.id) > 0
    dest = tmp_path.joinpath("animal.json")
    animal.to_file(dest)
    animal2 = AnimalSubject.from_file(dest)
    assert animal == animal2
    # also ensure for change to any attribute:
    for k, v in animal.__dict__.items():
        if k == "version":  # forced on write
            continue
        if isinstance(v, bool):
            v = not v
        elif isinstance(v, (int, float)):
            v += 1
        elif isinstance(v, str):
            v += "1"
        else:
            continue
        setattr(animal, k, v)
        assert animal != animal2
        animal.to_file(dest)
        animal2 = AnimalSubject.from_file(dest)
        assert animal == animal2, f"attribute {k} is not correctly saved/loaded"


def test_v5_persists_trial_progress_without_retired_fields(tmp_path):
    animal = AnimalSubject(name="animal1")
    animal.training.current_protocol = "protocol-1"
    animal.training.ordered_protocol_id = "ordered-a"
    animal.training.protocols = [{
        "plan_id": "protocol-1",
        "progress": [{"session_count": 3}],
    }]
    destination = tmp_path / "animal.json"

    animal.to_file(destination)

    persisted = json.loads(destination.read_text())
    assert persisted["version"] == 8
    assert persisted["training"]["selectedProtocol"] == "protocol-1"
    assert persisted["training"]["selectedOrderedProtocol"] == "ordered-a"
    phase = persisted["training"]["protocolProgress"][0]["progress"][0]
    assert phase["trial_count"] == 3
    assert "session_count" not in phase
    assert "autoclampEvasionPelletsConsumed" not in persisted
    loaded = AnimalSubject.from_file(destination)
    assert loaded.training.protocols[0]["progress"][0]["session_count"] == 3


def test_v4_migration_preserves_backup_and_resets_progress(tmp_path):
    destination = tmp_path / "animal.json"
    legacy = {
        "version": 4,
        "id": "animal-id",
        "name": "animal1",
        "reach": {"pelletDcs": {"x": 1, "y": 2, "z": 3}},
        "training": {
            "currentProtocol": "protocol-1",
            "protocols": [{"plan_id": "protocol-1", "session_count": 99}],
        },
        "targetYLimit": 4,
        "autoclampEvasionPelletsConsumed": 7,
    }
    destination.write_text(json.dumps(legacy))

    animal = AnimalSubject.from_file(destination)
    assert animal.training.current_protocol == "protocol-1"
    assert animal.training.protocols == []
    animal.to_file(destination)

    assert json.loads(destination.read_text())["version"] == 8
    assert json.loads(
        destination.with_suffix(".json.v4-backup").read_text()
    ) == legacy


@pytest.mark.parametrize("version", [None, 0, 1, 2, 3])
def test_unsupported_animal_versions_fail_clearly(tmp_path, version):
    destination = tmp_path / "animal.json"
    destination.write_text(json.dumps({"version": version, "name": "old"}))

    with pytest.raises(ValueError, match="only v4/v5/v6/v7 migration and v8"):
        AnimalSubject.from_file(destination)


def test_v5_migrates_with_backup(tmp_path):
    destination = tmp_path / "animal.json"
    legacy = {
        "version": 5,
        "id": "animal-id",
        "name": "animal1",
        "pellet": {
            "coordinateSpace": "device",
            "position": {"x": 1, "y": 2, "z": 3},
        },
        "training": {"selectedProtocol": None, "protocolProgress": []},
        "limits": {"targetY": None},
    }
    destination.write_text(json.dumps(legacy))

    animal = AnimalSubject.from_file(destination)
    animal.to_file(destination)

    assert json.loads(destination.read_text())["version"] == 8
    assert json.loads(destination.with_suffix(".json.v5-backup").read_text()) == legacy


def test_external_identity_metadata_and_session_snapshot_round_trip(tmp_path):
    animal = AnimalSubject(
        name="stable-local-name",
        notes="animal-level note",
        external_identity=ExternalIdentity("PT-42"),
        external_metadata=ExternalMetadataSnapshot(
            rfid="360002353933099",
            physical_tag="PT-42",
            sex="F",
            genotype=("Cre+", "WT"),
            state="Stock",
            source_hash="record-hash",
            registry_import_id="import-id",
            source_file_sha256="file-hash",
            imported_utc="2026-08-10T00:00:00Z",
        ),
    )
    destination = tmp_path / "animal.json"

    animal.to_file(destination)
    loaded = AnimalSubject.from_file(destination)

    assert loaded == animal
    snapshot = loaded.session_snapshot(snapshot_utc="2026-08-10T01:00:00Z")
    assert snapshot["name"] == "stable-local-name"
    assert snapshot["notes"] == "animal-level note"
    assert snapshot["externalIdentity"]["subjectId"] == "PT-42"
    assert snapshot["externalIdentity"]["rfid"] == "360002353933099"
    assert snapshot["provenance"]["sourceRecordHash"] == "record-hash"


def test_v6_migrates_with_empty_notes_and_backup(tmp_path):
    destination = tmp_path / "animal.json"
    legacy = {
        "version": 6,
        "id": "animal-id",
        "name": "animal1",
        "pellet": {
            "coordinateSpace": "device",
            "position": {"x": 1, "y": 2, "z": 3},
        },
        "training": {"selectedProtocol": None, "protocolProgress": []},
        "limits": {"targetY": None},
        "externalIdentity": None,
        "externalMetadata": None,
    }
    destination.write_text(json.dumps(legacy))

    animal = AnimalSubject.from_file(destination)
    assert animal.notes == ""
    animal.to_file(destination)

    assert json.loads(destination.read_text())["version"] == 8
    assert json.loads(destination.with_suffix(".json.v6-backup").read_text()) == legacy


def test_v7_migrates_ordered_protocol_preference(tmp_path):
    destination = tmp_path / "animal.json"
    legacy = {
        "version": 7,
        "id": "animal-id",
        "name": "animal1",
        "notes": "keep",
        "pellet": {
            "coordinateSpace": "dcs",
            "position": {"x": 1, "y": 2, "z": 3},
        },
        "training": {
            "selectedProtocol": "training-plan",
            "selectedOrderedProtocol": "ordered-a",
            "protocolProgress": [],
        },
        "limits": {"targetY": None},
        "externalIdentity": None,
        "externalMetadata": None,
    }
    destination.write_text(json.dumps(legacy))

    animal = AnimalSubject.from_file(destination)
    animal.to_file(destination)

    persisted = json.loads(destination.read_text())
    assert persisted["version"] == 8
    assert persisted["training"]["selectedProtocol"] == "training-plan"
    assert persisted["training"]["selectedOrderedProtocol"] == "ordered-a"
    assert json.loads(destination.with_suffix(".json.v7-backup").read_text()) == legacy
