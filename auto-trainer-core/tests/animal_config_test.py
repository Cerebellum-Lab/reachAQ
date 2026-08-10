import json

import pytest

from autotrainer.core import AnimalSubject


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
    animal.training.protocols = [{
        "plan_id": "protocol-1",
        "progress": [{"session_count": 3}],
    }]
    destination = tmp_path / "animal.json"

    animal.to_file(destination)

    persisted = json.loads(destination.read_text())
    assert persisted["version"] == 5
    assert persisted["training"]["selectedProtocol"] == "protocol-1"
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

    assert json.loads(destination.read_text())["version"] == 5
    assert json.loads(
        destination.with_suffix(".json.v4-backup").read_text()
    ) == legacy


@pytest.mark.parametrize("version", [None, 0, 1, 2, 3])
def test_unsupported_animal_versions_fail_clearly(tmp_path, version):
    destination = tmp_path / "animal.json"
    destination.write_text(json.dumps({"version": version, "name": "old"}))

    with pytest.raises(ValueError, match="only v4 migration and v5"):
        AnimalSubject.from_file(destination)
