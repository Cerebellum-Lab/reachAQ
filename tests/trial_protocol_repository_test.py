import json

import pytest

from tools.acquisition.model.trial_protocol_repository import (
    TrialProtocolRepository,
)
from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    TrialProtocolDocument,
)


def _document(protocol_id="shared-a", name="Shared A"):
    return TrialProtocolDocument(
        protocol_id=protocol_id,
        name=name,
        trial_count=3,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )


def test_save_reload_and_revision_are_atomic(tmp_path):
    repository = TrialProtocolRepository(tmp_path / "protocols")

    first = repository.save(_document())
    second = repository.save(first, expected_revision=first.revision)
    loaded = TrialProtocolRepository(repository.root)

    assert first.revision == 1
    assert second.revision == 2
    assert loaded.reload()[0].to_record() == second.to_record()
    assert not tuple(repository.root.glob("*.tmp"))


def test_revision_conflict_does_not_overwrite(tmp_path):
    repository = TrialProtocolRepository(tmp_path)
    first = repository.save(_document())

    with pytest.raises(RuntimeError, match="changed"):
        repository.save(first, expected_revision=99)

    assert repository.get("shared-a").revision == 1


def test_corrupt_file_isolated_and_last_valid_cached(tmp_path):
    repository = TrialProtocolRepository(tmp_path)
    saved = repository.save(_document())
    other = repository.save(_document("shared-b", "Shared B"))
    (tmp_path / "shared-a.json").write_text("{bad", encoding="utf-8")

    loaded = repository.reload()

    assert {item.protocol_id for item in loaded} == {
        saved.protocol_id,
        other.protocol_id,
    }
    assert repository.get("shared-a") == saved
    assert "shared-a.json" in next(iter(repository.errors))


def test_fresh_repository_skips_corrupt_file(tmp_path):
    (tmp_path / "broken.json").write_text("[]", encoding="utf-8")
    repository = TrialProtocolRepository(tmp_path)

    assert repository.reload() == ()
    assert repository.errors


def test_duplicate_rename_import_and_export(tmp_path):
    repository = TrialProtocolRepository(tmp_path / "library")
    original = repository.save(_document())
    duplicated = repository.duplicate(
        original.protocol_id,
        protocol_id="shared-copy",
        name="Shared Copy",
    )
    renamed = repository.rename(
        duplicated.protocol_id,
        protocol_id="shared-renamed",
        name="Shared Renamed",
        expected_revision=duplicated.revision,
    )
    exported_path = repository.export_file(
        renamed.protocol_id,
        tmp_path / "exported.json",
    )

    imported_repo = TrialProtocolRepository(tmp_path / "imported")
    imported = imported_repo.import_file(exported_path)

    assert imported.protocol_id == "shared-renamed"
    assert imported.name == "Shared Renamed"
    assert repository.get("shared-copy") is None
    assert tuple((tmp_path / "library").glob("*.renamed-backup"))
    with exported_path.open("r", encoding="utf-8") as stream:
        assert json.load(stream)["protocol_id"] == "shared-renamed"
