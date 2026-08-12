import json
from pathlib import Path
import shutil

import pytest

from autotrainer.core import AnimalSubject, ExternalIdentity
from tools.acquisition.model.animal_reconciliation import (
    AnimalReconciliationChoices,
    reconcile_animals,
)


def choices(survivor, loser):
    return AnimalReconciliationChoices(
        name_from=survivor.id,
        pellet_position_from=loser.id,
        training_from=survivor.id,
        target_limit_from=loser.id,
        external_identity_from=loser.id,
    )


def test_reconciliation_uses_explicit_fields_and_always_preserves_survivor_uuid():
    survivor = AnimalSubject(name="manual", pellet_x=1, target_y_limit=2)
    loser = AnimalSubject(
        name="scanned",
        pellet_x=9,
        target_y_limit=8,
        external_identity=ExternalIdentity("PT-1"),
    )

    result = reconcile_animals(survivor, loser, choices(survivor, loser))

    assert result.id == survivor.id
    assert result.name == "manual"
    assert result.pellet_x == 9
    assert result.target_y_limit == 8
    assert result.external_identity.subject_id == "PT-1"
    assert survivor.pellet_x == 1


def test_app_model_condensation_archives_loser_and_writes_redirect(app_model):
    survivor = app_model.add_animal("manual")
    loser = app_model.add_animal("scanned")
    loser.external_identity = ExternalIdentity("PT-1")
    app_model._save_animal_metadata(loser, sender="test")

    merged = app_model.condense_animals(
        survivor.id, loser.id, choices(survivor, loser)
    )

    directory = Path(app_model._preferences.animal_location)
    redirect = json.loads(
        (directory / f"{loser.id}.merged-redirect").read_text()
    )
    assert merged.id == survivor.id
    assert app_model.get_animal_by_id(loser.id) is None
    assert redirect["survivingReachaqId"] == survivor.id
    assert not (directory / f"{loser.id}.json").exists()


def test_app_model_condensation_rolls_back_mid_transaction(app_model, monkeypatch):
    survivor = app_model.add_animal("manual")
    loser = app_model.add_animal("scanned")
    directory = Path(app_model._preferences.animal_location)
    survivor_path = directory / f"{survivor.id}.json"
    loser_path = directory / f"{loser.id}.json"
    original_survivor = survivor_path.read_bytes()
    original_loser = loser_path.read_bytes()

    from tools.acquisition.model import app_model as app_model_module

    real_replace = app_model_module.os.replace
    call_count = 0

    def fail_redirect_once(source, destination):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise OSError("injected redirect publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(app_model_module.os, "replace", fail_redirect_once)

    with pytest.raises(OSError, match="injected"):
        app_model.condense_animals(
            survivor.id, loser.id, choices(survivor, loser)
        )

    assert survivor_path.read_bytes() == original_survivor
    assert loser_path.read_bytes() == original_loser
    assert not (directory / f"{loser.id}.merged-redirect").exists()
    assert app_model.get_animal_by_id(survivor.id) is survivor
    assert app_model.get_animal_by_id(loser.id) is loser


def test_animal_load_refuses_duplicate_uuid_files(app_model):
    animal = app_model.add_animal("duplicate")
    directory = Path(app_model._preferences.animal_location)
    shutil.copy2(directory / f"{animal.id}.json", directory / "copied-animal.json")

    with pytest.raises(ValueError, match="Duplicate reachAQ animal UUID"):
        app_model._load_animals()


def test_animal_load_skips_corrupt_file_without_modifying_it(app_model):
    valid = app_model.add_animal("valid")
    directory = Path(app_model._preferences.animal_location)
    corrupt_path = directory / "corrupt.json"
    corrupt_contents = b'{"schemaVersion": 5, "broken":'
    corrupt_path.write_bytes(corrupt_contents)
    errors = []
    app_model.on_error = lambda title, detail: errors.append((title, detail))

    app_model._load_animals()

    assert [animal.id for animal in app_model.animals] == [valid.id]
    assert corrupt_path.read_bytes() == corrupt_contents
    assert errors == [
        (
            "Some animal files were skipped",
            "corrupt.json: Expecting value: line 1 column 31 (char 30)",
        )
    ]
