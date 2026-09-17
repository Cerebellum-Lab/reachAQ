import pytest

from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
    TrialProtocolDocument,
)
from tools.acquisition.model.trial_protocol_set import (
    document_from_set,
    set_from_document,
)


def make_document(**overrides):
    values = dict(
        protocol_id="draft",
        name="Draft",
        trial_count=4,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
        bulk_overrides=(
            ProtocolScope.create("late", [3, 4], {"automatic_window_size": 25}),
        ),
        trial_overrides=(
            TrialOverride.create(1, {"automatic_window_size": 5}),
        ),
    )
    values.update(overrides)
    return TrialProtocolDocument(**values)


def test_a_document_converts_to_a_set_with_the_same_trial_content():
    result = set_from_document(make_document(), set_id="baseline", name="Baseline")

    assert result.set_id == "baseline"
    assert result.trial_count == 4
    assert result.defaults.to_mapping() == {"enabled": True}
    assert [item.name for item in result.bulk_overrides] == ["late"]
    assert [item.trial_id for item in result.trial_overrides] == [1]


def test_converting_a_document_that_has_epochs_is_refused():
    document = make_document(
        epochs=(ProtocolScope.create("phase", [1, 2], {"enabled": True}),)
    )

    with pytest.raises(ValueError, match="epochs or blocks"):
        set_from_document(document, set_id="baseline", name="Baseline")


def test_converting_a_document_that_has_blocks_is_refused():
    document = make_document(
        epochs=(ProtocolScope.create("phase", [1, 2], {"enabled": True}),),
        blocks=(
            ProtocolScope.create(
                "half", [1], {"enabled": True}, parent_epoch="phase"
            ),
        ),
    )

    with pytest.raises(ValueError, match="epochs or blocks"):
        set_from_document(document, set_id="baseline", name="Baseline")


def test_a_set_converts_back_into_an_editable_document():
    original = make_document()
    protocol_set = set_from_document(original, set_id="baseline", name="Baseline")

    document = document_from_set(
        protocol_set, protocol_id="baseline-draft", name="Baseline draft"
    )

    assert document.protocol_id == "baseline-draft"
    assert document.trial_count == original.trial_count
    assert document.epochs == ()


def test_a_round_trip_preserves_every_resolved_row():
    original = make_document()

    protocol_set = set_from_document(original, set_id="baseline", name="Baseline")
    restored = document_from_set(protocol_set, protocol_id="draft", name="Draft")

    assert [item.row for item in restored.resolve()] == [
        item.row for item in original.resolve()
    ]


def test_app_model_captures_a_protocol_as_a_set(app_model, tmp_path):
    from tools.acquisition.model.trial_protocol_repository import (
        TrialProtocolRepository,
    )
    from tools.acquisition.model.trial_protocol_set_repository import (
        TrialProtocolSetRepository,
    )

    app_model._trial_protocol_repository = TrialProtocolRepository(tmp_path / "p")
    app_model._trial_protocol_repository.reload()
    app_model._trial_protocol_repository.save(make_document())
    app_model._trial_protocol_set_repository = TrialProtocolSetRepository(
        tmp_path / "s"
    )
    app_model._trial_protocol_set_repository.reload()

    saved = app_model.save_protocol_as_set("draft", "baseline", "Baseline")

    assert saved.set_id == "baseline"
    assert saved.trial_count == 4


def test_app_model_opens_a_set_as_an_editable_protocol(app_model, tmp_path):
    from tools.acquisition.model.trial_protocol_repository import (
        TrialProtocolRepository,
    )
    from tools.acquisition.model.trial_protocol_set_repository import (
        TrialProtocolSetRepository,
    )

    app_model._trial_protocol_repository = TrialProtocolRepository(tmp_path / "p")
    app_model._trial_protocol_repository.reload()
    sets = TrialProtocolSetRepository(tmp_path / "s")
    sets.reload()
    sets.save(set_from_document(make_document(), set_id="baseline", name="Baseline"))
    app_model._trial_protocol_set_repository = sets

    saved = app_model.open_set_as_protocol("baseline", "baseline-draft", "Draft")

    assert saved.protocol_id == "baseline-draft"
    assert app_model._trial_protocol_repository.get("baseline-draft") is not None


def test_capturing_an_unknown_protocol_is_refused(app_model, tmp_path):
    from tools.acquisition.model.trial_protocol_repository import (
        TrialProtocolRepository,
    )

    app_model._trial_protocol_repository = TrialProtocolRepository(tmp_path / "p")
    app_model._trial_protocol_repository.reload()

    with pytest.raises(ValueError, match="missing"):
        app_model.save_protocol_as_set("missing", "baseline", "Baseline")
