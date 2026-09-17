import pytest

from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
)
from tools.acquisition.model.trial_protocol_set import (
    SET_SCHEMA_VERSION,
    TrialProtocolSet,
)


def make_set(**overrides):
    values = dict(
        set_id="baseline",
        name="Baseline",
        trial_count=4,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )
    values.update(overrides)
    return TrialProtocolSet(**values)


def test_a_set_round_trips_through_its_record():
    original = make_set(
        bulk_overrides=(
            ProtocolScope.create("late", [3, 4], {"pre_reveal_ms": 50}),
        ),
        trial_overrides=(TrialOverride.create(2, {"enabled": False}),),
        description="Four warm-up trials",
    )

    restored = TrialProtocolSet.from_record(original.to_record())

    assert restored == original


def test_set_id_is_normalised():
    assert make_set(set_id="  BaseLine  ").set_id == "baseline"


def test_a_set_rejects_an_empty_name():
    with pytest.raises(ValueError, match="name cannot be empty"):
        make_set(name="   ")


def test_a_set_rejects_a_scope_outside_its_own_range():
    with pytest.raises(ValueError, match="out-of-range"):
        make_set(
            bulk_overrides=(ProtocolScope.create("late", [9], {"enabled": True}),)
        )


def test_a_set_rejects_a_scope_that_names_a_parent_epoch():
    with pytest.raises(ValueError, match="parent epoch"):
        make_set(
            bulk_overrides=(
                ProtocolScope.create(
                    "late", [1], {"enabled": True}, parent_epoch="phase"
                ),
            )
        )


def test_a_set_rejects_duplicate_trial_overrides():
    with pytest.raises(ValueError, match="Duplicate trial override"):
        make_set(
            trial_overrides=(
                TrialOverride.create(2, {"enabled": False}),
                TrialOverride.create(2, {"enabled": True}),
            )
        )


def test_a_set_rejects_a_trial_override_outside_its_range():
    with pytest.raises(ValueError, match="outside the set range"):
        make_set(trial_overrides=(TrialOverride.create(9, {"enabled": True}),))


def test_a_set_refuses_an_unknown_schema():
    record = make_set().to_record()
    record["schema_version"] = SET_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="Unsupported set schema"):
        TrialProtocolSet.from_record(record)
