import pytest

from autotrainer.core.stimulus_trigger_profile import (
    STIMULUS_TRIGGER_SAMPLING_ALGORITHM,
    StimulusTriggerCategory,
    distribute_evenly,
    enabled_categories,
    has_offset_trigger,
    normalize_profile,
    profile_record,
    renormalize,
    select_trigger,
    validate_profile,
)


def _profile(*entries):
    return [dict(entry) for entry in entries]


BALANCED = _profile(
    {
        "category_id": "first_reach",
        "trigger": "first_reach",
        "label": "First reach",
        "percentage": 50.0,
    },
    {
        "category_id": "pre_reveal_100",
        "trigger": "pre_reveal",
        "label": "100 ms before reveal",
        "percentage": 50.0,
        "offset_ms": 100,
    },
)


def test_normalize_preserves_order_and_fills_labels():
    normalized = normalize_profile(
        _profile(
            {"category_id": "b", "trigger": "tone_2", "percentage": 40},
            {"category_id": "a", "trigger": "first_reach", "percentage": 60},
        )
    )
    assert [category.category_id for category in normalized] == ["b", "a"]
    assert normalized[0].label == "b"
    assert normalized[0].percentage == 40.0


def test_normalize_rejects_duplicate_category_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_profile(
            _profile(
                {"category_id": "a", "trigger": "tone_2", "percentage": 50},
                {"category_id": "a", "trigger": "first_reach", "percentage": 50},
            )
        )


def test_normalize_rejects_an_empty_profile():
    with pytest.raises(ValueError, match="at least one category"):
        normalize_profile([])


def test_category_requires_an_id_and_a_trigger():
    with pytest.raises(ValueError, match="category ID cannot be empty"):
        StimulusTriggerCategory(category_id="", trigger="tone_2")
    with pytest.raises(ValueError, match="has no trigger"):
        StimulusTriggerCategory(category_id="a", trigger="")


@pytest.mark.parametrize("percentage", [-1, 101, float("nan"), "abc", True])
def test_out_of_range_percentages_are_rejected(percentage):
    with pytest.raises(ValueError):
        normalize_profile(
            _profile(
                {"category_id": "a", "trigger": "tone_2", "percentage": percentage}
            )
        )


@pytest.mark.parametrize("offset", [-1, 10.5, "100", True])
def test_invalid_offsets_are_rejected(offset):
    with pytest.raises(ValueError):
        normalize_profile(
            _profile(
                {
                    "category_id": "a",
                    "trigger": "pre_reveal",
                    "percentage": 100,
                    "offset_ms": offset,
                }
            )
        )


def test_validate_requires_enabled_categories():
    disabled = _profile(
        {
            "category_id": "a",
            "trigger": "tone_2",
            "percentage": 100,
            "enabled": False,
        }
    )
    with pytest.raises(ValueError, match="Enable at least one"):
        validate_profile(disabled)


def test_validate_rejects_enabled_zero_weight_categories():
    with pytest.raises(ValueError, match="greater than zero"):
        validate_profile(
            _profile(
                {"category_id": "a", "trigger": "tone_2", "percentage": 100},
                {"category_id": "b", "trigger": "first_reach", "percentage": 0},
            )
        )


def test_validate_requires_enabled_weights_to_total_100():
    with pytest.raises(ValueError, match=r"total 90\.000%"):
        validate_profile(
            _profile(
                {"category_id": "a", "trigger": "tone_2", "percentage": 50},
                {"category_id": "b", "trigger": "first_reach", "percentage": 40},
            )
        )


def test_validate_ignores_disabled_categories_in_the_total():
    profile = _profile(
        {"category_id": "a", "trigger": "tone_2", "percentage": 100},
        {
            "category_id": "b",
            "trigger": "first_reach",
            "percentage": 80,
            "enabled": False,
        },
    )
    assert len(enabled_categories(validate_profile(profile))) == 1


def test_offsets_must_be_shorter_than_the_minimum_cue_interval():
    with pytest.raises(ValueError, match="shorter than the minimum cue interval"):
        validate_profile(BALANCED, offset_upper_bound_ms=100)


def test_offsets_inside_the_minimum_cue_interval_are_accepted():
    assert validate_profile(BALANCED, offset_upper_bound_ms=305)


def test_has_offset_trigger_ignores_disabled_and_zero_weight_rows():
    assert has_offset_trigger(BALANCED)
    disabled_offset = _profile(
        {"category_id": "a", "trigger": "tone_2", "percentage": 100},
        {
            "category_id": "b",
            "trigger": "pre_reveal",
            "percentage": 0,
            "offset_ms": 100,
            "enabled": False,
        },
    )
    assert not has_offset_trigger(disabled_offset)


def test_distribute_evenly_totals_100_with_a_residual():
    profile = _profile(
        {"category_id": "a", "trigger": "tone_2"},
        {"category_id": "b", "trigger": "tone_1"},
        {"category_id": "c", "trigger": "first_reach"},
    )
    result = distribute_evenly(profile)
    assert sum(category.percentage for category in result) == pytest.approx(100.0)
    assert validate_profile(result)


def test_distribute_evenly_only_touches_enabled_rows():
    profile = _profile(
        {"category_id": "a", "trigger": "tone_2"},
        {"category_id": "b", "trigger": "tone_1", "enabled": False, "percentage": 12},
    )
    result = distribute_evenly(profile)
    assert result[0].percentage == pytest.approx(100.0)
    assert result[1].percentage == pytest.approx(12.0)


def test_renormalize_preserves_ratios():
    profile = _profile(
        {"category_id": "a", "trigger": "tone_2", "percentage": 30},
        {"category_id": "b", "trigger": "tone_1", "percentage": 10},
    )
    result = renormalize(profile)
    assert sum(category.percentage for category in result) == pytest.approx(100.0)
    assert result[0].percentage == pytest.approx(75.0)
    assert result[1].percentage == pytest.approx(25.0)


def test_renormalize_rejects_all_zero_weights():
    profile = _profile(
        {"category_id": "a", "trigger": "tone_2", "percentage": 0},
        {"category_id": "b", "trigger": "tone_1", "percentage": 0},
    )
    with pytest.raises(ValueError, match="total zero"):
        renormalize(profile)


def test_selection_is_deterministic_and_covers_every_category():
    first = select_trigger(BALANCED, 0.0)
    assert first.category.category_id == "first_reach"
    assert first.index == 0
    assert first.probability == pytest.approx(0.5)
    assert first.sampling_algorithm == STIMULUS_TRIGGER_SAMPLING_ALGORITHM

    second = select_trigger(BALANCED, 0.5)
    assert second.category.category_id == "pre_reveal_100"
    assert second.index == 1

    seen = {
        select_trigger(BALANCED, index / 1000.0).category.category_id
        for index in range(1000)
    }
    assert seen == {"first_reach", "pre_reveal_100"}


def test_selection_skips_disabled_categories():
    profile = _profile(
        {
            "category_id": "skipped",
            "trigger": "tone_1",
            "percentage": 90,
            "enabled": False,
        },
        {"category_id": "kept", "trigger": "tone_2", "percentage": 100},
    )
    assert select_trigger(profile, 0.99).category.category_id == "kept"


@pytest.mark.parametrize("random_value", [-0.01, 1.0, 1.5, float("nan")])
def test_selection_rejects_out_of_range_uniform_values(random_value):
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        select_trigger(BALANCED, random_value)


def test_selection_validates_the_profile_first():
    with pytest.raises(ValueError, match="total 50.000%"):
        select_trigger(
            _profile({"category_id": "a", "trigger": "tone_2", "percentage": 50}),
            0.0,
        )


def test_profile_record_carries_the_sampling_algorithm():
    record = profile_record(validate_profile(BALANCED))
    assert record["sampling_algorithm"] == STIMULUS_TRIGGER_SAMPLING_ALGORITHM
    assert record["schema_version"] == 1
    assert [entry["category_id"] for entry in record["categories"]] == [
        "first_reach",
        "pre_reveal_100",
    ]
