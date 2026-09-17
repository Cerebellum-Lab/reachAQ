import math

import pytest

from autotrainer.core.delay_distribution import (
    DEFAULT_DELAY_PRESET,
    EXPONENTIAL_CDF_OFFSET_MS,
    EXPONENTIAL_CDF_TAU_MS,
    MAXIMUM_EXPONENTIAL_CDF_TAU_MS,
    MINIMUM_CUE_INTERVAL_MS,
    MINIMUM_EXPONENTIAL_CDF_TAU_MS,
    PUBLISHED_4S_PROBABILITIES,
    PUBLISHED_4S_VALUES,
    PUBLISHED_5S_PROBABILITIES,
    PUBLISHED_5S_VALUES,
    PUBLISHED_SHORT_REPORTED_PROBABILITIES,
    PUBLISHED_SHORT_VALUES,
    DelayPreset,
    DelayProbabilityMode,
    build_delay_distribution_profile,
    infer_delay_preset,
    normalize_cue_intervals,
    normalize_delay_preset,
    normalize_probability_mode,
    normalize_tau_ms,
    preset_values,
    select_delay,
    shifted_exponential_cdf,
    shifted_exponential_pdf_per_second,
    validate_manual_probabilities,
)


def test_published_4s_support_and_weights_are_exact():
    profile = build_delay_distribution_profile(PUBLISHED_4S_VALUES)
    assert profile.preset is DelayPreset.PUBLISHED_4S
    assert profile.probability_mode is DelayProbabilityMode.PUBLISHED
    assert profile.values == PUBLISHED_4S_VALUES
    assert profile.probabilities == PUBLISHED_4S_PROBABILITIES
    assert math.isclose(sum(profile.probabilities), 1.0, abs_tol=1e-12)


def test_published_5s_support_and_weights_are_exact():
    profile = build_delay_distribution_profile(PUBLISHED_5S_VALUES)
    assert profile.preset is DelayPreset.PUBLISHED_5S
    assert profile.probabilities == PUBLISHED_5S_PROBABILITIES
    assert math.isclose(sum(profile.probabilities), 1.0, abs_tol=1e-12)


def test_published_short_weights_are_renormalized_from_reported_sum():
    profile = build_delay_distribution_profile(PUBLISHED_SHORT_VALUES)
    assert profile.preset is DelayPreset.PUBLISHED_SHORT
    assert profile.reported_probabilities == PUBLISHED_SHORT_REPORTED_PROBABILITIES
    assert math.isclose(profile.reported_probability_sum, 0.99, abs_tol=1e-9)
    assert math.isclose(sum(profile.probabilities), 1.0, abs_tol=1e-12)
    # Renormalization is proportional, so ordering and ratios are preserved.
    assert math.isclose(
        profile.probabilities[1] / profile.probabilities[0],
        PUBLISHED_SHORT_REPORTED_PROBABILITIES[1]
        / PUBLISHED_SHORT_REPORTED_PROBABILITIES[0],
        rel_tol=1e-12,
    )


def test_published_preset_rejects_edited_support():
    edited = PUBLISHED_4S_VALUES[:-1] + (4100,)
    with pytest.raises(ValueError, match="fixed support"):
        build_delay_distribution_profile(edited, DelayPreset.PUBLISHED_4S)


def test_published_preset_rejects_manual_probabilities():
    with pytest.raises(ValueError, match="do not accept manual"):
        build_delay_distribution_profile(
            PUBLISHED_4S_VALUES,
            DelayPreset.PUBLISHED_4S,
            manual_probabilities=[1.0 / 8] * 8,
        )


def test_explicit_custom_preset_recalculates_a_published_support():
    profile = build_delay_distribution_profile(
        PUBLISHED_4S_VALUES, DelayPreset.CUSTOM
    )
    assert profile.preset is DelayPreset.CUSTOM
    assert profile.probability_mode is DelayProbabilityMode.CALCULATED
    assert profile.probabilities != PUBLISHED_4S_PROBABILITIES


def test_unknown_explicit_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown cue interval delay preset"):
        build_delay_distribution_profile(PUBLISHED_4S_VALUES, "published_9s")


def test_calculated_weights_total_one_and_carry_the_tail():
    values = (305, 600, 1200, 2400)
    profile = build_delay_distribution_profile(values, DelayPreset.CUSTOM)
    assert math.isclose(sum(profile.probabilities), 1.0, abs_tol=1e-12)
    assert profile.bucket_boundaries_ms[0] == float(EXPONENTIAL_CDF_OFFSET_MS)
    assert profile.bucket_boundaries_ms[-1] == "+infinity"
    # Internal boundaries are adjacent-value midpoints.
    assert profile.bucket_boundaries_ms[1] == pytest.approx((305 + 600) / 2.0)


def test_single_value_support_takes_all_mass():
    profile = build_delay_distribution_profile((900,), DelayPreset.CUSTOM)
    assert profile.probabilities == (1.0,)
    assert profile.bucket_boundaries_ms == (float(EXPONENTIAL_CDF_OFFSET_MS), "+infinity")


def test_calculated_weights_follow_tau():
    values = (305, 600, 1200, 2400)
    short_tau = build_delay_distribution_profile(
        values, DelayPreset.CUSTOM, tau_ms=400
    )
    long_tau = build_delay_distribution_profile(
        values, DelayPreset.CUSTOM, tau_ms=4000
    )
    # A shorter tau concentrates mass at the short end of the support.
    assert short_tau.probabilities[0] > long_tau.probabilities[0]
    assert short_tau.probabilities[-1] < long_tau.probabilities[-1]


def test_manual_probabilities_must_total_one():
    values = (305, 600, 1200)
    with pytest.raises(ValueError, match="must total 100.000%"):
        build_delay_distribution_profile(
            values, DelayPreset.CUSTOM, manual_probabilities=(0.5, 0.4, 0.05)
        )


def test_manual_probabilities_accept_zero_rows():
    values = (305, 600, 1200)
    profile = build_delay_distribution_profile(
        values, DelayPreset.CUSTOM, manual_probabilities=(0.5, 0.0, 0.5)
    )
    assert profile.probability_mode is DelayProbabilityMode.MANUAL
    assert profile.probabilities == (0.5, 0.0, 0.5)
    assert profile.bucket_boundaries_ms is None


def test_manual_probabilities_must_match_support_length():
    with pytest.raises(ValueError, match="must match"):
        validate_manual_probabilities((305, 600), (1.0,))


@pytest.mark.parametrize(
    "values",
    [
        (304,),
        (305, 200),
        (100, 900),
    ],
)
def test_intervals_below_the_safety_floor_are_rejected(values):
    with pytest.raises(ValueError, match=f"{MINIMUM_CUE_INTERVAL_MS} ms or longer"):
        normalize_cue_intervals(values)


def test_duplicate_intervals_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        normalize_cue_intervals((305, 600, 305))


def test_empty_support_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        normalize_cue_intervals(())


@pytest.mark.parametrize("value", [True, 305.5, "305", None])
def test_non_integer_intervals_are_rejected(value):
    with pytest.raises(ValueError, match="integer milliseconds"):
        normalize_cue_intervals((value,))


def test_intervals_are_sorted():
    assert normalize_cue_intervals((1200, 305, 600)) == (305, 600, 1200)


def test_infer_preset_round_trips_published_support():
    assert infer_delay_preset(PUBLISHED_4S_VALUES) is DelayPreset.PUBLISHED_4S
    assert infer_delay_preset(PUBLISHED_5S_VALUES) is DelayPreset.PUBLISHED_5S
    assert infer_delay_preset((305, 999)) is DelayPreset.CUSTOM


def test_normalize_preset_falls_back_to_legacy_support_then_default():
    assert (
        normalize_delay_preset("nonsense", PUBLISHED_5S_VALUES)
        is DelayPreset.PUBLISHED_5S
    )
    assert normalize_delay_preset(None) is DEFAULT_DELAY_PRESET
    assert normalize_delay_preset("published_short") is DelayPreset.PUBLISHED_SHORT


def test_preset_values_returns_none_for_custom():
    assert preset_values(DelayPreset.PUBLISHED_4S) == PUBLISHED_4S_VALUES
    assert preset_values(DelayPreset.CUSTOM) is None


def test_normalize_probability_mode_defaults_to_calculated():
    assert normalize_probability_mode("manual") is DelayProbabilityMode.MANUAL
    assert normalize_probability_mode("nonsense") is DelayProbabilityMode.CALCULATED


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, MINIMUM_EXPONENTIAL_CDF_TAU_MS),
        (10**9, MAXIMUM_EXPONENTIAL_CDF_TAU_MS),
        ("nonsense", EXPONENTIAL_CDF_TAU_MS),
        (True, EXPONENTIAL_CDF_TAU_MS),
        (1400, 1400),
    ],
)
def test_tau_is_clamped_into_the_supported_range(raw, expected):
    assert normalize_tau_ms(raw) == expected


def test_reference_cdf_and_pdf_are_zero_below_the_offset():
    assert shifted_exponential_cdf(EXPONENTIAL_CDF_OFFSET_MS) == 0.0
    assert shifted_exponential_pdf_per_second(EXPONENTIAL_CDF_OFFSET_MS - 1) == 0.0
    assert shifted_exponential_cdf(math.inf) == 1.0


def test_reference_cdf_rejects_non_positive_tau():
    with pytest.raises(ValueError, match="greater than zero"):
        shifted_exponential_cdf(900, 0)


def test_selection_is_deterministic_for_a_given_uniform_value():
    profile = build_delay_distribution_profile(PUBLISHED_4S_VALUES)
    first = select_delay(profile, 0.0)
    assert first.delay_ms == 305
    assert first.index == 0
    # 0.125 + 0.125 = 0.25 lands on the third value.
    assert select_delay(profile, 0.25).delay_ms == 700
    last = select_delay(profile, 0.999999)
    assert last.delay_ms == 4000


def test_selection_skips_zero_probability_rows():
    profile = build_delay_distribution_profile(
        (305, 600, 1200), DelayPreset.CUSTOM, manual_probabilities=(0.5, 0.0, 0.5)
    )
    assert select_delay(profile, 0.0).delay_ms == 305
    assert select_delay(profile, 0.5).delay_ms == 1200
    assert select_delay(profile, 0.999).delay_ms == 1200


@pytest.mark.parametrize("random_value", [-0.1, 1.0, 1.5])
def test_selection_rejects_out_of_range_uniform_values(random_value):
    profile = build_delay_distribution_profile(PUBLISHED_4S_VALUES)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        select_delay(profile, random_value)


def test_selection_covers_every_value_across_the_unit_interval():
    profile = build_delay_distribution_profile(PUBLISHED_4S_VALUES)
    seen = {
        select_delay(profile, index / 10000.0).delay_ms
        for index in range(10000)
    }
    assert seen == set(PUBLISHED_4S_VALUES)


def test_profile_record_is_plain_and_carries_provenance():
    profile = build_delay_distribution_profile(PUBLISHED_SHORT_VALUES)
    record = profile.to_record()
    assert record["preset"] == "published_short"
    assert record["probability_mode"] == "published"
    assert record["source"].startswith("paper_table_short_weights")
    assert record["schema_version"] == 1
    assert record["plot_contract"].startswith("discrete_pmf_cdf_vs_continuous")
    assert math.isclose(
        record["continuous_hazard_rate_per_second"], 1000.0 / 900.0, rel_tol=1e-12
    )


def test_calculated_profile_tracks_distance_from_the_reference_cdf():
    profile = build_delay_distribution_profile(
        (305, 600, 1200, 2400), DelayPreset.CUSTOM
    )
    assert 0.0 <= profile.cdf_max_abs_error < 1.0
    assert profile.categorical_cdf_at_values[-1] == 1.0
    assert profile.continuous_mass_below_first_value > 0.0
    assert profile.continuous_mass_above_last_value > 0.0
