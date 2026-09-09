import pytest

from autotrainer.core.delay_distribution import (
    PUBLISHED_4S_VALUES,
    DelayPreset,
    build_delay_distribution_profile,
)
from tools.acquisition.model.delay_distribution_report import (
    RealizedDelay,
    build_delay_report,
)


PROFILE = build_delay_distribution_profile(PUBLISHED_4S_VALUES)
CUSTOM = build_delay_distribution_profile(
    (305, 600, 1200), DelayPreset.CUSTOM, manual_probabilities=(0.5, 0.25, 0.25)
)


def test_counts_and_percentages_reflect_the_realized_draws():
    report = build_delay_report("published", CUSTOM, [305, 305, 600, 1200])
    assert report.total_draws == 4
    assert report.values == (305, 600, 1200)
    assert report.counts == (2, 1, 1)
    assert [item.percentage for item in report.occurrences] == [50.0, 25.0, 25.0]


def test_configured_probabilities_are_carried_alongside():
    report = build_delay_report("published", CUSTOM, [305, 600, 1200, 305])
    assert [item.configured_probability for item in report.occurrences] == [
        0.5,
        0.25,
        0.25,
    ]


def test_deviation_is_realized_minus_configured():
    report = build_delay_report("published", CUSTOM, [305, 305, 305, 305])
    deviations = [item.deviation for item in report.occurrences]
    assert deviations[0] == pytest.approx(0.5)
    assert deviations[1] == pytest.approx(-0.25)
    assert report.max_abs_deviation == pytest.approx(0.5)


def test_configured_values_that_never_occurred_are_still_reported():
    report = build_delay_report("published", CUSTOM, [305])
    assert report.counts == (1, 0, 0)
    assert report.values == (305, 600, 1200)


def test_an_empty_session_reports_zero_without_dividing_by_zero():
    report = build_delay_report("published", CUSTOM, [])
    assert report.total_draws == 0
    assert report.counts == (0, 0, 0)
    assert all(item.percentage == 0.0 for item in report.occurrences)
    assert report.max_abs_deviation == pytest.approx(0.5)


def test_intervals_outside_the_support_are_surfaced_not_folded_in():
    report = build_delay_report("published", CUSTOM, [305, 999, 999])
    assert report.unexpected_delays == (999,)
    assert report.total_draws == 3
    # The unexpected draws still count toward the denominator.
    assert report.occurrences[0].percentage == pytest.approx(100.0 / 3)


def test_realized_delays_accept_records_objects_and_integers():
    report = build_delay_report(
        "published",
        CUSTOM,
        [
            RealizedDelay(delay_ms=305, logical_trial_id=1),
            {"delay_ms": 600, "probability": 0.25, "index": 1},
            1200,
        ],
    )
    assert report.counts == (1, 1, 1)


def test_a_record_without_a_delay_is_rejected():
    with pytest.raises(ValueError, match="missing delay_ms"):
        build_delay_report("published", CUSTOM, [{"index": 0}])


@pytest.mark.parametrize("item", [True, 305.5, "305", None])
def test_non_integer_delays_are_rejected(item):
    with pytest.raises(ValueError, match="integer milliseconds"):
        build_delay_report("published", CUSTOM, [item])


def test_plot_series_carries_realized_and_reference_curves():
    report = build_delay_report("published", PROFILE, [305, 500, 500, 4000])
    series = report.plot_series()
    assert series["values_ms"] == list(PUBLISHED_4S_VALUES)
    assert series["realized_counts"][:2] == [1, 2]
    assert series["realized_cdf"][-1] == pytest.approx(1.0)
    assert len(series["configured_cdf"]) == len(PUBLISHED_4S_VALUES)
    assert len(series["reference_pdf_per_second"]) == len(PUBLISHED_4S_VALUES)
    assert series["plot_contract"].startswith("discrete_pmf_cdf_vs_continuous")


def test_plot_series_realized_cdf_is_flat_for_an_empty_session():
    series = build_delay_report("published", CUSTOM, []).plot_series()
    assert series["realized_cdf"] == [0.0, 0.0, 0.0]


def test_record_is_plain_and_carries_configuration_and_results():
    record = build_delay_report("published", PROFILE, [305, 4000]).to_record()
    assert record["schema_version"] == 1
    assert record["profile_id"] == "published"
    assert record["total_draws"] == 2
    assert record["distribution"]["preset"] == "published_4s"
    assert record["distribution"]["source"].startswith("paper_table_4s")
    assert record["occurrences"][0]["delay_ms"] == 305
    assert record["occurrences"][0]["count"] == 1
    assert "plot_series" in record
    assert record["unexpected_delays_ms"] == []
