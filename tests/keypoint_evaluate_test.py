"""Scoring a pose model on held-out labelled frames.

Raw median error rewards a model that reports only the easy keypoints, so the
score is taken at matched coverage: the same share of keypoints, each model's
most confident first. Detection rate is reported beside it at the gate live
inference uses.
"""

import numpy
import pytest

from autotrainer.inference.keypoint.evaluate import (  # noqa: E402
    INDICATIVE_BELOW,
    match_parts,
    score,
)

PARTS = ["a", "b"]


def _case(errors, scores):
    """Labels at the origin, predictions `errors` px away along x, part a only."""
    n = len(errors)
    labels = numpy.full((n, 2, 2), numpy.nan, "float32")
    labels[:, 0] = 0
    predicted = numpy.zeros((n, 2, 3), "float32")
    predicted[:, 0, 0] = errors
    predicted[:, 0, 2] = scores
    return predicted, labels


def test_coverage_median_takes_the_most_confident_share_first():
    errors = [1, 1, 1, 1, 9, 9, 9, 9]
    scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15]
    result = score(*_case(errors, scores), PARTS, gate=0.1)
    assert result["n"] == 8
    assert result["coverage_median"][50] == pytest.approx(1)
    assert result["coverage_median"][100] == pytest.approx(5)
    assert result["coverage_median"][25] == pytest.approx(1)


def test_detection_rate_counts_keypoints_at_or_above_the_gate():
    result = score(*_case([1, 1, 1, 1], [0.5, 0.1, 0.09, 0.0]), PARTS, gate=0.1)
    assert result["detected"] == pytest.approx(0.5)
    assert result["parts"]["a"]["detected"] == pytest.approx(0.5)


def test_a_part_with_few_held_out_labels_is_indicative():
    result = score(*_case([1] * 3, [0.9] * 3), PARTS, gate=0.1)
    assert INDICATIVE_BELOW == 5
    assert result["parts"]["a"]["indicative"]
    assert result["parts"]["b"]["n"] == 0


def test_an_empty_held_out_set_scores_without_raising():
    predicted = numpy.zeros((0, 2, 3), "float32")
    labels = numpy.zeros((0, 2, 2), "float32")
    result = score(predicted, labels, PARTS, gate=0.1)
    assert result["n"] == 0
    assert numpy.isnan(result["coverage_median"][100])


def test_parts_are_matched_by_name_and_missing_ones_reported():
    model_parts = ["b", "x", "a"]
    predicted = numpy.arange(2 * 3 * 3, dtype="float32").reshape(2, 3, 3)
    mapped, missing = match_parts(predicted, model_parts, ["a", "b", "c"])
    numpy.testing.assert_array_equal(mapped[:, 0], predicted[:, 2])
    numpy.testing.assert_array_equal(mapped[:, 1], predicted[:, 0])
    assert numpy.isnan(mapped[:, 2, :2]).all()
    assert (mapped[:, 2, 2] == 0).all()
    assert missing == ["c"]
