"""Equivalence tests for the numpy live 2D confidence vote.

`_select_most_likely_2d` replaced a pandas implementation that cost a scalar
`.loc` read and write per part per camera on every live batch. These tests pin
the new path against the retained pandas `_take_cams_most_likely` so the
optimization cannot silently change which frame wins or which parts are
considered confident.
"""

import numpy
import pandas
import pytest

from autotrainer.inference.pose_algorithm import PoseAlgorithm


THRESHOLD = PoseAlgorithm.MIN_CONFIDENCE_PRESENT_THRESHOLD


def _algorithm():
    return PoseAlgorithm()


def _pandas_reference(algorithm, per_cam_detection):
    """The original pandas path, as `_handle_3d_triangulate` used to call it."""
    frames_per_cam = len(per_cam_detection[0])
    columns = algorithm._measure_offset_parts_columns
    dfs = [
        pandas.DataFrame(cam.reshape(frames_per_cam, -1), columns=columns)
        for cam in per_cam_detection
    ]
    df_2d = algorithm._take_cams_most_likely(*dfs)
    confident_parts = [
        part
        for part in df_2d.columns.levels[0]
        if all(df_2d[part]["likelihood"] >= THRESHOLD)
    ]
    values = df_2d.values.astype(float).reshape(len(dfs), -1, 3)
    return values, confident_parts


def _make_detection(rng, frames, part_count, camera_count=2, likelihood=None):
    data = rng.uniform(0.0, 500.0, size=(camera_count, frames, part_count, 3))
    if likelihood is None:
        # Straddle the threshold so both the confident and rejected paths run.
        data[:, :, :, 2] = rng.uniform(0.0, 1.0, size=(camera_count, frames, part_count))
    else:
        data[:, :, :, 2] = likelihood
    return data


def _assert_equivalent(algorithm, detection):
    expected_values, expected_parts = _pandas_reference(algorithm, detection)
    selected, mask = algorithm._select_most_likely_2d(detection)
    # Built the way `_handle_3d_triangulate` builds it, but compared against the
    # independent pandas reference above.
    actual_parts = [
        part
        for part, column in zip(
            algorithm._measure_offset_level_parts,
            algorithm._measure_offset_level_order,
        )
        if mask[column]
    ]

    numpy.testing.assert_allclose(selected, expected_values, equal_nan=True)
    assert actual_parts == expected_parts
    return selected, mask


@pytest.mark.parametrize("frames", [1, 2, 3, 5])
def test_matches_pandas_reference_across_frame_counts(frames):
    algorithm = _algorithm()
    rng = numpy.random.default_rng(20260908 + frames)
    part_count = len(algorithm._measure_offset_parts)
    _assert_equivalent(algorithm, _make_detection(rng, frames, part_count))


def test_matches_pandas_reference_over_many_random_batches():
    algorithm = _algorithm()
    part_count = len(algorithm._measure_offset_parts)
    for seed in range(40):
        rng = numpy.random.default_rng(seed)
        _assert_equivalent(algorithm, _make_detection(rng, 3, part_count))


def test_all_high_confidence_keeps_every_part():
    algorithm = _algorithm()
    rng = numpy.random.default_rng(7)
    part_count = len(algorithm._measure_offset_parts)
    detection = _make_detection(rng, 3, part_count, likelihood=0.99)
    _, mask = _assert_equivalent(algorithm, detection)
    assert mask.all()


def test_all_low_confidence_rejects_every_part():
    algorithm = _algorithm()
    rng = numpy.random.default_rng(8)
    part_count = len(algorithm._measure_offset_parts)
    detection = _make_detection(rng, 3, part_count, likelihood=0.10)
    selected, mask = _assert_equivalent(algorithm, detection)
    assert not mask.any()
    # Rejected parts are blanked to (nan, nan, 0), not left at their raw values.
    assert numpy.isnan(selected[:, :, 0:2]).all()
    assert (selected[:, :, 2] == 0).all()


def test_threshold_is_inclusive_at_the_boundary():
    algorithm = _algorithm()
    rng = numpy.random.default_rng(9)
    part_count = len(algorithm._measure_offset_parts)
    # Exactly at the threshold: combined score equals n_cams * threshold.
    detection = _make_detection(rng, 2, part_count, likelihood=THRESHOLD)
    _, mask = _assert_equivalent(algorithm, detection)
    assert mask.all()


def test_ties_resolve_to_the_most_recent_frame():
    algorithm = _algorithm()
    part_count = len(algorithm._measure_offset_parts)
    detection = numpy.zeros((2, 3, part_count, 3), dtype=float)
    # Identical likelihoods across all three frames force a tie.
    detection[:, :, :, 2] = 0.95
    # Distinct coordinates so the winning frame is identifiable.
    for frame in range(3):
        detection[:, frame, :, 0] = 10.0 * frame
        detection[:, frame, :, 1] = 100.0 * frame

    selected, mask = _assert_equivalent(algorithm, detection)
    assert mask.all()
    # Frame index 2 is the most recent, matching sorted(...)[-1] on a stable sort.
    assert (selected[:, :, 0] == 20.0).all()
    assert (selected[:, :, 1] == 200.0).all()


def test_per_part_winners_are_independent():
    algorithm = _algorithm()
    part_count = len(algorithm._measure_offset_parts)
    assert part_count >= 2
    detection = numpy.zeros((2, 2, part_count, 3), dtype=float)
    detection[:, :, :, 2] = 0.95
    # Part 0 is strongest in frame 0; part 1 is strongest in frame 1.
    detection[:, 0, 0, 2] = 0.99
    detection[:, 1, 0, 2] = 0.95
    detection[:, 0, 1, 2] = 0.95
    detection[:, 1, 1, 2] = 0.99
    detection[:, 0, :, 0] = 1.0
    detection[:, 1, :, 0] = 2.0

    selected, _ = _assert_equivalent(algorithm, detection)
    assert (selected[:, 0, 0] == 1.0).all()
    assert (selected[:, 1, 0] == 2.0).all()


def test_input_array_is_not_mutated():
    algorithm = _algorithm()
    rng = numpy.random.default_rng(11)
    part_count = len(algorithm._measure_offset_parts)
    detection = _make_detection(rng, 3, part_count, likelihood=0.10)
    original = detection.copy()
    algorithm._select_most_likely_2d(detection)
    # Blanking low-confidence parts must not write back into the caller's frames.
    numpy.testing.assert_allclose(detection, original)
def test_part_order_follows_the_sorted_multiindex_level_not_declaration_order():
    """Pin the ordering the triangulation input depends on.

    `MultiIndex.from_product` sorts its levels, so the original pandas path
    yielded `confident_parts` in sorted order rather than in
    `_measure_offset_parts` declaration order. Triangulation receives that list
    as `body_parts`/`bpts`, so the order is behaviour, not cosmetics.
    """
    algorithm = _algorithm()
    declared = list(algorithm._measure_offset_parts)
    level_order = list(algorithm._measure_offset_level_parts)

    assert sorted(map(str, declared)) == list(map(str, level_order))
    # Guard the actual trap: the two orders genuinely differ for these parts.
    assert declared != level_order
    # The index map must round-trip every part back to its array column.
    for part, column in zip(level_order, algorithm._measure_offset_level_order):
        assert declared[column] == part
