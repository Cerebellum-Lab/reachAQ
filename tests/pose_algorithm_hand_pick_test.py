"""Picking the most likely hand row must not depend on sorting the frame.

`process_frames` chooses one row per hand per camera - the one with the highest
likelihood. It did that with

    raw[elem].sort_values(by="likelihood", ascending=False).reset_index().iloc[0]

which sorts every row to read one. Measured on the rig that is 313 us per call
and it runs once per hand per camera on every live batch: 1.25 ms of a 6.57 ms
PoseAlgorithm.process, in a loop with a 10-20 ms budget.

`iloc[argmax(likelihood)]` selects the same row. These tests pin that, because
the two are only equivalent as long as both resolve a tie to the first row at
the maximum, and a future change to either side could silently diverge.
"""

import numpy
import pandas
import pytest

COLUMNS = ["x", "y", "likelihood"]


def previous(frame):
    """What the code did before."""
    return frame.sort_values(by="likelihood", ascending=False).reset_index().iloc[0]


def current(frame):
    """What the code does now."""
    return frame.iloc[int(numpy.argmax(frame["likelihood"].to_numpy()))]


def _frame(rows):
    return pandas.DataFrame(rows, columns=COLUMNS)


def test_the_same_row_is_chosen():
    frame = _frame([[1.0, 2.0, 0.10],
                    [3.0, 4.0, 0.90],
                    [5.0, 6.0, 0.50]])
    assert current(frame)["likelihood"] == pytest.approx(0.90)
    for column in COLUMNS:
        assert current(frame)[column] == pytest.approx(previous(frame)[column])


def test_a_tie_resolves_to_the_first_row_either_way():
    """sort_values is stable and argmax returns the first maximum."""
    frame = _frame([[1.0, 2.0, 0.80],
                    [3.0, 4.0, 0.80],
                    [5.0, 6.0, 0.10]])
    assert current(frame)["x"] == pytest.approx(1.0)
    assert current(frame)["x"] == pytest.approx(previous(frame)["x"])


def test_a_single_row_is_returned_unchanged():
    frame = _frame([[7.0, 8.0, 0.25]])
    for column in COLUMNS:
        assert current(frame)[column] == pytest.approx(previous(frame)[column])


def test_the_maximum_in_the_last_row_is_found():
    """A descending frame is the case a first-row shortcut would get wrong."""
    frame = _frame([[1.0, 1.0, 0.10],
                    [2.0, 2.0, 0.20],
                    [3.0, 3.0, 0.95]])
    assert current(frame)["likelihood"] == pytest.approx(0.95)
    assert current(frame)["x"] == pytest.approx(previous(frame)["x"])


@pytest.mark.parametrize("seed", range(25))
def test_they_agree_on_random_frames(seed):
    rng = numpy.random.default_rng(seed)
    rows = rng.integers(1, 6)
    frame = _frame(numpy.column_stack([
        rng.uniform(0, 256, rows),
        rng.uniform(0, 256, rows),
        # Quantised so ties actually occur rather than being measure zero.
        rng.integers(0, 4, rows) / 4.0,
    ]))
    for column in COLUMNS:
        assert current(frame)[column] == pytest.approx(previous(frame)[column])


def test_the_algorithm_selects_hands_without_pandas():
    """A revert would be silent: the result is identical, only slower."""
    import inspect

    from autotrainer.inference.pose_algorithm import PoseAlgorithm

    source = inspect.getsource(PoseAlgorithm)
    assert 'sort_values(by="likelihood"' not in source
    assert "process_hand_data(" not in source, (
        "the live path builds DataFrames again")
    assert "numpy.argmax(likelihood)" in source


# --- the hand choice itself --------------------------------------------------
#
# The live path no longer calls process_hand_data. It takes, per camera, the
# (frame, sub-part) pair with the highest likelihood, which is the same answer
# process_hand_data reached in two stages: best sub-part per frame, then best
# frame. These tests hold the two implementations against each other on random
# input, because the equivalence is the whole justification for the change.

BASES = ["H_flat", "H_spread", "H_grab"]


def _pandas_choice(frames, hand):
    """What process_hand_data plus the old frame pick produced."""
    from autotrainer.inference.analysis.prepare_jetson_data import process_hand_data

    parts = [option + base for option in ("R", "L") for base in BASES]
    columns = pandas.MultiIndex.from_product(
        [parts, COLUMNS], names=["bodyparts", "coordinates"])
    rows = numpy.asarray(frames, dtype=float).reshape(len(frames), -1)
    df = pandas.DataFrame(rows, columns=columns)
    out_columns = pandas.MultiIndex.from_product(
        [["R_Hand", "L_Hand"], COLUMNS], names=["bodyparts", "coordinates"])
    result = process_hand_data(
        df, hand_base_names=BASES, hand_options=["R", "L"], dlc_seg="_raw2D",
        newdf=pandas.DataFrame(columns=out_columns, index=range(len(df))),
        additional_names=[])
    picked = result[f"{hand}_Hand"]
    best = picked.iloc[int(numpy.argmax(picked["likelihood"].to_numpy()))]
    return float(best["x"]), float(best["y"]), float(best["likelihood"])


def _numpy_choice(frames, hand):
    """What the live path does now."""
    rows = [0, 1, 2] if hand == "R" else [3, 4, 5]
    poses = numpy.asarray(frames, dtype=float)[:, rows, :]
    likelihood = poses[:, :, 2]
    frame_index, sub_index = divmod(
        int(numpy.argmax(likelihood)), likelihood.shape[1])
    x, y, score = poses[frame_index, sub_index]
    return float(x), float(y), float(score)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("hand", ["R", "L"])
def test_the_numpy_choice_matches_process_hand_data(seed, hand):
    rng = numpy.random.default_rng(seed)
    count = int(rng.integers(1, 4))
    frames = rng.uniform(0, 256, (count, 6, 3))
    # Quantised likelihoods so ties actually occur.
    frames[:, :, 2] = rng.integers(0, 4, (count, 6)) / 4.0
    assert _numpy_choice(frames, hand) == pytest.approx(
        _pandas_choice(frames, hand))


def test_a_tie_across_frames_and_subparts_resolves_the_same_way():
    """Flat argmax must break ties to the first frame, then the first sub-part."""
    frames = numpy.zeros((2, 6, 3))
    frames[:, :, 2] = 0.5           # every candidate equally likely
    frames[0, 0, 0:2] = (11.0, 12.0)
    assert _numpy_choice(frames, "R")[:2] == (11.0, 12.0)
    assert _numpy_choice(frames, "R") == pytest.approx(
        _pandas_choice(frames, "R"))


def test_the_most_likely_subpart_wins_over_frame_order():
    frames = numpy.zeros((2, 6, 3))
    frames[0, 0] = (1.0, 1.0, 0.20)
    frames[1, 2] = (9.0, 9.0, 0.90)     # later frame, third sub-part
    assert _numpy_choice(frames, "R") == pytest.approx((9.0, 9.0, 0.90))
    assert _numpy_choice(frames, "R") == pytest.approx(
        _pandas_choice(frames, "R"))
