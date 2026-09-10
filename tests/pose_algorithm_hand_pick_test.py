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


def test_the_algorithm_no_longer_sorts_to_pick_a_hand():
    """A revert would be silent: the result is identical, only slower."""
    import inspect

    from autotrainer.inference.pose_algorithm import PoseAlgorithm

    source = inspect.getsource(PoseAlgorithm)
    assert 'sort_values(by="likelihood"' not in source
    assert 'numpy.argmax(hand["likelihood"].to_numpy())' in source
