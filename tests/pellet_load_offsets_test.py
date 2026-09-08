import pytest

from tools.acquisition.model.pellet_load_offsets import (
    MAX_ABS_OFFSET_MM,
    PelletLoadOffset,
    PelletLoadOffsets,
    apply_load_offsets,
)


# The shape of the tracked load_pellet sequence: absolute axis moves
# interleaved with arm, tone, and relative steps.
LOAD_STEPS = [
    {"y": 0},
    {"barrier_arm": 110},
    {"x": 25},
    {"load_arm": 5},
    {"z": 22},
    {"load_arm": "114,25"},
    {"z": 10},
    {"tone": "5000,0.3"},
    {"barrier_arm": 80},
]


def test_no_offsets_preserve_the_sequence_exactly():
    assert apply_load_offsets(LOAD_STEPS, PelletLoadOffsets()) == LOAD_STEPS


def test_zero_offsets_are_identity():
    offsets = PelletLoadOffsets(
        offsets=({"axis": "z", "occurrence": 1, "offset_mm": 0.0},)
    )
    assert offsets.is_identity is True
    assert apply_load_offsets(LOAD_STEPS, offsets) == LOAD_STEPS


def test_occurrences_of_one_axis_are_addressed_independently():
    offsets = PelletLoadOffsets(
        offsets=(
            {"axis": "z", "occurrence": 1, "offset_mm": 1.0},
            {"axis": "z", "occurrence": 2, "offset_mm": -2.0},
        )
    )
    result = apply_load_offsets(LOAD_STEPS, offsets)
    assert result[4] == {"z": 23}
    assert result[6] == {"z": 8}
    assert offsets.is_identity is False


def test_negative_offsets_move_toward_home():
    offsets = PelletLoadOffsets(
        offsets=({"axis": "x", "occurrence": 1, "offset_mm": -5.0},)
    )
    assert apply_load_offsets(LOAD_STEPS, offsets)[2] == {"x": 20}


def test_non_positional_steps_are_left_untouched():
    offsets = PelletLoadOffsets(
        offsets=({"axis": "z", "occurrence": 1, "offset_mm": 1.0},)
    )
    result = apply_load_offsets(LOAD_STEPS, offsets)
    assert result[1] == {"barrier_arm": 110}
    assert result[5] == {"load_arm": "114,25"}
    assert result[7] == {"tone": "5000,0.3"}


def test_relative_moves_are_not_offset():
    steps = [{"y_rel": -25}, {"y": 10}]
    offsets = PelletLoadOffsets(
        offsets=({"axis": "y", "occurrence": 1, "offset_mm": 3.0},)
    )
    result = apply_load_offsets(steps, offsets)
    assert result[0] == {"y_rel": -25}
    assert result[1] == {"y": 13}


def test_non_numeric_positional_values_are_left_as_authored():
    steps = [{"z": "22,5"}]
    offsets = PelletLoadOffsets(
        offsets=({"axis": "z", "occurrence": 1, "offset_mm": 1.0},)
    )
    with pytest.raises(ValueError, match="do not exist"):
        apply_load_offsets(steps, offsets)


def test_fractional_offsets_are_preserved():
    offsets = PelletLoadOffsets(
        offsets=({"axis": "z", "occurrence": 1, "offset_mm": 0.5},)
    )
    assert apply_load_offsets(LOAD_STEPS, offsets)[4] == {"z": 22.5}


def test_integral_results_stay_integral():
    offsets = PelletLoadOffsets(
        offsets=({"axis": "z", "occurrence": 1, "offset_mm": 1.0},)
    )
    value = apply_load_offsets(LOAD_STEPS, offsets)[4]["z"]
    assert value == 23
    assert isinstance(value, int)


def test_an_offset_for_a_missing_step_is_an_error():
    offsets = PelletLoadOffsets(
        offsets=({"axis": "z", "occurrence": 5, "offset_mm": 1.0},)
    )
    with pytest.raises(ValueError, match="z occurrence 5"):
        apply_load_offsets(LOAD_STEPS, offsets)


def test_a_malformed_step_is_rejected():
    with pytest.raises(ValueError, match="exactly one action"):
        apply_load_offsets([{"x": 1, "y": 2}], PelletLoadOffsets())


@pytest.mark.parametrize("axis", ["a", "barrier_arm", "", "X1"])
def test_offsets_apply_only_to_positional_axes(axis):
    with pytest.raises(ValueError, match="apply to x, y, z"):
        PelletLoadOffset(axis=axis, occurrence=1, offset_mm=1.0)


def test_axis_names_are_normalized():
    assert PelletLoadOffset(axis=" Z ", occurrence=1, offset_mm=1.0).axis == "z"


@pytest.mark.parametrize("occurrence", [0, -1])
def test_occurrence_is_one_based(occurrence):
    with pytest.raises(ValueError, match="1-based"):
        PelletLoadOffset(axis="z", occurrence=occurrence, offset_mm=1.0)


@pytest.mark.parametrize(
    "offset", [MAX_ABS_OFFSET_MM + 0.1, -MAX_ABS_OFFSET_MM - 0.1, float("nan")]
)
def test_offset_magnitude_is_bounded(offset):
    with pytest.raises(ValueError, match="within"):
        PelletLoadOffset(axis="z", occurrence=1, offset_mm=offset)


def test_duplicate_offsets_are_rejected():
    with pytest.raises(ValueError, match="Duplicate pellet load offset"):
        PelletLoadOffsets(
            offsets=(
                {"axis": "z", "occurrence": 1, "offset_mm": 1.0},
                {"axis": "z", "occurrence": 1, "offset_mm": 2.0},
            )
        )


def test_records_round_trip():
    offsets = PelletLoadOffsets(
        offsets=(
            {"axis": "z", "occurrence": 1, "offset_mm": 1.5},
            {"axis": "x", "occurrence": 1, "offset_mm": -2.0},
        )
    )
    restored = PelletLoadOffsets.from_record(offsets.to_record())
    assert restored == offsets
    assert offsets.to_record()["schema_version"] == 1


def test_an_unsupported_schema_is_rejected():
    with pytest.raises(ValueError, match="Unsupported pellet load offset schema"):
        PelletLoadOffsets(schema_version=99)
