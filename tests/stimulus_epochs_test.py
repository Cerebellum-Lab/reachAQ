import pytest

from tools.acquisition.model.stimulus_epochs import (
    BASELINE,
    DEFAULT_EPOCH_BLOCK_SIZE,
    MAX_EPOCH_BLOCK_SIZE,
    STIMULATION,
    WASHOUT,
    build_epoch_scopes,
    epoch_assignment,
    epoch_kind,
)
from tools.acquisition.model.trial_protocol_schedule import (
    TrialProtocolDocument,
)


STIM_VALUES = {
    "stimulus_assignment": "randomized",
    "stimulus_trigger_profile_id": "mixed",
}


@pytest.mark.parametrize(
    "trial_id,expected",
    [
        (1, BASELINE),
        (15, BASELINE),
        (16, STIMULATION),
        (30, STIMULATION),
        (31, WASHOUT),
        (45, WASHOUT),
        (46, STIMULATION),
    ],
)
def test_blocks_alternate_after_a_baseline_block(trial_id, expected):
    assert epoch_kind(trial_id, DEFAULT_EPOCH_BLOCK_SIZE) == expected


def test_assignment_reports_position_and_label():
    assignment = epoch_assignment(17, 15)
    assert assignment["block_index"] == 1
    assert assignment["epoch_number"] == 1
    assert assignment["epoch_kind"] == STIMULATION
    assert assignment["trial_index_in_block"] == 2
    assert assignment["progress_label"] == "2/15 Stimulation epoch #1"
    assert assignment["stimulation_allowed"] is True


def test_baseline_never_allows_stimulation():
    assignment = epoch_assignment(1, 15)
    assert assignment["epoch_kind"] == BASELINE
    assert assignment["epoch_number"] == 0
    assert assignment["stimulation_allowed"] is False


def test_washout_does_not_allow_stimulation():
    assert epoch_assignment(31, 15)["stimulation_allowed"] is False


def test_trial_ids_below_one_clamp_into_the_first_block():
    assert epoch_kind(0, 15) == BASELINE
    assert epoch_assignment(-5, 15)["trial_index_in_block"] == 1


@pytest.mark.parametrize("block_size", [0, -1, MAX_EPOCH_BLOCK_SIZE + 1])
def test_invalid_block_sizes_are_rejected(block_size):
    with pytest.raises(ValueError, match="block size must be between"):
        epoch_kind(1, block_size)


def test_scopes_cover_every_trial_exactly_once():
    scopes = build_epoch_scopes(45, 15, stimulation_values=STIM_VALUES)
    covered = [trial for scope in scopes for trial in scope.trial_ids]
    assert sorted(covered) == list(range(1, 46))
    assert len(covered) == len(set(covered))


def test_scopes_are_named_by_kind_and_number():
    scopes = build_epoch_scopes(45, 15, stimulation_values=STIM_VALUES)
    assert [scope.name for scope in scopes] == [
        "baseline",
        "stimulation-1",
        "washout-1",
    ]


def test_a_partial_final_block_still_becomes_a_scope():
    scopes = build_epoch_scopes(20, 15, stimulation_values=STIM_VALUES)
    assert len(scopes) == 2
    assert scopes[1].trial_ids == (16, 17, 18, 19, 20)


def test_baseline_and_washout_disable_stimulation_explicitly():
    scopes = build_epoch_scopes(45, 15, stimulation_values=STIM_VALUES)
    baseline, stimulation, washout = scopes
    assert dict(baseline.patch.to_mapping())["stimulus_assignment"] == "disabled"
    assert dict(washout.patch.to_mapping())["stimulus_assignment"] == "disabled"
    assert dict(stimulation.patch.to_mapping())["stimulus_assignment"] == (
        "randomized"
    )


def test_caller_can_override_baseline_and_washout_values():
    scopes = build_epoch_scopes(
        30,
        15,
        stimulation_values=STIM_VALUES,
        baseline_values={"stimulus_assignment": "disabled", "enabled": True},
    )
    assert dict(scopes[0].patch.to_mapping())["enabled"] is True


def test_stimulation_values_are_required():
    with pytest.raises(ValueError, match="require stimulation values"):
        build_epoch_scopes(30, 15, stimulation_values={})


def test_at_least_one_trial_is_required():
    with pytest.raises(ValueError, match="at least one trial"):
        build_epoch_scopes(0, 15, stimulation_values=STIM_VALUES)


def test_generated_scopes_load_into_a_protocol_document():
    scopes = build_epoch_scopes(30, 15, stimulation_values=STIM_VALUES)
    document = TrialProtocolDocument(
        protocol_id="p",
        name="epochs",
        trial_count=30,
        epochs=scopes,
    )
    resolved = document.resolve()
    assert len(resolved) == 30
    # The scope, not the row default, is the source of the assignment.
    stimulated = resolved[15].row
    assert stimulated.stimulus_assignment.value == "randomized"
    assert resolved[0].row.stimulus_assignment.value == "disabled"


def test_generated_scopes_record_their_provenance():
    scopes = build_epoch_scopes(30, 15, stimulation_values=STIM_VALUES)
    document = TrialProtocolDocument(
        protocol_id="p",
        name="epochs",
        trial_count=30,
        epochs=scopes,
    )
    sources = dict(document.resolve()[15].sources)
    assert "stimulation-1" in sources["stimulus_assignment"]
