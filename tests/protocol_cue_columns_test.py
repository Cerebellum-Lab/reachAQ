"""Tests for the cue and trigger-profile controls in the Protocol panel.

The schema and compiler have supported these fields since the cue pair landed,
but nothing exposed them, so an operator could not reach the features at all.
These assert the columns exist, edit the right row fields, and read back in
terms an operator can act on rather than raw values.

The panel needs Qt, so the column table and the pure formatting are tested
directly instead of building a widget.
"""

import pytest

from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow


@pytest.fixture(scope="module")
def content():
    return pytest.importorskip(
        "tools.acquisition.view.protocol_content",
        reason="Protocol panel requires PySide6",
    )


def _column(content, field):
    for column in content.ProtocolContent.COLUMNS:
        if column.field == field:
            return column
    raise AssertionError(f"no column edits {field!r}")


@pytest.mark.parametrize("field,label", [
    ("cue_tone_profile_id", "Cue tone"),
    ("cue_interval_profile_id", "Cue interval"),
    ("cue_interval_fixed_ms", "Cue fixed (ms)"),
    ("cue_lock_timing", "Lock timing"),
    ("cue_post_clear_delay_ms", "Post-clear (ms)"),
    ("stimulus_trigger_profile_id", "Trigger profile"),
])
def test_every_cue_field_has_a_column(content, field, label):
    assert _column(content, field).label == label


def test_every_column_field_is_a_real_row_field(content):
    """A column editing a field the row does not have would fail silently."""
    row = TrialProtocolRow(trial_id=1)
    for column in content.ProtocolContent.COLUMNS:
        if column.field is None:
            continue
        assert hasattr(row, column.field), column.field


def test_the_profile_columns_resolve_to_state_keys(content):
    """A delegate kind with no state key raises KeyError when the editor opens."""
    known = {
        "tone_profile", "laser_profile", "automatic_shift_profile",
        "cue_interval_profile", "stimulus_trigger_profile",
    }
    for column in content.ProtocolContent.COLUMNS:
        if isinstance(column.delegate, str) and column.delegate.endswith("_profile"):
            assert column.delegate in known, column.delegate


def test_lock_timing_reads_as_its_behaviour_not_a_boolean(content):
    display = content.ProtocolContent._display_value
    assert display("cue_lock_timing", True) == "Locked"
    assert display("cue_lock_timing", False) == "Unlocked"


def test_a_zero_fixed_interval_reads_as_deferring_to_the_profile(content):
    """Zero is not a zero-length cue interval, and must not look like one."""
    display = content.ProtocolContent._display_value
    assert display("cue_interval_fixed_ms", 0) == "From profile"
    assert display("cue_interval_fixed_ms", 750) == "750 ms"


def test_no_post_clear_delay_reads_as_none(content):
    display = content.ProtocolContent._display_value
    assert display("cue_post_clear_delay_ms", 0) == "None"
    assert display("cue_post_clear_delay_ms", 250) == "250 ms"


def test_the_cue_columns_sit_with_the_tone_they_extend(content):
    """Tone 2 belongs next to Tone 1, not at the far end of a 25 column table."""
    fields = [column.field for column in content.ProtocolContent.COLUMNS]
    assert fields.index("cue_tone_profile_id") == fields.index("tone_phase") + 1


def test_the_trigger_profile_follows_the_single_trigger(content):
    fields = [column.field for column in content.ProtocolContent.COLUMNS]
    assert (
        fields.index("stimulus_trigger_profile_id")
        == fields.index("stimulus_trigger") + 1
    )
