"""A protocol saved before the cue pair must still load.

`PROTOCOL_SCHEMA_VERSION` went to 2 when the Tone 1 to Tone 2 cue pair and the
weighted stimulus trigger profiles landed. Until now `from_record` passed the
stored version straight through and `__post_init__` rejected anything that was
not exactly 2, so every protocol document written before that change failed to
load outright - not with a migration prompt, with a ValueError.

Schema 2 added only optional fields, all of which default to the behaviour a
schema 1 document already had, so the document is upgraded on load. That is the
same thing StimulusProfileLibrary has always done for schema 1 to 4.

These tests assert the upgrade preserves content, that an unknown schema is
still refused rather than guessed at, and that the defaults chosen for a
migrated document cannot change how it behaves.
"""

import pytest

from tools.acquisition.model.trial_protocol_schedule import (
    PROTOCOL_SCHEMA_VERSION,
    SUPPORTED_PROTOCOL_SCHEMA_VERSIONS,
    TrialProtocolDocument,
)


def _record(schema_version, **changes):
    record = {
        "schema_version": schema_version,
        "protocol_id": "legacy-protocol",
        "name": "Legacy protocol",
        "description": "written before the cue pair",
        "revision": 3,
        "trial_count": 25,
    }
    record.update(changes)
    return record


def test_the_current_schema_still_loads():
    document = TrialProtocolDocument.from_record(_record(PROTOCOL_SCHEMA_VERSION))
    assert document.schema_version == PROTOCOL_SCHEMA_VERSION


def test_a_schema_1_document_loads_and_is_upgraded():
    """The whole point: an existing saved protocol must not become unloadable."""
    document = TrialProtocolDocument.from_record(_record(1))
    assert document.schema_version == PROTOCOL_SCHEMA_VERSION


def test_the_upgrade_preserves_the_document_content():
    document = TrialProtocolDocument.from_record(_record(1))
    assert document.protocol_id == "legacy-protocol"
    assert document.name == "Legacy protocol"
    assert document.description == "written before the cue pair"
    assert document.revision == 3
    assert document.trial_count == 25


def test_a_migrated_document_has_no_cue_pair():
    """cue_lock_timing defaults to True, which must not matter here.

    A schema 1 document has no cue tone, so no cue pair is ever armed and the
    lock-timing default cannot change how the protocol behaves.
    """
    document = TrialProtocolDocument.from_record(_record(1))
    row = document.resolved_row(1) if hasattr(document, "resolved_row") else None
    if row is None:
        pytest.skip("document does not expose resolved rows directly")
    assert not row.cue_tone_profile_id
    assert not row.cue_interval_profile_id
    assert row.cue_interval_fixed_ms == 0


@pytest.mark.parametrize("schema_version", [0, 3, 99, -1])
def test_an_unknown_schema_is_refused_not_guessed(schema_version):
    with pytest.raises(ValueError, match="Unsupported protocol schema"):
        TrialProtocolDocument.from_record(_record(schema_version))


def test_a_document_with_no_schema_is_refused():
    """A missing version is not the same as an old one."""
    record = _record(1)
    del record["schema_version"]
    with pytest.raises(ValueError, match="Unsupported protocol schema"):
        TrialProtocolDocument.from_record(record)


def test_the_supported_set_names_every_loadable_version():
    assert PROTOCOL_SCHEMA_VERSION in SUPPORTED_PROTOCOL_SCHEMA_VERSIONS
    assert 1 in SUPPORTED_PROTOCOL_SCHEMA_VERSIONS


def test_a_round_trip_writes_the_current_schema():
    """A migrated document saves as schema 2, so it upgrades once and stays."""
    document = TrialProtocolDocument.from_record(_record(1))
    record = document.to_record()
    assert record["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert TrialProtocolDocument.from_record(record).schema_version == (
        PROTOCOL_SCHEMA_VERSION
    )
