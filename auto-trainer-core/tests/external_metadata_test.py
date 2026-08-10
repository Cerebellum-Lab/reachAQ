import pytest

from autotrainer.core.animal.external_metadata import (
    ExternalAnimalRecord,
    ExternalIdentity,
    NormalizedAnimalBatch,
    normalize_rfid,
    split_genotype,
)


TAG_A = "D4D47231005A30010000000000"


def test_rfid_normalization_uses_validated_26_character_payload():
    assert normalize_rfid(TAG_A.lower()) == TAG_A
    with pytest.raises(ValueError):
        normalize_rfid("D4D47231005A3001")


def test_external_identity_has_stable_normalized_key():
    identity = ExternalIdentity(" PT-42 ", provider="SoftMouse")
    assert identity.key == ("softmouse", "physical_tag", "PT-42")
    assert ExternalIdentity.from_dict(identity.to_dict()) == identity


def test_split_genotype_preserves_multiple_values():
    assert split_genotype("Cre+; GFP-, WT") == ("Cre+", "GFP-", "WT")


def test_batch_rejects_duplicate_current_rfid():
    def record(subject_id):
        return ExternalAnimalRecord(
            identity=ExternalIdentity(subject_id),
            physical_rfid=TAG_A,
            new_animal_name_candidate=subject_id,
            state="Stock",
            source_payload={},
            source_hash=subject_id,
        )

    with pytest.raises(ValueError, match="duplicate active RFID"):
        NormalizedAnimalBatch(
            import_id="import-1",
            provider="softmouse",
            schema_version=1,
            mapping_profile_id="default",
            mapping_profile_version=1,
            source_filename="animals.xlsx",
            source_sheet="Animal List",
            source_file_sha256="abc",
            header_signature="def",
            imported_utc="2026-08-10T00:00:00Z",
            total_source_rows=2,
            ignored_missing_rfid_rows=0,
            ignored_ended_rows=0,
            records=(record("PT-1"), record("PT-2")),
        )
