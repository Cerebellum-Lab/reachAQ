from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


SOFTMOUSE_PROVIDER = "softmouse"
PHYSICAL_TAG_ID_KIND = "physical_tag"
RFID_HEX_LENGTH = 26
_RFID_RE = re.compile(r"^[0-9A-F]{26}$")


def normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    return text


def normalize_external_subject_id(value: Any) -> str:
    text = normalize_optional_text(value)
    if text is None:
        raise ValueError("External subject ID cannot be empty")
    return text


def normalize_rfid(value: Any) -> str:
    text = normalize_optional_text(value)
    if text is None:
        raise ValueError("RFID cannot be empty")
    normalized = text.upper()
    if _RFID_RE.fullmatch(normalized) is None:
        raise ValueError(
            "RFID must be exactly 26 hexadecimal characters from a validated "
            "reader payload"
        )
    return normalized


def normalize_state(value: Any) -> str:
    text = normalize_optional_text(value)
    return "unknown" if text is None else text.casefold()


@dataclass(frozen=True)
class ExternalIdentity:
    subject_id: str
    provider: str = SOFTMOUSE_PROVIDER
    subject_id_kind: str = PHYSICAL_TAG_ID_KIND

    def __post_init__(self) -> None:
        provider = normalize_external_subject_id(self.provider).casefold()
        subject_id_kind = normalize_external_subject_id(
            self.subject_id_kind
        ).casefold()
        subject_id = normalize_external_subject_id(self.subject_id)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "subject_id_kind", subject_id_kind)
        object.__setattr__(self, "subject_id", subject_id)

    @property
    def key(self) -> Tuple[str, str, str]:
        return self.provider, self.subject_id_kind, self.subject_id

    def to_dict(self) -> Dict[str, str]:
        return {
            "provider": self.provider,
            "subjectIdKind": self.subject_id_kind,
            "subjectId": self.subject_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalIdentity":
        return cls(
            provider=value.get("provider", SOFTMOUSE_PROVIDER),
            subject_id_kind=value.get("subjectIdKind", PHYSICAL_TAG_ID_KIND),
            subject_id=value.get("subjectId"),
        )


@dataclass(frozen=True)
class ExternalMetadataSnapshot:
    rfid: Optional[str] = None
    physical_tag: Optional[str] = None
    sex: Optional[str] = None
    date_of_birth: Optional[str] = None
    strain: Optional[str] = None
    genotype: Tuple[str, ...] = ()
    cage: Optional[str] = None
    protocol: Optional[str] = None
    state: str = "unknown"
    source_hash: Optional[str] = None
    registry_import_id: Optional[str] = None
    source_file_sha256: Optional[str] = None
    imported_utc: Optional[str] = None
    source_payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        rfid = None if self.rfid is None else normalize_rfid(self.rfid)
        object.__setattr__(self, "rfid", rfid)
        object.__setattr__(
            self,
            "physical_tag",
            normalize_optional_text(self.physical_tag),
        )
        object.__setattr__(self, "sex", normalize_optional_text(self.sex))
        object.__setattr__(
            self,
            "date_of_birth",
            normalize_optional_text(self.date_of_birth),
        )
        object.__setattr__(self, "strain", normalize_optional_text(self.strain))
        object.__setattr__(self, "cage", normalize_optional_text(self.cage))
        object.__setattr__(self, "protocol", normalize_optional_text(self.protocol))
        object.__setattr__(self, "state", normalize_state(self.state))
        object.__setattr__(
            self,
            "genotype",
            tuple(
                text
                for item in self.genotype
                if (text := normalize_optional_text(item)) is not None
            ),
        )
        object.__setattr__(self, "source_payload", dict(self.source_payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rfid": self.rfid,
            "physicalTag": self.physical_tag,
            "sex": self.sex,
            "dateOfBirth": self.date_of_birth,
            "strain": self.strain,
            "genotype": list(self.genotype),
            "cage": self.cage,
            "protocol": self.protocol,
            "state": self.state,
            "sourceHash": self.source_hash,
            "registryImportId": self.registry_import_id,
            "sourceFileSha256": self.source_file_sha256,
            "importedUtc": self.imported_utc,
            "sourcePayload": dict(self.source_payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalMetadataSnapshot":
        genotype = value.get("genotype") or ()
        if isinstance(genotype, str):
            genotype = (genotype,)
        return cls(
            rfid=value.get("rfid"),
            physical_tag=value.get("physicalTag"),
            sex=value.get("sex"),
            date_of_birth=value.get("dateOfBirth"),
            strain=value.get("strain"),
            genotype=tuple(genotype),
            cage=value.get("cage"),
            protocol=value.get("protocol"),
            state=value.get("state", "unknown"),
            source_hash=value.get("sourceHash"),
            registry_import_id=value.get("registryImportId"),
            source_file_sha256=value.get("sourceFileSha256"),
            imported_utc=value.get("importedUtc"),
            source_payload=value.get("sourcePayload") or {},
        )


@dataclass(frozen=True)
class ExternalAnimalRecord:
    identity: ExternalIdentity
    physical_rfid: str
    new_animal_name_candidate: Optional[str]
    state: str
    source_payload: Mapping[str, Any]
    source_hash: str
    sex: Optional[str] = None
    date_of_birth: Optional[str] = None
    strain: Optional[str] = None
    genotype: Tuple[str, ...] = ()
    cage: Optional[str] = None
    protocol: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_rfid", normalize_rfid(self.physical_rfid))
        object.__setattr__(
            self,
            "new_animal_name_candidate",
            normalize_optional_text(self.new_animal_name_candidate),
        )
        object.__setattr__(self, "state", normalize_state(self.state))
        object.__setattr__(self, "source_payload", dict(self.source_payload))
        object.__setattr__(self, "genotype", tuple(self.genotype))

    def metadata_snapshot(
        self,
        *,
        registry_import_id: Optional[str] = None,
        source_file_sha256: Optional[str] = None,
        imported_utc: Optional[str] = None,
    ) -> ExternalMetadataSnapshot:
        return ExternalMetadataSnapshot(
            rfid=self.physical_rfid,
            physical_tag=self.identity.subject_id,
            sex=self.sex,
            date_of_birth=self.date_of_birth,
            strain=self.strain,
            genotype=self.genotype,
            cage=self.cage,
            protocol=self.protocol,
            state=self.state,
            source_hash=self.source_hash,
            registry_import_id=registry_import_id,
            source_file_sha256=source_file_sha256,
            imported_utc=imported_utc,
            source_payload=self.source_payload,
        )


@dataclass(frozen=True)
class NormalizedAnimalBatch:
    import_id: str
    provider: str
    schema_version: int
    mapping_profile_id: str
    mapping_profile_version: int
    source_filename: str
    source_sheet: str
    source_file_sha256: str
    header_signature: str
    imported_utc: str
    total_source_rows: int
    ignored_missing_rfid_rows: int
    ignored_ended_rows: int
    records: Tuple[ExternalAnimalRecord, ...]
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", self.provider.casefold())
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        identities = [record.identity.key for record in self.records]
        rfids = [record.physical_rfid for record in self.records]
        if len(identities) != len(set(identities)):
            raise ValueError("Batch contains duplicate external subject identities")
        if len(rfids) != len(set(rfids)):
            raise ValueError("Batch contains duplicate active RFID values")
        if any(record.state == "ended" for record in self.records):
            raise ValueError("Ended animals must not enter the current RFID cache")

    @property
    def accepted_rows(self) -> int:
        return len(self.records)


def split_genotype(value: Any) -> Tuple[str, ...]:
    text = normalize_optional_text(value)
    if text is None:
        return ()
    return tuple(part.strip() for part in re.split(r"[;,]", text) if part.strip())
