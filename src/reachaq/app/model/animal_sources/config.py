"""The one source a computer's user has set up, as kept in their preferences.

Settings and the column mapping live in the user's preferences (never the
shared rig config or the repository); secrets live in the
:class:`CredentialStore`. A computer set up before sources existed, with only
``softmouse/manifest_path``, reads that publication as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Type

from .base import AnimalMetadataSource, available_source_kinds
from .credentials import CredentialStore
from .mapping import AnimalTableMapper, ColumnMapping, softmouse_mapping
from .publication import PublicationSource


@dataclass(frozen=True)
class SourceConfiguration:
    kind: str
    settings: Mapping[str, str]
    mapping: Optional[ColumnMapping]
    #: Also write a shared publication here after each refresh ("" = no).
    publish_directory: str = ""
    #: Refuse to publish unless this path is a mount point ("" = no check).
    publish_mount: str = ""
    legacy: bool = field(default=False, compare=False)

    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "settings": dict(self.settings),
                "mapping": None if self.mapping is None else self.mapping.to_dict(),
                "publishDirectory": self.publish_directory,
                "publishMount": self.publish_mount,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "SourceConfiguration":
        value = json.loads(text)
        mapping = value.get("mapping")
        return cls(
            kind=str(value["kind"]),
            settings={str(k): str(v) for k, v in (value.get("settings") or {}).items()},
            mapping=None if mapping is None else ColumnMapping.from_dict(mapping),
            publish_directory=str(value.get("publishDirectory") or ""),
            publish_mount=str(value.get("publishMount") or ""),
        )


def load_source_configuration(preferences) -> Optional[SourceConfiguration]:
    """The configured source, or None when the user has set none up."""
    text = getattr(preferences, "animal_metadata_source", "")
    if text:
        return SourceConfiguration.from_json(text)
    manifest = getattr(preferences, "softmouse_manifest_path", "")
    if manifest:
        return SourceConfiguration(
            kind=PublicationSource.kind,
            settings={"manifest": manifest},
            mapping=softmouse_mapping(
                getattr(preferences, "softmouse_name_column", "") or "Physical Tag"
            ),
            legacy=True,
        )
    return None


def save_source_configuration(preferences, configuration: Optional[SourceConfiguration]) -> None:
    preferences.animal_metadata_source = "" if configuration is None else configuration.to_json()


@dataclass(frozen=True)
class ConfiguredSource:
    configuration: SourceConfiguration
    source: AnimalMetadataSource

    def mapper_for(self, table_mapping: Optional[ColumnMapping], provider: Optional[str]):
        mapping = table_mapping or self.configuration.mapping
        if mapping is None:
            raise ValueError("The source has no column mapping; set it up again")
        return AnimalTableMapper(mapping, provider=provider or self.source.provider)


def build_source(
    configuration: SourceConfiguration,
    *,
    credentials: Optional[CredentialStore] = None,
    kinds: Optional[Dict[str, Type[AnimalMetadataSource]]] = None,
) -> ConfiguredSource:
    kinds = available_source_kinds() if kinds is None else kinds
    kind = kinds.get(configuration.kind)
    if kind is None:
        raise ValueError(
            f"Animal metadata source kind {configuration.kind!r} is not installed"
        )
    if configuration.publish_directory and configuration.kind == PublicationSource.kind:
        raise ValueError("A shared publication cannot also be published")
    source = kind(configuration.settings, credentials or CredentialStore())
    return ConfiguredSource(configuration, source)


def publish_target(configuration: SourceConfiguration):
    """(directory, mount) for publishing, or (None, None)."""
    if not configuration.publish_directory:
        return None, None
    return (
        Path(configuration.publish_directory).expanduser(),
        Path(configuration.publish_mount) if configuration.publish_mount else None,
    )
