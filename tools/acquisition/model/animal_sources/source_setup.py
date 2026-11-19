"""Setting a source up, shared by the setup dialog and the CLI.

Secrets entered during setup are used from memory to list choices and to
read the source's columns; they are stored only when the setup is saved.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type

from .base import AnimalMetadataSource
from .config import SourceConfiguration, save_source_configuration
from .credentials import CredentialStore
from .mapping import ColumnMapping, SourceTable


class _EnteredSecrets:
    """A credential store that answers with not-yet-saved secrets first."""

    def __init__(self, base: CredentialStore, entered: Mapping[str, str]):
        self._base = base
        self._entered = dict(entered)

    def get(self, key: str) -> Optional[str]:
        return self._entered.get(key) or self._base.get(key)

    def legacy_keyring_password(self, service: str, account: str) -> Optional[str]:
        return self._base.legacy_keyring_password(service, account)


def split_settings(
    kind: Type[AnimalMetadataSource], values: Mapping[str, str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(settings, secrets) from one form's values; empty secrets are dropped."""
    secret_names = {item.name for item in kind.setup_fields() if item.kind == "secret"}
    settings = {
        name: str(value).strip()
        for name, value in values.items()
        if name not in secret_names and str(value).strip()
    }
    secrets = {name: str(value) for name, value in values.items() if name in secret_names and value}
    return settings, secrets


def source_for_setup(
    kind: Type[AnimalMetadataSource],
    values: Mapping[str, str],
    credentials: CredentialStore,
) -> AnimalMetadataSource:
    settings, secrets = split_settings(kind, values)
    entered = {probe_key(kind, settings, name): value for name, value in secrets.items()}
    return kind(settings, _EnteredSecrets(credentials, entered))


def probe_key(kind: Type[AnimalMetadataSource], settings: Mapping[str, str], name: str) -> str:
    holder = kind.__new__(kind)
    holder.settings = dict(settings)
    return holder.secret_key(name)


def list_choices(
    kind: Type[AnimalMetadataSource],
    values: Mapping[str, str],
    field_name: str,
    credentials: CredentialStore,
) -> List[str]:
    return list(source_for_setup(kind, values, credentials).choices(field_name))


def read_columns(
    kind: Type[AnimalMetadataSource],
    values: Mapping[str, str],
    credentials: CredentialStore,
) -> SourceTable:
    """Fetch once so the operator can map the source's real column names."""
    source = source_for_setup(kind, values, credentials)
    with tempfile.TemporaryDirectory(prefix="animal-source-setup-") as temporary:
        return source.fetch(Path(temporary))


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


_GUESSES = {
    "external_identity_column": ("subjectid", "animalid", "mouseid", "id", "physicaltag", "eartag", "tag"),
    "new_animal_name_column": ("name", "animalname", "subjectid", "animalid", "mouseid", "physicaltag"),
    "rfid_column": ("rfid", "plateid", "chip", "transponder"),
    "state_column": ("state", "status"),
    "sex_column": ("sex", "gender"),
    "date_of_birth_column": ("dateofbirth", "dob", "birthdate", "born"),
    "strain_column": ("strain", "line", "mouseline"),
    "genotype_column": ("genotype",),
    "cage_column": ("cage", "cagetag", "cageid"),
    "protocol_column": ("protocol", "iacuc"),
}


def suggest_mapping(
    kind: Type[AnimalMetadataSource], headers: Sequence[str]
) -> Dict[str, Optional[str]]:
    """Column choices to pre-fill: the kind's own mapping, else name matches."""
    default = kind.default_mapping()
    if default is not None:
        available = {_key(header) for header in headers}
        if all(_key(column) in available for column in default.required_columns()):
            return {name: getattr(default, name) for name, _, _ in ColumnMapping.COLUMN_FIELDS}
    by_key = {_key(header): header for header in headers if header}
    suggestion = {}
    for name, candidates in _GUESSES.items():
        suggestion[name] = next((by_key[key] for key in candidates if key in by_key), None)
    return suggestion


@dataclass(frozen=True)
class SavedSetup:
    configuration: SourceConfiguration
    secret_locations: Tuple[str, ...]


def save_setup(
    preferences,
    credentials: CredentialStore,
    kind: Type[AnimalMetadataSource],
    values: Mapping[str, str],
    mapping: ColumnMapping,
    *,
    publish_directory: str = "",
    publish_mount: str = "",
) -> SavedSetup:
    settings, secrets = split_settings(kind, values)
    source = kind(settings, credentials)
    for item in kind.setup_fields():
        if item.required and item.kind != "secret" and not settings.get(item.name):
            raise ValueError(f"{kind.label} setup is missing: {item.label}")
        if item.required and item.kind == "secret" and not (
            secrets.get(item.name) or source.secret(item.name)
        ):
            raise ValueError(f"{kind.label} setup is missing: {item.label}")
    configuration = SourceConfiguration(
        kind=kind.kind,
        settings=settings,
        mapping=mapping,
        publish_directory=str(publish_directory or "").strip(),
        publish_mount=str(publish_mount or "").strip(),
    )
    if configuration.publish_directory and kind.kind == "publication":
        raise ValueError("A shared publication cannot also be published")
    locations = tuple(
        credentials.set(source.secret_key(name), value) for name, value in secrets.items()
    )
    save_source_configuration(preferences, configuration)
    return SavedSetup(configuration, locations)
