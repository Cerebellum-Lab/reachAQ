"""Pluggable sources of animal metadata: SoftMouse, a spreadsheet, a database,
a shared publication, or a lab's own kind (see :mod:`.base`)."""

from .base import (
    ENTRY_POINT_GROUP,
    AnimalMetadataSource,
    SetupField,
    available_source_kinds,
)
from .config import (
    ConfiguredSource,
    SourceConfiguration,
    build_source,
    load_source_configuration,
    save_source_configuration,
)
from .credentials import CredentialStore
from .mapping import (
    AnimalTableMapper,
    ColumnMapping,
    ImportGuardrails,
    SourceTable,
    SpreadsheetImportPreview,
    read_table_file,
    softmouse_mapping,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "AnimalMetadataSource",
    "AnimalTableMapper",
    "ColumnMapping",
    "ConfiguredSource",
    "CredentialStore",
    "ImportGuardrails",
    "SetupField",
    "SourceConfiguration",
    "SourceTable",
    "SpreadsheetImportPreview",
    "available_source_kinds",
    "build_source",
    "load_source_configuration",
    "read_table_file",
    "save_source_configuration",
    "softmouse_mapping",
]
