"""The interface every animal-metadata source implements, and how kinds are found.

A source kind is a class. Setup describes its settings with
:class:`SetupField`; the setup dialog and the CLI both build their prompts
from that list. Plain settings are stored in the user's preferences; a
``secret`` field is stored only through :class:`CredentialStore`.

A lab adds a kind by publishing a class under the
``reachaq.animal_metadata_sources`` entry-point group, for example in its
package's ``pyproject.toml``::

    [project.entry-points."reachaq.animal_metadata_sources"]
    labdb = "lab_package.sources:LabDatabaseSource"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, Mapping, Optional, Sequence, Tuple, Type

from reachaq.core.logging import get_verbose_logger

from .mapping import ColumnMapping, SourceTable

if TYPE_CHECKING:
    from .credentials import CredentialStore


logger = get_verbose_logger(__name__)

ENTRY_POINT_GROUP = "reachaq.animal_metadata_sources"


@dataclass(frozen=True)
class SetupField:
    """One setting a source kind asks for at setup.

    ``kind`` is ``text``, ``path``, ``secret``, ``choice`` or ``multiline``.
    A ``choice`` field with ``dynamic_choices`` gets its options from
    :meth:`AnimalMetadataSource.choices`, which may contact the source.
    """

    name: str
    label: str
    kind: str = "text"
    required: bool = True
    default: str = ""
    help: str = ""
    choices: Tuple[str, ...] = ()
    dynamic_choices: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"text", "path", "secret", "choice", "multiline"}:
            raise ValueError(f"Unknown setup field kind {self.kind!r}")


class AnimalMetadataSource:
    """Base class for a source of animal metadata.

    Subclasses set ``kind`` (the stored identifier), ``label`` (what the
    operator sees) and ``provider`` (the external-identity provider of the
    records it yields), list their settings in :meth:`setup_fields`, and
    implement :meth:`fetch`.
    """

    kind: ClassVar[str] = ""
    label: ClassVar[str] = ""
    provider: ClassVar[str] = ""

    def __init__(self, settings: Mapping[str, str], credentials: "CredentialStore"):
        self.settings: Dict[str, str] = {
            str(key): str(value) for key, value in dict(settings).items()
        }
        self.credentials = credentials
        missing = [
            item.label
            for item in self.setup_fields()
            if item.required
            and item.kind != "secret"
            # A dynamic choice (a SoftMouse colony) is picked after the source
            # exists; fetch() checks it.
            and not item.dynamic_choices
            and not self.settings.get(item.name)
        ]
        if missing:
            raise ValueError(f"{self.label} setup is missing: {', '.join(missing)}")

    @classmethod
    def setup_fields(cls) -> Tuple[SetupField, ...]:
        return ()

    @classmethod
    def default_mapping(cls) -> Optional[ColumnMapping]:
        """A mapping to offer at setup, or None when columns vary per lab."""
        return None

    def secret_key(self, field_name: str) -> str:
        """The credential-store key for one secret of this source's account."""
        account = self.settings.get("username", "")
        return f"{self.kind}/{field_name}/{account}"

    def secret(self, field_name: str) -> Optional[str]:
        return self.credentials.get(self.secret_key(field_name))

    def choices(self, field_name: str) -> Sequence[str]:
        """Options for a ``dynamic_choices`` field; may contact the source."""
        raise ValueError(f"{self.label} has no choices for {field_name!r}")

    def describe(self) -> str:
        return self.label

    def current_version(self) -> Optional[str]:
        """A cheap content identifier, if the source has one.

        When it matches the last import, a catch-up refresh is skipped.
        None means the source must be fetched to know.
        """
        return None

    def fetch(self, work_directory: Path) -> SourceTable:
        """Return one complete snapshot; may write scratch files to the directory."""
        raise NotImplementedError


def builtin_source_kinds() -> Dict[str, Type[AnimalMetadataSource]]:
    from .database import DatabaseSource
    from .publication import PublicationSource
    from .softmouse import SoftMouseSource
    from .spreadsheet import SpreadsheetSource

    return {
        kind.kind: kind
        for kind in (SoftMouseSource, SpreadsheetSource, DatabaseSource, PublicationSource)
    }


def available_source_kinds(entry_points=None) -> Dict[str, Type[AnimalMetadataSource]]:
    """Built-in kinds plus any a lab package registers as an entry point.

    A broken or misnamed plug-in is logged and left out, so it cannot stop
    the app from starting; it cannot replace a built-in kind.
    """
    kinds = builtin_source_kinds()
    if entry_points is None:
        from importlib.metadata import entry_points as installed

        try:
            entry_points = installed(group=ENTRY_POINT_GROUP)
        except TypeError:  # Python < 3.10
            entry_points = installed().get(ENTRY_POINT_GROUP, ())
    for entry_point in entry_points:
        if entry_point.name in kinds:
            logger.warning(
                "Animal metadata source plug-in ignored: %s would replace a built-in kind",
                entry_point.name,
            )
            continue
        try:
            kind = entry_point.load()
        except Exception:
            logger.exception(
                "Animal metadata source plug-in failed to load: %s", entry_point.name
            )
            continue
        if not (isinstance(kind, type) and issubclass(kind, AnimalMetadataSource)):
            logger.warning(
                "Animal metadata source plug-in ignored: %s is not an "
                "AnimalMetadataSource subclass",
                entry_point.name,
            )
            continue
        if kind.kind != entry_point.name:
            logger.warning(
                "Animal metadata source plug-in ignored: entry point %s names a "
                "class whose kind is %r",
                entry_point.name,
                kind.kind,
            )
            continue
        kinds[entry_point.name] = kind
    return kinds
