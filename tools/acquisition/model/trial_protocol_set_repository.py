"""Atomic storage for reusable trial sets and the experiments built on them.

Both repositories mirror TrialProtocolRepository: one file per document, an
atomic write, and a reload that keeps the cached copy of a file that has become
unreadable rather than making the document vanish from the library.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Tuple

from tools.acquisition.model.atomic_session_io import atomic_write_json
from tools.acquisition.model.trial_protocol_set import (
    ExperimentComposition,
    TrialProtocolSet,
)


class _JsonDocumentRepository:
    """Shared mechanics for a directory of individually isolated documents."""

    #: Subclasses set these three.
    document_type = None
    id_field = ""
    label = ""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._documents = {}
        self._path_to_id: Dict[Path, str] = {}
        self._errors: Dict[str, str] = {}

    @property
    def documents(self) -> Tuple:
        with self._lock:
            return tuple(self._documents[key] for key in sorted(self._documents))

    @property
    def errors(self) -> Mapping[str, str]:
        with self._lock:
            return dict(self._errors)

    def get(self, document_id: str):
        with self._lock:
            return self._documents.get(str(document_id).strip().lower())

    def reload(self) -> Tuple:
        """Reload valid files while retaining cached versions of corrupt files."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            previous = dict(self._documents)
            previous_paths = dict(self._path_to_id)
            loaded = {}
            path_to_id: Dict[Path, str] = {}
            errors: Dict[str, str] = {}
            for path in sorted(self.root.glob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as stream:
                        document = self.document_type.from_record(json.load(stream))
                    identifier = getattr(document, self.id_field)
                    expected_name = "{}.json".format(identifier)
                    if path.name != expected_name:
                        raise ValueError(
                            "filename must be {!r} for {} {!r}".format(
                                expected_name, self.label, identifier
                            )
                        )
                    if identifier in loaded:
                        raise ValueError(
                            "duplicate {} ID {!r}".format(self.label, identifier)
                        )
                    loaded[identifier] = document
                    path_to_id[path.resolve()] = identifier
                except Exception as error:
                    errors[path.as_posix()] = "{}: {}".format(
                        type(error).__name__, error
                    )
                    cached_id = previous_paths.get(path.resolve())
                    if cached_id is not None and cached_id in previous:
                        loaded.setdefault(cached_id, previous[cached_id])
                        path_to_id[path.resolve()] = cached_id
            self._documents = loaded
            self._path_to_id = path_to_id
            self._errors = errors
            return self.documents

    def save(self, document):
        """Save the next revision, mirroring TrialProtocolRepository.save."""
        with self._lock:
            identifier = getattr(document, self.id_field)
            current = self._documents.get(identifier)
            if current is None:
                next_revision = max(1, int(document.revision))
            else:
                next_revision = int(current.revision) + 1
            saved = replace(document, revision=next_revision)
            path = self.root / "{}.json".format(identifier)
            atomic_write_json(path, saved.to_record())
            self._documents[identifier] = saved
            self._path_to_id[path.resolve()] = identifier
            self._errors.pop(path.as_posix(), None)
            return saved


class TrialProtocolSetRepository(_JsonDocumentRepository):
    document_type = TrialProtocolSet
    id_field = "set_id"
    label = "set"

    def library(self) -> Dict[str, TrialProtocolSet]:
        """Mapping the compiler accepts as its set_library argument."""
        with self._lock:
            return dict(self._documents)


class ExperimentCompositionRepository(_JsonDocumentRepository):
    document_type = ExperimentComposition
    id_field = "experiment_id"
    label = "experiment"
