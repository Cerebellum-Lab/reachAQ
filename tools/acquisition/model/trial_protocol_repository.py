"""Atomic reusable storage for ordered pellet-trial protocols."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from tools.acquisition.model.atomic_session_io import (
    atomic_publish_file,
    atomic_write_json,
    fsync_directory,
)
from tools.acquisition.model.trial_protocol_schedule import TrialProtocolDocument


class TrialProtocolRepository:
    """Own a shared directory of individually isolated protocol documents."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._documents: Dict[str, TrialProtocolDocument] = {}
        self._path_to_id: Dict[Path, str] = {}
        self._errors: Dict[str, str] = {}

    @property
    def documents(self) -> Tuple[TrialProtocolDocument, ...]:
        with self._lock:
            return tuple(
                self._documents[key] for key in sorted(self._documents)
            )

    @property
    def errors(self) -> Mapping[str, str]:
        with self._lock:
            return dict(self._errors)

    def get(self, protocol_id: str) -> Optional[TrialProtocolDocument]:
        with self._lock:
            return self._documents.get(str(protocol_id).strip().lower())

    def reload(self) -> Tuple[TrialProtocolDocument, ...]:
        """Reload valid files while retaining cached versions of corrupt files."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            previous = dict(self._documents)
            previous_paths = dict(self._path_to_id)
            loaded: Dict[str, TrialProtocolDocument] = {}
            path_to_id: Dict[Path, str] = {}
            errors: Dict[str, str] = {}
            for path in sorted(self.root.glob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as stream:
                        document = TrialProtocolDocument.from_record(json.load(stream))
                    expected_name = f"{document.protocol_id}.json"
                    if path.name != expected_name:
                        raise ValueError(
                            f"filename must be {expected_name!r} for protocol "
                            f"{document.protocol_id!r}"
                        )
                    existing = loaded.get(document.protocol_id)
                    if existing is not None:
                        raise ValueError(
                            f"duplicate protocol ID {document.protocol_id!r}"
                        )
                    loaded[document.protocol_id] = document
                    path_to_id[path.resolve()] = document.protocol_id
                except Exception as error:
                    errors[path.as_posix()] = f"{type(error).__name__}: {error}"
                    cached_id = previous_paths.get(path.resolve())
                    if cached_id is not None and cached_id in previous:
                        loaded.setdefault(cached_id, previous[cached_id])
                        path_to_id[path.resolve()] = cached_id
            self._documents = loaded
            self._path_to_id = path_to_id
            self._errors = errors
            return self.documents

    def save(
        self,
        document: TrialProtocolDocument,
        *,
        expected_revision: Optional[int] = None,
    ) -> TrialProtocolDocument:
        """Save a new revision with optimistic concurrency protection."""
        with self._lock:
            current = self._documents.get(document.protocol_id)
            if expected_revision is not None:
                actual = None if current is None else current.revision
                if actual != int(expected_revision):
                    raise RuntimeError(
                        f"Protocol {document.protocol_id!r} changed: expected "
                        f"revision {expected_revision}, found {actual}"
                    )
            if current is None:
                next_revision = max(1, int(document.revision))
            else:
                next_revision = int(current.revision) + 1
            saved = replace(document, revision=next_revision)
            # Expand and validate the complete schedule before touching disk.
            saved.resolve()
            path = self._path(saved.protocol_id)
            atomic_write_json(path, saved.to_record())
            self._documents[saved.protocol_id] = saved
            self._path_to_id[path.resolve()] = saved.protocol_id
            self._errors.pop(path.as_posix(), None)
            return saved

    def duplicate(
        self,
        source_id: str,
        *,
        protocol_id: str,
        name: str,
    ) -> TrialProtocolDocument:
        with self._lock:
            source = self._require(source_id)
            if self.get(protocol_id) is not None or self._path(protocol_id).exists():
                raise FileExistsError(f"Protocol {protocol_id!r} already exists")
            duplicated = replace(
                source,
                protocol_id=protocol_id,
                name=str(name).strip(),
                revision=1,
            )
            return self.save(duplicated)

    def rename(
        self,
        source_id: str,
        *,
        protocol_id: str,
        name: str,
        expected_revision: Optional[int] = None,
    ) -> TrialProtocolDocument:
        """Atomically publish the new identity before archiving the old file."""
        with self._lock:
            source = self._require(source_id)
            if expected_revision is not None and source.revision != expected_revision:
                raise RuntimeError("Protocol changed before rename")
            destination = replace(
                source,
                protocol_id=protocol_id,
                name=str(name).strip(),
                revision=1,
            )
            if destination.protocol_id != source.protocol_id:
                if self.get(destination.protocol_id) is not None:
                    raise FileExistsError(
                        f"Protocol {destination.protocol_id!r} already exists"
                    )
                destination = self.save(destination)
                source_path = self._path(source.protocol_id)
                archive = source_path.with_name(
                    f"{source_path.name}.revision-{source.revision}.renamed-backup"
                )
                source_path.replace(archive)
                fsync_directory(source_path.parent)
                self._documents.pop(source.protocol_id, None)
                self._path_to_id.pop(source_path.resolve(), None)
                return destination
            return self.save(
                replace(source, name=destination.name),
                expected_revision=source.revision,
            )

    def import_file(
        self,
        source: Path,
        *,
        replace_existing: bool = False,
    ) -> TrialProtocolDocument:
        source = Path(source)
        with source.open("r", encoding="utf-8") as stream:
            document = TrialProtocolDocument.from_record(json.load(stream))
        with self._lock:
            current = self.get(document.protocol_id)
            if current is not None and not replace_existing:
                raise FileExistsError(
                    f"Protocol {document.protocol_id!r} already exists"
                )
            return self.save(
                document,
                expected_revision=(None if current is None else current.revision),
            )

    def export_file(self, protocol_id: str, destination: Path) -> Path:
        document = self._require(protocol_id)
        destination = Path(destination)

        def write(path: Path) -> None:
            with path.open("w", encoding="utf-8") as stream:
                json.dump(document.to_record(), stream, indent=2, sort_keys=True)
                stream.write("\n")

        def validate(path: Path) -> None:
            with path.open("r", encoding="utf-8") as stream:
                TrialProtocolDocument.from_record(json.load(stream)).resolve()

        return atomic_publish_file(destination, write, validate=validate)

    def _path(self, protocol_id: str) -> Path:
        # TrialProtocolDocument performs the authoritative identifier validation.
        protocol_id = TrialProtocolDocument(
            protocol_id=protocol_id,
            name="Path validation",
            trial_count=1,
        ).protocol_id
        return self.root / f"{protocol_id}.json"

    def _require(self, protocol_id: str) -> TrialProtocolDocument:
        document = self.get(protocol_id)
        if document is None:
            raise KeyError(f"Unknown protocol {protocol_id!r}")
        return document
