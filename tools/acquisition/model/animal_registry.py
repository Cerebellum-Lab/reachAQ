from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.animal.external_metadata import (
    ExternalAnimalRecord,
    ExternalIdentity,
    NormalizedAnimalBatch,
    normalize_rfid,
)


logger = get_verbose_logger(__name__)


@dataclass(frozen=True)
class RegistryImportResult:
    import_id: str
    source_file_sha256: str
    imported_records: int
    unchanged: bool = False


class RegistrySchemaError(RuntimeError):
    pass


class AnimalRegistry:
    """Per-computer, rebuildable cache of current external animal records."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.recovered_corrupt_path: Optional[Path] = None
        try:
            self._initialize()
        except RegistrySchemaError:
            raise
        except sqlite3.DatabaseError:
            if not self.path.exists():
                raise
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            recovery = self.path.with_name(
                f"{self.path.name}.{timestamp}.corrupt-backup"
            )
            os.replace(self.path, recovery)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(self.path) + suffix)
                if sidecar.exists():
                    os.replace(sidecar, Path(str(recovery) + suffix))
            self.recovered_corrupt_path = recovery
            logger.warning(
                "SoftMouse registry was corrupt and rebuilt: path=%s backup=%s",
                self.path,
                recovery,
            )
            self._initialize()
        logger.info(
            "SoftMouse registry ready: path=%s recovered=%s",
            self.path,
            self.recovered_corrupt_path is not None,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            version = int(db.execute("PRAGMA user_version").fetchone()[0])
            if version > self.SCHEMA_VERSION:
                raise RegistrySchemaError(
                    f"Registry schema version {version} is newer than supported "
                    f"version {self.SCHEMA_VERSION}"
                )
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS imports (
                    import_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    source_file_sha256 TEXT NOT NULL,
                    source_sheet TEXT NOT NULL,
                    header_signature TEXT NOT NULL,
                    mapping_profile_id TEXT NOT NULL,
                    mapping_profile_version INTEGER NOT NULL,
                    imported_utc TEXT NOT NULL,
                    total_source_rows INTEGER NOT NULL,
                    accepted_rows INTEGER NOT NULL,
                    ignored_missing_rfid_rows INTEGER NOT NULL,
                    ignored_ended_rows INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    UNIQUE(provider, source_file_sha256, status)
                );
                CREATE TABLE IF NOT EXISTS external_animals (
                    provider TEXT NOT NULL,
                    subject_id_kind TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    physical_rfid TEXT NOT NULL,
                    name_candidate TEXT,
                    state TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    first_seen_import_id TEXT NOT NULL,
                    last_seen_import_id TEXT NOT NULL,
                    current INTEGER NOT NULL CHECK(current IN (0, 1)),
                    PRIMARY KEY(provider, subject_id_kind, subject_id)
                );
                CREATE TABLE IF NOT EXISTS external_identifiers (
                    provider TEXT NOT NULL,
                    identifier_kind TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    subject_id_kind TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    import_id TEXT NOT NULL,
                    active INTEGER NOT NULL CHECK(active IN (0, 1)),
                    PRIMARY KEY(provider, identifier_kind, normalized_value)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_rfid
                    ON external_identifiers(provider, identifier_kind, normalized_value)
                    WHERE active = 1;
                """
            )
            db.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            check = db.execute("PRAGMA quick_check").fetchone()[0]
            if check != "ok":
                raise sqlite3.DatabaseError(f"SQLite quick_check failed: {check}")

    def replace(self, batch: NormalizedAnimalBatch) -> RegistryImportResult:
        with self._lock, self._connect() as db:
            existing = db.execute(
                "SELECT import_id FROM imports WHERE provider=? AND "
                "source_file_sha256=? AND status='complete'",
                (batch.provider, batch.source_file_sha256),
            ).fetchone()
            if existing is not None:
                logger.info(
                    "SoftMouse registry import unchanged: import_id=%s records=%d "
                    "source_sha256=%s",
                    existing["import_id"],
                    len(batch.records),
                    batch.source_file_sha256,
                )
                return RegistryImportResult(
                    existing["import_id"], batch.source_file_sha256, len(batch.records), True
                )
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE external_animals SET current=0 WHERE provider=?",
                    (batch.provider,),
                )
                db.execute(
                    "UPDATE external_identifiers SET active=0 WHERE provider=?",
                    (batch.provider,),
                )
                for record in batch.records:
                    encoded = json.dumps(self._record_to_dict(record), sort_keys=True)
                    db.execute(
                        """
                        INSERT INTO external_animals(
                            provider, subject_id_kind, subject_id, physical_rfid,
                            name_candidate, state, record_json, source_hash,
                            first_seen_import_id, last_seen_import_id, current
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(provider, subject_id_kind, subject_id) DO UPDATE SET
                            physical_rfid=excluded.physical_rfid,
                            name_candidate=excluded.name_candidate,
                            state=excluded.state,
                            record_json=excluded.record_json,
                            source_hash=excluded.source_hash,
                            last_seen_import_id=excluded.last_seen_import_id,
                            current=1
                        """,
                        (
                            *record.identity.key,
                            record.physical_rfid,
                            record.new_animal_name_candidate,
                            record.state,
                            encoded,
                            record.source_hash,
                            batch.import_id,
                            batch.import_id,
                        ),
                    )
                    db.execute(
                        """
                        INSERT INTO external_identifiers(
                            provider, identifier_kind, normalized_value,
                            subject_id_kind, subject_id, import_id, active
                        ) VALUES (?, 'rfid', ?, ?, ?, ?, 1)
                        ON CONFLICT(provider, identifier_kind, normalized_value)
                        DO UPDATE SET subject_id_kind=excluded.subject_id_kind,
                            subject_id=excluded.subject_id,
                            import_id=excluded.import_id, active=1
                        """,
                        (
                            record.identity.provider,
                            record.physical_rfid,
                            record.identity.subject_id_kind,
                            record.identity.subject_id,
                            batch.import_id,
                        ),
                    )
                db.execute(
                    """
                    INSERT INTO imports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'complete')
                    """,
                    (
                        batch.import_id,
                        batch.provider,
                        batch.source_filename,
                        batch.source_file_sha256,
                        batch.source_sheet,
                        batch.header_signature,
                        batch.mapping_profile_id,
                        batch.mapping_profile_version,
                        batch.imported_utc,
                        batch.total_source_rows,
                        len(batch.records),
                        batch.ignored_missing_rfid_rows,
                        batch.ignored_ended_rows,
                    ),
                )
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(
                    "SoftMouse registry replacement rolled back: import_id=%s "
                    "records=%d source_sha256=%s",
                    batch.import_id,
                    len(batch.records),
                    batch.source_file_sha256,
                )
                raise
        logger.info(
            "SoftMouse registry replaced: import_id=%s records=%d source_sha256=%s",
            batch.import_id,
            len(batch.records),
            batch.source_file_sha256,
        )
        return RegistryImportResult(
            batch.import_id, batch.source_file_sha256, len(batch.records)
        )

    def resolve_rfid(self, rfid: str) -> Optional[ExternalAnimalRecord]:
        normalized = normalize_rfid(rfid)
        with self._lock, self._connect() as db:
            row = db.execute(
                """
                SELECT animal.record_json
                FROM external_identifiers identifier
                JOIN external_animals animal
                  ON animal.provider=identifier.provider
                 AND animal.subject_id_kind=identifier.subject_id_kind
                 AND animal.subject_id=identifier.subject_id
                WHERE identifier.provider='softmouse'
                  AND identifier.identifier_kind='rfid'
                  AND identifier.normalized_value=?
                  AND identifier.active=1 AND animal.current=1
                """,
                (normalized,),
            ).fetchone()
        record = None if row is None else self._record_from_dict(json.loads(row[0]))
        logger.info(
            "SoftMouse registry RFID lookup: rfid=%s matched=%s subject_id=%s",
            normalized,
            record is not None,
            None if record is None else record.identity.subject_id,
        )
        return record

    def list_current_records(self) -> List[ExternalAnimalRecord]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT record_json FROM external_animals WHERE current=1 "
                "ORDER BY name_candidate, subject_id"
            ).fetchall()
        return [self._record_from_dict(json.loads(row[0])) for row in rows]

    def last_complete_import(self) -> Optional[sqlite3.Row]:
        with self._lock, self._connect() as db:
            return db.execute(
                "SELECT * FROM imports WHERE status='complete' "
                "ORDER BY imported_utc DESC LIMIT 1"
            ).fetchone()

    @staticmethod
    def _record_to_dict(record: ExternalAnimalRecord):
        return {
            "identity": record.identity.to_dict(),
            "physicalRfid": record.physical_rfid,
            "newAnimalNameCandidate": record.new_animal_name_candidate,
            "state": record.state,
            "sourcePayload": dict(record.source_payload),
            "sourceHash": record.source_hash,
            "sex": record.sex,
            "dateOfBirth": record.date_of_birth,
            "strain": record.strain,
            "genotype": list(record.genotype),
            "cage": record.cage,
            "protocol": record.protocol,
        }

    @staticmethod
    def _record_from_dict(value) -> ExternalAnimalRecord:
        return ExternalAnimalRecord(
            identity=ExternalIdentity.from_dict(value["identity"]),
            physical_rfid=value["physicalRfid"],
            new_animal_name_candidate=value.get("newAnimalNameCandidate"),
            state=value["state"],
            source_payload=value.get("sourcePayload") or {},
            source_hash=value["sourceHash"],
            sex=value.get("sex"),
            date_of_birth=value.get("dateOfBirth"),
            strain=value.get("strain"),
            genotype=tuple(value.get("genotype") or ()),
            cage=value.get("cage"),
            protocol=value.get("protocol"),
        )
