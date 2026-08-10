import dataclasses
import datetime as dt
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Dict, Any, List, Type
from typing_extensions import Self

from .external_metadata import ExternalIdentity, ExternalMetadataSnapshot

from .. import Offset3DTuple, get_verbose_logger

logger = get_verbose_logger(__name__)


@dataclass
class AnimalTraining:
    """Animal Training configuration"""

    # NB: protocol == plan ; todo: could/should better be moved to auto-trainer-training repo

    current_protocol: Optional[str] = None
    protocols: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def get_plan_progress(self, plan_id: str) -> Optional[Dict[str, Any]]:
        # {"plan_id": self.plan_id,
        #                 "progress_state": self.progress_state,
        #                 "current_phase_id": None if self.current_phase is None else self.current_phase.phase_id,
        #                 "progress": progress
        #                 }
        for prot in self.protocols:
            if prot.get('plan_id') == plan_id:
                return prot
        return None

    def set_plan_progress(self, plan_id: str, progress: Dict[str, Any]):
        for idx, prog in enumerate(self.protocols):
            if prog['plan_id'] == plan_id:
                self.protocols[idx] = progress
                return
        self.protocols.append(progress)


@dataclass
class _AnimalSubject:
    """A subject in an animal experiment."""

    version: int = 6

    name: str = ""
    id: str = None   # handled in post_init

    is_pellet_dcs: bool = False
    pellet_x: float = 0
    pellet_y: float = 0
    pellet_z: float = 0

    training: AnimalTraining = dataclasses.field(default_factory=AnimalTraining)

    target_y_limit: Optional[float] = None  # in DCS

    external_identity: Optional[ExternalIdentity] = None
    external_metadata: Optional[ExternalMetadataSnapshot] = None

    _legacy_v4_path: Optional[Path] = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _legacy_v4_content: Optional[bytes] = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    _legacy_version: Optional[int] = dataclasses.field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if not self.name:
            self.name = f"Mouse-{self.id}"

@dataclass
class AnimalSubject(_AnimalSubject):
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, id={self.id!r})"

    @staticmethod
    def _rename_progress_count(value, *, to_persisted: bool):
        if isinstance(value, list):
            return [
                AnimalSubject._rename_progress_count(
                    item,
                    to_persisted=to_persisted,
                )
                for item in value
            ]
        if not isinstance(value, dict):
            return value
        source = "session_count" if to_persisted else "trial_count"
        target = "trial_count" if to_persisted else "session_count"
        return {
            (target if key == source else key): AnimalSubject._rename_progress_count(
                item,
                to_persisted=to_persisted,
            )
            for key, item in value.items()
        }

    @classmethod
    def _from_v4(cls, data: Dict[str, Any]) -> Self:
        if "id" not in data:
            raise ValueError("Animal v4 requires an id; older no-id files are unsupported")
        reach = data.get("reach")
        if not isinstance(reach, dict):
            raise ValueError("Animal v4 requires reach coordinates")
        pellet_dcs = reach.get("pelletDcs")
        position = pellet_dcs or reach.get("pelletDevice")
        if not isinstance(position, dict):
            raise ValueError("Animal v4 requires pelletDcs or pelletDevice coordinates")
        training = data.get("training") or {}
        # Recording-count progress cannot be converted to pellet-trial progress.
        return cls(
            id=data["id"],
            name=data["name"],
            is_pellet_dcs=pellet_dcs is not None,
            target_y_limit=data.get("targetYLimit"),
            pellet_x=position["x"],
            pellet_y=position["y"],
            pellet_z=position["z"],
            training=AnimalTraining(
                current_protocol=training.get("currentProtocol"),
                protocols=[],
            ),
        )

    @classmethod
    def _from_v5(cls, data: Dict[str, Any]) -> Self:
        pellet = data["pellet"]
        position = pellet["position"]
        training = data.get("training") or {}
        limits = data.get("limits") or {}
        protocol_progress = cls._rename_progress_count(
            training.get("protocolProgress", []),
            to_persisted=False,
        )
        return cls(
            id=data["id"],
            name=data["name"],
            is_pellet_dcs=pellet["coordinateSpace"] == "dcs",
            target_y_limit=limits.get("targetY"),
            pellet_x=position["x"],
            pellet_y=position["y"],
            pellet_z=position["z"],
            training=AnimalTraining(
                current_protocol=training.get("selectedProtocol"),
                protocols=protocol_progress,
            ),
        )

    @classmethod
    def _from_v6(cls, data: Dict[str, Any]) -> Self:
        animal = cls._from_v5(data)
        identity = data.get("externalIdentity")
        metadata = data.get("externalMetadata")
        animal.external_identity = (
            None if identity is None else ExternalIdentity.from_dict(identity)
        )
        animal.external_metadata = (
            None if metadata is None else ExternalMetadataSnapshot.from_dict(metadata)
        )
        return animal

    @classmethod
    def from_file(cls: Type[Self], file_path: Path) -> Optional[Self]:
        original = file_path.read_bytes()
        data = json.loads(original)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid animal file {file_path}: expected an object")
        file_version = data.get("version")
        if file_version == 4:
            animal = cls._from_v4(data)
            animal._legacy_v4_path = file_path.resolve()
            animal._legacy_v4_content = original
            animal._legacy_version = 4
            logger.notice(
                "Loaded animal v4 for one-way migration to v6; protocol "
                "recording-count progress was reset"
            )
        elif file_version == 5:
            animal = cls._from_v5(data)
            animal._legacy_v4_path = file_path.resolve()
            animal._legacy_v4_content = original
            animal._legacy_version = 5
            logger.notice("Loaded animal v5 for one-way migration to v6")
        elif file_version == cls.version:
            animal = cls._from_v6(data)
        else:
            raise ValueError(
                f"Unsupported animal schema version {file_version!r} in "
                f"{file_path}; only v4/v5 migration and v6 are supported"
            )

        logger.debug("loaded animal id=%r name=%r pellet=%s is_dcs=%s current_protocol=%s",
                     animal.id, animal.name,
                     (animal.pellet_x, animal.pellet_y, animal.pellet_z), animal.is_pellet_dcs,
                     animal.training.current_protocol)

        return animal

    def to_api_status(self) -> Dict[str, Any]:
        return {
            "identifier": self.id,
            "name": self.name,
            "pellet": {
                "coordinateSpace": "dcs" if self.is_pellet_dcs else "device",
                "position": {
                    "x": self.pellet_x,
                    "y": self.pellet_y,
                    "z": self.pellet_z,
                },
            },
            "targetYLimit": self.target_y_limit,
            "selectedProtocol": self.training.current_protocol,
        }

    def session_snapshot(self, *, snapshot_utc: Optional[str] = None) -> Dict[str, Any]:
        if snapshot_utc is None:
            snapshot_utc = (
                dt.datetime.now(dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        identity = self.external_identity
        metadata = self.external_metadata
        return {
            "reachaqId": self.id,
            "name": self.name,
            "externalIdentity": (
                None
                if identity is None
                else {
                    **identity.to_dict(),
                    "rfid": None if metadata is None else metadata.rfid,
                }
            ),
            "metadata": (
                None
                if metadata is None
                else {
                    "physicalTag": metadata.physical_tag,
                    "sex": metadata.sex,
                    "dateOfBirth": metadata.date_of_birth,
                    "strain": metadata.strain,
                    "genotype": list(metadata.genotype),
                    "cage": metadata.cage,
                    "protocol": metadata.protocol,
                    "state": metadata.state,
                }
            ),
            "provenance": {
                "registryImportId": (
                    None if metadata is None else metadata.registry_import_id
                ),
                "sourceFileSha256": (
                    None if metadata is None else metadata.source_file_sha256
                ),
                "sourceRecordHash": None if metadata is None else metadata.source_hash,
                "importedUtc": None if metadata is None else metadata.imported_utc,
                "snapshotUtc": snapshot_utc,
            },
        }

    def to_file(self, file_path: Path):
        data = {
            "version": self.version,
            "id": self.id,
            "name": self.name,
            "pellet": {
                "coordinateSpace": "dcs" if self.is_pellet_dcs else "device",
                "position": {
                    "x": self.pellet_x,
                    "y": self.pellet_y,
                    "z": self.pellet_z,
                },
            },
            "training": {
                "selectedProtocol": self.training.current_protocol,
                "protocolProgress": self._rename_progress_count(
                    self.training.protocols,
                    to_persisted=True,
                ),
            },
            "limits": {
                "targetY": self.target_y_limit,
            },
            "externalIdentity": (
                None
                if self.external_identity is None
                else self.external_identity.to_dict()
            ),
            "externalMetadata": (
                None
                if self.external_metadata is None
                else self.external_metadata.to_dict()
            ),
        }
        xyz = Offset3DTuple(self.pellet_x, self.pellet_y, self.pellet_z)
        logger.debug("Saving %s to %s ; xyz=%s", self.name, file_path.as_posix(), xyz.humanize())
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            self._legacy_v4_content is not None
            and self._legacy_v4_path == file_path.resolve()
        ):
            legacy_version = self._legacy_version or 4
            backup_path = file_path.with_suffix(
                file_path.suffix + f".v{legacy_version}-backup"
            )
            if not backup_path.exists():
                with NamedTemporaryFile(
                    "wb",
                    delete=False,
                    dir=file_path.parent,
                ) as backup:
                    backup.write(self._legacy_v4_content)
                os.replace(backup.name, backup_path)
        with NamedTemporaryFile("w", delete=False, dir=file_path.parent) as fh:
            json.dump(data, fh, indent=4)
        os.replace(fh.name, file_path)
        self._legacy_v4_path = None
        self._legacy_v4_content = None
        self._legacy_version = None
