import dataclasses
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Dict, Any, List, Type
from typing_extensions import Self

from autotrainer.api.api_system_status import ApiAnimalStatus, ApiReachStatus

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

    version: int = 5

    name: str = ""
    id: str = None   # handled in post_init

    is_pellet_dcs: bool = False
    pellet_x: float = 0
    pellet_y: float = 0
    pellet_z: float = 0

    training: AnimalTraining = dataclasses.field(default_factory=AnimalTraining)

    target_y_limit: Optional[float] = None  # in DCS

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
            logger.notice(
                "Loaded animal v4 for one-way migration to v5; protocol "
                "recording-count progress was reset"
            )
        elif file_version == cls.version:
            animal = cls._from_v5(data)
        else:
            raise ValueError(
                f"Unsupported animal schema version {file_version!r} in "
                f"{file_path}; only v4 migration and v5 are supported"
            )

        logger.debug("loaded animal id=%r name=%r pellet=%s is_dcs=%s current_protocol=%s",
                     animal.id, animal.name,
                     (animal.pellet_x, animal.pellet_y, animal.pellet_z), animal.is_pellet_dcs,
                     animal.training.current_protocol)

        return animal

    def to_api_status(self) -> ApiAnimalStatus:
        return ApiAnimalStatus(
            identifier=self.id,
            name=self.name,
            dcs_send_x=self.pellet_x,
            dcs_send_y=self.pellet_y,
            dcs_send_z=self.pellet_z,
            target_y_limit=self.target_y_limit,
            # Keep the public API schema intact while retiring persisted
            # day/lifetime counters. Session counts are reported in the
            # system behavior status instead.
            reach_status_total=ApiReachStatus(),
            reach_status_day=ApiReachStatus(),
        )

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
        }
        xyz = Offset3DTuple(self.pellet_x, self.pellet_y, self.pellet_z)
        logger.debug("Saving %s to %s ; xyz=%s", self.name, file_path.as_posix(), xyz.humanize())
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            self._legacy_v4_content is not None
            and self._legacy_v4_path == file_path.resolve()
        ):
            backup_path = file_path.with_suffix(file_path.suffix + ".v4-backup")
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
