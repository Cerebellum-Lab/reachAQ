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

    version: int = 4

    name: str = ""
    id: str = None   # handled in post_init

    is_pellet_dcs: bool = False
    pellet_x: float = 0
    pellet_y: float = 0
    pellet_z: float = 0

    training: AnimalTraining = dataclasses.field(default_factory=AnimalTraining)

    target_y_limit: Optional[float] = None  # in DCS

    autoclamp_evasion_pellets_consumed: int = 0

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if not self.name:
            self.name = f"Mouse-{self.id}"

    @classmethod
    def _load_old_format(cls, data: Dict[str, Any]) -> Self:
        kw = {}
        if "name" in data:
            kw['name'] = data["name"]
        animal = cls(**kw)
        if "pellet_x" in data and "pellet_y" in data and "pellet_z" in data:
            animal.pellet_x = data["pellet_x"]
            animal.pellet_y = data["pellet_y"]
            animal.pellet_z = data["pellet_z"]
        return animal


@dataclass
class AnimalSubject(_AnimalSubject):

    def __init__(self, **kwargs):
        kwargs.pop("baseline_magnet_intensity", None)
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, id={self.id!r})"

    @classmethod
    def from_file(cls: Type[Self], file_path: Path) -> Optional[Self]:
        with file_path.open("r") as file:
            try:
                data = json.load(file)
                if not isinstance(data, dict):
                    if not data:
                        # assuming empty file
                        return cls()
                    raise ValueError(f"Invalid file format in {file_path} ; not a dict: %s", type(data))
                file_version = data.get("version", None)
                if file_version is None or file_version != AnimalSubject.version:
                    logger.notice("Loaded animal version %s vs current Animal version %s",
                                   file_version, AnimalSubject.version)
                if "id" not in data:
                    # old format
                    return cls._load_old_format(data)
                # new "id" format:
                reach = data.pop('reach')
                pellet_dev = reach.pop('pelletDevice', None)
                pellet_dcs = reach.pop('pelletDcs', None)
                if pellet_dcs is None:
                    src = pellet_dev
                else:
                    src = pellet_dcs
                pellet_x, pellet_y, pellet_z = src['x'], src['y'], src['z']
                training = data.pop('training')
                # Removed in v4. Continue accepting v3 animal files without
                # carrying accumulated day/lifetime counts into the new model.
                data.pop("pelletCountsDay", None)
                data.pop("pelletCountsTotal", None)
                data.pop("pelletCountsDayDate", None)
                autoclamp_evasion_pellets_consumed = data.pop("autoclampEvasionPelletsConsumed", 0)
                animal = cls(
                    id=data.pop('id'),
                    name=data.pop('name'),
                    is_pellet_dcs=pellet_dcs is not None,
                    target_y_limit=data.pop('targetYLimit', None),
                    pellet_x=pellet_x,
                    pellet_y=pellet_y,
                    pellet_z=pellet_z,
                    training=AnimalTraining(
                        current_protocol=training.pop('currentProtocol'),
                        protocols=training.pop('protocols'),
                    ),
                    autoclamp_evasion_pellets_consumed=autoclamp_evasion_pellets_consumed,
                )
            except Exception as err:
                logger.error("Error loading animal subject from %s: %s", file_path, err)
                return None

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
        reach: Dict[str, Any] = {}
        key = "pelletDcs" if self.is_pellet_dcs else "pelletDevice"
        reach[key] = {
            'x': self.pellet_x,
            'y': self.pellet_y,
            'z': self.pellet_z,
        }
        data = {
            "version": self.version,
            "id": self.id,
            "name": self.name,
            "reach": reach,
            "training": {
                'currentProtocol': self.training.current_protocol,
                'protocols': self.training.protocols,
            },
            "targetYLimit": self.target_y_limit,
            "autoclampEvasionPelletsConsumed": self.autoclamp_evasion_pellets_consumed,
        }
        xyz = Offset3DTuple(self.pellet_x, self.pellet_y, self.pellet_z)
        logger.debug("Saving %s to %s ; xyz=%s", self.name, file_path.as_posix(), xyz.humanize())
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, dir=file_path.parent) as fh:
            json.dump(data, fh, indent=4)
        os.replace(fh.name, file_path)
