from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

from autotrainer.core import AnimalSubject


@dataclass(frozen=True)
class AnimalReconciliationChoices:
    """Source UUID selected for each conflicting reachAQ-owned field group."""

    name_from: str
    pellet_position_from: str
    training_from: str
    target_limit_from: str
    external_identity_from: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name_from,
            "pelletPosition": self.pellet_position_from,
            "training": self.training_from,
            "targetLimit": self.target_limit_from,
            "externalIdentity": self.external_identity_from,
        }


def reconcile_animals(
    survivor: AnimalSubject,
    loser: AnimalSubject,
    choices: AnimalReconciliationChoices,
) -> AnimalSubject:
    if survivor.id == loser.id:
        raise ValueError("Survivor and losing animal must be different")
    sources = {survivor.id: survivor, loser.id: loser}
    for field, source_id in choices.to_dict().items():
        if source_id not in sources:
            raise ValueError(f"Invalid {field} source UUID {source_id!r}")
    result = copy.deepcopy(survivor)
    result.name = sources[choices.name_from].name
    pellet = sources[choices.pellet_position_from]
    result.is_pellet_dcs = pellet.is_pellet_dcs
    result.pellet_x = pellet.pellet_x
    result.pellet_y = pellet.pellet_y
    result.pellet_z = pellet.pellet_z
    result.training = copy.deepcopy(sources[choices.training_from].training)
    result.target_y_limit = sources[choices.target_limit_from].target_y_limit
    external = sources[choices.external_identity_from]
    result.external_identity = copy.deepcopy(external.external_identity)
    result.external_metadata = copy.deepcopy(external.external_metadata)
    # The surviving reachAQ UUID is never selectable as an overwrite field.
    result.id = survivor.id
    return result
