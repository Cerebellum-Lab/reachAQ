import json
from pathlib import Path
from typing import Dict, Any

from autotrainer.core.logging import get_verbose_logger


logger = get_verbose_logger(__name__)


def get_plan_id(dct: Dict) -> str:
    return dct['planId']


def get_phase_id(dct: Dict) -> str:
    return dct['phaseId']


def load_training_plans(dir_path: Path) -> Dict[Path, Dict[str, Any]]:
    files = list(dir_path.glob("*.json"))
    plans: Dict[Path, Dict] = {}
    logger.verbose("Loading training plans: %s", files)
    for path in files:
        try:
            with path.open("r") as fh:
                plans[path] = json.load(fh)
        except Exception as err:
            logger.error("Could not load plan %s: %s", path, err)
            raise
    phase_ids = {}
    plan_ids = {}
    for path, plan in plans.items():
        plan_id = get_plan_id(plan)
        prev_path, prev_plan = plan_ids.setdefault(plan_id, (path, plan))
        if path != prev_path:
            raise RuntimeError(f"duplicated plan_id {plan_id} in {path} vs {prev_path}")
        for phase in plan['phases']:
            phase_id = get_phase_id(phase)
            prev_path, prev_phase = phase_ids.setdefault(phase_id, (path, phase))
            if path != prev_path:
                raise RuntimeError(f"Duplicated phase_id: {phase_id} in path {path} vs {prev_path}")
            if id(prev_phase) != id(phase):
                raise RuntimeError(f"Duplicated phase_id: {phase_id} in path {path}: {prev_phase} vs {phase}")
    return plans
