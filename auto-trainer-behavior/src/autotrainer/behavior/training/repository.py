import dataclasses
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from autotrainer.core.logging import get_verbose_logger

from .plan import TrainingPlan
from .schema import PlanDocument, PlanFormatError, parse_plan

logger = get_verbose_logger(__name__)


@dataclasses.dataclass(frozen=True)
class PlanInfo:
    plan_id: str
    name: str
    description: str
    path: Path


class PlanRepository:
    """The ``*.json`` plan files of one directory.

    ``get_plans`` reads them all and fails on the first unreadable or
    duplicated plan, so a bad library is never half-loaded. ``get_plan``
    returns a new ``TrainingPlan`` on every call: progress belongs to the
    animal it is attached for, never to the shared document.
    """

    def __init__(self, root: Optional[Path] = None):
        self._root = None if root is None else Path(root)
        self._lock = threading.RLock()
        self._documents: Dict[str, PlanDocument] = {}
        self._infos: List[PlanInfo] = []
        self._loaded = False

    @property
    def root(self) -> Optional[Path]:
        return self._root

    @root.setter
    def root(self, value: Optional[Path]):
        with self._lock:
            self._root = None if value is None else Path(value)
            # Never serve another directory's plans after a failed reload.
            self._documents, self._infos = {}, []
            self._loaded = False

    def get_plans(self, *, refresh: bool = False) -> List[PlanInfo]:
        with self._lock:
            if refresh or not self._loaded:
                self._documents, self._infos = self._read()
                self._loaded = True
            return list(self._infos)

    def get_plan(self, plan_id: str) -> Optional[TrainingPlan]:
        with self._lock:
            if not self._loaded:
                try:
                    self.get_plans()
                except PlanFormatError as err:
                    logger.error("Training plans not loaded: %s", err)
                    return None
            document = self._documents.get(plan_id)
        return None if document is None else TrainingPlan(document)

    def _read(self):
        root = self._root
        if root is None or not root.is_dir():
            return {}, []
        documents: Dict[str, PlanDocument] = {}
        infos: List[PlanInfo] = []
        paths: Dict[str, Path] = {}
        phase_owner: Dict[str, str] = {}
        for path in sorted(root.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    document = parse_plan(json.load(fh), source=path.name)
            except (OSError, json.JSONDecodeError, PlanFormatError) as err:
                raise PlanFormatError(f"{path}: {err}") from None
            if document.plan_id in paths:
                raise PlanFormatError(
                    f"duplicated planId {document.plan_id} in {path} and {paths[document.plan_id]}")
            # Phase ids name a phase in saved progress and session evidence,
            # so they are unique across the library, not only within a plan.
            for phase in document.phases:
                owner = phase_owner.setdefault(phase.phase_id, document.plan_id)
                if owner != document.plan_id:
                    raise PlanFormatError(
                        f"duplicated phaseId {phase.phase_id} in plans {owner} and {document.plan_id}")
            paths[document.plan_id] = path
            documents[document.plan_id] = document
            infos.append(PlanInfo(document.plan_id, document.name, document.description, path))
        logger.info("Loaded %d training plan(s) from %s", len(infos), root)
        return documents, infos
