"""reachAQ training plans: phases with predicates and per-trial actions.

Plan files live in the loaded configuration's ``training_plans/`` directory;
the format is documented in ``schema``. Progress is saved in the animal file.
"""

from .plan import (
    LoadProgressResult,
    PhaseProgress,
    SessionResult,
    TrainingPhase,
    TrainingPlan,
    TrainingProgressState,
    TrialDirective,
    TrialKind,
    TrialResult,
)
from .repository import PlanInfo, PlanRepository
from .schema import PlanFormatError, parse_plan

__all__ = [
    "LoadProgressResult",
    "PhaseProgress",
    "PlanFormatError",
    "PlanInfo",
    "PlanRepository",
    "SessionResult",
    "TrainingPhase",
    "TrainingPlan",
    "TrainingProgressState",
    "TrialDirective",
    "TrialKind",
    "TrialResult",
    "parse_plan",
]
