from dataclasses import dataclass
from typing import Protocol, Optional

from autotrainer.core import ProjectInfo
from autotrainer.core.interfaces import (  # noqa
    # actually for autotrainer.training only
    CaptureAnalysisResult, RecordingEndingReason
)


# keeping top level atm, given not quite sure where to put

class CompleteCallbackT(Protocol):

    def __call__(self, success: bool, *, error: Optional[str] = None):
        """Signature of Segmentation/Detection Complete Callback"""


class _UnconfiguredCompleteAction:

    def __call__(self, success: bool, *, error: Optional[str] = None):
        raise RuntimeError("complete attribute unconfigured")


@dataclass
class SegmentationConfiguration:
    project: ProjectInfo
    frame_rate: Optional[int] = None
    complete: CompleteCallbackT = _UnconfiguredCompleteAction()


@dataclass
class DetectionConfiguration:
    project: ProjectInfo
    frame_rate: Optional[int] = None
    complete: CompleteCallbackT = _UnconfiguredCompleteAction()


@dataclass
class IntersessionBlock:
    configuration: SegmentationConfiguration
    frame_count: int = 0


@dataclass
class IntersessionDetection:
    configuration: DetectionConfiguration


# Protocol first (less strict)
from .inference_protocol import InferenceProtocol
from .pellet_device_protocol import PelletDeviceProtocol
from .pellet_trial import (
    AttemptAssignmentPolicy,
    HardwareErrorKind,
    PelletTrialAttempt,
    PelletTrialLedger,
    RetrySettingsPolicy,
    TrialAccountingConfiguration,
    TrialCountBasis,
    TrialOutcome,
)

from .intersession import IntersessionState
from .intersession.intersession_machine import IntersessionMachine
from .behavior_algorithm import BehaviorAlgorithm

from .system_machine import SystemMachine
from .system_machine_state import SystemState
