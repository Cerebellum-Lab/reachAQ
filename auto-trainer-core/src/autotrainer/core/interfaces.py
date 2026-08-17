import enum
from typing import Callable, List, Protocol, Optional, TypeVar
from uuid import UUID

from autotrainer.core import Offset3DTuple, ObservableObjectProtocol, ProjectInfo
from autotrainer.core.observable_object import EventHandler
from autotrainer.core.reach_event import ReachEvent


class CoverServoStatus(int, enum.Enum):
    OK = 0
    COVER_POSITION_ERROR = 1
    RELEASE_POSITION_ERROR = 2

    COVER_AND_RELEASE_POS_ERROR = COVER_POSITION_ERROR | RELEASE_POSITION_ERROR

    @property
    def is_error(self):
        return self is not CoverServoStatus.OK


class RecordingEndingReason(str, enum.Enum):

    NA = "NA"
    MANUAL_STOP = "ManualStop"
    MANUAL_ABORT = "ManualAbort"
    DURATION_LIMIT = "DurationLimit"
    TRIAL_LIMIT = "TrialLimit"
    PROTOCOL_COMPLETE = "ProtocolComplete"
    STOP_DRAIN_TIMEOUT = "StopDrainTimeout"
    STORAGE_LIMIT = "StorageLimit"
    REQUIRED_SOURCE_FAILURE = "RequiredSourceFailure"
    ALGO_PAUSED = "AlgoPaused"


class CaptureAnalysisResult(str, enum.Enum):

    CAPTURE_ONLY = "capture_only"
    ANALYSIS_SUCCEEDED = "analysis_succeeded"
    ANALYSIS_FAILED = "analysis_failed"
    ANALYSIS_DELAYED = "analysis_delayed"


class PelletHardwareProtocol(Protocol):
    """
    Defines an expected/required set of commands from the pellet device that are used as part of the behavior algorithm
    and state machine.
    """

    def delay(self, amount: float):
        """Request to delay that amount of seconds"""

    @property
    def last_set_position(self) -> Optional[Offset3DTuple]:
        """Give the last SET position (deliver position, used with SEND_PELLET command"""

    @property
    def last_position(self) -> Optional[Offset3DTuple]:
        """Given the last actual position"""

    @property
    def last_dcs_set_position(self) -> Optional[Offset3DTuple]:
        """Give the last SET position, in DCS, (deliver position), used with SEND_PELLET command"""

    @property
    def last_dcs_position(self) -> Optional[Offset3DTuple]:
        """Given the last actual, in DCS, triangle position"""

    def set_x(self, value: float, *, absolute: bool = True, sender: str = "NA") -> Optional[UUID]:
        """
        Change the X stepper location and set it as the X-axis pellet release location.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def set_y(self, value: float, *, absolute: bool = True, sender: str = "NA") -> Optional[UUID]:
        """
        Change the Y stepper location and set it as the Y-axis pellet release location.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def set_z(self, value: float, *, absolute: bool = True, sender: str = "NA") -> Optional[UUID]:
        """
        Change the Z stepper location and set it as the Z-axis pellet release location.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def set_dcs_x(self, value: float, *, absolute: bool = True, sender: str = "NA") -> Optional[UUID]:
        """
        Change the DCS-X stepper location and set it as the X-axis pellet release location.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def set_dcs_y(self, value: float, *, absolute: bool = True, sender: str = "NA") -> Optional[UUID]:
        """
        Change the DCS-Y stepper location and set it as the Y-axis pellet release location.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def set_dcs_z(self, value: float, *, absolute: bool = True, sender: str = "NA") -> Optional[UUID]:
        """
        Change the DCS-Z stepper location and set it as the Z-axis pellet release location.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def move_x(self, value: float, *, absolute: bool = True) -> Optional[UUID]:
        """
        Move the X stepper.  This may not be supported on all device platforms.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def move_y(self, value: float, *, absolute: bool = True) -> Optional[UUID]:
        """
        Move the Y stepper.  This may not be supported on all device platforms.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def move_z(self, value: float, *, absolute: bool = True) -> Optional[UUID]:
        """
        Move the Z stepper.  This may not be supported on all device platforms.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def send_home(self) -> Optional[UUID]:
        """
        Request a move to 0, 0, 0.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def send_retract(self) -> Optional[UUID]:
        """Request a move to y - 10 (relative)
        :return: A token to expect from the device message handler when the request is complete.
        """

    def load_pellet(self) -> Optional[UUID]:
        """
        Request a full load cycle to scoop the pellet from the bin.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def send_pellet(self, *, embedded_tone=None) -> Optional[UUID]:
        """
        Request a move from the current position to the "send" location stored in the hardware.

        ``embedded_tone`` optionally requests a frequency/duration pair inside
        the same board-owned compound sequence.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def cover_pellet(self) -> Optional[UUID]:
        """
        Request the barrier arm close and cover the pellet.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def release_pellet(self) -> Optional[UUID]:
        """
        Request the barrier arm open and expose the pellet.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def play_tone(self, frequency: int, duration: float) -> Optional[UUID]:
        """
        Request the device play a tone.

        :return: A token to expect from the device message handler when the request is complete.
        """

    def set_motors_drift(self, drift: Offset3DTuple):
        """Set the motor drift offset"""

    def set_auto_correct_motor_drift(self, enabled: bool):
        """Set autocorrect motor drift"""


class TunnelHardwareProtocol(Protocol):
    """Empty compatibility type for the installed training package.

    ReachAQ no longer implements or supplies tunnel/head-fix hardware. The
    external training package still imports this type while accepting ``None``
    for the corresponding attachment.
    """

#

class BehaviorAlgoEvents:
    """Define the behavior algo events and their signature"""
    # NB: *assigned/defined* here,
    # but used as typehints in BehaviorAlgoProtocol below.

    session_starting_before_record_start = EventHandler[Callable[[], None]]
    session_starting = EventHandler[Callable[[], None]]
    session_capture_ending = EventHandler[Callable[[RecordingEndingReason], None]]

    session_processing_starting = EventHandler[Callable[[], None]]

    session_ending = EventHandler[Callable[[ProjectInfo, CaptureAnalysisResult], None]]

    cover_servo_status_changed = EventHandler[Callable[[CoverServoStatus], None]]

    # NB:
    # these events receive as single param/arg the **increment** applied to the previous value (whatever it was):
    pellets_presented_evt = EventHandler[Callable[[int], None]]
    pellets_consumed_evt = EventHandler[Callable[[int], None]]
    successful_reaches_evt = EventHandler[Callable[[int], None]]
    total_reaches_evt = EventHandler[Callable[[int], None]]


class BehaviorAlgorithmProtocol(ObservableObjectProtocol, Protocol):

    # 1) attributes, or properties:

    @property
    def pellet_delivery_enabled(self) -> bool: ...
    @pellet_delivery_enabled.setter
    def pellet_delivery_enabled(self, value: bool): ...

    @property
    def pellet_cover_enabled(self) -> bool: ...
    @pellet_cover_enabled.setter
    def pellet_cover_enabled(self, value: bool): ...

    @property
    def intersession_pellet_shift_enabled(self) -> bool: ...
    @intersession_pellet_shift_enabled.setter
    def intersession_pellet_shift_enabled(self, value: bool): ...

    @property
    def pellet_hands_min_distance(self) -> float: ...
    @pellet_hands_min_distance.setter
    def pellet_hands_min_distance(self, value: float): ...

    @property
    def trial_reaches(self) -> List[ReachEvent]:
        """The list of reaches of the previously analyzed trial"""

    # 2) commands :

    def reset_configuration(self) -> None:
        """Reset the configuration to what it was when it was first loaded"""

    # 3) events:

    session_starting_before_record_start: BehaviorAlgoEvents.session_starting_before_record_start

    session_starting: BehaviorAlgoEvents.session_starting
    """Emitted when a new recording session starts"""

    session_capture_ending: BehaviorAlgoEvents.session_capture_ending
    """Emitted when recording for the current session ends"""

    session_ending: BehaviorAlgoEvents.session_ending
    """Emitted when a session and its optional post-session analysis have ended"""

    pellets_presented_evt: BehaviorAlgoEvents.pellets_presented_evt
    """When a pellet is "presented" ; i.e: when it's arrived at deliver/send position"""

    pellets_consumed_evt: BehaviorAlgoEvents.pellets_consumed_evt
    """When a pellet is consumed"""

    successful_reaches_evt: BehaviorAlgoEvents.successful_reaches_evt
    """Successful reaches"""

    total_reaches_evt: BehaviorAlgoEvents.total_reaches_evt
    """Total reaches"""
