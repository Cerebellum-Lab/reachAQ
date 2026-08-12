import dataclasses
from dataclasses import dataclass, field
from typing import ClassVar, Type, Optional

import yaml

from .. import make_camelize_representer, make_decamelize_constructor, Offset3DTuple


@dataclasses.dataclass
class ShiftXYZTarget:

    # should be in Diamond coordinate system
    x: float = 1.5
    y: float = -3
    z: float = 1


def default_shift_xyz_target() -> Offset3DTuple:
    return Offset3DTuple(**dataclasses.asdict(ShiftXYZTarget()))


def default_tongue_eaten_shift() -> Offset3DTuple:
    return Offset3DTuple(0, 0.5, 0)


@dataclasses.dataclass
class _ShiftXYZBufferHandlerConfig:

    minimum_reach_fail: int = 10  # minimum nbr of failed reach, to make the mean/an entire processing of them
    target: Offset3DTuple = field(default_factory=default_shift_xyz_target)


@dataclasses.dataclass
class ShiftXYZBufferHandlerConfig(_ShiftXYZBufferHandlerConfig):
    pass


@dataclasses.dataclass
class ShiftXYZHandlerConfig:

    use_tongue_eaten: bool = True
    tongue_eaten_shift: Offset3DTuple = field(default_factory=default_tongue_eaten_shift)

    use_reach_buffer: bool = True
    selected: str = "ShiftXYZBufferHandler"
    buffer: ShiftXYZBufferHandlerConfig = field(default_factory=ShiftXYZBufferHandlerConfig)


@dataclass
class HomeOnExcessiveDriftDistanceConfiguration:
    """Execute, when in monitoring (should equal to be in deliver position), home to reset motors if measured drift distance is higher than threshold"""

    enabled: bool = False
    excessive_distance_threshold: float = 5  # mm

    min_samples: int = 30
    # only considerate if/when nbr of samples is greater than this.
    # This can compensate small unsync between inference results and motor status positions,
    # when the start of the sampling would be done right after the pellet-arm finished moving.
    # The current inference is giving us ~15 datapoints per second,
    # and almost same for the motor status position: ~10 / sec.
    # So this requires/takes ~2 seconds of duration to get what's necessary.


@dataclass
class PelletUncoverConfiguration:
    min_y_dcs: float = 0  # mm,  minimum Y dcs for all hand parts to be "valid" for uncover
    trigger_delay: float = 1  # seconds, duration before real active/trigger to uncover when it's "valid"


@dataclass
class PelletDeliveryConfiguration:
    """
    Behavior model options related to pellet delivery.
    """

    is_enabled: bool = False
    """When disabled no automatic behavior movement will be peformed, but eventually on application start"""

    pellet_send_wait_delay: float = 1
    """Seconds after recording starts before an automatic pellet send."""

    is_pellet_cover_enabled: bool = False
    """If enabled: cover pellet when session starts, and wait uncover condition"""

    retract_enabled: bool = True
    """If enabled: pellet is retracted by default. And wait send condition before pellet-send"""

    # not really related to pellet delivery but has been here since start:
    is_intersession_analysis_enabled: bool = False
    is_intersession_pellet_shift_enabled: bool = True

    max_pellet_missing_seconds: float = 1.0  # how long to wait before load pellet when pellet missing/not seen
    # this help ensure we don't execute a load pellet if we get an incorrect pose_result with pellet seen == False,
    # which can happen eventually (missed inference detection basically).

    auto_correct_motors_drift: bool = False  # attempt "live" motor drift correction -- DISABLED in code

    use_triangle_pellet_distance_too_far: bool = False
    """If enabled then a triangle-pellet too far distance also trigger a load-pellet"""
    triangle_pellet_expected_distance: float = 5  # mm
    triangle_pellet_diff_too_far_threshold: float = 1  # mm

@dataclass
class SessionControlConfiguration:
    """Continuous recording and pellet-trial policy exposed to operators."""

    automatic_pellet_cycles_enabled: bool = False
    automatic_protocol_advance_enabled: bool = False
    attempt_assignment: str = "retry_within_trial"
    retry_settings: str = "reuse"
    trial_count_basis: str = "completed"
    counted_trial_outcomes: tuple = (
        "success",
        "failure",
        "pellet_missing",
        "no_reach",
    )
    duration_limit_seconds: Optional[float] = None
    trial_limit: Optional[int] = None
    stop_on_protocol_complete: bool = False
    stop_drain_timeout_seconds: float = 15.0

    MAX_DURATION_SECONDS: ClassVar[float] = 2 * 60 * 60

    ATTEMPT_ASSIGNMENTS = (
        "retry_within_trial",
        "every_attempt_is_trial",
        "successful_presentations_only",
    )
    RETRY_SETTINGS = ("reuse", "resample")
    TRIAL_COUNT_BASES = ("started", "presented", "completed", "scored")
    TRIAL_OUTCOMES = (
        "success",
        "failure",
        "pellet_missing",
        "no_reach",
        "incomplete",
        "aborted",
    )

    def __post_init__(self):
        self.counted_trial_outcomes = tuple(self.counted_trial_outcomes)
        if self.attempt_assignment not in self.ATTEMPT_ASSIGNMENTS:
            raise ValueError(
                "Unknown attempt assignment policy: "
                f"{self.attempt_assignment}"
            )
        if self.retry_settings not in self.RETRY_SETTINGS:
            raise ValueError(f"Unknown retry settings policy: {self.retry_settings}")
        if self.trial_count_basis not in self.TRIAL_COUNT_BASES:
            raise ValueError(f"Unknown trial count basis: {self.trial_count_basis}")
        unknown_outcomes = sorted(
            set(self.counted_trial_outcomes) - set(self.TRIAL_OUTCOMES)
        )
        if unknown_outcomes:
            raise ValueError(
                "Unknown counted trial outcome(s): " + ", ".join(unknown_outcomes)
            )
        if self.duration_limit_seconds is not None and self.duration_limit_seconds <= 0:
            raise ValueError("Recording duration limit must be positive")
        if (
            self.duration_limit_seconds is not None
            and self.duration_limit_seconds > self.MAX_DURATION_SECONDS
        ):
            raise ValueError("Recording duration limit cannot exceed 2 hours")
        if self.trial_limit is not None and self.trial_limit <= 0:
            raise ValueError("Trial target must be positive")
        if self.stop_drain_timeout_seconds <= 0:
            raise ValueError("Stop drain timeout must be positive")


@dataclass
class BehaviorConfiguration:
    pellet_delivery: PelletDeliveryConfiguration = field(default_factory=PelletDeliveryConfiguration)
    session_control: SessionControlConfiguration = field(default_factory=SessionControlConfiguration)
    pellet_uncover: PelletUncoverConfiguration = field(default_factory=PelletUncoverConfiguration)
    shift_xyz_handler: ShiftXYZHandlerConfig = field(default_factory=ShiftXYZHandlerConfig)
    home_on_excessive_drift_distance: HomeOnExcessiveDriftDistanceConfiguration = field(default_factory=HomeOnExcessiveDriftDistanceConfiguration)


_tag_2_cls = dict(
    PelletDeliveryConfiguration=PelletDeliveryConfiguration,
    SessionControlConfiguration=SessionControlConfiguration,
    PelletUncoverConfiguration=PelletUncoverConfiguration,
    BehaviorConfiguration=BehaviorConfiguration,
    HomeOnExcessiveDriftDistance=HomeOnExcessiveDriftDistanceConfiguration,  # missed Configuration suffix
    # ShiftXYZTarget="ShiftXYZTarget",  # replaced by Offset3dTuple.
    ShiftXYZHandlerConfiguration=ShiftXYZHandlerConfig,
    ShiftXYZBufferHandlerConfiguration=ShiftXYZBufferHandlerConfig,
)


def add_behavior_configuration_representers(dumper: Type[yaml.SafeDumper]):
    def add(klass, tagname):
        dumper.add_representer(klass, make_camelize_representer(f"!{tagname}"))

    for tag, cls in _tag_2_cls.items():
        add(cls, tag)

    from autotrainer.core.configuration import repr_offset3d_tuple

    dumper.add_representer(ShiftXYZTarget, repr_offset3d_tuple)
    # changed to use Offset3dTuple


def add_behavior_configuration_constructors(safe_loader: Type[yaml.SafeLoader]):

    def add(klass, tagname):
        safe_loader.add_constructor(f"!{tagname}", make_decamelize_constructor(klass))

    for tag, cls in _tag_2_cls.items():
        add(cls, tag)
