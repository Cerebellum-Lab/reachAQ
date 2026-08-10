import dataclasses
import datetime
import enum
from dataclasses import dataclass, field
from typing import Type, Optional, Dict
from typing_extensions import Self

import yaml

from autotrainer.core.logging import get_verbose_logger
from .animal_presence_configuration import GlobalAnimalPresenceConfig
from .autoclamp_evasion_config import AutoClampEvasionDetectorConfig
from .external_doors_monitor_configuration import ExternalDoorsAlarmConfig
from .presence_detection_configuration import PresenceDetectionConfig
from .system_fault_config import SystemFaultConfig
from .system_maintenance_config import SystemMaintenanceConfig
from .. import build_kwargs_apply_mapping, make_camelize_representer, make_decamelize_constructor, Offset3DTuple

from .headbar_pressure_config import HeadbarPressureConfiguration
from .audio_thrash_config import AudioSpectrumThrashMonitorConfig
from .alarm_configuration import EmergencyAlarmConfiguration
from .tunnel_sweep_config import AutoTunnelSweepConfiguration
from .animal_thrash_config import AnimalThrashAlarmConfig
from .device_comm_alarm_config import DeviceCommAlarmConfig

logger = get_verbose_logger(__name__)


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

    def __init__(self, **kwargs):
        for c in "xyz":
            kwargs.pop(f"target_{c}", None)  # old config
        super().__init__(**kwargs)


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
    """In seconds, how long to wait before send_pellet when autoclamp disabled"""

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

    @classmethod
    def from_version_zero(cls, content: dict) -> Self:
        return cls(**build_kwargs_apply_mapping(content, (
            *(f.name for f in dataclasses.fields(cls)),
            ('is_enabled', 'is_deliver_pellet_enabled'),
            ('is_pellet_cover_enabled', 'is_cover_pellet_enabled'),
        ), skip_remaining=True))


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
    )
    duration_limit_seconds: Optional[float] = None
    trial_limit: Optional[int] = None
    stop_on_protocol_complete: bool = False
    stop_drain_timeout_seconds: float = 15.0

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
        if self.trial_limit is not None and self.trial_limit <= 0:
            raise ValueError("Trial target must be positive")
        if self.stop_drain_timeout_seconds <= 0:
            raise ValueError("Stop drain timeout must be positive")


class HeadClampReleaseMode(str, enum.Enum):
    ACTIVITY = "Activity"
    FIXED_DURATION = "Fixed duration"


@dataclass
class HeadClampConfiguration:
    """
    Behavior model options related to the head clamp magnet including standard intensity and auto-clamp actions.
    """

    enabled: bool = False
    wait_engaged_before_send_pellet: bool = True

    baseline_intensity: float = 0.0
    auto_clamp_intensity: float = 100.0
    auto_clamp_release_tone_freq: int = 7000
    auto_clamp_release_tone_delay: float = 0.1
    before_reengage_delay: float = 5  # how long to wait/delay before allow/execute a re-engage after a disengage.

    prerelease_intensity: float = 70  # absolute % value
    prerelease_duration: float = 0  # seconds, if 0 then this pre-release is disabled / does not occur.

    release_mode: str = HeadClampReleaseMode.ACTIVITY.value

    # HeadClampReleaseMode.ACTIVITY
    auto_clamp_no_activity_release_delay: float = 30  # seconds
    auto_clamp_release_load_count: int = 100_000

    # HeadClampReleaseMode.FIXED_DURATION
    fixed_duration_release_delay: float = 30  # seconds


    @classmethod
    def from_version_zero(cls, content: dict) -> Self:
        return cls(**build_kwargs_apply_mapping(
            content,
            tuple(f.name for f in dataclasses.fields(cls)),
            skip_remaining=True,
        ))


@dataclass
class CageCleaningConfig:

    clean_days_interval: int = 14


@dataclass
class LEDAlarmConfig:

    start_ignore_hour: datetime.time = datetime.time(10, 0)
    stop_ignore_hour: datetime.time = datetime.time(20, 0)


@dataclass
class _BehaviorConfiguration:
    pellet_delivery: PelletDeliveryConfiguration = field(default_factory=PelletDeliveryConfiguration)
    session_control: SessionControlConfiguration = field(default_factory=SessionControlConfiguration)
    pellet_uncover: PelletUncoverConfiguration = field(default_factory=PelletUncoverConfiguration)
    shift_xyz_handler: ShiftXYZHandlerConfig = field(default_factory=ShiftXYZHandlerConfig)
    head_clamp: HeadClampConfiguration = field(default_factory=HeadClampConfiguration)
    headbar_pressure: HeadbarPressureConfiguration = field(default_factory=HeadbarPressureConfiguration)
    audio: AudioSpectrumThrashMonitorConfig = field(default_factory=AudioSpectrumThrashMonitorConfig)
    emergency_alarm: EmergencyAlarmConfiguration = field(default_factory=EmergencyAlarmConfiguration)
    topcam_presence_detection: PresenceDetectionConfig = field(default_factory=PresenceDetectionConfig)
    auto_tunnel_sweep: AutoTunnelSweepConfiguration = field(default_factory=AutoTunnelSweepConfiguration)
    home_on_excessive_drift_distance: HomeOnExcessiveDriftDistanceConfiguration = field(default_factory=HomeOnExcessiveDriftDistanceConfiguration)
    cage_cleaning: CageCleaningConfig = field(default_factory=CageCleaningConfig)
    autoclamp_evasion_detector: AutoClampEvasionDetectorConfig = field(default_factory=AutoClampEvasionDetectorConfig)
    led_alarm: LEDAlarmConfig = field(default_factory=LEDAlarmConfig)

    @classmethod
    def from_version_zero(cls, content: Dict) -> Self:
        configuration = cls()

        if "head_fix" in content:
            if "headbar_pressure" in content["head_fix"]:
                configuration.headbar_pressure = HeadbarPressureConfiguration.from_version_zero(
                    content["head_fix"]["headbar_pressure"]
                )

        if "behavior" in content:
            configuration.head_clamp = HeadClampConfiguration.from_version_zero(content["behavior"])
            configuration.pellet_delivery = PelletDeliveryConfiguration.from_version_zero(content["behavior"])

        return configuration

    @classmethod
    def from_version_one(cls, content: Dict):
        headclamp = content.get("head_clamp", {})
        pellet_delivery = dict(content.get("pellet_delivery", {}))
        pellet_delivery.pop("max_pellets_per_session", None)
        pellet_delivery.pop("max_pellets_per_day", None)
        headclamp.pop('max_baseline_intensity')
        headclamp.pop('baseline_intensity_increment')
        baseline = headclamp.pop('min_baseline_intensity')
        if baseline is not None:
            headclamp['baseline_intensity'] = baseline
        return cls(
            headbar_pressure=HeadbarPressureConfiguration(**content.get("headbar_pressure", {})),
            head_clamp=HeadClampConfiguration(**headclamp),
            pellet_delivery=PelletDeliveryConfiguration(**pellet_delivery),
        )


@dataclasses.dataclass
class BehaviorConfiguration(_BehaviorConfiguration):
    # NB: having to subclass _BehaviorConfiguration dataclass type to allow to customize init signature (and body):

    def __init__(self,
                 *,
                 mouse_presence=None,  # temporarily to be back-compatible with previous
                 auto_end_session=None,
                 batch_session_recording=None,
                 auto_close_gate_on_intersession=None,
                 load_cell=None,
                 auto_tare=None,
                 **kwargs):
        if mouse_presence is not None:
            logger.notice("Dropping previous mouse_presence config, new default one will be used. dropped entry: %s",
                          mouse_presence)
        if (
            auto_end_session is not None
            or batch_session_recording is not None
            or auto_close_gate_on_intersession is not None
        ):
            logger.notice("Dropping obsolete automatic recording trigger configuration")
        if load_cell is not None or auto_tare is not None:
            logger.notice("Dropping obsolete weight-sensor configuration")
        super().__init__(**kwargs)


_tag_2_cls = dict(
    PelletDeliveryConfiguration=PelletDeliveryConfiguration,
    SessionControlConfiguration=SessionControlConfiguration,
    PelletUncoverConfiguration=PelletUncoverConfiguration,
    HeadClampConfiguration=HeadClampConfiguration,
    HeadbarPressureConfiguration=HeadbarPressureConfiguration,
    BehaviorConfiguration=BehaviorConfiguration,
    AudioMonitorConfiguration=AudioSpectrumThrashMonitorConfig,
    AnimalPresenceConfiguration=GlobalAnimalPresenceConfig,
    EmergencyAlarmConfiguration=EmergencyAlarmConfiguration,
    PresenceDetectionConfiguration=PresenceDetectionConfig,
    ExternalDoorsMonitorConfiguration=ExternalDoorsAlarmConfig,
    AutoTunnelSweepConfiguration=AutoTunnelSweepConfiguration,
    HomeOnExcessiveDriftDistance=HomeOnExcessiveDriftDistanceConfiguration,  # missed Configuration suffix
    # ShiftXYZTarget="ShiftXYZTarget",  # replaced by Offset3dTuple.
    ShiftXYZHandlerConfiguration=ShiftXYZHandlerConfig,
    ShiftXYZBufferHandlerConfiguration=ShiftXYZBufferHandlerConfig,
    SystemMaintenanceConfig=SystemMaintenanceConfig,
    SystemFaultConfig=SystemFaultConfig,
    CageCleaningConfig=CageCleaningConfig,
    AnimalThrashAlarmConfig=AnimalThrashAlarmConfig,
    DeviceCommAlarmConfig=DeviceCommAlarmConfig,
    LEDAlarmConfig=LEDAlarmConfig,
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

    def obsolete_recording_trigger(loader, node):
        return loader.construct_mapping(node, deep=True)

    safe_loader.add_constructor(
        "!AutoEndSessionConfiguration",
        obsolete_recording_trigger,
    )
    safe_loader.add_constructor(
        "!BatchSessionRecordingConfiguration",
        obsolete_recording_trigger,
    )
    safe_loader.add_constructor(
        "!AutoCloseGateOnIntersessionConfiguration",
        obsolete_recording_trigger,
    )
    safe_loader.add_constructor(
        "!LoadCellConfiguration",
        obsolete_recording_trigger,
    )
    safe_loader.add_constructor(
        "!LoadCellAutoTareConfiguration",
        obsolete_recording_trigger,
    )
    add(GlobalAnimalPresenceConfig, "MousePresenceConfiguration")
    # keeping temporarily MousePresenceConfiguration, was renamed to AnimalPresenceConfiguration. Back-compatibility.
    # todo: remove some when later.
    #
