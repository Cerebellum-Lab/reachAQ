from __future__ import annotations

import dataclasses
from typing import ClassVar, Optional, Tuple

from autotrainer.core import make_camelize_representer, make_decamelize_constructor
from autotrainer.core.configuration import SystemConfigurationDumper, SystemConfigurationLoader


@dataclasses.dataclass(frozen=True)
class NidaqDeviceIdentity:
    """Portable identity for one configured NI-DAQ timing role."""

    logical_name: str
    runtime_name: Optional[str] = None
    product_type: Optional[str] = None
    serial_number: Optional[int] = None
    #: The breakout block cabled to this device, by name - BNC-2090A,
    #: BNC-2110. Optional, and a device without one behaves exactly as it did
    #: before there was a name for this: card terms are canonical either way,
    #: and this only buys the label printed on the thing being touched and a
    #: warning when a connector is a dead end on this particular card.
    breakout: Optional[str] = None

    def __post_init__(self):
        if not self.logical_name.strip():
            raise ValueError("NI-DAQ logical device name must be provided")
        for name in ("logical_name", "runtime_name", "product_type",
                     "breakout"):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, value.strip() or None)


@dataclasses.dataclass(frozen=True)
class NidaqTimingConfiguration:
    """Requested timing policy; resolved physical routes are runtime metadata."""

    VALID_SYNC_MODES: ClassVar[tuple[str, ...]] = (
        "auto",
        "backplane",
        "external",
        "independent",
    )

    sync_mode: str = "auto"
    timing_master: Optional[NidaqDeviceIdentity] = None
    #: Whether the boards must share a hardware clock. Unset means derive
    #: it from sync_mode, which is the whole of C5: choosing "independent"
    #: is already saying they do not, and having to say it twice is how the
    #: two came to contradict each other.
    require_hardware_synchronization: Optional[bool] = None
    reference_clock_source: Optional[str] = None
    start_trigger_source: Optional[str] = None
    sample_clock_source: Optional[str] = None
    task_strategy: str = "per_device"
    require_distinct_start_trigger: bool = False
    sample_clock_export_terminal: Optional[str] = None
    start_trigger_export_terminal: Optional[str] = None
    external_routes: Tuple["NidaqTimingRoute", ...] = tuple()
    transfer_mechanism_overrides: Tuple[Tuple[str, str], ...] = tuple()

    def _refuse_meaningless_combinations(self, sync_mode, strategy) -> None:
        """C5. Stop the configuration expressing decisions that contradict.

        Four settings describe one decision and most of their combinations
        say nothing: "run the boards independently, and require them to be
        hardware synchronized" is not a configuration, it is a contradiction
        that produced an invalid plan several layers later with a reason
        nobody traced back to here.

        The fields are kept rather than folded away, because they are written
        into every system configuration on disk and the loader does not
        filter unknown keys - removing one would stop an existing rig
        loading. What is removed is the ability to set them to a combination
        that means nothing.
        """
        if sync_mode == "independent" and self.require_hardware_synchronization:
            raise ValueError(
                "NI-DAQ timing asks for independent boards and for hardware "
                "synchronization at the same time. Independent means each "
                "board runs on its own clock. Leave "
                "requireHardwareSynchronization unset to accept that, or "
                "choose syncMode auto, backplane or external to get "
                "synchronization."
            )
        if sync_mode == "independent" and strategy != "per_device":
            raise ValueError(
                f"NI-DAQ timing asks for independent boards and for "
                f"{strategy} tasks. A task spanning devices is one task on "
                "one clock, which is the opposite of independent; use "
                "per_device."
            )
        if self.require_distinct_start_trigger:
            # Read by nothing, from the day it was added. A knob that does
            # nothing is worse than one that refuses, because somebody sets
            # it and believes they have changed something.
            raise ValueError(
                "NI-DAQ requireDistinctStartTrigger is not implemented - "
                "nothing has ever read it. Leave it false; the start trigger "
                "each device uses comes from the resolved topology."
            )

    def __post_init__(self):
        sync_mode = self.sync_mode.strip().lower()
        if sync_mode not in self.VALID_SYNC_MODES:
            raise ValueError(
                "NI-DAQ sync_mode must be one of: "
                + ", ".join(self.VALID_SYNC_MODES)
            )
        object.__setattr__(self, "sync_mode", sync_mode)
        strategy = self.task_strategy.strip().lower()
        if strategy not in {"per_device", "auto_multidevice", "forced_multidevice"}:
            raise ValueError(
                "NI-DAQ task_strategy must be per_device, auto_multidevice, or "
                "forced_multidevice"
            )
        object.__setattr__(self, "task_strategy", strategy)
        self._refuse_meaningless_combinations(sync_mode, strategy)
        # Resolved after the refusal, so the contradiction is reported
        # against what was written rather than against what was derived.
        if self.require_hardware_synchronization is None:
            object.__setattr__(self, "require_hardware_synchronization",
                               sync_mode != "independent")
        for name in (
            "reference_clock_source",
            "start_trigger_source",
            "sample_clock_source",
            "sample_clock_export_terminal",
            "start_trigger_export_terminal",
        ):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, value.strip() or None)
        object.__setattr__(self, "external_routes", tuple(self.external_routes))
        normalized_overrides = tuple(
            (str(resource).strip(), str(mechanism).strip().lower())
            for resource, mechanism in self.transfer_mechanism_overrides
        )
        if any(not resource or not mechanism for resource, mechanism in normalized_overrides):
            raise ValueError("NI transfer overrides require resource and mechanism")
        object.__setattr__(self, "transfer_mechanism_overrides", normalized_overrides)


@dataclasses.dataclass(frozen=True)
class NidaqTimingRoute:
    signal: str
    source: str
    destinations: Tuple[str, ...] = tuple()

    def __post_init__(self):
        object.__setattr__(self, "signal", str(self.signal).strip().lower())
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(
            self,
            "destinations",
            tuple(str(value).strip() for value in self.destinations if str(value).strip()),
        )
        if not self.signal or not self.source or not self.destinations:
            raise ValueError("NI timing routes require signal, source, and destinations")


@dataclasses.dataclass(frozen=True)
class NidaqTaskSpecification:
    task_id: str
    device: str
    subsystem: str
    channels: Tuple[str, ...]
    mode: str
    sample_clock_source: Optional[str] = None
    start_trigger_source: Optional[str] = None
    reference_clock_source: Optional[str] = None
    required: bool = True
    transfer_mechanism: Optional[str] = None
    resources: Tuple[str, ...] = tuple()

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "resources", tuple(self.resources))


@dataclasses.dataclass(frozen=True)
class NidaqTaskGraph:
    graph_id: str
    strategy: str
    tasks: Tuple[NidaqTaskSpecification, ...]
    routes: Tuple[NidaqTimingRoute, ...] = tuple()
    create_order: Tuple[str, ...] = tuple()
    start_order: Tuple[str, ...] = tuple()
    shutdown_order: Tuple[str, ...] = tuple()

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "create_order", tuple(self.create_order))
        object.__setattr__(self, "start_order", tuple(self.start_order))
        object.__setattr__(self, "shutdown_order", tuple(self.shutdown_order))


@dataclasses.dataclass(frozen=True)
class NidaqTimingPlan:
    """Validated immutable timing plan passed into the NI-DAQ worker."""

    requested_mode: str
    resolved_mode: str
    is_valid: bool
    master_device: Optional[str] = None
    slave_devices: Tuple[str, ...] = tuple()
    reference_clock_source: Optional[str] = None
    reference_clock_rate_hz: Optional[float] = None
    sample_clock_source: Optional[str] = None
    #: Rate of sample_clock_source. A consumer clocked by that terminal ticks
    #: at this rate whatever rate it asked for, so a waveform built for a
    #: different one comes out stretched: the laser built at 100 kHz and ran
    #: on the stream's 10 kHz, and a two-second pulse train took twenty.
    sample_clock_rate_hz: Optional[float] = None
    start_trigger_source: Optional[str] = None
    sample_clock_export_terminal: Optional[str] = None
    start_trigger_export_terminal: Optional[str] = None
    routes: Tuple[NidaqTimingRoute, ...] = tuple()
    task_start_order: Tuple[str, ...] = tuple()
    resolved_devices: Tuple[NidaqDeviceIdentity, ...] = tuple()
    synchronization_quality: str = "unresolved"
    clock_producer: str = "unresolved"
    clock_producer_device: Optional[str] = None
    consumer_devices: Tuple[str, ...] = tuple()
    hardware_output_devices: Tuple[str, ...] = tuple()
    hardware_output_timing_status: str = "not_configured"
    hardware_output_timing_reason: str = ""
    reason: str = ""
    task_graph: Optional[NidaqTaskGraph] = None
    multidevice_probe_status: str = "not_requested"

    def __post_init__(self):
        object.__setattr__(self, "slave_devices", tuple(self.slave_devices))
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "task_start_order", tuple(self.task_start_order))
        object.__setattr__(self, "consumer_devices", tuple(self.consumer_devices))
        object.__setattr__(
            self, "hardware_output_devices", tuple(self.hardware_output_devices),
        )
        object.__setattr__(
            self,
            "resolved_devices",
            tuple(self.resolved_devices),
        )


@dataclasses.dataclass(frozen=True)
class NidaqPortConfiguration:
    """Named NI-DAQ channels used by the reachAQ operator workflow."""

    device_name: Optional[str] = None
    tone1: Optional[str] = None
    tone2: Optional[str] = None
    tone3_r: Optional[str] = None
    tone3_l: Optional[str] = None
    cam_frames: Optional[str] = None
    barcode: Optional[str] = None
    device_identities: Tuple[NidaqDeviceIdentity, ...] = tuple()
    timing: NidaqTimingConfiguration = dataclasses.field(
        default_factory=NidaqTimingConfiguration,
    )

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, str):
                value = value.strip() or None
                object.__setattr__(self, field.name, value)
        object.__setattr__(
            self,
            "device_identities",
            tuple(self.device_identities),
        )


for _tag, _cls in (
    ("NidaqDeviceIdentity", NidaqDeviceIdentity),
    ("NidaqTimingConfiguration", NidaqTimingConfiguration),
    ("NidaqTimingRoute", NidaqTimingRoute),
    ("NidaqPortConfiguration", NidaqPortConfiguration),
):
    SystemConfigurationDumper.add_representer(
        _cls,
        make_camelize_representer(f"!{_tag}"),
    )
    SystemConfigurationLoader.add_constructor(
        f"!{_tag}",
        make_decamelize_constructor(_cls),
    )
