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

    def __post_init__(self):
        if not self.logical_name.strip():
            raise ValueError("NI-DAQ logical device name must be provided")
        for name in ("logical_name", "runtime_name", "product_type"):
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
    require_hardware_synchronization: bool = True
    reference_clock_source: Optional[str] = None
    start_trigger_source: Optional[str] = None
    sample_clock_source: Optional[str] = None
    task_strategy: str = "per_device"
    require_distinct_start_trigger: bool = False
    sample_clock_export_terminal: Optional[str] = None
    start_trigger_export_terminal: Optional[str] = None
    external_routes: Tuple["NidaqTimingRoute", ...] = tuple()
    transfer_mechanism_overrides: Tuple[Tuple[str, str], ...] = tuple()

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
    start_trigger_source: Optional[str] = None
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
