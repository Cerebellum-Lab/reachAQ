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

    def __post_init__(self):
        sync_mode = self.sync_mode.strip().lower()
        if sync_mode not in self.VALID_SYNC_MODES:
            raise ValueError(
                "NI-DAQ sync_mode must be one of: "
                + ", ".join(self.VALID_SYNC_MODES)
            )
        object.__setattr__(self, "sync_mode", sync_mode)
        for name in (
            "reference_clock_source",
            "start_trigger_source",
            "sample_clock_source",
        ):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, value.strip() or None)


@dataclasses.dataclass(frozen=True)
class NidaqTimingRoute:
    signal: str
    source: str
    destinations: Tuple[str, ...] = tuple()


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
    synchronization_quality: str = "unresolved"
    reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, "slave_devices", tuple(self.slave_devices))
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "task_start_order", tuple(self.task_start_order))


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
    timing: NidaqTimingConfiguration = dataclasses.field(
        default_factory=NidaqTimingConfiguration,
    )

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, str):
                value = value.strip() or None
                object.__setattr__(self, field.name, value)


for _tag, _cls in (
    ("NidaqDeviceIdentity", NidaqDeviceIdentity),
    ("NidaqTimingConfiguration", NidaqTimingConfiguration),
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
