from __future__ import annotations

import dataclasses
from typing import Optional

from autotrainer.core import make_camelize_representer, make_decamelize_constructor
from autotrainer.core.configuration import SystemConfigurationDumper, SystemConfigurationLoader


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

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, str):
                value = value.strip() or None
                object.__setattr__(self, field.name, value)


SystemConfigurationDumper.add_representer(
    NidaqPortConfiguration,
    make_camelize_representer("!NidaqPortConfiguration"),
)
SystemConfigurationLoader.add_constructor(
    "!NidaqPortConfiguration",
    make_decamelize_constructor(NidaqPortConfiguration),
)
