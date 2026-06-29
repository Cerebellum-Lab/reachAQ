from __future__ import annotations

import dataclasses
import enum
from typing import Any, Mapping, Optional, Protocol, Union


class CanTransportKind(str, enum.Enum):
    """Supported CAN transport families.

    `pyjerrycan` is the existing Jetson/custom-board path. The other transport
    kinds define the new Ubuntu desktop boundary without changing current
    runtime behavior yet.
    """

    PYJERRYCAN = "pyjerrycan"
    SOCKETCAN = "socketcan"
    PCAN_BASIC = "pcan_basic"
    EMULATION = "emulation"


def normalize_can_transport_kind(value: Union[CanTransportKind, str]) -> CanTransportKind:
    if isinstance(value, CanTransportKind):
        return value
    return CanTransportKind(str(value).lower())


@dataclasses.dataclass(frozen=True)
class CanTransportConfiguration:
    """Configuration for selecting and opening a CAN transport backend."""

    kind: CanTransportKind = CanTransportKind.PYJERRYCAN
    channel: str = "can0"
    bitrate: Optional[int] = None
    data_bitrate: Optional[int] = None
    fd: bool = False
    receive_timeout_seconds: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "kind", normalize_can_transport_kind(self.kind))
        if self.bitrate is not None and self.bitrate <= 0:
            raise ValueError("bitrate must be positive when provided")
        if self.data_bitrate is not None and self.data_bitrate <= 0:
            raise ValueError("data_bitrate must be positive when provided")
        if self.receive_timeout_seconds < 0:
            raise ValueError("receive_timeout_seconds must be non-negative")

    @classmethod
    def from_mapping(cls, content: Mapping[str, Any]) -> "CanTransportConfiguration":
        values = dict(content)
        if "type" in values and "kind" not in values:
            values["kind"] = values.pop("type")
        return cls(**values)

    @property
    def uses_linux_can_stack(self) -> bool:
        return self.kind in {CanTransportKind.SOCKETCAN, CanTransportKind.PCAN_BASIC}


class CanTransportProtocol(Protocol):
    """Minimal backend contract for CAN transport implementations."""

    @property
    def configuration(self) -> CanTransportConfiguration:
        """Transport configuration used by this backend."""

    @property
    def is_open(self) -> bool:
        """Whether the backend currently has an open CAN connection."""

    def open(self) -> bool:
        """Open the transport backend."""

    def close(self) -> None:
        """Close the transport backend."""

    def read(self, limit: int = 1) -> list[Any]:
        """Read up to `limit` backend-native CAN messages."""

    def write(self, message: Any) -> bool:
        """Write a backend-native CAN message."""
