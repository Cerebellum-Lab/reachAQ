"""GPU capability detection, so precision is chosen per card rather than assumed.

reachAQ rigs do not share a GPU. The known fleet spans compute capability 7.5
(T1000), 8.6 (RTX A2000) and 12.0 (RTX 5060 Ti), and the fastest precision
differs between them, so nothing here may hardcode a card or a capability.

The specific trap: measured on an NVIDIA T1000, FP16 runs 3-4x *slower* than
FP32, confirmed by two independent methods (a cuBLAS GEMM and a convolution
stack). That card uses the TU117 die, which omits tensor cores even though it
reports capability 7.5 like the tensor-core-equipped Turing parts. Capability
alone is therefore not sufficient, so the small set of tensor-core-less dies that
report a tensor-core capability is named explicitly.
"""

from __future__ import annotations

import dataclasses
import typing

from autotrainer.core.logging import get_verbose_logger


logger = get_verbose_logger(__name__)

TENSOR_CORE_EXCLUDED_DEVICES = frozenset({
    # TU117 and TU116: Turing dies without tensor cores, which still report 7.5.
    # Entries are matched case-insensitively against the device name, so they
    # must be lowercase.
    "t1000", "t600", "t400", "t500",
    "gtx 1650", "gtx 1660",
})

_TENSOR_CORE_MIN_MAJOR = 7  # Volta and later


@dataclasses.dataclass(frozen=True)
class GpuCapability:
    """What a visible CUDA device is, and what precision suits it."""

    name: str
    major: int
    minor: int
    total_memory_bytes: int

    @property
    def compute_capability(self) -> typing.Tuple[int, int]:
        return self.major, self.minor

    @property
    def has_tensor_cores(self) -> bool:
        if self.major < _TENSOR_CORE_MIN_MAJOR:
            return False
        lowered = self.name.lower()
        return not any(excluded in lowered for excluded in TENSOR_CORE_EXCLUDED_DEVICES)

    @property
    def preferred_precision(self) -> str:
        """"fp16" only where it is actually faster; "fp32" otherwise."""
        return "fp16" if self.has_tensor_cores else "fp32"


def detect_gpu_capability(index: int = 0) -> typing.Optional[GpuCapability]:
    """Describe a visible CUDA device, or None when torch cannot see one.

    Never raises: callers use this to pick a precision, and a detection failure
    should fall back to a safe default rather than stop inference from starting.
    """
    try:
        import torch
    except Exception as exc:
        logger.info("torch unavailable for capability detection: %s", exc)
        return None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
            return None
        major, minor = torch.cuda.get_device_capability(index)
        properties = torch.cuda.get_device_properties(index)
        capability = GpuCapability(
            name=torch.cuda.get_device_name(index),
            major=int(major),
            minor=int(minor),
            total_memory_bytes=int(properties.total_memory),
        )
        logger.info(
            "GPU %d: %s sm_%d%d %.1f GiB tensor_cores=%s precision=%s",
            index, capability.name, capability.major, capability.minor,
            capability.total_memory_bytes / 1024 ** 3,
            capability.has_tensor_cores, capability.preferred_precision,
        )
        return capability
    except Exception as exc:
        logger.warning("GPU capability detection failed: %s", exc)
        return None
