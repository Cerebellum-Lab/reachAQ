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
import os
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

# Device names measured faster in FP16 than FP32 at the live workload - one
# frame per camera, batch 2 at 256x256 - matched case-insensitively.
#
# Deliberately empty. Tensor cores are necessary for FP16 to win but they are
# not sufficient, and this set records measurement rather than architecture:
#
#   T1000 (no tensor cores, Linux)   FP16 is 3-4x SLOWER. Measured twice, by a
#                                    cuBLAS GEMM and by a convolution stack.
#   RTX 5060 Ti (tensor cores, but   FP16 is 0.78-0.85x, i.e. slower, for
#   measured on Windows)             cspnext_s and rtmpose_s. That machine is
#                                    launch-bound at this batch, though: batch
#                                    1 and batch 8 cost the same wall clock
#                                    (9.43 vs 9.16 ms), so the GPU is idle
#                                    waiting on WDDM kernel launches and no
#                                    precision change can show through. FP16
#                                    does win at batch 32 (1.23x), once compute
#                                    dominates. The result says nothing about
#                                    the same card under Linux, which has no
#                                    WDDM launch penalty.
#
# So no card in the fleet is yet measured faster in FP16 at the live batch, and
# the honest default is FP32 everywhere. Add an entry only with a measurement
# from that card on its production operating system, at the live batch and
# input size. PRECISION_ENVIRONMENT_VARIABLE is how you take that measurement.
MEASURED_FASTER_IN_FP16: typing.FrozenSet[str] = frozenset()

SUPPORTED_PRECISIONS = frozenset({"fp32", "fp16"})

# Per-rig override, for benchmarking a card before it earns an entry above.
PRECISION_ENVIRONMENT_VARIABLE = "AUTOTRAINER_POSE_PRECISION"


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
        """The precision measured fastest on this card, defaulting to "fp32".

        Having tensor cores is not the question - whether FP16 is faster on
        this card, at this batch and input size, under this operating system
        is. See MEASURED_FASTER_IN_FP16 for why those come apart.
        """
        if not self.has_tensor_cores:
            return "fp32"
        lowered = self.name.lower()
        if any(entry in lowered for entry in MEASURED_FASTER_IN_FP16):
            return "fp16"
        return "fp32"


def resolve_precision(
    capability: typing.Optional["GpuCapability"] = None,
    environ: typing.Optional[typing.Mapping[str, str]] = None,
) -> str:
    """The precision to run at: the rig's override, else the card's default.

    The override exists so a precision can be benchmarked on a rig before it
    is made that card's default, which is the only way an entry in
    MEASURED_FASTER_IN_FP16 can ever be justified. An unrecognised value is
    ignored with a warning rather than raising: a typo in a rig's environment
    must not stop inference from starting.
    """
    environ = os.environ if environ is None else environ
    requested = (environ.get(PRECISION_ENVIRONMENT_VARIABLE) or "").strip().lower()
    if requested:
        if requested in SUPPORTED_PRECISIONS:
            logger.notice("pose precision overridden to %s by %s",
                          requested, PRECISION_ENVIRONMENT_VARIABLE)
            return requested
        logger.warning("%s=%r is not one of %s; ignoring",
                       PRECISION_ENVIRONMENT_VARIABLE, requested,
                       sorted(SUPPORTED_PRECISIONS))
    return capability.preferred_precision if capability is not None else "fp32"


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
