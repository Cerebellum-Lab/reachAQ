"""Precision must be chosen per GPU from measurement, never assumed.

Two separate traps, and the second was found by taking the first too far.

The T1000 uses the TU117 die, which omits tensor cores while still reporting
compute capability 7.5, and FP16 there is 3-4x SLOWER than FP32 - measured
twice, by a cuBLAS GEMM and by a convolution stack. So capability alone cannot
answer the question and the device name has to be consulted.

Having tensor cores then looked sufficient, and it is not. On an RTX 5060 Ti,
which has them, FP16 measured 0.78-0.85x - slower - for cspnext_s and
rtmpose_s at the live batch. That machine turned out to be launch-bound rather
than compute-bound (batch 1 and batch 8 cost the same wall clock), so the
result is really about WDDM launch overhead on Windows and says nothing about
the same card on a Linux rig. Either way, no fleet card is yet measured faster
in FP16 at the live workload, so the default is FP32 everywhere and
MEASURED_FASTER_IN_FP16 is empty until a real measurement fills it.
"""

import pytest

from autotrainer.inference import gpu_capability
from autotrainer.inference.gpu_capability import (
    PRECISION_ENVIRONMENT_VARIABLE,
    TENSOR_CORE_EXCLUDED_DEVICES,
    GpuCapability,
    detect_gpu_capability,
    resolve_precision,
)


def _cap(name, major, minor, memory_gb=4):
    return GpuCapability(
        name=name, major=major, minor=minor,
        total_memory_bytes=int(memory_gb * 1024 ** 3),
    )


def test_compute_capability_tuple():
    assert _cap("NVIDIA T1000", 7, 5).compute_capability == (7, 5)


def test_t1000_has_no_tensor_cores_despite_capability_7_5():
    assert _cap("NVIDIA T1000", 7, 5).has_tensor_cores is False


def test_other_turing_cards_do_have_tensor_cores():
    assert _cap("Tesla T4", 7, 5).has_tensor_cores is True


def test_ampere_and_blackwell_have_tensor_cores():
    assert _cap("NVIDIA RTX A2000", 8, 6).has_tensor_cores is True
    assert _cap("NVIDIA GeForce RTX 5060 Ti", 12, 0).has_tensor_cores is True


def test_pre_volta_has_no_tensor_cores():
    assert _cap("NVIDIA GeForce GTX 1080", 6, 1).has_tensor_cores is False


def test_precision_is_fp32_without_tensor_cores():
    assert _cap("NVIDIA T1000", 7, 5).preferred_precision == "fp32"


def test_tensor_cores_alone_do_not_select_fp16():
    """The regression this guards: selecting FP16 from the architecture would
    have made the 5060 Ti 15-22% slower at the live batch."""
    assert _cap("NVIDIA RTX A2000", 8, 6).preferred_precision == "fp32"
    assert _cap("NVIDIA GeForce RTX 5060 Ti", 12, 0).preferred_precision == "fp32"


def test_nothing_is_measured_faster_in_fp16_yet():
    """An entry here is a claim about a card, and needs a measurement from
    that card on its production OS at the live batch and input size."""
    assert gpu_capability.MEASURED_FASTER_IN_FP16 == frozenset()


def test_a_measured_card_selects_fp16(monkeypatch):
    monkeypatch.setattr(gpu_capability, "MEASURED_FASTER_IN_FP16",
                        frozenset({"rtx a2000"}))
    assert _cap("NVIDIA RTX A2000", 8, 6).preferred_precision == "fp16"
    assert _cap("NVIDIA GeForce RTX 5060 Ti", 12, 0).preferred_precision == "fp32"


def test_a_measured_card_without_tensor_cores_still_gets_fp32():
    """A mistaken entry must not be able to select FP16 on a card that
    physically cannot do it quickly."""
    import unittest.mock as mock

    with mock.patch.object(gpu_capability, "MEASURED_FASTER_IN_FP16",
                           frozenset({"t1000"})):
        assert _cap("NVIDIA T1000", 7, 5).preferred_precision == "fp32"


# --- the per-rig override ----------------------------------------------------


def test_the_environment_overrides_the_card_default():
    """How a rig gets benchmarked before it earns an entry."""
    capability = _cap("NVIDIA T1000", 7, 5)
    assert resolve_precision(
        capability, {PRECISION_ENVIRONMENT_VARIABLE: "fp16"}) == "fp16"


@pytest.mark.parametrize("value", ["FP16", " fp16 "])
def test_the_override_is_case_and_space_insensitive(value):
    assert resolve_precision(
        _cap("NVIDIA T1000", 7, 5),
        {PRECISION_ENVIRONMENT_VARIABLE: value}) == "fp16"


@pytest.mark.parametrize("value", ["fp8", "half", "", "  "])
def test_an_unusable_override_is_ignored_not_raised(value):
    """A typo in a rig's environment must not stop inference from starting."""
    assert resolve_precision(
        _cap("NVIDIA T1000", 7, 5),
        {PRECISION_ENVIRONMENT_VARIABLE: value}) == "fp32"


def test_without_an_override_the_card_decides():
    assert resolve_precision(_cap("NVIDIA T1000", 7, 5), {}) == "fp32"


def test_without_a_card_at_all_it_is_fp32():
    """Detection returns None when torch cannot see a GPU, and that must not
    become an exception on the path that starts inference."""
    assert resolve_precision(None, {}) == "fp32"


def test_any_capability_from_maxwell_up_is_describable():
    """Not just fleet members: nothing may be hardcoded as the only support."""
    for major, minor in ((5, 2), (6, 1), (7, 5), (8, 6), (8, 9), (9, 0), (12, 0)):
        cap = _cap("Some NVIDIA Card", major, minor)
        assert cap.compute_capability == (major, minor)
        assert cap.preferred_precision in {"fp32", "fp16"}


def test_exclusion_matching_is_case_insensitive():
    assert _cap("nvidia t1000", 7, 5).has_tensor_cores is False
    assert _cap("NVIDIA T1000 8GB", 7, 5).has_tensor_cores is False


def test_excluded_device_list_is_lowercase():
    """Matching lowercases the device name, so entries must be lowercase."""
    for entry in TENSOR_CORE_EXCLUDED_DEVICES:
        assert entry == entry.lower()


def test_capability_is_frozen():
    cap = _cap("NVIDIA T1000", 7, 5)
    try:
        cap.name = "other"
    except Exception:
        return
    raise AssertionError("GpuCapability should be immutable")


def test_detect_returns_none_without_torch(monkeypatch):
    """Detection must degrade quietly, not raise, when torch is absent."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert detect_gpu_capability() is None
