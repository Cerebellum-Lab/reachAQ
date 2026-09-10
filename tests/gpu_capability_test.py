"""Precision must be chosen per GPU, never assumed.

Measured on the rig's T1000: FP16 is 3-4x SLOWER than FP32, by two independent
methods (a cuBLAS GEMM and a convolution stack). The T1000 uses the TU117 die,
which omits tensor cores while still reporting compute capability 7.5 - so
capability alone does not answer the question and the device name has to be
consulted.
"""

from autotrainer.inference.gpu_capability import (
    TENSOR_CORE_EXCLUDED_DEVICES,
    GpuCapability,
    detect_gpu_capability,
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


def test_precision_is_fp16_with_tensor_cores():
    assert _cap("NVIDIA RTX A2000", 8, 6).preferred_precision == "fp16"
    assert _cap("NVIDIA GeForce RTX 5060 Ti", 12, 0).preferred_precision == "fp16"


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
