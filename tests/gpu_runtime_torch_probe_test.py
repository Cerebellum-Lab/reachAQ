"""The torch probe must survive a cuDNN failure, not just catch an exception.

`torch.cuda.is_available()` only checks the driver and CUDA runtime. On the
reachAQ rig it returns True, and a cuBLAS GEMM succeeds, while cuDNN cannot load
at all - `nvidia-cudnn-cu11` and `nvidia-cudnn-cu12` both install
`libcudnn.so.8` and whichever lands last wins.

Critically, that failure is **not a Python exception**. Measured on the rig, the
process prints `Could not load library libcudnn_ops_infer.so.8` from cuDNN's own
loader and then aborts with a core dump. A try/except around the convolution
cannot catch it, so the check has to run in a subprocess where an abort is
contained and observable as a non-zero exit status. `_detect_nvidia_driver` in
the same module already shells out for the same reason.
"""

import subprocess
import sys
import types

from autotrainer.inference import gpu_runtime


def _install_fake_torch(monkeypatch, *, available=True, devices=("FakeGPU",)):
    """Only the in-process checks are faked; the cuDNN probe is a subprocess."""
    torch = types.ModuleType("torch")

    class _Cuda:
        def is_available(self):
            return available

        def device_count(self):
            return len(devices)

        def get_device_name(self, index):
            return devices[index]

    torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_cudnn_probe_runs_in_a_subprocess(monkeypatch):
    """An abort in cuDNN must not take down the calling process."""
    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(gpu_runtime.subprocess, "run", fake_run)
    assert gpu_runtime._probe_torch_cudnn() is None
    assert calls["args"][0] == sys.executable
    assert "-c" in calls["args"]
    assert calls["kwargs"].get("timeout")


def test_cudnn_probe_reports_a_nonzero_exit(monkeypatch):
    """A core dump surfaces as a non-zero return code, not an exception."""

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, -6,
            stdout="",
            stderr="Could not load library libcudnn_ops_infer.so.8. "
                   "Error: libnvrtc.so: cannot open shared object file",
        )

    monkeypatch.setattr(gpu_runtime.subprocess, "run", fake_run)
    error = gpu_runtime._probe_torch_cudnn()
    assert error is not None
    assert "libcudnn" in error


def test_cudnn_probe_reports_a_timeout(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 60)

    monkeypatch.setattr(gpu_runtime.subprocess, "run", fake_run)
    error = gpu_runtime._probe_torch_cudnn()
    assert error is not None
    assert "timed out" in error.lower()


def test_probe_reports_unavailable_when_cudnn_probe_fails(monkeypatch):
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(
        gpu_runtime, "_probe_torch_cudnn",
        lambda: "Could not load library libcudnn_ops_infer.so.8",
    )
    status = gpu_runtime._detect_torch_cuda()
    assert status.is_available is False
    assert "libcudnn" in status.error
    # The device list stays: it says which card failed.
    assert status.devices == ("FakeGPU",)


def test_probe_reports_available_when_cudnn_probe_succeeds(monkeypatch):
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(gpu_runtime, "_probe_torch_cudnn", lambda: None)
    status = gpu_runtime._detect_torch_cuda()
    assert status.is_available is True
    assert status.devices == ("FakeGPU",)
    assert status.error == ""


def test_probe_reports_unavailable_when_cuda_is_absent(monkeypatch):
    _install_fake_torch(monkeypatch, available=False)
    called = {"probe": False}

    def should_not_run():
        called["probe"] = True
        return None

    monkeypatch.setattr(gpu_runtime, "_probe_torch_cudnn", should_not_run)
    status = gpu_runtime._detect_torch_cuda()
    assert status.is_available is False
    # No point paying for a subprocess when CUDA is not even present.
    assert called["probe"] is False


def test_probe_backend_name_is_unchanged(monkeypatch):
    """detect_gpu_runtime filters detectors by this name; it must not drift."""
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(gpu_runtime, "_probe_torch_cudnn", lambda: None)
    assert gpu_runtime._detect_torch_cuda().backend == "torch"
