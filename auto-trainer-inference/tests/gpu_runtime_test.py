from autotrainer.inference import gpu_runtime
from autotrainer.inference.gpu_runtime import GpuRuntimeStatus


def _capture_hardware_messages(monkeypatch):
    messages = []

    def capture(_logger, message, *args, **_kwargs):
        messages.append(message % args if args else message)

    monkeypatch.setattr(gpu_runtime, "log_hardware_initialization", capture)
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_nvidia_driver",
        lambda: GpuRuntimeStatus(True, backend="nvidia-driver", devices=("Test GPU, 535.0",)),
    )
    return messages


def test_gpu_runtime_reports_each_backend_and_selected_device(monkeypatch):
    messages = _capture_hardware_messages(monkeypatch)
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_tensorflow_gpu",
        lambda: GpuRuntimeStatus(False, backend="tensorflow", error="no GPU devices found"),
    )
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_torch_cuda",
        lambda: GpuRuntimeStatus(True, backend="torch", devices=("CUDA:0",)),
    )

    status = gpu_runtime.detect_gpu_runtime()

    assert status.is_available is True
    assert status.backend == "torch"
    assert any("START | GPU runtime probe | backend=tensorflow" in message for message in messages)
    assert any("UNAVAILABLE | GPU runtime probe | backend=tensorflow" in message for message in messages)
    assert any("START | GPU runtime probe | backend=torch" in message for message in messages)
    assert any("READY | live inference GPU | backend=torch" in message for message in messages)


def test_gpu_runtime_reports_combined_failure(monkeypatch):
    messages = _capture_hardware_messages(monkeypatch)
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_tensorflow_gpu",
        lambda: GpuRuntimeStatus(False, backend="tensorflow", error="missing driver"),
    )
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_torch_cuda",
        lambda: GpuRuntimeStatus(False, backend="torch", error="CUDA is not available"),
    )

    status = gpu_runtime.detect_gpu_runtime()

    assert status.is_available is False
    assert status.backend == "tensorflow/torch"
    assert "tensorflow: missing driver" in status.error
    assert "torch: CUDA is not available" in status.error
    assert any("FAILED | live inference GPU" in message for message in messages)


def test_required_tensorflow_backend_does_not_import_torch(monkeypatch):
    _capture_hardware_messages(monkeypatch)
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_tensorflow_gpu",
        lambda: GpuRuntimeStatus(False, backend="tensorflow", error="no GPU devices found"),
    )

    def unexpected_torch_probe():
        raise AssertionError("torch should not be probed for a TensorFlow model")

    monkeypatch.setattr(gpu_runtime, "_detect_torch_cuda", unexpected_torch_probe)

    status = gpu_runtime.detect_gpu_runtime(required_backend="tensorflow")

    assert status.is_available is False
    assert status.backend == "tensorflow"


def test_nouveau_driver_fails_before_framework_import(monkeypatch):
    messages = []
    monkeypatch.setattr(
        gpu_runtime,
        "log_hardware_initialization",
        lambda _logger, message, *args, **_kwargs: messages.append(message % args if args else message),
    )
    monkeypatch.setattr(gpu_runtime, "_linux_nvidia_display_drivers", lambda: ("nouveau",))

    def unexpected_framework_probe():
        raise AssertionError("framework should not be imported when nouveau is active")

    monkeypatch.setattr(gpu_runtime, "_detect_tensorflow_gpu", unexpected_framework_probe)

    status = gpu_runtime.detect_gpu_runtime(required_backend="tensorflow")

    assert status.is_available is False
    assert status.backend == "nvidia-driver"
    assert "nouveau" in status.error
    assert any("FAILED | NVIDIA driver preflight" in message for message in messages)
