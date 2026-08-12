import time

import pytest

from autotrainer.video import video_manager
from autotrainer.video.camera_discovery import (
    CameraDiscoveryError,
    CameraDiscoveryFailure,
    CameraDiscoveryTimeout,
    discover_spin_cameras,
)


def _successful_worker(result_queue):
    result_queue.put(("ok", ("111", "222")))


def _failing_worker(result_queue):
    result_queue.put(
        (
            "error",
            CameraDiscoveryFailure(
                stage="enumerate",
                exception_type="PySpinException",
                message="USB transport failed",
                traceback_text="child traceback",
            ),
        )
    )


def _hanging_worker(_result_queue):
    time.sleep(30)


def test_discovery_returns_serials_from_disposable_process():
    assert discover_spin_cameras(
        timeout_seconds=2,
        _worker_target=_successful_worker,
    ) == ("111", "222")


def test_discovery_preserves_child_stage_and_first_error():
    with pytest.raises(CameraDiscoveryError) as caught:
        discover_spin_cameras(
            timeout_seconds=2,
            _worker_target=_failing_worker,
        )

    assert caught.value.stage == "enumerate"
    assert caught.value.exception_type == "PySpinException"
    assert "USB transport failed" in str(caught.value)
    assert caught.value.traceback_text == "child traceback"


def test_discovery_terminates_hung_vendor_process_at_deadline():
    started = time.perf_counter()
    with pytest.raises(CameraDiscoveryTimeout) as caught:
        discover_spin_cameras(
            timeout_seconds=0.1,
            _worker_target=_hanging_worker,
        )

    assert caught.value.stage == "timeout"
    assert time.perf_counter() - started < 3


def test_video_manager_does_not_convert_backend_failure_to_empty_list(monkeypatch):
    def fail_backend_load():
        raise RuntimeError("SDK load failed")

    monkeypatch.setattr(video_manager, "_get_spincam_cls", fail_backend_load)

    with pytest.raises(RuntimeError, match="SDK load failed"):
        video_manager.VideoManager.list_spin_cameras()
