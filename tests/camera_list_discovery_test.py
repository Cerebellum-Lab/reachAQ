import pytest

from autotrainer.video.camera_discovery import CameraDiscoveryError
from tools.acquisition.model import video_capture_model


def test_camera_list_distinguishes_zero_hardware_from_discovery_failure(monkeypatch):
    monkeypatch.setattr(
        video_capture_model,
        "discover_spin_cameras",
        lambda **_kwargs: (),
    )
    sources = video_capture_model.create_camera_list(include_hardware=True)
    assert all(not source.url.startswith("spinnaker://") for source in sources)

    def fail(**_kwargs):
        raise CameraDiscoveryError(
            "SDK unavailable",
            stage="load_backend",
            elapsed_seconds=0.1,
            exception_type="ImportError",
        )

    monkeypatch.setattr(video_capture_model, "discover_spin_cameras", fail)
    with pytest.raises(CameraDiscoveryError, match="SDK unavailable"):
        video_capture_model.create_camera_list(include_hardware=True)
