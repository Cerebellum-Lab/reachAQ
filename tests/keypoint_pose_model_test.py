"""The keypoint backend: its coordinate convention, its model directory, its routing.

The convention test is the one that matters most. Training encodes a label as a
heatmap cell plus an offset and live inference decodes it back; if the two
disagree by half a cell, every keypoint is off by two pixels and nothing else
in the pipeline would notice.
"""

import numpy
import pytest
import yaml

torch = pytest.importorskip("torch")

from reachaq.inference import backend_selection, pose_plugins  # noqa: E402
from reachaq.inference.backend_selection import (  # noqa: E402
    KEYPOINT_CONFIDENCE_THRESHOLD,
    TORCH_BACKEND,
    build_pose_model,
    confidence_threshold,
    pre_validate_model,
    selected_backend,
)
from reachaq.inference.keypoint import KeypointPoseModel  # noqa: E402
from reachaq.inference.keypoint.network import STRIDE, KeypointNet, decode  # noqa: E402
from reachaq.inference.keypoint.train import SIDECAR, WEIGHTS, _targets  # noqa: E402


def _ideal_output(points, cells):
    """The output a perfect network would give for these labels."""
    n, k, _ = points.shape
    u, cell, valid = _targets(points, cells)
    out = torch.zeros(n, 3 * k, cells, cells)
    out[:, :k] = -10.0
    for i in range(n):
        for j in range(k):
            if not valid[i, j]:
                continue
            cx, cy = int(cell[i, j, 0]), int(cell[i, j, 1])
            out[i, j, cy, cx] = 10.0
            out[i, k + j, cy, cx] = u[i, j, 0] - (cx + 0.5)
            out[i, 2 * k + j, cy, cx] = u[i, j, 1] - (cy + 0.5)
    return out, valid


def test_decode_inverts_the_training_encoding():
    points = torch.tensor([[[0.0, 0.0], [17.3, 200.9], [255.0, 128.5]],
                           [[63.99, 64.0], [3.5, 251.25], [float("nan"), float("nan")]]])
    out, valid = _ideal_output(points, 256 // STRIDE)
    decoded = decode(out)
    assert torch.allclose(decoded[..., :2][valid], points[valid], atol=1e-4)
    assert (decoded[..., 2][valid] > 0.99).all()
    assert (decoded[..., 2][~valid] < 0.01).all()


def test_the_network_output_matches_what_decode_reads():
    net = KeypointNet(5, widths=(8, 16, 16, 16), blocks=(1, 1, 1, 1), neck=8).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 1, 128, 128))
    assert out.shape == (2, 15, 128 // STRIDE, 128 // STRIDE)
    assert decode(out).shape == (2, 5, 3)


def _model_dir(tmp_path, parts=("Nose", "Pellet", "RH_grab"), declared=None):
    net = KeypointNet(len(parts), widths=(8, 16, 16, 16), blocks=(1, 1, 1, 1), neck=8)
    torch.save(net.state_dict(), tmp_path / WEIGHTS)
    spec = {"weights": WEIGHTS, "imgsz": 128, "network": net.config(),
            "bodyparts": list(parts) if declared is None else declared}
    (tmp_path / SIDECAR).write_text(yaml.safe_dump(spec))
    return tmp_path


def test_a_model_directory_loads_and_predicts_full_frame_parts(tmp_path):
    model = KeypointPoseModel(str(_model_dir(tmp_path)), batch_size=2, device="cpu")
    model.load()
    assert model.body_parts == ["Nose", "Pellet", "RH_grab"]
    assert "no CUDA graph" in model.runtime_detail()
    frames = numpy.random.default_rng(0).integers(0, 255, (2, 128, 128, 3), dtype=numpy.uint8)
    out = model.predict(frames)
    assert len(out) == 2 and out[0].shape == (3, 3) and out[0].dtype == numpy.float32


def test_a_larger_frame_is_scaled_back_about_pixel_centres(tmp_path):
    """A 256 frame through a 128 model reports in the 256 frame's pixels."""
    model = KeypointPoseModel(str(_model_dir(tmp_path)), batch_size=1, device="cpu")
    model.load()
    small = numpy.random.default_rng(1).integers(0, 255, (1, 128, 128, 3), dtype=numpy.uint8)
    large = small.repeat(2, axis=1).repeat(2, axis=2)
    # A 2x nearest upscale resized back by bilinear interpolation is the
    # original frame, so the two must land on the same points.
    at_small = model.predict(small)[0]
    at_large = model.predict(large)[0]
    assert numpy.allclose(at_large[:, :2], (at_small[:, :2] + 0.5) * 2 - 0.5, atol=1e-3)
    assert numpy.allclose(at_large[:, 2], at_small[:, 2], atol=1e-5)


def test_a_non_square_frame_is_refused(tmp_path):
    model = KeypointPoseModel(str(_model_dir(tmp_path)), device="cpu")
    model.load()
    with pytest.raises(ValueError, match="square"):
        model.predict(numpy.zeros((1, 128, 96, 3), dtype=numpy.uint8))


def test_a_sidecar_that_names_other_parts_than_the_network_fails_to_load(tmp_path):
    model_dir = _model_dir(tmp_path, declared=["Nose", "Pellet"])
    with pytest.raises(ValueError, match="keypoints"):
        KeypointPoseModel(str(model_dir), device="cpu").load()


def test_a_keypoint_model_routes_to_its_backend_and_threshold(tmp_path, monkeypatch):
    monkeypatch.delenv(backend_selection.POSE_BACKEND_ENV_VAR, raising=False)
    monkeypatch.delenv(backend_selection.POSE_CONFIDENCE_THRESHOLD_ENV_VAR, raising=False)
    path = str(_model_dir(tmp_path))
    pre_validate_model(path)
    assert isinstance(build_pose_model(path, batch_size=2), KeypointPoseModel)
    assert selected_backend(model_path=path) == TORCH_BACKEND
    assert confidence_threshold(model_path=path) == pytest.approx(KEYPOINT_CONFIDENCE_THRESHOLD)


def test_a_keypoint_model_missing_its_weights_fails_pre_validation(tmp_path):
    (_model_dir(tmp_path) / WEIGHTS).unlink()
    with pytest.raises(FileNotFoundError):
        pre_validate_model(str(tmp_path))


def test_a_yolo_model_without_its_plugin_says_why(tmp_path, monkeypatch):
    """ultralytics is not a dependency any more; the refusal has to name the
    plugin that carries it and the keypoint trainer that replaces it."""
    (tmp_path / "yolo_pose.yaml").write_text("weights: best.pt\nbodyparts: [Nose]\n")
    (tmp_path / "best.pt").write_bytes(b"")
    monkeypatch.setattr(pose_plugins, "_entry_points", lambda: [])
    pose_plugins.reset_discovery()
    try:
        with pytest.raises(ImportError, match="reachaq-pose-yolo") as raised:
            pre_validate_model(str(tmp_path))
        assert "reachaq.inference.keypoint.train" in str(raised.value)
        with pytest.raises(ImportError, match="reachaq-plugin-install.sh yolo"):
            build_pose_model(str(tmp_path))
    finally:
        pose_plugins.reset_discovery()
