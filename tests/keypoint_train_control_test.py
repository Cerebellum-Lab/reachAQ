"""Training that reports progress, stops on request and resumes exactly.

Pausing is what keeps training off the GPU while a session records, so a
paused and resumed run has to end where an uninterrupted one would: otherwise
an operator who recorded mid-training gets a different model for it.
"""

import json
import os

import numpy
import pytest
import yaml

torch = pytest.importorskip("torch")

from autotrainer.inference import pose_plugins  # noqa: E402
from autotrainer.inference.keypoint import KeypointPoseModel  # noqa: E402
from autotrainer.inference.keypoint.train import (  # noqa: E402
    CHECKPOINT,
    SIDECAR,
    WEIGHTS,
    autocast_dtype,
    train_arrays,
)

PARTS = ["a", "b"]
TINY = dict(widths=(8, 8, 8, 8), blocks=(1, 1, 1, 1), neck=8, batch=8,
            device="cpu", seed=3)


def _data(n=16, size=64):
    rng = numpy.random.default_rng(0)
    frames = rng.integers(0, 255, (n, size, size)).astype("uint8")
    points = rng.uniform(8, size - 8, (n, len(PARTS), 2)).astype("float32")
    points[0, 1] = numpy.nan
    return frames, points


def _run(output, **kwargs):
    frames, points = _data()
    return train_arrays(frames, points, frames[:4], points[:4], PARTS, str(output),
                        epochs=4, **TINY, **kwargs)


def _weights(output):
    return torch.load(os.path.join(output, WEIGHTS), weights_only=True)


def test_progress_is_reported_once_per_epoch(tmp_path):
    seen = []
    result = _run(tmp_path / "m", progress=lambda epoch, epochs, line: seen.append((epoch, epochs)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert result["status"] == "finished"


def test_a_stopped_run_resumes_to_the_uninterrupted_result(tmp_path):
    _run(tmp_path / "straight")

    epochs_done = []
    result = _run(tmp_path / "paused",
                  progress=lambda epoch, epochs, line: epochs_done.append(epoch),
                  should_stop=lambda: len(epochs_done) >= 2)
    assert result["status"] == "stopped"
    assert os.path.isfile(tmp_path / "paused" / CHECKPOINT)
    assert not os.path.exists(tmp_path / "paused" / WEIGHTS)

    result = _run(tmp_path / "paused", resume=True)
    assert result["status"] == "finished"
    assert not os.path.exists(tmp_path / "paused" / CHECKPOINT)
    straight, resumed = _weights(tmp_path / "straight"), _weights(tmp_path / "paused")
    for key in straight:
        assert torch.allclose(straight[key].float(), resumed[key].float(), atol=1e-5), key


def test_a_stop_before_the_first_epoch_resumes_from_scratch(tmp_path):
    result = _run(tmp_path / "m", should_stop=lambda: True)
    assert result["status"] == "stopped"
    assert result["epoch"] == 0
    result = _run(tmp_path / "m", resume=True)
    assert result["status"] == "finished"


def test_resume_without_a_checkpoint_starts_fresh(tmp_path):
    assert _run(tmp_path / "m", resume=True)["status"] == "finished"


def test_no_labelled_training_frames_is_refused(tmp_path):
    frames, points = _data()
    with pytest.raises(ValueError, match="no labelled"):
        train_arrays(frames[:0], points[:0], frames, points, PARTS, str(tmp_path / "m"),
                     epochs=1, **TINY)


def test_training_without_held_out_frames_still_writes_a_model(tmp_path):
    frames, points = _data()
    result = train_arrays(frames, points, frames[:0], points[:0], PARTS, str(tmp_path / "m"),
                          epochs=2, **TINY)
    assert result["status"] == "finished"
    assert os.path.isfile(tmp_path / "m" / WEIGHTS)


def test_the_output_is_a_live_model_with_its_provenance(tmp_path):
    _run(tmp_path / "m", provenance={"dataset": "demo", "content_hash": "abc"})
    path = str(tmp_path / "m")
    assert pose_plugins.KEYPOINT_PLUGIN.claims(path)
    with open(os.path.join(path, SIDECAR), encoding="utf-8") as handle:
        sidecar = yaml.safe_load(handle)
    assert sidecar["training"]["dataset"] == "demo"
    assert sidecar["training"]["content_hash"] == "abc"
    assert sidecar["training"]["epochs"] == 4
    with open(os.path.join(path, "history.json"), encoding="utf-8") as handle:
        assert json.load(handle)
    model = KeypointPoseModel(path, batch_size=1, device="cpu")
    model.load()
    assert model.body_parts == PARTS


def test_cpu_training_runs_without_autocast():
    assert autocast_dtype("cpu") is None


def test_turing_cards_train_in_fp32_and_newer_ones_in_bf16(monkeypatch):
    # Measured on christielab10's Quadro T1000 (7.5), full network, batch 32:
    # fp32 126 ms per step, bf16 489, fp16 417. bf16 is emulated before 8.0.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    assert autocast_dtype("cuda") is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 9))
    assert autocast_dtype("cuda") is torch.bfloat16
