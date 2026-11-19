"""The YOLO model's side of the live path.

Moved from core's tests/live_graph_batch_test.py and
tests/cropped_pose_model_test.py with the plugin; the pipeline side (the hook
on PoseModel, the order in pose_process, cropping a model that preserves
scale) stays in core.
"""

from autotrainer.inference.cropped_pose_model import (
    INFERENCE_ROI_ENV_VAR,
    InferenceRoi,
    maybe_crop,
)
from autotrainer.pose_plugins.yolo import YoloPoseModel


def test_yolo_takes_the_live_batch_for_its_graph():
    model = YoloPoseModel("/unused", batch_size=6)
    assert model._batch_size == 6
    model.prepare_live_batch(2)
    assert model._batch_size == 2


def test_a_meaningless_batch_is_ignored():
    """Nothing usable is better left alone than captured at a zero shape."""
    model = YoloPoseModel("/unused", batch_size=6)
    model.prepare_live_batch(0)
    assert model._batch_size == 6


def test_runtime_detail_names_the_batch_the_graph_is_captured_for():
    """The batch matters as much as the graph, so both belong in the line."""
    model = YoloPoseModel("/unused", batch_size=6)
    model.prepare_live_batch(2)
    model._graph = object()
    detail = model.runtime_detail()
    assert "CUDA graph" in detail
    assert "batch 2" in detail


def test_a_yolo_model_is_never_cropped_and_the_roi_is_said_to_be_ignored(monkeypatch):
    """A YOLO graph runs at its imgsz whatever it is handed, so a crop saves no
    time and only changes the scale the model was trained at. YOLO is the live
    model because it is the fastest; it keeps running, on the full frame."""
    from autotrainer.inference import cropped_pose_model

    warnings = []
    monkeypatch.setattr(cropped_pose_model.logger, "warning",
                        lambda message, *args: warnings.append(message % args))
    yolo = YoloPoseModel("/unused", batch_size=2)

    result = maybe_crop(yolo, (256, 256), roi=InferenceRoi(16, 16, 128, 128))

    assert result is yolo
    assert any(INFERENCE_ROI_ENV_VAR in warning for warning in warnings)
