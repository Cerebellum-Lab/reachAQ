"""Live frames reach a YOLO model at its own imgsz, and come back in frame pixels.

The 320 and 512 candidates were trained on the same 256x256 frames as the 256
models, which ultralytics upscaled to imgsz, and its predict() upscales live
frames the same way before dividing the keypoints back by the gain. The CUDA
graph path instead copied each 256 frame into an input captured at imgsz. On
the rig the first live call raised "The size of tensor a (320) must match the
size of tensor b (256)", the pose process exited, and a 120 s recording posed
none of its frames.

The graph is replaced by a CPU stand-in that reports the brightest input pixel
as the keypoint, in input pixels as the real head does, so the round trip from
frame to model and back is checked without a GPU.
"""

import numpy
import pytest

torch = pytest.importorskip("torch")

from autotrainer.inference.yolo import YoloPoseModel


class _BrightestPixelGraph:
    def __init__(self, model):
        self._model = model

    def replay(self):
        graph_in = self._model._graph_in
        size = graph_in.shape[-1]
        brightest = graph_in.sum(dim=1).flatten(1).argmax(dim=1)
        out = self._model._graph_out
        out.zero_()
        out[:, 4, 0] = 1.0  # class score of the only anchor
        out[:, 5, 0] = (brightest % size).float()  # x
        out[:, 6, 0] = (brightest // size).float()  # y
        out[:, 7, 0] = 1.0  # keypoint confidence


def _graphed_model(imgsz):
    model = YoloPoseModel("/unused", batch_size=2)
    model._imgsz = imgsz
    model._body_parts_count = 1
    model._graph_in = torch.zeros((2, 3, imgsz, imgsz))
    model._graph_out = torch.zeros((2, 5 + 3, 1))
    model._graph = _BrightestPixelGraph(model)
    return model


def _frames_with_dots(rows, cols, dots):
    frames = numpy.zeros((len(dots), rows, cols, 3))
    for index, (x, y) in enumerate(dots):
        frames[index, y, x] = 255.0
    return frames


DOTS = [(40, 100), (200, 7)]


@pytest.mark.parametrize("imgsz", [320, 512])
def test_a_base_frame_is_scaled_to_the_model_and_back(imgsz):
    poses = _graphed_model(imgsz).predict(_frames_with_dots(256, 256, DOTS))

    for pose, (x, y) in zip(poses, DOTS):
        assert pose[0, 0] == pytest.approx(x, abs=1.0)
        assert pose[0, 1] == pytest.approx(y, abs=1.0)
        assert pose[0, 2] == 1.0


@pytest.mark.parametrize("imgsz", [320, 512])
def test_the_resize_is_the_one_ultralytics_letterbox_makes(imgsz):
    """LetterBox resizes with cv2 INTER_LINEAR on half-pixel centres. A dot
    test cannot tell align_corners=True from False, or a half-pixel shift; a
    comparison of every input value can."""
    cv2 = pytest.importorskip("cv2")
    model = _graphed_model(imgsz)
    frames = numpy.random.default_rng(0).uniform(0, 255, (2, 256, 256, 3))

    model.predict(frames)

    letterboxed = numpy.stack([
        cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        for frame in frames])
    expected = torch.from_numpy(numpy.ascontiguousarray(
        letterboxed[..., ::-1].transpose(0, 3, 1, 2))).float().div_(255.0)
    assert (model._graph_in - expected).abs().max().item() < 1e-4


def test_a_frame_already_at_imgsz_reaches_the_graph_unchanged():
    """The 256 models, which is what the rig runs live, must not move at all."""
    model = _graphed_model(256)
    frames = _frames_with_dots(256, 256, DOTS)

    poses = model.predict(frames)

    expected = torch.from_numpy(numpy.ascontiguousarray(
        frames[..., ::-1].transpose(0, 3, 1, 2))).float().div_(255.0)
    assert torch.equal(model._graph_in, expected)
    assert [tuple(pose[0, :2]) for pose in poses] == [(40.0, 100.0), (200.0, 7.0)]


def test_a_non_square_frame_is_refused_rather_than_stretched():
    """Every model was trained on square frames; stretching one would mislabel
    every keypoint while still looking plausible."""
    with pytest.raises(ValueError, match="256x200"):
        _graphed_model(320).predict(numpy.zeros((2, 256, 200, 3)))
