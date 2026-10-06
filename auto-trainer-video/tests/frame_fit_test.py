import numpy
import pytest

from autotrainer.video.frame_fit import QueueFit, fit_factor


def _rounded_box_mean(frame, k):
    rows, cols = frame.shape[0] // k, frame.shape[1] // k
    blocks = frame.reshape(rows, k, cols, k).astype(numpy.uint32).sum(axis=(1, 3))
    return ((blocks + (k * k) // 2) // (k * k)).astype(numpy.uint8)


@pytest.mark.parametrize("k", (2, 4))
def test_a_larger_frame_becomes_the_rounded_mean_of_each_k_by_k_block(k):
    rng = numpy.random.default_rng(1)
    frame = rng.integers(0, 256, (256 * k, 256 * k), dtype=numpy.uint8)

    fit = QueueFit(frame.shape, (256, 256))
    fitted = fit(frame)

    assert fit.factor == k
    assert fitted.shape == (256, 256) and fitted.dtype == numpy.uint8
    # OpenCV's fixed-point area average is within one grey level of the exact
    # rounded mean (measured on christielab10: 0 at k=2, 1 at k=4).
    assert numpy.abs(fitted.astype(int) - _rounded_box_mean(frame, k).astype(int)).max() <= 1


def test_a_queue_the_same_size_as_the_frame_gets_the_frame_itself():
    frame = numpy.zeros((256, 256), numpy.uint8)
    fit = QueueFit(frame.shape, (256, 256))
    assert fit.factor == 1 and fit.shape == (256, 256)
    assert fit(frame) is frame


def test_a_queue_without_a_shape_gets_the_frame_itself():
    frame = numpy.zeros((200, 300), numpy.uint8)
    fit = QueueFit(frame.shape, None)
    assert fit.factor == 1 and fit.shape == (200, 300)
    assert fit(frame) is frame


@pytest.mark.parametrize("frame_shape,target", [
    ((500, 500), (256, 256)),    # not a whole multiple
    ((512, 1024), (256, 256)),   # different factor per direction
    ((256, 256), (512, 512)),    # a larger queue would need upscaling
    ((512, 512), (0, 256)),
])
def test_a_queue_that_is_not_the_frame_divided_by_one_integer_is_refused(frame_shape, target):
    with pytest.raises(ValueError, match="one integer factor"):
        fit_factor(frame_shape, target)
