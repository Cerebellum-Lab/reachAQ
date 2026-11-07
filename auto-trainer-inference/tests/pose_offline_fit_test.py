import multiprocessing

import numpy
import pytest

from autotrainer.inference.pose_offline_input import OfflineInputProcess, _fit_to_shape


class _Capture:
    def __init__(self, frame):
        self._frame = frame

    def read(self):
        return True, self._frame


def _offline():
    offline = OfflineInputProcess(
        stop_recorded=multiprocessing.Event(),
        frame_shape=(256, 256),
        frames_per_cam=1,
        nr_cams=2,
        msg_queue=multiprocessing.Queue(),
        event_cb_ack=multiprocessing.Event(),
    )
    offline._sema_free.release()  # set_project_info normally frees the buffers
    return offline


def test_a_512_recording_reaches_the_pose_buffer_averaged_to_256():
    rng = numpy.random.default_rng(2)
    gray = rng.integers(0, 256, (512, 512), dtype=numpy.uint8)
    bgr = numpy.repeat(gray[:, :, None], 3, axis=2)  # what cv2 decodes from the mp4
    offline = _offline()

    assert offline._put_intersession_frame(_Capture(bgr), 0, 0)

    expected = (gray.reshape(256, 2, 256, 2).astype(numpy.uint32).sum(axis=(1, 3)) + 2) // 4
    plane = offline._buffer1[0, :, :, 0]
    assert numpy.abs(plane - expected).max() <= 1
    assert (offline._buffer1[0, :, :, 2] == plane).all()


def test_a_recording_at_the_pose_size_is_untouched():
    frame = numpy.zeros((256, 256), numpy.uint8)
    assert _fit_to_shape(frame, (256, 256)) is frame


def test_a_recording_that_is_not_an_integer_multiple_is_refused():
    with pytest.raises(ValueError, match="one integer factor"):
        _fit_to_shape(numpy.zeros((500, 500), numpy.uint8), (256, 256))


@pytest.mark.parametrize("k", (2, 4))
def test_the_replay_averages_exactly_as_live_capture_does(k):
    # autotrainer.inference does not depend on autotrainer.video, so the two
    # implementations are separate; this pins them to the same output.
    from autotrainer.video.frame_fit import QueueFit

    frame = numpy.random.default_rng(3).integers(0, 256, (256 * k, 256 * k), dtype=numpy.uint8)
    assert (_fit_to_shape(frame, (256, 256)) == QueueFit(frame.shape, (256, 256))(frame)).all()
