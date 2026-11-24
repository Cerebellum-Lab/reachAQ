"""Pellet presence from an ROI of one camera's frames, for the cue gate."""

import multiprocessing

import numpy
import pytest

from autotrainer.video import (
    PelletRoiConfiguration,
    PelletRoiDetector,
    PelletRoiState,
)


def _state():
    return PelletRoiState(multiprocessing.get_context("spawn"))


def _frame(level, shape=(20, 20)):
    frame = numpy.zeros(shape, dtype=numpy.uint8)
    frame[5:10, 5:10] = level
    return frame


ROI = PelletRoiConfiguration(x=5, y=5, width=5, height=5, threshold=100, hysteresis=20)


def test_nothing_is_known_before_a_frame():
    assert _state().latest() is None


def test_a_bright_roi_is_a_present_pellet_at_the_frames_time():
    state = _state()
    detector = PelletRoiDetector(ROI, state)

    assert detector.process(_frame(150), 12.5) is True
    present, perf, value = state.latest()
    assert present is True and perf == 12.5 and value == 150.0


def test_hysteresis_holds_a_present_pellet_near_the_threshold():
    state = _state()
    detector = PelletRoiDetector(ROI, state)

    assert detector.process(_frame(150), 1.0) is True
    assert detector.process(_frame(90), 1.1) is True     # within the 20 of hysteresis
    assert detector.process(_frame(80), 1.2) is False    # back across it
    assert detector.process(_frame(95), 1.3) is False    # must exceed the threshold again
    assert detector.process(_frame(101), 1.4) is True


def test_a_pellet_darker_than_its_post_reads_present_below():
    detector = PelletRoiDetector(
        PelletRoiConfiguration(x=5, y=5, width=5, height=5, threshold=100,
                               present_when="below"),
        _state())

    assert detector.process(_frame(40), 1.0) is True
    assert detector.process(_frame(160), 1.1) is False


def test_an_roi_outside_the_frame_publishes_unknown_and_does_not_raise():
    state = _state()
    detector = PelletRoiDetector(
        PelletRoiConfiguration(x=50, y=50, width=5, height=5, threshold=1), state)

    assert detector.process(_frame(150), 1.0) is None
    assert state.latest() is None


def test_the_configuration_comes_from_the_cameras_params():
    assert PelletRoiConfiguration.from_params({}) is None
    configuration = PelletRoiConfiguration.from_params({
        "pellet_roi_x": "4", "pellet_roi_y": 6, "pellet_roi_width": 10,
        "pellet_roi_height": 8, "pellet_roi_threshold": "75.5",
        "pellet_roi_hysteresis": 5, "pellet_roi_present_when": "Below",
    })
    assert configuration == PelletRoiConfiguration(4, 6, 10, 8, 75.5, 5.0, "below")
    with pytest.raises(ValueError, match="threshold"):
        PelletRoiConfiguration.from_params({"pellet_roi_width": 4, "pellet_roi_height": 4})
    with pytest.raises(ValueError, match="present_when"):
        PelletRoiConfiguration(0, 0, 1, 1, 1, present_when="sideways")


def test_the_state_crosses_into_a_spawned_process():
    # Made in the parent and handed to the capture child, as CaptureAttrs is.
    context = multiprocessing.get_context("spawn")
    state = PelletRoiState(context)
    process = context.Process(target=_publish_present, args=(state,))
    process.start()
    process.join(30)

    assert process.exitcode == 0
    assert state.latest() == (True, 3.0, 150.0)


def _publish_present(state):
    PelletRoiDetector(ROI, state).process(_frame(150), 3.0)
