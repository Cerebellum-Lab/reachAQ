"""The pose overlay must belong to the frame it is drawn over.

Poses and images reach the camera panel independently: poses from inference at
up to the capture rate, images on a queue rate limited to the display rate.
Drawing whichever of each was newest meant the dots could sit on a frame up to
a display tick away - 66 ms, ten frames at 150 fps - with nothing correlating
them, in demo mode and in real acquisition alike.

Both now carry a camera frame id. These cover the matching rule, the fallback
when an id is missing, and the queue that carries the id across the process
boundary. The closed loop is untouched: it consumes pose_response_ready
directly and never goes through this widget.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy
import pytest
from PySide6.QtWidgets import QApplication

from autotrainer.core.fixed_array_queue import BufferResult, FixedArrayQueue
from autotrainer.pyside.capture.QtCaptureView import (
    POSE_FRAME_TOLERANCE,
    ImageData,
    QCaptureView,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    return app or QApplication([])


@pytest.fixture
def view(qapp):
    widget = QCaptureView()
    widget._display_dots_detection = True
    return widget


def _image(frame_id):
    return ImageData(numpy.zeros((4, 4), dtype=numpy.uint8), 4, 4, frame_id)


def _drawn(view):
    """The points the widget actually pushed to its image."""
    return view._image.points if hasattr(view._image, "points") else view._last_set


# --- the matching rule -------------------------------------------------------


def test_an_overlay_for_the_displayed_frame_is_drawn(view):
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, 100)
    view.update_pose()
    assert view._are_points_dirty is False, "the overlay was not drawn"


def test_the_default_tolerance_is_twenty_milliseconds_at_150_fps():
    """The bound the overlay is held to: below what the eye reads as lag.

    Acquisition runs ahead of inference by design, so the two are rarely on the
    same frame; what matters is that the gap stays small enough not to show.
    """
    assert POSE_FRAME_TOLERANCE / 150.0 <= 0.020 + 1e-9
    assert POSE_FRAME_TOLERANCE > 0, (
        "an exact match would drop most overlays; the display sees about one "
        "frame in ten")


def test_an_overlay_within_tolerance_is_drawn(view):
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, 100 + POSE_FRAME_TOLERANCE)
    view.update_pose()
    assert view._are_points_dirty is False


def test_an_overlay_just_past_tolerance_is_held(view):
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, 100 + POSE_FRAME_TOLERANCE + 1)
    view.update_pose()
    assert view._are_points_dirty is True


def test_an_overlay_from_a_distant_frame_is_held_back(view):
    """This is the misalignment the change exists to prevent."""
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, 130)
    view.update_pose()
    assert view._are_points_dirty is True, (
        "a pose 30 frames from the displayed one was drawn over it")


def test_a_held_overlay_is_drawn_once_its_frame_arrives(view):
    """Held back, not discarded: the frame it belongs to may still be coming."""
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, 130)
    view.update_pose()
    assert view._are_points_dirty is True

    view.refresh_image(_image(130), 15.0)
    view.update_pose()
    assert view._are_points_dirty is False


def test_tolerance_of_zero_demands_an_exact_frame(view):
    view.set_pose_frame_tolerance(0)
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, 101)
    view.update_pose()
    assert view._are_points_dirty is True


# --- falling back ------------------------------------------------------------


def test_an_unstamped_frame_falls_back_to_drawing(view):
    """A source with no frame id must not lose its overlay entirely."""
    view.refresh_image(_image(-1), 15.0)
    view.refresh_pose({"a": object()}, 100)
    view.update_pose()
    assert view._are_points_dirty is False


def test_an_unstamped_pose_falls_back_to_drawing(view):
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, -1)
    view.update_pose()
    assert view._are_points_dirty is False


def test_dots_disabled_clears_rather_than_holds(view):
    view._display_dots_detection = False
    view.refresh_image(_image(100), 15.0)
    view.refresh_pose({"a": object()}, 999)
    view.update_pose()  # must not raise


# --- the id crossing the process boundary ------------------------------------


def test_the_display_queue_carries_the_frame_id():
    queue = FixedArrayQueue(2, (4, 4), name="test")
    frame = numpy.full((4, 4), 7, dtype=numpy.uint8)
    assert queue.put(frame, 1234) == BufferResult.Ok
    data, frame_id = queue.get(timeout=0.1, with_frame_id=True)
    assert frame_id == 1234
    assert int(data[0, 0]) == 7


def test_an_unstamped_put_reads_back_as_unknown():
    queue = FixedArrayQueue(2, (4, 4), name="test")
    queue.put(numpy.zeros((4, 4), dtype=numpy.uint8))
    _data, frame_id = queue.get(timeout=0.1, with_frame_id=True)
    assert frame_id == -1


def test_existing_callers_still_get_a_bare_frame():
    """The id is opt-in; nothing that ignores it needs to change."""
    queue = FixedArrayQueue(2, (4, 4), name="test")
    queue.put(numpy.full((4, 4), 3, dtype=numpy.uint8), 99)
    data = queue.get(timeout=0.1)
    assert isinstance(data, numpy.ndarray)
    assert int(data[0, 0]) == 3


def test_ids_stay_with_their_own_buffer_slot():
    """Depth > 1, so a stale id must not leak onto the next frame."""
    queue = FixedArrayQueue(3, (4, 4), name="test")
    for value, frame_id in ((1, 10), (2, 20), (3, 30)):
        queue.put(numpy.full((4, 4), value, dtype=numpy.uint8), frame_id)
    for value, frame_id in ((1, 10), (2, 20), (3, 30)):
        data, got = queue.get(timeout=0.1, with_frame_id=True)
        assert (int(data[0, 0]), got) == (value, frame_id)
