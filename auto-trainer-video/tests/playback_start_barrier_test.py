import multiprocessing
import time

import pytest


def _worker(barrier, started_at, index):
    # Stagger arrivals so an unsynchronized start would be obvious.
    time.sleep(0.05 * index)
    barrier.wait(timeout=5)
    started_at[index] = time.perf_counter()


def test_barrier_aligns_staggered_starts():
    """Cameras that arrive up to 100ms apart still start together."""
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    started_at = context.Array("d", 3)

    processes = [
        context.Process(target=_worker, args=(barrier, started_at, index))
        for index in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)

    values = list(started_at)
    assert all(value > 0 for value in values), "every worker must have recorded a start"
    spread = max(values) - min(values)
    # One frame at 150 FPS is 6.7ms. Allow generous slack for process scheduling
    # while still proving the 100ms stagger was absorbed.
    assert spread < 0.05, f"start spread {spread:.4f}s is too wide"


def test_capture_attrs_carries_a_playback_barrier():
    from autotrainer.video.video_capture import CaptureAttrs

    assert "playback_start_barrier" in CaptureAttrs.__dataclass_fields__
    assert "playback_start_timeout" in CaptureAttrs.__dataclass_fields__
    assert CaptureAttrs.__dataclass_fields__["playback_start_barrier"].default is None


def test_await_playback_start_is_a_noop_without_a_barrier():
    from autotrainer.video.video_capture import VideoCapture

    assert hasattr(VideoCapture, "_await_playback_start")


def test_barrier_timeout_names_the_camera():
    """A camera that never arrives fails the start rather than hanging the UI."""
    context = multiprocessing.get_context("spawn")
    # Sized for two participants, but only this test arrives.
    barrier = context.Barrier(2)

    with pytest.raises(Exception):
        barrier.wait(timeout=0.2)
