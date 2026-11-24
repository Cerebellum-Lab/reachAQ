import threading
import time

import numpy
import pytest

from reachaq.core.simulation import ENV_SIM_EPOCH
from reachaq.video.camera.camera_backend import CameraBackend
from reachaq.video.camera.sim_cam import HOST_BUFFER_FRAMES, SimulatedCam
from reachaq.video.video_manager import CameraKind, VideoManager

FPS = 150
EXPOSURE_US = 2000


@pytest.fixture
def epoch(monkeypatch):
    value = time.perf_counter() + 0.05
    monkeypatch.setenv(ENV_SIM_EPOCH, repr(value))
    return value


def _make_cam(name="cam"):
    return VideoManager.create_camera(f"sim://?fps={FPS}&exposure={EXPOSURE_US}&primary=yes", name)


def _grid_position(stamp_ns, exposure, epoch, period):
    return (stamp_ns / 1e9 - exposure - epoch) / period


def test_sim_scheme_creates_simulated_cam(epoch):
    cam = _make_cam()
    assert isinstance(cam, SimulatedCam)
    assert isinstance(cam, CameraBackend)
    assert CameraKind("sim") is CameraKind.Simulated
    assert (cam.fps, cam.width, cam.height, cam.is_primary) == (FPS, 640, 480, True)
    assert cam.exposure == pytest.approx(EXPOSURE_US * 1e-6)


def test_default_params_carry_exposure_in_us(monkeypatch):
    # The app reads the URL's exposure as microseconds (_camera_exposure_seconds).
    monkeypatch.setenv("REACHAQ_SIM_CAMERA_EXPOSURE_S", "0.003")
    monkeypatch.setenv("REACHAQ_SIM_CAMERA_FPS", "100")
    _, params = VideoManager.parse_params("sim://")
    assert float(params["exposure"]) == pytest.approx(3000)
    assert int(params["fps"]) == 100


def test_stamps_sit_on_exposure_grid_end(epoch):
    cam = _make_cam()
    cam.prepare_capture()
    period = 1.0 / FPS
    exposure = EXPOSURE_US * 1e-6
    previous = None
    for _ in range(10):
        frame, stamp_ns = cam.capture()
        assert frame.shape == (480, 640) and frame.dtype == numpy.uint8
        position = _grid_position(stamp_ns, exposure, epoch, period)
        assert abs(position - round(position)) < 1e-6
        assert cam.frame_id == round(position)
        assert cam.frame_perf_c == pytest.approx(stamp_ns / 1e9 + 0.001, abs=1e-9)
        assert cam.frame_arrival_perf_c == cam.frame_perf_c
        assert cam.frame_poll_perf_c <= cam.frame_arrival_perf_c
        assert time.perf_counter() >= cam.frame_arrival_perf_c
        if previous is not None:
            assert cam.frame_id == previous + 1  # on time: no frame skipped
        previous = cam.frame_id
    cam.end_capture()


def test_a_stall_the_buffer_holds_loses_no_frame(epoch):
    cam = _make_cam()
    cam.prepare_capture()
    cam.capture()
    first = cam.frame_id
    time.sleep(5.0 / FPS)
    for offset in range(1, 7):
        cam.capture()
        assert cam.frame_id == first + offset


def test_a_stall_past_the_buffer_leaves_a_frame_id_gap(epoch):
    cam = _make_cam()
    cam.prepare_capture()
    cam.capture()
    first = cam.frame_id
    time.sleep((HOST_BUFFER_FRAMES + 5.0) / FPS)
    cam.capture()
    assert cam.frame_id >= first + HOST_BUFFER_FRAMES + 4


def test_two_cameras_share_frame_ids_for_same_exposure(epoch):
    cams = [_make_cam("left"), _make_cam("right")]
    seen = [dict(), dict()]

    def run(index):
        cam = cams[index]
        cam.prepare_capture()
        for _ in range(15):
            _, stamp_ns = cam.capture()
            seen[index][stamp_ns] = cam.frame_id

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    common = set(seen[0]) & set(seen[1])
    assert len(common) >= 10
    for stamp_ns in common:
        assert seen[0][stamp_ns] == seen[1][stamp_ns]


def test_latch_clock_brackets_perf(epoch):
    cam = _make_cam()
    perf_before, camera_ns, perf_after, raw_before, raw_after = cam.latch_clock()
    assert perf_before <= camera_ns / 1e9 <= perf_after
    assert raw_before <= raw_after or numpy.isnan(raw_before)
    assert isinstance(camera_ns, int)


def test_is_trigger_synchronized(epoch):
    assert _make_cam().is_trigger_synchronized is True


def test_frames_differ_as_the_disc_moves(epoch):
    cam = _make_cam()
    cam.prepare_capture()
    first, _ = cam.capture()
    second, _ = cam.capture()
    assert not numpy.array_equal(first, second)
    assert first is not second


def test_latch_raw_bracket_is_the_simulated_clock_on_a_slewing_host(epoch, monkeypatch):
    # The simulated camera clock is perf_counter, so its RAW bracket is perf
    # reads too; a real RAW read would carry the host's perf-vs-RAW slew into
    # the RAW latch fit, which a real camera's oscillator does not have.
    from reachaq.video.camera import sim_cam

    monkeypatch.setattr(sim_cam, "monotonic_raw_time", lambda: time.perf_counter() * 0.998 + 5.0,
                        raising=False)
    cam = _make_cam()
    perf_before, camera_ns, perf_after, raw_before, raw_after = cam.latch_clock()
    assert raw_before <= camera_ns / 1e9 <= raw_after
    assert (raw_before, raw_after) == (perf_before, perf_after)
