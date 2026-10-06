import math

from autotrainer.video.camera.random_cam import RandomCam


def test_base_capture_reports_its_own_return_as_arrival():
    camera = RandomCam("random")
    camera.width, camera.height, camera.fps = 8, 6, 1000
    camera.prepare_capture()
    camera.capture()

    assert math.isfinite(camera.frame_arrival_perf_c)
    assert camera.frame_poll_perf_c == camera.frame_arrival_perf_c == camera.frame_perf_c
