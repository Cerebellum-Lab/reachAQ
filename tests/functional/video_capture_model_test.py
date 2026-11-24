import argparse
import logging
import time

import pytest

from reachaq.core.project import ProjectInfo
from reachaq.video import CaptureCameraAttrs
from reachaq.app.model.video_capture_model import VideoCaptureModel

logging.basicConfig(level=logging.INFO)
logging.getLogger('reachaq').setLevel(logging.DEBUG)
logging.getLogger('tools').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def camera_url():
    return CaptureCameraAttrs(name="Random", url="random://0")


@pytest.fixture
def iterations():
    return 1


@pytest.fixture
def capture_duration():
    return 2


@pytest.fixture
def output_location():
    return None


def test_video_capture_model(camera_url: CaptureCameraAttrs, iterations: int, capture_duration: int, output_location: str):
    assert camera_url != ""

    model = VideoCaptureModel("camera-1")
    model.camera_source = camera_url
    model.set_display_fcn(lambda x, y: None)

    count = 0

    while count < iterations:
        logger.info(f"video capture model test iteration {count + 1} of {iterations}")

        project_info = ProjectInfo(root=output_location)

        assert model.on_prepare_capture(project_info, None)

        model.on_capture_start()

        time.sleep(capture_duration)

        model.on_capture_notify_end()

        model.on_capture_stop()

        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--camera", help="camera 1", default="random://0")
    parser.add_argument("-i", "--iterations", help="the number of process iterations (default 2)", type=int,
                        default=2)
    parser.add_argument("-d", "--duration", help="the capture duration in seconds for each iteration (default 2)",
                        type=int, default=2)
    parser.add_argument("-o", "--output", help="the output location for recorded files")

    args = parser.parse_args()

    source = CaptureCameraAttrs("camera-1", args.camera)

    test_video_capture_model(source, args.iterations, args.duration, args.output)
