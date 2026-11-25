import argparse
import logging
import time
from typing import List

import pytest

from reachaq.core.project import ProjectInfo
from reachaq.video import CaptureCameraAttrs
from reachaq.app.model.video_capture_model import VideoCaptureModel

logging.basicConfig(level=logging.INFO)
logging.getLogger('reachaq').setLevel(logging.DEBUG)
logging.getLogger('reachaq.app').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


@pytest.fixture
def camera_url_1():
    return CaptureCameraAttrs(name="Left", url="random://0")


@pytest.fixture
def camera_url_2():
    return CaptureCameraAttrs(name="Right", url="random://0")


@pytest.fixture
def camera_url_3():
    return CaptureCameraAttrs(name="Web", url="random://0")


@pytest.fixture
def iterations():
    return 1


@pytest.fixture
def duration():
    return 2


@pytest.fixture
def output_location():
    return None


def test_video_capture_model(camera_url_1: CaptureCameraAttrs, camera_url_2: CaptureCameraAttrs,
                             camera_url_3: CaptureCameraAttrs, iterations: int, duration: int,
                             output_location: str):
    models: List[VideoCaptureModel] = []

    if camera_url_1 != "":
        model_1 = VideoCaptureModel("1")
        model_1.camera_source = camera_url_1
        model_1.set_display_fcn(lambda x, y: None)
        models.append(model_1)

    if camera_url_2 != "":
        model_2 = VideoCaptureModel("2")
        model_2.camera_source = camera_url_2
        model_2.set_display_fcn(lambda x, y: None)
        models.append(model_2)

    if camera_url_3 != "":
        model_3 = VideoCaptureModel("3")
        model_3.camera_source = camera_url_3
        model_3.set_display_fcn(lambda x, y: None)
        models.append(model_3)

    count = 0

    project_info = ProjectInfo(root=output_location)

    while count < iterations:
        logger.info(f"video capture process starting (iteration {count + 1})")

        success = True

        for model in models:
            res = model.on_prepare_capture(project_info, None)

            if not res:
                success = False
                logger.error(f"video capture model failed to prepare: {model.camera_source}")
                break

        if success is True:
            for model in models:
                model.on_capture_start()

            time.sleep(duration)

        for model in models:
            model.on_capture_stop()

        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--left", help="camera 1", default="random://0")
    parser.add_argument("-r", "--right", help="camera 2", default="random://0")
    parser.add_argument("-t", "--top", help="camera 3", default="random://0")
    parser.add_argument("-i", "--iterations", help="the number of process iterations (default 2)", type=int,
                        default=2)
    parser.add_argument("-d", "--duration", help="the capture duration in seconds for each iteration (default 2)",
                        type=int, default=2)
    parser.add_argument("-o", "--output", help="the output location for recorded files")

    args = parser.parse_args()

    camera_url_1 = CaptureCameraAttrs(name="Left", url=args.left)
    camera_url_2 = CaptureCameraAttrs(name="Right", url=args.right)
    camera_url_3 = CaptureCameraAttrs(name="Web", url=args.top)

    test_video_capture_model(camera_url_1, camera_url_2, camera_url_3, args.iterations, args.duration, args.output)
