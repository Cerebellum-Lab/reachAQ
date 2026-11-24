import argparse
import logging
import time
from multiprocessing import Queue, Value

import pytest

from reachaq.video import VideoCapture, CaptureCommandKind, CaptureCameraAttrs, CaptureAttrs
from reachaq.core.capture import CaptureProcessStatus
from reachaq.core import clear_queue

logging.basicConfig(level=logging.INFO)
logging.getLogger('reachaq').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def camera_url():
    return "random://0"


@pytest.fixture
def iterations():
    return 1


@pytest.fixture
def capture_duration():
    return 2


def wait_for_status(status: Value, expected: CaptureProcessStatus, timeout: int):
    start_ns = time.perf_counter_ns()
    elapsed = 0

    while status.value != expected and elapsed < timeout:
        time.sleep(0.05)
        elapsed = (time.perf_counter_ns() - start_ns) / 1e9

    return elapsed <= timeout


@pytest.mark.functional
def test_video_capture_process(camera_url: str, iterations: int, capture_duration: int):
    cmd_queue = Queue()
    image_queue = Queue()
    status = Value("i", CaptureProcessStatus.UNKNOWN)
    frame = Value("i", 0)

    count = 0

    while count < iterations:
        logger.info(f"video capture process test iteration {count + 1} of {iterations}")

        camera = CaptureCameraAttrs(name="test", url=camera_url)

        attrs = CaptureAttrs(command_queue=cmd_queue, status=status, image_queue=image_queue, camera=camera,
                             frame=frame, errors=None)

        process = VideoCapture(attrs)

        assert status.value == CaptureProcessStatus.INITIALIZED

        process.start()

        assert wait_for_status(status, CaptureProcessStatus.RUNNING, timeout=4)

        cmd_queue.put((CaptureCommandKind.ENABLE_CAPTURE, None))

        time.sleep(capture_duration)

        cmd_queue.put((CaptureCommandKind.DISABLE_CAPTURE, None))

        time.sleep(0.25)

        cmd_queue.put((CaptureCommandKind.TERMINATE, None))

        assert wait_for_status(status, CaptureProcessStatus.TERMINATED, timeout=4)

        clear_queue(image_queue)

        time.sleep(0.25)

        while process.is_alive():
            logger.warning("still alive...")
            time.sleep(0.5)

        logger.info("process fully terminated")

        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("cameraurl", help="the camera to use")
    parser.add_argument("-i", "--iterations", help="the number of process iterations (default 10)", type=int,
                        default=2)
    parser.add_argument("-d", "--duration", help="the capture duration in seconds for each iteration (default 2)",
                        type=int, default=2)

    args = parser.parse_args()

    test_video_capture_process(args.cameraurl, args.iterations, args.duration)
