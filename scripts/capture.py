import argparse
import logging
import sys
from queue import Queue

import cv2

from reachaq.core.project import ProjectInfo
from reachaq.video import VideoManager, VideoRecordProperties, VideoRecordMode
from reachaq.video import VideoRecord

logging.basicConfig(level=logging.INFO)
logging.getLogger('reachaq').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def capture_video(camera_url: str, frame_count: int, output_path: str, image_interval: int,
                  record_batch_size: int):
    logger.info(f"Camera URL: {camera_url}")
    logger.info(f"Output Path: {output_path}")

    camera = VideoManager.create_camera(camera_url, "capture_camera")
    if camera is None:
        raise RuntimeError(f"Could not create camera with url={camera_url!r}")

    camera.prepare_capture()

    if output_path:
        record_queue = Queue()
        record_project = ProjectInfo(root=output_path, device_id="CameraCapture", ensure_exists=True)
        record_properties = VideoRecordProperties(project_info=record_project, name="camera",
                                                  record_mode=VideoRecordMode.CONTINUOUS, video_rotate_interval=3600,
                                                  frame_size=(camera.width, camera.height), fps=camera.fps,
                                                  image_interval=image_interval)
        record = VideoRecord(record_properties, record_queue)

        record.start()
    else:
        record_queue = None
        record = None

    queue_list = list()
    queue_list_count = 0

    for idx in range(frame_count):
        image, when = camera.capture()

        if image is not None:
            cv2.imshow("window", image)
            cv2.waitKey(1)

            if record_queue is not None:
                queue_list.append((image, when))
                queue_list_count += 1
                if queue_list_count >= record_batch_size:
                    record_queue.put(queue_list)
                    queue_list = list()
                    queue_list_count = 0

    camera.end_capture()

    if record is not None:
        record.cancel()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("cameraurl", help="the camera to use")
    parser.add_argument("-f", "--frame-count", type=int, default=30,
                        help="the number of frames to capture (default 30)")
    parser.add_argument("-o", "--output", help="image and video output location if record is enabled")
    parser.add_argument("-i", "--image-interval", type=int, default=1,
                        help="the interval between image capture in seconds (default 1)")
    parser.add_argument("-b", "--record-batch-size", type=int, default=60,
                        help="record queue batch size (default 60)")

    args = parser.parse_args()

    capture_video(args.cameraurl, args.frame_count, args.output, args.image_interval,
                  args.record_batch_size)

    return True


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
