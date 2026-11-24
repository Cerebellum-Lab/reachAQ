import logging
import sys
import os
import argparse
import time

from cProfile import Profile
from pstats import SortKey, Stats

import cv2
import numpy

from reachaq.pose_plugins.deeplabcut import DlcPoseModel
from numpy import fromfile

logging.basicConfig(level=logging.INFO)
logging.getLogger("root").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def prepare_video_capture(file_name):
    if os.path.isfile(file_name):
        capture = cv2.VideoCapture(file_name)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return capture, frame_count

    return None, 0


def process_video(network, front_source, side_source, batch_size: int, user_max_frames: int, report_profile: bool,
                  output_file_name: str, input_file_name: str, width: int, height: int):
    front_capture, front_frame_count = prepare_video_capture(front_source)

    side_capture, side_frame_count = prepare_video_capture(side_source)

    max_frames = int(min(front_frame_count, side_frame_count))

    configuration = DlcPoseModel(network, 1, 0, batch_size * 2)

    configuration.load()

    idx = 0

    frames = numpy.ndarray((batch_size * 2, width, height, 3))

    frame_count = 0

    if user_max_frames > 0:
        max_frames = min(max_frames, user_max_frames)

    output_file = None

    if output_file_name is not None and len(output_file_name) > 0:
        output_file = open(output_file_name, "wb")

    reference = None
    reference_index = 0

    if input_file_name is not None:
        input_file = open(input_file_name, "rb")
        reference = fromfile(input_file, dtype=numpy.float64)
        reference = numpy.array_split(reference, int(len(reference) / (batch_size * 2 * 30)))

    start_time = time.perf_counter()

    with Profile() as profile:
        while True:
            for fdx in range(batch_size):
                if idx >= max_frames:
                    break

                ret1, front_frame = front_capture.read()
                ret2, side_frame = side_capture.read()

                frames[fdx * 2, :, :, :] = front_frame
                frames[fdx * 2 + 1, :, :, :] = side_frame

                idx += 1

            pose = configuration.predict(frames)
            pose = numpy.asarray(pose)

            if output_file is not None:
                output_file.write(pose.tobytes())

            if reference is not None:
                current = reference[reference_index].reshape(batch_size * 2, 30)
                if not numpy.array_equal(pose, current):
                    for pdx in range(batch_size * 2):
                        pose_p = pose[pdx, :].reshape(10, 3)
                        current_p = current[pdx, :].reshape(10, 3)
                        dist_p = numpy.round(numpy.linalg.norm(pose_p[0, 0:1] - current_p[0, 0:1]))
                        if dist_p > 0:
                            logger.error(f"batch {reference_index} frame {pdx} dist: {dist_p}")
                reference_index += 1

            frame_count += batch_size

            if frame_count % 100 == 0:
                print(f"{(frame_count / (time.perf_counter() - start_time)):.1f} Network FPS")

            if idx >= max_frames:
                break

        if output_file is not None:
            output_file.close()

        if report_profile:
            Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("pellet", help="the DeepLabCut pellet to use")
    parser.add_argument("front", help="the front camera video source file")
    parser.add_argument("side", help="the side camera video source file")
    parser.add_argument("-n", "--network", help="the DLC model dir")
    parser.add_argument("-b", "--batchsize", help="the batch size for DLC", type=int, default=5)
    parser.add_argument("-f", "--frames", help="the number of frames to process", type=int, default=-1)
    parser.add_argument("-p", "--profile", help="report profiling data", type=bool, default=False)
    parser.add_argument("-o", "--output", help="save pose output to specified file")
    parser.add_argument("-i", "--input", help="read pose output from specified file for comparison")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)

    args = parser.parse_args()

    if not os.path.exists(args.front):
        logger.error("The front/left camera video file does not exist")

    if not os.path.exists(args.side):
        logger.error("The side/right camera video file does not exist")

    if args.output is not None:
        logger.info("Saving pose output.  Performance values less accurate than usual")

    process_video(args.network, args.front, args.side, args.batchsize, args.frames, args.profile, args.output,
                  args.input, args.width, args.height)

    return True


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
