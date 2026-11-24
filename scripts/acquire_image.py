import argparse
import sys

import cv2

from reachaq.video import VideoManager


def acquire_image(camera_url: str):
    print("Camera URL:", camera_url)

    camera = VideoManager.create_camera(camera_url)

    camera.prepare_capture()

    frame, when = camera.capture()

    camera.end_capture()

    print(f"image captured: frame_when={when} frame={frame}")

    if frame is not None:
        window_name = "capture"
        cv2.imshow(window_name, frame)
        while True:
            # Wait for 25ms for a key press
            key = cv2.waitKey(25) & 0xFF

            # Check if the 'q' key was pressed (ASCII 113)
            if key == ord('q'):
                break  # Exit the loop
            # Check if the ESC key was pressed (ASCII 27)
            if key == 27:
                break
            # Check if the window was closed by the user's mouse click
            # getWindowProperty returns -1 if the window is closed.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <= 0:
                break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("cameraurl", help="the camera to use")

    args = parser.parse_args()
    acquire_image(args.cameraurl)

    return True


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
