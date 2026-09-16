"""Check that a demo will actually work on this rig, before the meeting.

Demo mode substitutes recorded video for the cameras. Two things can be wrong in
ways that do not crash and do not announce themselves:

1. The configured pose model may not fire on the chosen video. The keypoint
   confidence gate is calibrated per backbone, and an unmeasured or mismatched
   backbone can hold every hand flag False. The demo then runs, shows video, and
   silently never closes the loop.
2. The rig calibration may not match the geometry the video was recorded under.
   Calibration loading degrades gracefully when absent, so a mismatch produces
   wrong 3D underneath a correct-looking 2D overlay.

This turns both into a pass/fail check that can be run once on the rig rather
than discovered during the demo.

Run on the rig, in the reachAQ environment:

    python scripts/verify_demo_readiness.py
    python scripts/verify_demo_readiness.py --frames 600
    python scripts/verify_demo_readiness.py --skip-pose     # geometry only, no GPU

Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.configuration.demo_sources import (
    DEFAULT_DEMO_SOURCES_PATH,
    DemoSources,
    DemoSourcesError,
    load_demo_sources,
)
from autotrainer.core.pose_elements import SceneElement

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_demo_readiness")

DEFAULT_CALIB_PATH = Path("~/Autotrainer") / DEFAULT_3D_CALIB_DIR_NAME
DEFAULT_CONFIG_PATH = Path("~/Autotrainer/system_configuration.yaml")

#: The parts the cue gate actually depends on. A demo whose model never puts
#: these above threshold will run, and will never close the loop.
GATE_PARTS = (SceneElement.RH_grab, SceneElement.LH_grab)

#: Fraction of frames a gate part must clear the threshold in for the demo to be
#: worth showing. Not a correctness bound, a "will the audience see anything" bound.
MIN_GATE_DETECTION_RATE = 0.05

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


class Report:
    """Collects check results so every check runs before anything exits."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str) -> None:
        self.rows.append((status, check, detail))
        print(f"[{status}] {check}: {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status in (row[0] for row in self.rows))

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for status, _, _ in self.rows:
            counts[status] = counts.get(status, 0) + 1
        return "  ".join(f"{status}={counts[status]}" for status in sorted(counts))


# ----------------------------------------------------------------- video


def video_geometry(path: Path) -> Optional[Dict[str, float]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def check_videos(report: Report, sources: DemoSources) -> Dict[str, Dict[str, float]]:
    geometries: Dict[str, Dict[str, float]] = {}
    for name, video in sorted(sources.cameras.items()):
        geometry = video_geometry(video)
        if geometry is None:
            report.add(FAIL, f"video {name}", f"could not be opened: {video}")
            continue
        geometries[name] = geometry
        report.add(
            PASS, f"video {name}",
            f"{geometry['width']}x{geometry['height']} "
            f"@{geometry['fps']:.1f}fps, {geometry['frames']} frames "
            f"({geometry['frames'] / max(geometry['fps'], 1):.1f}s)",
        )

    rates = {geometry["fps"] for geometry in geometries.values()}
    if len(rates) > 1:
        report.add(WARN, "video frame rates",
                   f"sources disagree: {sorted(rates)}; playback paces each camera "
                   "to the spec fps, so mismatched sources will drift apart")

    if geometries:
        spec_fps = float(sources.fps)
        native = next(iter(geometries.values()))["fps"]
        if abs(spec_fps - native) > 1.0:
            report.add(
                WARN, "spec fps",
                f"spec says {spec_fps:g} but the video is {native:.1f}; playback uses "
                "the spec value, so live inference will see a different load than the rig",
            )
        else:
            report.add(PASS, "spec fps", f"{spec_fps:g} matches the source video")

    _check_lengths(report, geometries)
    return geometries


def _check_lengths(report: Report, geometries: Dict[str, Dict[str, float]]) -> None:
    """Length mismatch matters differently for the triangulated pair than elsewhere.

    The start barrier aligns the first frame, but nothing re-aligns a loop. When
    the reach pair differ in length the shorter one restarts while the other runs
    on, so triangulation keeps pairing frames that are aligned in time and no
    longer show the same moment. That is a break, not a nuisance.

    Any other camera looping separately is cosmetic: the stim camera is never
    triangulated.
    """
    pair = {name: geometries[name]["frames"]
            for name in ("left", "right") if name in geometries}
    if len(set(pair.values())) > 1:
        report.add(
            FAIL, "reach pair length",
            f"left and right differ in length {pair}. The start barrier aligns the "
            "first frame but nothing re-aligns a loop, so after the shorter one "
            "restarts, triangulation pairs frames that no longer show the same moment.",
        )
    elif pair:
        report.add(PASS, "reach pair length",
                   f"left and right are both {next(iter(pair.values()))} frames, "
                   "so they loop together")

    others = {name: int(geometry["frames"]) for name, geometry in geometries.items()
              if name not in ("left", "right")}
    if others and set(others.values()) - set(pair.values()):
        report.add(WARN, "other camera length",
                   f"{others} differs from the reach pair; that camera loops at a "
                   "different point. Cosmetic: it is not triangulated.")


# ----------------------------------------------------------------- calibration


def check_calibration(
    report: Report,
    calib_dir: Path,
    geometries: Dict[str, Dict[str, float]],
    *,
    live_2d_only: bool = False,
) -> None:
    calib_dir = calib_dir.expanduser()
    if not calib_dir.is_dir():
        report.add(FAIL, "calibration", f"not a directory: {calib_dir}")
        return

    params_path = calib_dir / "camera_matrix" / "stereo_params.pickle"
    if not params_path.is_file():
        report.add(FAIL, "calibration", f"missing stereo params: {params_path}")
        return

    try:
        with params_path.open("rb") as handle:
            outer = pickle.load(handle)
        key = next(iter(outer))
        inner = outer[key]
    except Exception as err:
        report.add(FAIL, "calibration", f"could not read {params_path}: {err}")
        return

    report.add(PASS, "calibration source", f"built from {key}")

    image_shape = inner.get("image_shape")
    if not image_shape:
        report.add(WARN, "calibration geometry",
                   "stereo params carry no image_shape; cannot compare against the video")
        return

    # image_shape is [(w, h), (w, h)] for the left and right cameras. Confirmed
    # against a real calibration: 1440x1080 source video at the 4x oversample in
    # the directory name stores (360, 270), which is width-first.
    calib_dims = [tuple(int(v) for v in pair) for pair in image_shape]
    report.add(PASS, "calibration geometry",
               f"image_shape={calib_dims} (width, height)")

    ordered = [geometries.get("left"), geometries.get("right")]
    for index, (label, geometry) in enumerate(zip(("left", "right"), ordered)):
        if geometry is None:
            continue
        if index >= len(calib_dims):
            continue
        calib_w, calib_h = calib_dims[index]
        video_w, video_h = int(geometry["width"]), int(geometry["height"])
        if (calib_w, calib_h) == (video_w, video_h):
            report.add(PASS, f"calibration vs {label}",
                       f"{video_w}x{video_h} matches the calibration")
        else:
            # Calibration only feeds triangulation. A per-view 2D overlay never
            # touches it, so a mismatch is fatal to 3D and irrelevant to 2D.
            report.add(
                WARN if live_2d_only else FAIL, f"calibration vs {label}",
                f"video is {video_w}x{video_h} but calibration was built at "
                f"{calib_w}x{calib_h}. The intrinsics are in pixels of a different "
                "frame, so 3D would be wrong. Live 2D per view is unaffected."
                + (" Ignored: this run is live-2D only." if live_2d_only else ""),
            )


# ----------------------------------------------------------------- pose model


def resolve_model_location(report: Report, explicit: Optional[str],
                           config_path: Path) -> Optional[str]:
    if explicit:
        return explicit
    config_path = config_path.expanduser()
    if not config_path.is_file():
        report.add(WARN, "pose model", f"no system configuration at {config_path}; "
                                       "pass --model to name the model explicitly")
        return None
    try:
        from autotrainer.core.configuration.system_configuration import SystemConfiguration
        configuration = SystemConfiguration.load_yaml_file(config_path)
        return configuration.inference.pose_model_location
    except Exception as err:
        report.add(WARN, "pose model", f"could not read {config_path}: {err}")
        return None


def check_pose(
    report: Report,
    sources: DemoSources,
    model_location: Optional[str],
    frame_limit: int,
) -> None:
    from autotrainer.inference.backend_selection import (
        build_pose_model,
        confidence_threshold,
        selected_backend,
        trained_model_name,
    )

    backend = selected_backend()
    report.add(PASS, "pose backend", backend)

    if not model_location:
        report.add(FAIL, "pose model",
                   "no model configured; live inference would run the in-memory "
                   "rig-checkout model, which produces synthetic pose, not real detections")
        return

    model_name = None
    try:
        model_name = trained_model_name(model_location)
    except Exception as err:
        logger.debug("could not read the trained model name: %s", err)
    threshold = confidence_threshold(backend, model_name=model_name)
    report.add(PASS, "confidence threshold",
               f"{threshold:.2f} for backend={backend} model={model_name or 'unmeasured'}")

    try:
        model = build_pose_model(model_location, backend=backend)
        if not model.is_valid():
            report.add(FAIL, "pose model", f"{model_location} did not validate for {backend}")
            return
        model.load()
    except Exception as err:
        report.add(FAIL, "pose model", f"{model_location} could not be loaded: {err}")
        return

    parts = list(model.body_parts)
    report.add(PASS, "pose model", f"{model_location} loaded, {len(parts)} body parts")

    missing = [part for part in GATE_PARTS if part not in parts]
    if missing:
        report.add(FAIL, "gate parts",
                   f"the model has no {missing}; the cue gate can never be answered "
                   f"from it. Model parts: {parts}")
        return

    # Every camera, not just the primary. The live overlay is drawn per view, so a
    # camera the model cannot read is a dead panel on screen regardless of how well
    # its partner does - and on a real rig the two views can differ enormously.
    any_camera_detects = False
    gate_cameras = []
    for name, video in sorted(sources.cameras.items()):
        above, total, best = _score_video(model, video, parts, threshold, frame_limit)
        if total == 0:
            report.add(FAIL, f"pose {name}", f"no frames could be read from {video}")
            continue

        print(f"      {name}:")
        for part in parts:
            rate = above.get(part, 0) / total
            marker = "  <- gate part" if part in GATE_PARTS else ""
            print(f"        {part:<14} {rate * 100:5.1f}% of {total} frames"
                  f"   best={best.get(part, 0.0):.3f}{marker}")

        detected = [part for part in parts if above.get(part, 0) > 0]
        gate_hits = max(above.get(part, 0) for part in GATE_PARTS)
        if not detected:
            report.add(
                FAIL, f"pose {name}",
                f"no body part cleared {threshold:.2f} in any of {total} sampled "
                f"frames (best seen {max(best.values(), default=0.0):.3f}). This view "
                "is not footage the model recognises; its live overlay will be noise.",
            )
            continue

        any_camera_detects = True
        if gate_hits:
            gate_cameras.append(name)
        report.add(PASS, f"pose {name}",
                   f"{len(detected)}/{len(parts)} parts detected, hand gate in "
                   f"{gate_hits / total * 100:.1f}% of {total} sampled frames")

    if not any_camera_detects:
        report.add(FAIL, "pose detection",
                   "no camera view is readable by this model; pick a different session")
        return

    # Live 2D is drawn per view, so one good camera is a usable demo. Live 3D is
    # not: triangulation needs confident keypoints on both views of the same frame.
    if len(gate_cameras) >= 2:
        report.add(PASS, "gate detection",
                   f"hand gate detected on {', '.join(gate_cameras)} - 2D overlay and "
                   "3D triangulation both have what they need")
    elif gate_cameras:
        report.add(
            WARN, "gate detection",
            f"hand gate detected on {gate_cameras[0]} only. Live 2D on that view is "
            "fine, which is what a per-view overlay demo needs. Live 3D is not: "
            "triangulation needs confident keypoints on both views of the same frame.",
        )
    else:
        report.add(
            WARN, "gate detection",
            f"no view put a gate part ({', '.join(GATE_PARTS)}) above {threshold:.2f}. "
            "Reaches are sparse, so this may be sampling - but confirm the clip "
            "contains reaches or the hand overlay will stay empty.",
        )


def _score_video(model, video: Path, parts: List[str], threshold: float,
                 frame_limit: int) -> Tuple[Dict[str, int], int, Dict[str, float]]:
    """Per body part: how often it clears the gate, and its best confidence.

    Samples across the whole clip rather than reading consecutively from the
    start. A run of consecutive frames covers a second or two of a ten-minute
    recording, and reaches are sparse, so a consecutive sample reports zero for
    the hand parts almost regardless of the video.
    """
    above: Dict[str, int] = {part: 0 for part in parts}
    best: Dict[str, float] = {part: 0.0 for part in parts}
    capture = cv2.VideoCapture(str(video))
    total = 0
    try:
        if not capture.isOpened():
            return above, 0, best
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return above, 0, best
        wanted = min(frame_limit, frame_count)
        for index in range(wanted):
            position = int(frame_count * (index + 0.5) / wanted)
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                continue
            try:
                pose = numpy.asarray(model.predict(numpy.expand_dims(frame, axis=0)))
            except Exception as err:
                logger.error("predict failed at frame %s: %s", position, err)
                break
            # (x, y, confidence) per part, in body_parts order.
            values = pose.reshape(-1, 3)
            for part_index, part in enumerate(parts):
                if part_index >= len(values):
                    continue
                confidence = float(values[part_index, 2])
                best[part] = max(best[part], confidence)
                if confidence >= threshold:
                    above[part] += 1
            total += 1
            if total % 50 == 0:
                print(f"        ... {total}/{wanted} sampled", flush=True)
    finally:
        capture.release()
    return above, total, best


# ----------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sources", type=Path, default=DEFAULT_DEMO_SOURCES_PATH,
                        help="demo source spec (default: %(default)s)")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIB_PATH,
                        help="3D calibration directory (default: %(default)s)")
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="system configuration, read for the pose model location "
                             "(default: %(default)s)")
    parser.add_argument("--model", default=None,
                        help="pose model location, overriding the configuration")
    parser.add_argument("--frames", type=int, default=300,
                        help="frames to score for detection rate (default: %(default)s)")
    parser.add_argument("--skip-pose", action="store_true",
                        help="check videos and calibration only; no GPU needed")
    parser.add_argument("--live-2d-only", action="store_true",
                        help="the demo shows a per-view 2D overlay and no 3D, so a "
                             "calibration geometry mismatch warns instead of failing")
    args = parser.parse_args()

    report = Report()

    print("=== demo readiness ===\n")

    try:
        sources = load_demo_sources(args.sources)
    except DemoSourcesError as err:
        report.add(FAIL, "demo sources", str(err))
        print(f"\n{report.summary()}")
        return 1
    report.add(PASS, "demo sources",
               f"{args.sources} -> {sorted(sources.enabled_camera_names())}")

    geometries = check_videos(report, sources)
    check_calibration(report, args.calibration, geometries,
                      live_2d_only=args.live_2d_only)

    if args.skip_pose:
        report.add(SKIP, "pose model", "--skip-pose was given")
    else:
        try:
            check_pose(report, sources, resolve_model_location(
                report, args.model, args.configuration), args.frames)
        except Exception as err:
            report.add(FAIL, "pose model", f"check could not run: {err}")

    print(f"\n{report.summary()}")
    if report.failed:
        print("\nNot ready. Fix the FAIL rows before demoing.")
        return 1
    print("\nReady.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
