"""Stitch recorded trials into one demo clip per camera, with an event index.

Demo playback loops a single file per camera, which suits a long session
recording. Trials are the opposite shape: a few seconds each, hundreds of them.
Playing them one at a time would need a playlist in the capture process;
concatenating them needs nothing new, and gives the event navigation something
to navigate - each trial boundary is an event.

The reel is built by stream copy, so frames are preserved exactly and the
cumulative frame offsets written into the spec address the same frames the
model will see. That only holds while every input shares a codec and frame
rate, which is checked rather than assumed.

Left and right are concatenated in the same trial order, and trials whose two
cameras disagree on length are skipped: a reel where the cameras drift apart
would silently break the stereo pairing that 3D relies on, and one shared event
index could not address both.

    python scripts/stage_trial_reel.py ~/newtrials --count 20
    python scripts/stage_trial_reel.py ~/newtrials --count 20 --dry-run
    python scripts/stage_trial_reel.py ~/newtrials --trials trial011 trial014

Nothing is overwritten unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from autotrainer.core.configuration.demo_sources import (
    DEFAULT_DEMO_SOURCES_PATH,
    DemoSourcesError,
    load_demo_sources,
)

CAMERAS = ("left", "right")

#: One trial is one event in a reel, so the boundaries are trial starts rather
#: than the pellet deliveries a session recording carries. The UI words its
#: controls from whichever kind the clip has.
EVENT_KIND = "trial_start"


def _probe(path: Path) -> Dict[str, object]:
    """Frame count, rate and codec for one video."""
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frames <= 0:
        raise DemoSourcesError(f"{path} reports {frames} frames; is it readable?")
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=False).stdout
    try:
        name = json.loads(codec)["streams"][0]["codec_name"]
    except Exception:
        name = "unknown"
    return {"frames": frames, "fps": fps, "size": size, "codec": name}


def find_trials(root: Path, wanted: Optional[Sequence[str]]) -> List[Path]:
    """Trial directories holding both cameras, in name order."""
    found = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted and not any(name in directory.name for name in wanted):
            continue
        if all(list(directory.glob(f"*_{camera}.mp4")) for camera in CAMERAS):
            found.append(directory)
    return found


def _video(trial: Path, camera: str) -> Path:
    matches = sorted(trial.glob(f"*_{camera}.mp4"))
    if not matches:
        raise DemoSourcesError(f"{trial.name} has no {camera} video")
    return matches[0]


def plan_reel(trials: Sequence[Path]) -> Tuple[List[Path], List[int], Dict]:
    """Which trials go in, where each starts, and the shared properties.

    Returns the usable trials, the start frame of each, and the codec/fps/size
    they agree on. A trial whose cameras disagree on length is dropped with a
    reason rather than silently stretching one side.
    """
    usable: List[Path] = []
    starts: List[int] = []
    shared: Optional[Dict[str, object]] = None
    offset = 0
    for trial in trials:
        probes = {camera: _probe(_video(trial, camera)) for camera in CAMERAS}
        counts = {camera: probe["frames"] for camera, probe in probes.items()}
        if len(set(counts.values())) != 1:
            print(f"  skipping {trial.name}: cameras disagree on length {counts}")
            continue
        left = probes["left"]
        if shared is None:
            shared = {"fps": left["fps"], "size": left["size"], "codec": left["codec"]}
        elif (left["fps"], left["size"], left["codec"]) != (
                shared["fps"], shared["size"], shared["codec"]):
            print(f"  skipping {trial.name}: {left} differs from {shared}")
            continue
        usable.append(trial)
        starts.append(offset)
        offset += int(left["frames"])
    if not usable:
        raise DemoSourcesError("no usable trials found")
    return usable, starts, shared


def concatenate(trials: Sequence[Path], camera: str, destination: Path) -> None:
    """Stream-copy the trials into one file, preserving frames exactly."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for trial in trials:
            # ffconcat quoting: single quotes, doubled to escape.
            path = str(_video(trial, camera)).replace("'", "'\\''")
            listing.write(f"file '{path}'\n")
        listing_path = listing.name
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", listing_path, "-c", "copy", str(destination)],
            check=True)
    finally:
        Path(listing_path).unlink(missing_ok=True)


def write_spec(path: Path, videos: Dict[str, Path], fps: float,
               starts: Sequence[int], trials: Sequence[Path]) -> None:
    lines = [
        "# reachAQ demo playback sources.",
        "#",
        "# Generated by scripts/stage_trial_reel.py from recorded trials.",
        "# Regenerate rather than editing by hand, so the media, the frame",
        "# offsets and the spec stay in step.",
        "",
        f"fps: {fps}",
        "loop: true",
        "",
        "cameras:",
    ]
    for camera in CAMERAS:
        lines.append(f"  {camera}: {videos[camera]}")
    lines += [
        "",
        "# Where each trial starts in the reel, as a frame number. Read by",
        "# Tools -> Next/Previous in demo mode; one trial is one event.",
        "events:",
        f"  {EVENT_KIND}:",
    ]
    for start, trial in zip(starts, trials):
        lines.append(f"  - {start}  # {trial.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory holding trial folders")
    parser.add_argument("--count", type=int, default=20,
                        help="how many trials to include (default: 20)")
    parser.add_argument("--trials", nargs="*", default=None,
                        help="name fragments selecting specific trials")
    parser.add_argument("--dest", type=Path,
                        default=DEFAULT_DEMO_SOURCES_PATH.expanduser().parent)
    parser.add_argument("--spec", type=Path, default=None,
                        help="spec to write (default: <dest>/demo_sources.yaml)")
    parser.add_argument("--name", default="trial_reel")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    trials = find_trials(root, args.trials)
    if not trials:
        print(f"no trials with both cameras under {root}", file=sys.stderr)
        return 2
    if args.trials is None:
        # Spread across the run rather than taking the first N, so a reel shows
        # more than one animal and one day.
        step = max(1, len(trials) // args.count)
        trials = trials[::step][:args.count]

    print(f"planning a reel from {len(trials)} trials")
    usable, starts, shared = plan_reel(trials)
    total = starts[-1] + _probe(_video(usable[-1], "left"))["frames"]
    fps = float(shared["fps"])
    print(f"  {len(usable)} trials, {total} frames, {total / fps:.1f}s at {fps} fps")
    print(f"  codec={shared['codec']} size={shared['size']}")

    dest = args.dest.expanduser()
    spec_path = (args.spec or dest / "demo_sources.yaml").expanduser()
    videos = {camera: dest / f"{args.name}_{camera}.mp4" for camera in CAMERAS}

    if args.dry_run:
        print(f"  would write {spec_path} and {list(videos.values())}")
        for start, trial in zip(starts, usable):
            print(f"    frame {start:>7}  {trial.name}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    for camera, destination in videos.items():
        if destination.exists() and not args.force:
            print(f"refusing to overwrite {destination}; pass --force",
                  file=sys.stderr)
            return 1
    if spec_path.exists() and not args.force:
        print(f"refusing to overwrite {spec_path}; pass --force", file=sys.stderr)
        return 1
    if spec_path.exists():
        backup = spec_path.with_suffix(spec_path.suffix + ".bak")
        shutil.copy2(spec_path, backup)
        print(f"  previous spec kept at {backup}")

    for camera, destination in videos.items():
        print(f"  writing {destination}")
        concatenate(usable, camera, destination)

    # The reel must come back with the frames the offsets assume.
    for camera, destination in videos.items():
        got = _probe(destination)["frames"]
        if got != total:
            print(f"{destination} has {got} frames, expected {total}; "
                  f"the event offsets would not line up", file=sys.stderr)
            return 1

    write_spec(spec_path, videos, fps, starts, usable)
    # Validate through the loader the application uses, so a spec that would
    # fail at startup fails here instead.
    sources = load_demo_sources(spec_path)
    print(f"  wrote {spec_path}: {len(sources.event_frames())} events, "
          f"cameras {sorted(sources.enabled_camera_names())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
