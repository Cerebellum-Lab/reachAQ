"""Generate presentation assets.

Diagrams are drawn rather than hand-made so the deck can be rebuilt. Stills are
extracted from material already on disk under temp/, which is untracked: the
generator degrades to a labelled placeholder when a source is absent, so the deck
always builds.

Run from the repository root:
    python docs/presentation/make_assets.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"

#: Where the session video lives. It is untracked, so a git worktree will not have
#: it even though the main checkout does. Override with --media-root or
#: REACHAQ_MEDIA_ROOT to point at whichever checkout actually holds the media.
MEDIA_ROOT = Path(os.environ.get("REACHAQ_MEDIA_ROOT") or (REPO_ROOT / "temp"))

REACH = "#6b7f9e"
AUTO = "#4a8c7a"
NEW = "#b3593f"
INK = "#26282b"
MUTED = "#7a7f87"


def _fig(width: float, height: float):
    figure, axes = plt.subplots(figsize=(width, height), dpi=200)
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100 * height / width)
    axes.axis("off")
    figure.patch.set_facecolor("white")
    return figure, axes


def _box(axes, x, y, w, h, label, color, text_color="white", size=9):
    axes.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        facecolor=color, edgecolor="none",
    ))
    axes.text(x + w / 2, y + h / 2, label, ha="center", va="center",
              color=text_color, fontsize=size, weight="bold")


def _arrow(axes, start, end, color=MUTED):
    axes.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=1.6, shrinkA=2, shrinkB=2,
    ))


def _save(figure, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def lineage():
    figure, axes = _fig(12, 5)
    stages = [
        (4, REACH, "reach-training", [
            "PySpin multi-camera capture",
            "Arduino serial pellet control",
            "ROI threshold reach detection",
            "Offline DeepLabCut + curation",
        ]),
        (36, AUTO, "auto-trainer", [
            "Modular package split",
            "Synchronized VideoCapture",
            "Pose inference + 3D calibration",
            "CAN device, sessions, PySide6",
        ]),
        (68, NEW, "reachAQ", [
            "Dell/Ubuntu, SocketCAN/PCAN",
            "NI-DAQ laser + signal streaming",
            "Live 3D pose closes the cue gate",
            "Protocols, epochs, validation",
        ]),
    ]
    for x, color, title, bullets in stages:
        _box(axes, x, 26, 28, 9, title, color, size=13)
        for index, bullet in enumerate(bullets):
            axes.text(x + 1, 22 - index * 4.2, f"— {bullet}", ha="left", va="center",
                      color=INK, fontsize=8.5)
    _arrow(axes, (32.5, 30.5), (35.5, 30.5))
    _arrow(axes, (64.5, 30.5), (67.5, 30.5))
    axes.text(50, 39, "What each layer contributed", ha="center",
              color=MUTED, fontsize=10, style="italic")
    _save(figure, "lineage.png")


def pipeline():
    figure, axes = _fig(12, 4.5)
    stages = [
        (2, "Cameras\n150 FPS", AUTO),
        (18, "VideoCapture\nper-process", AUTO),
        (34, "Pose\ninference", AUTO),
        (50, "3D\ntriangulation", NEW),
        (66, "Behavior\nstate machine", AUTO),
        (82, "Pellet / Laser\noutput", NEW),
    ]
    for x, label, color in stages:
        _box(axes, x, 14, 14, 11, label, color, size=9)
    for index in range(len(stages) - 1):
        _arrow(axes, (stages[index][0] + 14.5, 19.5), (stages[index + 1][0] - 0.5, 19.5))
    axes.text(50, 31, "Acquisition pipeline", ha="center", color=INK,
              fontsize=12, weight="bold")
    axes.text(50, 8, "Green: inherited from auto-trainer     Red: new in reachAQ",
              ha="center", color=MUTED, fontsize=8.5)
    _save(figure, "pipeline.png")


def two_tier():
    figure, axes = _fig(12, 5)
    axes.text(50, 39, "Two-tier timing architecture", ha="center", color=INK,
              fontsize=12, weight="bold")

    _box(axes, 2, 26, 22, 8, "Tier 1 — stim loop", NEW, size=10)
    axes.text(2, 22, "stimCam at 900 Hz on a small ROI", color=INK, fontsize=8.5)
    axes.text(2, 18.5, "decision inline in the capture thread", color=INK, fontsize=8.5)
    axes.text(2, 15, "pre-armed hardware-timed NI-DAQ output", color=INK, fontsize=8.5)
    axes.text(2, 11.5, "no process queue may be crossed", color=INK, fontsize=8.5)
    axes.text(2, 7, "target 5 ms p99", color=NEW, fontsize=10, weight="bold")

    _box(axes, 52, 26, 22, 8, "Tier 2 — behavior loop", AUTO, size=10)
    axes.text(52, 22, "reach cameras at 150 FPS", color=INK, fontsize=8.5)
    axes.text(52, 18.5, "pose inference and 3D triangulation", color=INK, fontsize=8.5)
    axes.text(52, 15, "behavior state machine", color=INK, fontsize=8.5)
    axes.text(52, 11.5, "trial, pellet, and protocol decisions", color=INK, fontsize=8.5)
    axes.text(52, 7, "20-100 ms", color=AUTO, fontsize=10, weight="bold")

    axes.text(50, 2, "The 5 ms figure is a design target, not a measured result.",
              ha="center", color=MUTED, fontsize=8.5, style="italic")
    _save(figure, "two_tier.png")


def demo_substitution():
    figure, axes = _fig(12, 4.5)
    stages = [
        (2, "Recorded\nvideo", NEW),
        (18, "VideoCapture\nper-process", AUTO),
        (34, "Pose\ninference", AUTO),
        (50, "3D\ntriangulation", AUTO),
        (66, "Behavior\nstate machine", AUTO),
        (82, "Pellet / Laser\noutput", AUTO),
    ]
    for x, label, color in stages:
        _box(axes, x, 14, 14, 11, label, color, size=9)
    for index in range(len(stages) - 1):
        _arrow(axes, (stages[index][0] + 14.5, 19.5), (stages[index + 1][0] - 0.5, 19.5))
    axes.text(50, 31, "Demo mode substitutes exactly one stage", ha="center",
              color=INK, fontsize=12, weight="bold")
    axes.text(9, 8, "substituted", ha="center", color=NEW, fontsize=9, weight="bold")
    axes.text(58, 8, "unchanged, live, real", ha="center", color=AUTO,
              fontsize=9, weight="bold")
    _save(figure, "demo_substitution.png")


def _first_frame(video: Path):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ok, frame = capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _placeholder(axes, message):
    axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes,
              color=MUTED, fontsize=9, style="italic", wrap=True)
    axes.set_xticks([])
    axes.set_yticks([])


def _stills(sources, out_name, title, captions):
    figure, axes_list = plt.subplots(1, len(sources), figsize=(11, 4.6), dpi=200)
    if len(sources) == 1:
        axes_list = [axes_list]
    for axes, source, caption in zip(axes_list, sources, captions):
        frame = _first_frame(source) if source.is_file() else None
        if frame is None:
            _placeholder(axes, f"{caption}\n\nsource not on this machine:\n{source.name}")
        else:
            axes.imshow(frame, cmap="gray")
            axes.set_xticks([])
            axes.set_yticks([])
        axes.set_title(caption, fontsize=10, color=INK)
        for spine in axes.spines.values():
            spine.set_edgecolor("#d8dade")
    figure.suptitle(title, fontsize=12, color=INK, weight="bold")
    figure.patch.set_facecolor("white")
    _save(figure, out_name)


def demo_frames():
    session = MEDIA_ROOT / "2p_sessions/20260914_christie2P_session001"
    _stills(
        [
            session / "20260914_christie2P_session001_left-0000.mp4",
            session / "20260914_christie2P_session001_right-0000.mp4",
        ],
        "demo_frames.png",
        "Demo source: a real recorded session, 150 FPS",
        ["left camera", "right camera"],
    )


def pose_backbones():
    """Real keypoint output from four candidate backbones on one frame.

    Each panel is a 2x2 grid of cspnext_s, cspnext_m, yolo11n and yolov8n with
    the count of keypoints above threshold. This is the evidence behind the
    per-backbone confidence calibration, so label it as the comparison it is
    rather than as a generic overlay.
    """
    overlays = MEDIA_ROOT / "overlays"
    _stills(
        [overlays / "compare_left.mp4", overlays / "compare_right.mp4"],
        "pose_backbones.png",
        "Live pose keypoints: four backbones on the same frame",
        ["left camera", "right camera"],
    )


def main():
    global MEDIA_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="directory holding the untracked session video (default: %(default)s "
             "or REACHAQ_MEDIA_ROOT, else <repo>/temp)",
    )
    args = parser.parse_args()
    if args.media_root is not None:
        MEDIA_ROOT = args.media_root
    print(f"media root: {MEDIA_ROOT}")
    if not MEDIA_ROOT.is_dir():
        print("  (not a directory; stills will render as placeholders)")

    lineage()
    pipeline()
    two_tier()
    demo_substitution()
    demo_frames()
    pose_backbones()
    print(f"\nassets in {ASSETS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
