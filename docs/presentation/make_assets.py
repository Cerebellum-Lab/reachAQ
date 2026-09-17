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
import numpy
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


# ----------------------------------------------------------------- benchmarks
#
# Every number below is copied from docs/linux-install/latency-tuning.md, which
# records what was measured on the reference workstation. Nothing here is
# illustrative: if the measurements are re-run, edit them there and here together.

#: 60 s at 900 Hz pinned to physical P-cores. Cost = scheduler wake-up lateness
#: plus decision time, in milliseconds.
STIM_LOOP_MS = [
    # label,                       p50,   p99,  p99.9,  max
    ("idle\nSCHED_OTHER", 0.129, 0.221, 0.262, 0.332),
    ("idle\nSCHED_FIFO", 0.077, 0.170, 0.183, 0.246),
    ("16 busy procs\nSCHED_OTHER", 0.099, 3.662, 4.481, 6.050),
    ("16 busy procs\nSCHED_FIFO", 0.047, 0.059, 0.122, 0.155),
]
STIM_BUDGET_MS = 5.0

#: Live pose inference, DeepLabCut TensorFlow, batch 2 at 128x128, on the GPU.
INFERENCE_MS = [
    ("powersave governor", 12.55, 17.58, 18.96),
    ("performance governor", 10.79, 11.53, 11.73),
]


def stim_latency():
    """Stim-loop cost by condition, against the 5 ms budget."""
    figure, axes = plt.subplots(figsize=(12, 5), dpi=200)
    figure.patch.set_facecolor("white")
    labels = [row[0] for row in STIM_LOOP_MS]
    series = [("p50", 0), ("p99", 1), ("p99.9", 2), ("max", 3)]
    colors = ["#9fb3c8", "#6b7f9e", "#c98c6a", NEW]
    width = 0.2
    positions = numpy.arange(len(labels))
    for index, ((name, offset), color) in enumerate(zip(series, colors)):
        values = [row[offset + 1] for row in STIM_LOOP_MS]
        bars = axes.bar(positions + (index - 1.5) * width, values, width,
                        label=name, color=color)
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:g}",
                      ha="center", va="bottom", fontsize=7.5, color=INK)

    axes.axhline(STIM_BUDGET_MS, color=NEW, linestyle="--", linewidth=1.4)
    axes.text(len(labels) - 0.45, STIM_BUDGET_MS + 0.12, "5 ms budget",
              ha="right", color=NEW, fontsize=9, weight="bold")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=9, color=INK)
    axes.set_ylabel("milliseconds", fontsize=9, color=INK)
    axes.set_ylim(0, 6.8)
    axes.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(colors=MUTED, labelsize=8)
    axes.set_title("Stim loop: 900 Hz, 60 s, wake-up lateness + decision time",
                   fontsize=11, color=INK, weight="bold", pad=12)
    _save(figure, "stim_latency.png")


def inference_latency():
    """Live pose inference cost, and what the CPU governor did to it."""
    figure, axes = plt.subplots(figsize=(11, 4.2), dpi=200)
    figure.patch.set_facecolor("white")
    labels = [row[0] for row in INFERENCE_MS]
    series = [("p50", 1), ("p99", 2), ("max", 3)]
    colors = ["#9fb3c8", AUTO, NEW]
    width = 0.24
    positions = numpy.arange(len(labels))
    for index, ((name, offset), color) in enumerate(zip(series, colors)):
        values = [row[offset] for row in INFERENCE_MS]
        bars = axes.barh(positions + (index - 1) * width, values, width,
                         label=name, color=color)
        for bar, value in zip(bars, values):
            axes.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:g} ms",
                      va="center", fontsize=8.5, color=INK)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=10, color=INK)
    axes.set_xlabel("milliseconds per inference", fontsize=9, color=INK)
    axes.set_xlim(0, 22)
    axes.invert_yaxis()
    axes.legend(frameon=False, fontsize=9, ncol=3, loc="lower right")
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(colors=MUTED, labelsize=8)
    axes.set_title("Live pose inference: DeepLabCut TensorFlow, batch 2 at 128x128, GPU",
                   fontsize=11, color=INK, weight="bold", pad=12)
    _save(figure, "inference_latency.png")


#: Author's rig measurements on an NVIDIA T1000. Before/after pairs only, so
#: every bar in a pair shares one unit and nothing has to be read across axes.
GATED = "#c98c6a"
DECLINED = "#a8adb4"

#: label, before, after, unit note, colour of the "after" bar
WINS = [
    ("Partial batch\n2 real frames, not a padded 6", 54.8, 25.3, "ms / batch", GATED),
    ("CPU decode rewrite\nargmax + numpy, not sort + pandas", 6.565, 2.675, "ms / batch", AUTO),
    ("CUDA-graph capture\nreplay the graph for the live batch", 7.4, 3.6, "ms / call", AUTO),
]


def measurements():
    """Tier 1 tail on the left, the Tier 2 wins on the right."""
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 4.6), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    figure.patch.set_facecolor("white")

    # --- Tier 1: worst case under load, against the budget
    labels = [row[0].replace("\n", " · ") for row in STIM_LOOP_MS]
    maxima = [row[4] for row in STIM_LOOP_MS]
    colors = [DECLINED, DECLINED, NEW, AUTO]
    positions = numpy.arange(len(labels))
    bars = left.barh(positions, maxima, 0.55, color=colors)
    for bar, value in zip(bars, maxima):
        left.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:g} ms",
                  va="center", fontsize=9, color=INK)
    left.axvline(STIM_BUDGET_MS, color=MUTED, linestyle="--", linewidth=1.2)
    # Clear space to the right of the line, between the two idle rows.
    left.text(STIM_BUDGET_MS + 0.14, 1.3, "5 ms budget", ha="left", va="center",
              fontsize=8.5, color=MUTED, style="italic")
    left.set_yticks(positions)
    left.set_yticklabels(labels, fontsize=8.5, color=INK)
    left.invert_yaxis()
    left.set_xlim(0, 7.4)
    left.set_xlabel("worst case over 60 s at 900 Hz, ms", fontsize=9, color=INK)
    left.set_title("Tier 1 — reflex loop", fontsize=11, color=INK, weight="bold", pad=10)
    left.spines[["top", "right"]].set_visible(False)
    left.tick_params(colors=MUTED, labelsize=8)

    # --- Tier 2: each change as its own before/after pair
    gap, positions = 0.42, []
    for index, (label, before, after, unit, color) in enumerate(WINS):
        base = index * 1.3
        positions.append(base + gap / 2)
        for offset, value, bar_color in ((0, before, DECLINED), (gap, after, color)):
            bar = right.barh(base + offset, value, gap * 0.92, color=bar_color)[0]
            right.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:g}",
                       va="center", fontsize=8.5, color=INK)
        right.text(58, base + gap / 2, unit, va="center", ha="right",
                   fontsize=7.5, color=MUTED, style="italic")
    right.set_yticks(positions)
    right.set_yticklabels([row[0] for row in WINS], fontsize=8.5, color=INK)
    right.invert_yaxis()
    right.set_xlim(0, 60)
    right.set_xlabel("before  →  after", fontsize=9, color=INK)
    right.set_title("Tier 2 — pose stream", fontsize=11, color=INK, weight="bold", pad=10)
    right.spines[["top", "right"]].set_visible(False)
    right.tick_params(colors=MUTED, labelsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (AUTO, GATED, DECLINED)]
    figure.legend(handles, ["adopted, always on", "gated behind a flag", "before / not taken"],
                  frameon=False, fontsize=9, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, -0.05))
    figure.tight_layout()
    _save(figure, "measurements.png")


def synchronization():
    """One clock, and what hangs off it."""
    # No title or footnote inside the figure: the slide supplies both, and
    # duplicating them reads as a mistake.
    figure, axes = _fig(12, 3.9)
    _box(axes, 36, 20, 28, 7, "NI-DAQ master clock", NEW, size=11)
    feeds = [
        (2, "Cameras\n150 FPS", AUTO),
        (20, "stimCam\n900 Hz", NEW),
        (38, "Tones\n1 / 2 / 3", AUTO),
        (56, "Pellet board\nFSR + events", NEW),
        (74, "Laser\nAO + diode", NEW),
    ]
    for x, label, color in feeds:
        _box(axes, x, 4, 16, 9, label, color, size=8.5)
        _arrow(axes, (x + 8, 13.5), (48, 19.5))
    _save(figure, "synchronization.png")


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

    # Only what the deck references. The other generators above are kept for
    # slides that may come back; add the call here if one does.
    measurements()
    synchronization()
    print(f"\nassets in {ASSETS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
