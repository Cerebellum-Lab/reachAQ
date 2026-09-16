"""Build the reachAQ overview deck.

Content is the prose in outline.md, kept in sync by hand: the outline is the
reviewable source, this script is the renderer.

Run from the repository root, after generating assets:
    python docs/presentation/make_assets.py --media-root <checkout>/temp
    python docs/presentation/build_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "reachAQ-overview.pptx"

W, H = Inches(13.333), Inches(7.5)
INK = RGBColor(0x26, 0x28, 0x2B)
MUTED = RGBColor(0x7A, 0x7F, 0x87)
REACH = RGBColor(0x6B, 0x7F, 0x9E)
AUTO = RGBColor(0x4A, 0x8C, 0x7A)
NEW = RGBColor(0xB3, 0x59, 0x3F)


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _text(slide, left, top, width, height, text, size, color=INK,
          bold=False, align=PP_ALIGN.LEFT, italic=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return box


def _title(slide, text, subtitle=None):
    _text(slide, Inches(0.7), Inches(0.4), Inches(12), Inches(0.9), text, 30, INK, bold=True)
    if subtitle:
        _text(slide, Inches(0.7), Inches(1.15), Inches(12), Inches(0.5), subtitle, 14,
              MUTED, italic=True)


def _bullets(slide, items, left=Inches(0.7), top=Inches(1.9),
             width=Inches(12.0), size=14, gap=0.46):
    for index, item in enumerate(items):
        _text(slide, left, top + Inches(index * gap), width, Inches(gap),
              f"•  {item}", size)


def _lineage(slide, reach, auto, new, top=Inches(5.25)):
    """The three-way strip every feature slide carries."""
    _text(slide, Inches(0.7), top - Inches(0.34), Inches(12), Inches(0.3),
          "Where it came from", 10, MUTED, italic=True)
    column = Inches(4.05)
    for index, (label, body, color) in enumerate((
        ("reach-training", reach, REACH),
        ("auto-trainer", auto, AUTO),
        ("reachAQ", new, NEW),
    )):
        left = Inches(0.7) + index * column
        _text(slide, left, top, column - Inches(0.3), Inches(0.3), label, 11, color, bold=True)
        _text(slide, left, top + Inches(0.32), column - Inches(0.3), Inches(1.4), body, 10, INK)


def _picture(slide, name, left, top, max_width, max_height):
    """Fit an image inside a box, preserving aspect ratio and centring it.

    Sizing by width alone silently overruns whatever sits below, which a bounds
    check does not catch because the shape is still on the slide. Fit to the box
    so the caller's layout is the thing that decides.
    """
    path = ASSETS / name
    if not path.is_file():
        _text(slide, left, top, max_width, Inches(0.5),
              f"[missing asset: {name} — run make_assets.py]", 11, MUTED, italic=True)
        return

    with Image.open(path) as image:
        native_w, native_h = image.size
    scale = min(max_width / native_w, max_height / native_h)
    width = int(native_w * scale)
    height = int(native_h * scale)
    offset_x = int((max_width - width) / 2)
    offset_y = int((max_height - height) / 2)
    slide.shapes.add_picture(
        str(path), int(left) + offset_x, int(top) + offset_y, width=width, height=height
    )


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------- slides


def slide_title(prs):
    slide = _blank(prs)
    _text(slide, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.2),
          "reachAQ", 54, INK, bold=True)
    _text(slide, Inches(0.9), Inches(3.6), Inches(11.5), Inches(0.6),
          "Automated mouse reach acquisition", 22, INK)
    _text(slide, Inches(0.9), Inches(4.3), Inches(11.5), Inches(0.5),
          "Cerebellum Lab  ·  Dell/Ubuntu rig", 14, MUTED)
    _notes(slide,
           "One sentence of welcome. Say that the last slide before backup is a live "
           "demo, and that everything in it is real except the video.")


def slide_what_it_does(prs):
    slide = _blank(prs)
    _title(slide, "What this system does")
    _bullets(slide, [
        "A head-fixed mouse reaches for a pellet",
        "Several cameras watch at 150 frames per second",
        "The system finds the paw in 3D while the reach is still happening",
        "It decides, and acts, inside the same reach: cue, pellet, laser",
        "Every trial is recorded, timestamped, and reproducible",
    ], top=Inches(2.2), size=18, gap=0.72)
    _notes(slide,
           "No architecture yet. The point is the closed loop: the system is a "
           "participant in the experiment, not a recorder of it. Everything after this "
           "slide is how that is achieved.")


def slide_where_from(prs):
    slide = _blank(prs)
    _title(slide, "Where it came from",
           "The only slide organised by repository — everything after this is by feature")
    _picture(slide, "lineage.png", Inches(0.6), Inches(1.85), Inches(12.1), Inches(5.2))
    _notes(slide,
           "reach-training was the original working rig: Python 3.8, PySpin cameras, an "
           "Arduino for the pellet, reach detection by camera ROI threshold, DeepLabCut "
           "offline afterwards, then manual curation.\n\n"
           "auto-trainer is the modular platform reachAQ is built from: core, video, "
           "device, inference, behavior, model and pyside packages, synchronized "
           "multi-process capture, integrated pose inference, stereo calibration, a CAN "
           "pellet device, and the session and metadata structure.\n\n"
           "reachAQ is this system: new hardware target, new stimulation subsystem, live "
           "3D pose in the control loop, and a protocol engine.\n\n"
           "Say plainly: we did not start over, and we did not just rename someone "
           "else's code.")


def slide_cameras(prs):
    slide = _blank(prs)
    _title(slide, "Camera acquisition")
    _bullets(slide, [
        "2 to 6 cameras, configurable, hardware-synchronized at 150 FPS",
        "Each camera runs in its own process, so one slow camera cannot stall the others",
        "A shared frame index keeps every camera on the same frame number",
        "A separate faster stim camera runs its own small-ROI detector",
    ], top=Inches(1.6), size=14, gap=0.4)
    _picture(slide, "pipeline.png", Inches(1.85), Inches(3.3), Inches(9.6), Inches(1.8))
    _lineage(
        slide,
        "PySpin capture, cameras configured in systemdata.yaml",
        "Per-process VideoCapture, shared synchronized frame index, Spinnaker "
        "camera-clock mapping",
        "2–6 cameras rather than a fixed pair, configurable frame rate, stim camera "
        "at a multiple of the reach rate",
        top=Inches(5.55),
    )
    _notes(slide,
           "The shared frame index is what makes 3D possible at all. Frame N on left "
           "must be the same instant as frame N on right.")


def slide_pellet(prs):
    slide = _blank(prs)
    _title(slide, "Pellet delivery and device control")
    _bullets(slide, [
        "The pellet arm is a motorized device on a CAN bus, not a serial gadget",
        "The application checks the board's firmware capabilities before using a feature",
        "Presence evidence: the system knows whether a pellet is actually there",
        "Manual controls stay available at all times: home, load, present, release",
    ], top=Inches(2.0), size=16, gap=0.6)
    _lineage(
        slide,
        "Arduino over USB serial, text commands, hotkeys H / P / M / R",
        "A CAN device abstraction behind a hardware interface",
        "SocketCAN/PCAN on Ubuntu x86_64 instead of the Jetson 40-pin adapter, plus "
        "firmware compatibility checking and typed failure causes",
    )
    _notes(slide,
           "The firmware capability check matters because a protocol feature that needs "
           "a newer board must refuse rather than silently misbehave.")


def slide_pose(prs):
    slide = _blank(prs)
    _title(slide, "Pose inference")
    _bullets(slide, [
        "A pose model finds keypoints on the paw and snout in every frame",
        "Two camera views are triangulated into a 3D position",
        "This runs live, during the session, not afterwards",
        "Confidence thresholds are measured per model, not guessed",
    ], top=Inches(1.55), size=13, gap=0.36)
    _picture(slide, "pose_backbones.png", Inches(2.6), Inches(3.05), Inches(8.1), Inches(2.1))
    _lineage(
        slide,
        "DeepLabCut run offline after the session, then curated by hand",
        "Pose inference and FLIR stereo calibration integrated into the application",
        "Live 3D computed in NumPy on the PyTorch backend, per-backbone confidence "
        "calibration, top-down model support",
        top=Inches(5.55),
    )
    _notes(slide,
           "The image shows four candidate backbones on one frame with how many "
           "keypoints each put above threshold. That spread is exactly why the "
           "confidence gate is calibrated per model: a cspnext_s never scores above "
           "0.33, so a single global threshold would hold the reach flag permanently "
           "false.")


def slide_closed_loop(prs):
    slide = _blank(prs)
    _title(slide, "Closed-loop reach detection",
           "The central capability — and the one that did not exist before")
    _bullets(slide, [
        "The cue gate asks a live question: has the paw crossed into the reach volume?",
        "The answer comes from the live 3D pose, on the current frame",
        "If the answer is no at the deadline, the trial resets rather than firing late",
        "The reach-state source is selectable, so the gate can be driven different ways",
    ], top=Inches(2.1), size=16, gap=0.6)
    _lineage(
        slide,
        "A brightness threshold inside a drawn ROI, checked against an Arduino serial "
        "handshake",
        "Pose-derived behavioural state feeding a state machine",
        "Live 3D pose answers the cue gate directly, with a deadline-accurate timer and "
        "a typed cancel reason when the gate is blocked or stale",
    )
    _notes(slide,
           "This is the central claim of the whole system. Everything on the previous "
           "slides exists to make this slide possible. Be clear that reachAQ replaced an "
           "ROI brightness test with a tracked 3D coordinate.")


def slide_stim(prs):
    slide = _blank(prs)
    _title(slide, "Optogenetic stimulation", "New in reachAQ — no ancestor in either repository")
    _bullets(slide, [
        "Up to four lasers, each modeled independently",
        "Analog and digital control paths per laser",
        "Per-laser diode input and a command copy recorded alongside the data",
        "Shutter control, and a PMT shutter output",
        "Hardware-timed NI-DAQ tasks, not Python-loop timing",
    ], top=Inches(2.1), size=16, gap=0.58)
    _lineage(
        slide,
        "None — external stimulation was a separate LabVIEW system",
        "None — no laser subsystem",
        "Entirely new, built against the NI-DAQ channel inventory taken from the "
        "original LabVIEW control system",
    )
    _notes(slide,
           "Say plainly that this is new and has no ancestor in either repository. The "
           "LabVIEW system was used as a requirements inventory, not as a source of "
           "logic — it is known to contain bugs.")


def slide_protocols(prs):
    slide = _blank(prs)
    _title(slide, "Trial protocols")
    _bullets(slide, [
        "A trial is Tone 1, a drawn delay, then Tone 2",
        "Delay distributions use the published presets and probability vectors",
        "Stimulation epochs: baseline, stimulation, washout, sized in trials",
        "Every per-trial draw is seeded from the trial identity, so sessions replay identically",
        "Protocols are edited in the application and versioned on disk",
    ], top=Inches(2.0), size=15, gap=0.55)
    _lineage(
        slide,
        "Fixed protocols in scripts, delays drawn live at runtime",
        "A training plan and phase structure",
        "A protocol table with cue pairs and trigger profiles, published delay "
        "distributions, stimulation epochs, and an inter-trial timing target",
    )
    _notes(slide,
           "The reproducibility point is worth dwelling on: reach-training drew at "
           "runtime, so a session could not be replayed. reachAQ derives every draw from "
           "the trial identity.")


def slide_timing(prs):
    slide = _blank(prs)
    _title(slide, "Timing architecture",
           "Two loops, separate budgets, deliberately kept apart")
    _picture(slide, "two_tier.png", Inches(0.6), Inches(1.8), Inches(12.1), Inches(4.8))
    _text(slide, Inches(0.7), Inches(6.75), Inches(12), Inches(0.4),
          "A stimulus condition that needs a 3D coordinate belongs to Tier 2, never to "
          "the 5 ms budget.", 12, MUTED, italic=True)
    _notes(slide,
           "THE 5 MS FIGURE IS A DESIGN TARGET AND A TAIL FIGURE, NOT A MEASURED "
           "RESULT. It needs thread pinning, realtime scheduling, and no allocation or "
           "logging on the trigger branch, and it is pending rig validation. Do not "
           "present it as achieved.")


def slide_data(prs):
    slide = _blank(prs)
    _title(slide, "Data and metadata")
    _bullets(slide, [
        "Every session writes video, timestamps, trial records, and events",
        "Session metadata is schema v2, and embeds the full system configuration that produced it",
        "Hardware state is recorded as both configured and runtime",
        "A validation command checks a finished session against a rule set",
    ], top=Inches(2.1), size=16, gap=0.6)
    _lineage(
        slide,
        "Per-session YAML plus loose output files",
        "Project and session structure with a metadata schema",
        "Schema v2 with configuration embedding, alignment and trial-summary artifacts, "
        "and reachaq-validate-session",
    )
    _notes(slide,
           "The embedded configuration means a session can be interpreted years later "
           "without the rig notes.")


def slide_demo(prs):
    slide = _blank(prs)
    _title(slide, "Demo", "No mouse today — so exactly one stage is substituted")
    _picture(slide, "demo_substitution.png", Inches(1.9), Inches(1.78), Inches(9.5), Inches(1.6))
    _picture(slide, "live_overlay.png", Inches(0.75), Inches(3.45), Inches(11.8), Inches(3.0))
    _text(slide, Inches(0.75), Inches(6.6), Inches(11.8), Inches(0.4),
          "Live 2D pose, per view, computed on the GPU from the played frames — "
          "captured from the running application on the rig.", 12, MUTED, italic=True)
    _notes(slide,
           "Run order:\n"
           "1. Launch with --demo, or toggle Tools > Demo Mode while idle, to show both "
           "routes.\n"
           "2. Show the live camera panels. Each view carries its own 2D pose overlay, "
           "computed on the GPU from the played frames. This is real inference running "
           "now, not a replayed result.\n"
           "3. Show the NI-DAQ live signal stream. Genuinely live.\n"
           "4. Drive the pellet device manually. Genuinely live.\n"
           "5. Open the protocol table; show cue timing and trigger profiles. Genuinely "
           "live.\n"
           "6. Start a recording session, let trials run, show trial records "
           "accumulating.\n"
           "7. Stop, and show the written session directory including its DEMO marker "
           "file.\n"
           "8. State explicitly which single element was pre-recorded, and why.\n\n"
           "Point at the red banner in the status bar. It is there so nobody mistakes a "
           "demo session for data.")


def slide_status(prs):
    slide = _blank(prs)
    _title(slide, "Current status and limitations")

    _text(slide, Inches(0.7), Inches(1.7), Inches(3.9), Inches(0.35),
          "Validated on the rig", 13, AUTO, bold=True)
    _bullets(slide, [
        "Camera acquisition, recording, session output",
        "Pellet delivery over CAN, with firmware compatibility checking",
        "NI-DAQ signal streaming and port configuration",
    ], left=Inches(0.7), top=Inches(2.15), width=Inches(3.8), size=11, gap=0.72)

    _text(slide, Inches(4.75), Inches(1.7), Inches(3.9), Inches(0.35),
          "Not yet validated on the rig", 13, NEW, bold=True)
    _bullets(slide, [
        "The 5 ms Tier 1 stim-loop budget, as a tail figure",
        "Protocol timing, precheck and cue behaviour from the current work, which needs "
        "TTL and video proof before experimental use",
    ], left=Inches(4.75), top=Inches(2.15), width=Inches(3.8), size=11, gap=1.15)

    _text(slide, Inches(8.8), Inches(1.7), Inches(3.9), Inches(0.35),
          "Deferred", 13, MUTED, bold=True)
    _bullets(slide, [
        "Offline curation, RFID, SoftMouse automation",
        "Tunnel and head-fix behaviour inherited from auto-trainer",
    ], left=Inches(8.8), top=Inches(2.15), width=Inches(3.8), size=11, gap=0.9)

    _notes(slide,
           "Be honest here. This audience will run experiments on this system, and a "
           "surprise later costs far more than a caveat now.")


def slide_backup(prs):
    slide = _blank(prs)
    _title(slide, "Getting started", "Backup — leave up during questions")
    _bullets(slide, [
        "Install:  linux-install-instructions.md  and  tools/install/reachaq-linux-install.sh",
        "Run:  python -m reachAQ.app -c ~/Autotrainer/system_configuration.yaml",
        "Demo mode:  docs/acquisition/demo-mode.md",
        "Session recording and synchronization:  docs/acquisition/session-recording-and-synchronization.md",
        "Trials and protocols:  docs/acquisition/session-trials-protocols.md",
    ], top=Inches(2.2), size=14, gap=0.7)
    _notes(slide, "Leave this up while taking questions.")


SLIDES = (
    slide_title,
    slide_what_it_does,
    slide_where_from,
    slide_cameras,
    slide_pellet,
    slide_pose,
    slide_closed_loop,
    slide_stim,
    slide_protocols,
    slide_timing,
    slide_data,
    slide_demo,
    slide_status,
    slide_backup,
)


def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    for build in SLIDES:
        build(prs)
    prs.save(OUT)
    print(f"wrote {OUT} with {len(prs.slides._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
