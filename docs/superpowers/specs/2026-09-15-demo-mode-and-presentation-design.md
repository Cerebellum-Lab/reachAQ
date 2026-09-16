# Demo Mode and Onboarding Presentation — Design

Date: 2026-09-15
Branch: `demo-mode-and-presentation`
Worktree: `../reachAQ-demo`
Base: `reach-training-protocol-features` at `69ed5e40`

## Purpose

Produce two deliverables for an upcoming meeting whose audience is brand-new users:

1. A slide deck explaining reachAQ as a progression from the `reach-training` setup,
   what was migrated from `auto-trainer`, and what is new here.
2. A runnable "test" session demo. No mouse is present, so the camera feed is
   pre-recorded video. Everything else stays live: pose inference runs on the played
   frames, and NI-DAQ streaming, pellet device control, protocol editing, and port
   configuration remain real.

The demo's credibility depends on it being the real pipeline with one substituted
input. Any design that builds a parallel demo path defeats its own purpose.

## Non-Goals

- Simulated CAN, NI-DAQ, or laser backends. The demo runs on the rig with real
  hardware attached; absent hardware is switched off in configuration, not faked.
- A CPU inference fallback. Live inference requires a working CUDA runtime and
  refuses to start without one. That gate stays as it is.
- Offline analysis, curation, or RFID/SoftMouse behavior in the demo path.
- Changing session identifier naming. Analysis conventions depend on it.

## Runtime Decisions

Settled before design:

- The demo runs on the Dell/Ubuntu rig, with GPU, CAN pellet board, and NI-DAQ present.
- Demo mode is reachable two ways: a `--demo` CLI flag and a UI toggle.
- The demo runs a full session: recording, trials, and protocol, tagged as demo data.
- The demo asset is a `christie2P` session from `temp/2p_sessions`: paired left/right
  (and stimCam) video at 150 FPS.

## Existing Capability This Builds On

`PlaybackCam` already exists and is a first-class camera type, not a test fixture
(`auto-trainer-video/src/autotrainer/video/camera/playback_cam.py`). It reads a video
through OpenCV, paces itself to real wall-clock FPS, loops at EOF, and synthesizes
precise nanosecond timestamps. It is registered under the `playback://` URL scheme in
`video_manager.py`, so a camera becomes a video player purely through configuration:

```yaml
scheme: playback
path: /path/to/session_left.mp4
```

Downstream, nothing distinguishes it. Frames enter the same `VideoCapture` process,
the same record queue, and the same live-inference queue as Spinnaker frames.

`AppModel._apply_random_camera_override` establishes the precedent for a runtime
camera-source override applied at configuration load, driven by the `--random-cameras`
flag. Demo mode is the same shape with a different target scheme.

Hardware subsystems are independently switchable in configuration (`canEnabled`,
`pelletControllerEnabled`, `nidaqEnabled`, `laser.backend`, `nidaqStream.isEnabled`),
so demo mode does not need to touch any of them.

## Architecture

Demo mode is a **camera-layer override**. Both entry points funnel into one function
that rewrites only the reach cameras' source, leaving hardware, NI-DAQ, laser,
protocol, and persistence exactly as the rig has them configured. Nothing downstream
of `VideoCapture` learns that demo mode exists.

Two alternatives were considered and rejected. A separate full demo configuration file
needs no code but replaces the rig's entire configuration, so the demo would show a
different hardware setup than the real one, and it cannot be toggled from the UI. A
dedicated demo-session controller offers the most control but builds a parallel code
path, which removes the demo's whole claim to authenticity.

### Component 1 — Demo source specification

New pure module in `auto-trainer-core`. No Qt imports, no hardware imports, so it stays
testable and upstream-exportable.

A dataclass plus YAML loader mapping camera name to video path, with playback options:

```yaml
# ~/Autotrainer/demo/demo_sources.yaml
fps: 150
loop: true
cameras:
  left: /home/christie07/Autotrainer/demo/session001_left-0000.mp4
  right: /home/christie07/Autotrainer/demo/session001_right-0000.mp4
  stimCam: /home/christie07/Autotrainer/demo/session001_stimCam-0000.mp4
```

Validation on load: every named camera must be a recognized camera name, every path
must exist and be readable, and `fps` must be positive. A malformed or missing spec
raises with the offending path named, rather than silently starting a demo with dead
cameras.

Demo video files are **not** committed. They are large session recordings and the repo
keeps `temp/` untracked. The repo ships the loader, an example spec, and a documented
path convention; the operator stages the actual media on the rig.

### Component 2 — Camera override

`AppModel._apply_demo_playback_override(configuration, spec)`, a direct sibling of
`_apply_random_camera_override`, called from `load_configuration` under a new
`demo_sources` argument.

For each reach camera in the configuration:

- If the spec supplies a video for it, set `scheme: playback`, `path: <video>`,
  `fps: <spec fps>`, and leave `primary` as configured.
- If the spec supplies no video for it, **disable the camera**. Leaving it enabled
  would point it at a Spinnaker handle that demo mode is not driving.

Hardware, inference, laser, NI-DAQ, persistence, and protocol sections are left
untouched. `configuration._camera_map` is reset, matching the existing override.

### Component 3 — Playback start barrier

This is the one genuinely new mechanism, and the demo's correctness depends on it.

Each camera runs in its own `VideoCapture` process, and each `PlaybackCam` paces
against its own `_capture_start`. Hardware-synchronized Spinnaker cameras guarantee
that frame *N* on left corresponds to frame *N* on right; two independently started
playback processes do not. At 150 FPS one frame is 6.7 ms, so process start skew lands
the streams a frame or more apart. Live 3D triangulation pairs by frame ID, so
unsynchronized starts produce silently wrong 3D from plausible-looking 2D.

Design: a `multiprocessing.Barrier` sized to the number of enabled playback cameras,
shared through `CaptureAttrs`. Each `PlaybackCam` waits on the barrier immediately
before capturing its first frame, so every camera's `_capture_start` is established
within microseconds of the others. Because each camera already paces to an absolute
target of `_capture_start + n/fps` rather than to an accumulating delta, aligned starts
keep frame indices aligned for the life of the run without further coordination.

The barrier is created only for playback sources. Spinnaker capture is untouched.

Barrier wait uses a bounded timeout. A camera that fails to arrive fails the capture
start with a named error instead of hanging the UI.

### Component 4 — Demo session tagging

A demo session writes real session files through the real recorder. Those files must
never be mistaken for experimental data.

Session metadata is schema v2 and already embeds the full `configuration`, so
`scheme: playback` is technically *detectable*. Implicit detection is not sufficient
for data that nobody should analyze. Three explicit markers:

- A `demoMode` block in the session metadata recording that demo mode was active and
  which source videos were played.
- A `DEMO` marker file written into the session directory.
- A persistent banner in the UI for as long as demo mode is active, so the operator
  cannot forget mid-presentation.

Session identifiers are unchanged. Prefixing them would ripple into the naming
conventions that downstream analysis depends on.

### Component 5 — UI toggle

A Demo Mode control in `tools/acquisition/view/camera_content.py`.

`AppModel.load_configuration` is already guarded by
`_require_session_ready_for_configuration`, which raises unless the recording session
is `SessionRecordingStatus.READY`. The toggle is therefore enabled only in that state
and disabled otherwise. This enforces a constraint the model already owns rather than
inventing a new one.

Toggling reloads configuration with or without the demo override. The banner from
Component 4 follows the toggle state.

Spec resolution for the toggle: it uses the spec path given on the command line if the
app was launched with `--demo`, and otherwise the default
`~/Autotrainer/demo/demo_sources.yaml`. So the toggle works in a session that did not
start in demo mode, which is the point of having it.

If the spec is missing or invalid when the toggle is switched on, the app reports the
validation error from Component 1 in a dialog naming the offending path, leaves the
toggle off, and keeps the current configuration loaded. It does not fall back to
physical cameras silently and it does not start a partially configured demo.

### Component 6 — CLI flag

`--demo` in `tools/acquisition/args.py`, optionally taking a spec path and defaulting
to `~/Autotrainer/demo/demo_sources.yaml`. Threaded through `run_acquisition.py` and
`main_window.py` to `load_configuration`, mirroring `--random-cameras` exactly.

`--demo` and `--random-cameras` are mutually exclusive; passing both is an argument
error, not a silent precedence rule.

## Testing

Unit tests, no hardware required:

- Demo source spec: valid load, missing file, unreadable path, unknown camera name,
  non-positive fps.
- Camera override: cameras with a video become `playback` with the right path and fps;
  cameras without one are disabled; hardware, inference, laser, and NI-DAQ sections are
  unchanged before and after.
- Argument parsing: `--demo` default path, explicit path, and mutual exclusion with
  `--random-cameras`.
- Start barrier: N playback cameras reach first frame within a tolerance; a missing
  participant produces the named timeout error rather than hanging.

Manual rig rehearsal, which is required regardless and is covered under Risks.

## Risks

**The pose model must actually detect reaches in the demo video.** The keypoint
confidence gate is calibrated per backbone: commits `ad13b7a4` and `681a3f12` add
measured thresholds for `cspnext_m` and `rtmpose_s`, and the code notes a `cspnext_s`
never scores above 0.33, which would hold `rh_grab_seen` permanently false. If the
rig's configured model does not fire on the `christie2P` video, the closed-loop portion
of the demo is silent. This must be verified in a rig rehearsal well before the
meeting, not on the day.

**Calibration must match the geometry that recorded the video.** `reload_calib` loads
stereo parameters from `~/Autotrainer/<calib dir>` and degrades gracefully when absent,
so a mismatch does not crash — it silently produces wrong 3D under a correct-looking 2D
overlay. The `christie2P` session's `systemdata_copy.yaml` records per-camera crops
(`[0,256,5,256]` left, `[57,256,7,256]` right). If the rig's current calibration was
produced under different crops, 3D will be subtly wrong. Verify during rehearsal.

**Session validation may flag demo recordings.** Recording playback frames exercises
`camera_recording_validation` and `session_validation_controller`. Demo sessions may
produce validation findings that are correct for real data and meaningless here.
Rehearsal should establish whether any appear, and they should be explained rather than
suppressed.

## Slide Deck

### Framing

Staged by feature, not by repository. Each feature slide carries its own three-way
lineage — what `reach-training` did, what `auto-trainer` contributed, what reachAQ
does now — so goals 1 through 3 are answered continuously rather than in three
segregated sections. Exactly one slide is organized by repository, and its job is to
give the audience the map before the feature slides assume it.

Audience is brand-new users. The deck explains what the system does and why it is
built this way. It is not a code walkthrough.

### Lineage facts the deck rests on

Grounded in the working tree and the reference material, so no slide overstates:

`reach-training` (`temp/reach-training`, read-only reference): Python 3.8 GUI
acquisition in `multiCam_RT_videoAcquisition_v5.py`/`v6`, PySpin multi-camera capture,
Arduino serial pellet control in `arduinoCtrl_v5.py`, reach detection by camera ROI
thresholding, offline DeepLabCut in `multiCam_DLC_PySpin_v2.py`, offline reach finding
and curation in `findReachEvents_v2.py` and `Reach_Curator_py38_autotrainer.py`, and a
`systemdata.yaml` configuration format. Operator hotkeys: H home, P load pellet,
M send mouse, R release.

`auto-trainer` (migrated, the `auto-trainer-*` packages in this repo): the modular
package split of core, video, device, inference, behavior, model, and pyside; the
multi-process `VideoCapture` with a shared synchronized frame index; the Spinnaker
driver including camera-clock to `perf_counter` mapping; DLC pose inference, FLIR
stereo calibration, and 3D triangulation; the CAN pellet device; the project, session,
and metadata structure; the PySide6 UI layer; and the behavior state machine.

reachAQ (new here): Dell/Ubuntu x86_64 replacing Jetson AGX aarch64; SocketCAN/PCAN
replacing the Jetson 40-pin adapter assumption; an NI-DAQ laser and optogenetics
subsystem modeling up to four independent lasers with analog and digital paths; NI-DAQ
signal streaming with live plots; a formalized two-tier latency architecture; live 3D
pose computed in NumPy on the PyTorch backend with per-backbone confidence calibration;
trial protocols with cue pairs, published delay distributions, and stimulation epochs;
live pose answering the cue gate through a selectable reach-state source; session
validation tooling; and pellet firmware compatibility checking.

### Slide plan

Approximately 14 slides plus backup.

1. **Title.** System name, rig, presenter, date.
2. **What this system does.** The experiment in plain terms: a mouse reaches for a
   pellet, cameras watch, the system decides and acts within milliseconds. No
   architecture yet.
3. **Where it came from.** The one repository-organized slide: reach-training →
   auto-trainer → reachAQ, with what each contributed. The map for everything after.
4. **Camera acquisition.** ROI-thresholded PySpin capture → multi-process synchronized
   `VideoCapture` → configurable 2–6 cameras at 150 FPS with a faster stimCam.
5. **Pellet delivery and device control.** Arduino serial → CAN device abstraction →
   SocketCAN/PCAN with firmware compatibility checking and presence evidence.
6. **Pose inference.** Offline DeepLabCut → integrated 2D pose and stereo triangulation
   → live 3D pose in NumPy on the PyTorch backend, calibrated per backbone.
7. **Closed-loop reach detection.** ROI threshold against an Arduino → pose-derived
   state → live 3D pose answering the cue gate. The central novel claim; own slide.
8. **Optogenetic stimulation.** Wholly new: NI-DAQ laser control, up to four
   independent lasers, analog and digital paths, hardware-timed tasks.
9. **Trial protocols.** Cue pairs with Tone 1 and Tone 2, published delay
   distributions, stimulation epochs, reproducible seeded per-trial draws.
10. **Timing architecture.** The two-tier design: Tier 1 stim loop at a 5 ms p99 target
    through a pre-armed hardware-timed NI-DAQ output, Tier 2 behavior loop at 20–100 ms
    carrying pose and trial logic. States the budget as a tail figure, and states
    plainly that it is a target pending rig validation.
11. **Data and metadata.** What a session writes, metadata schema v2, session
    validation tooling.
12. **Demo.** Live, driven from the deck. Speaker notes carry the run order.
13. **Current status and limitations.** What is validated on the rig, what is not, what
    is deferred. Honest, because the audience will use this system.
14. **Backup.** Configuration example, install path, where to get help.

### Demo run order (slide 12 speaker notes)

1. Launch with `--demo`, or toggle Demo Mode in the UI while idle, to show both routes.
2. Show the live camera panels with the pose overlay — real inference on played frames.
3. Show the NI-DAQ live signal stream. Genuinely live.
4. Drive the pellet device manually. Genuinely live.
5. Open the protocol table, show cue timing and trigger profiles. Genuinely live.
6. Start a recording session; let trials run; show trial records accumulating.
7. Stop, and show the written session directory including its `DEMO` marker.
8. State explicitly which single element was pre-recorded, and why.

### Build method

Built with the `pptx` skill. Stored in the repo under `docs/presentation/`: the built
`.pptx` alongside a markdown outline carrying the per-slide content and speaker notes,
so the deck is reviewable in a diff and rebuildable.

Screenshots are captured during the rig rehearsal, which is required anyway. The deck
is built structurally complete first, with a named asset list, so capture is one pass
against a known shot list rather than an open-ended hunt. Candidate existing assets:
`temp/overlays/compare_left.mp4` and the `_raw2D_live.h5` live-inference outputs in
`temp/examplevids`.

No slide claims rig validation that has not happened. Where a figure is a target rather
than a measurement — the 5 ms p99 especially — the slide says so.

## Sequencing

Demo mode is built and rehearsed on the rig before the deck is finalized, because the
rehearsal settles the model-detection and calibration risks and produces the
screenshots. The deck's structure can be drafted in parallel; its assets and its status
slide cannot be finished until rehearsal results exist.
