# reachAQ overview deck — outline

Audience: brand-new users. The deck explains what the system does and why it is built
this way. It is not a code walkthrough.

Staged by feature, not by repository. Each feature slide carries its own three-way
lineage strip, so "where did this come from" is answered continuously rather than in
three segregated sections. Slide 3 is the only repository-organized slide.

Rebuild with:

    python docs/presentation/make_assets.py --media-root <checkout>/temp
    python docs/presentation/build_deck.py

---

## 1. Title

**reachAQ** — Automated mouse reach acquisition
Subtitle: Cerebellum Lab · Dell/Ubuntu rig

Notes: One sentence of welcome. Say that the last slide before backup is a live demo,
and that everything in it is real except the video.

---

## 2. What this system does

- A head-fixed mouse reaches for a pellet
- Several cameras watch at 150 frames per second
- The system finds the paw in 3D while the reach is still happening
- It decides, and acts, inside the same reach: cue, pellet, laser
- Every trial is recorded, timestamped, and reproducible

Notes: No architecture yet. The point is the closed loop: the system is a participant in
the experiment, not a recorder of it. Everything after this slide is how that is
achieved.

---

## 3. Where it came from

Asset: `lineage.png`

- **reach-training** — the original working rig. Python 3.8, PySpin cameras, an Arduino
  for the pellet, reach detection by camera ROI threshold, DeepLabCut run offline
  afterwards, then manual curation.
- **auto-trainer** — the modular platform reachAQ is built from. Split into core, video,
  device, inference, behavior, model, and pyside packages, with synchronized
  multi-process capture, integrated pose inference, stereo calibration, a CAN pellet
  device, and the session and metadata structure.
- **reachAQ** — this system. New hardware target, new stimulation subsystem, live 3D
  pose in the control loop, and a protocol engine.

Notes: The only slide organized by repository. Everything after this is organized by
feature, and each carries its own lineage strip. Say plainly: we did not start over, and
we did not just rename someone else's code.

---

## 4. Camera acquisition

Asset: `pipeline.png`

- 2 to 6 cameras, configurable, hardware-synchronized at 150 FPS
- Each camera runs in its own process, so one slow camera cannot stall the others
- A shared frame index keeps every camera on the same frame number
- A separate faster stim camera runs its own small-ROI detector

Lineage strip:
- reach-training: PySpin capture, cameras configured in `systemdata.yaml`
- auto-trainer: per-process `VideoCapture`, shared synchronized frame index, Spinnaker
  driver with camera-clock to `perf_counter` mapping
- reachAQ: 2–6 cameras rather than a fixed pair, configurable frame rate, stim camera
  at a multiple of the reach rate

Notes: The shared frame index is what makes 3D possible at all. Frame N on left must be
the same instant as frame N on right.

---

## 5. Pellet delivery and device control

- The pellet arm is a motorized device on a CAN bus, not a serial gadget
- The application checks the board's firmware capabilities before it uses a feature
- Presence evidence: the system knows whether a pellet is actually there
- Manual controls stay available at all times: home, load, present, release

Lineage strip:
- reach-training: Arduino over a USB serial link, text commands, hotkeys H/P/M/R
- auto-trainer: a CAN device abstraction behind a hardware interface
- reachAQ: SocketCAN/PCAN on Ubuntu x86_64 instead of the Jetson 40-pin adapter, plus
  firmware compatibility checking and typed failure causes

Notes: The firmware capability check matters because a protocol feature that needs a
newer board must refuse rather than silently misbehave.

---

## 6. Pose inference

Asset: `pose_backbones.png`

- A pose model finds keypoints on the paw and snout in every frame
- Two camera views are triangulated into a 3D position
- This runs live, during the session, not afterwards
- Confidence thresholds are measured per model, not guessed

Lineage strip:
- reach-training: DeepLabCut run offline after the session, then curated by hand
- auto-trainer: pose inference and FLIR stereo calibration integrated into the
  application
- reachAQ: live 3D computed in NumPy on the PyTorch backend, per-backbone confidence
  calibration, top-down model support

Notes: The image shows four candidate backbones on one frame with how many keypoints
each put above threshold. That spread is exactly why the confidence gate is calibrated
per model: a `cspnext_s` never scores above 0.33, so a single global threshold would
hold the reach flag permanently false.

---

## 7. Closed-loop reach detection

- The cue gate asks a live question: has the paw crossed into the reach volume?
- The answer comes from the live 3D pose, on the current frame
- If the answer is no at the deadline, the trial resets rather than firing late
- The reach-state source is selectable, so the gate can be driven different ways

Lineage strip:
- reach-training: a brightness threshold inside a drawn ROI, evaluated against an
  Arduino serial handshake
- auto-trainer: pose-derived behavioral state feeding a state machine
- reachAQ: live 3D pose answers the cue gate directly, with a deadline-accurate timer
  and a typed cancel reason when the gate is blocked or stale

Notes: This is the central claim of the whole system, and the thing that did not exist
before. Everything else on the previous slides exists to make this slide possible. Be
clear that reachAQ replaced an ROI brightness test with a tracked 3D coordinate.

---

## 8. Optogenetic stimulation

- Up to four lasers, each modeled independently
- Analog and digital control paths per laser
- Per-laser diode input and a command copy recorded alongside the data
- Shutter control, and a PMT shutter output
- Hardware-timed NI-DAQ tasks, not Python-loop timing

Lineage strip:
- reach-training: none — external stimulation was a separate LabVIEW system
- auto-trainer: none — no laser subsystem
- reachAQ: entirely new, built against the NI-DAQ channel inventory from the original
  LabVIEW control system

Notes: Say plainly that this is new and has no ancestor in either repository. The
LabVIEW system was used as a requirements inventory, not as a source of logic.

---

## 9. Trial protocols

- A trial is Tone 1, a drawn delay, then Tone 2
- Delay distributions use the published presets and probability vectors
- Stimulation epochs: baseline, stimulation, washout, sized in trials
- Every per-trial draw is seeded from the trial identity, so sessions replay identically
- Protocols are edited in the application and versioned on disk

Lineage strip:
- reach-training: fixed protocols in scripts, delays drawn live at runtime
- auto-trainer: a training plan and phase structure
- reachAQ: a protocol table with cue pairs and trigger profiles, published delay
  distributions, stimulation epochs, and an inter-trial timing target

Notes: The reproducibility point is worth dwelling on: reach-training drew at runtime,
so a session could not be replayed. reachAQ derives every draw from the trial identity.

---

## 10. Timing architecture

Asset: `two_tier.png`

- Two loops with separate budgets, deliberately kept apart
- Tier 1, the stim loop: small ROI on the fast camera, decision inline in the capture
  thread, output through a pre-armed hardware-timed NI-DAQ task. No process queue may be
  crossed.
- Tier 2, the behavior loop: reach cameras, pose, 3D, state machine, trial and protocol
  decisions
- A stimulus condition that needs a 3D coordinate belongs to Tier 2, not to the fast
  budget

Notes: **The 5 ms figure is a design target and a tail figure, not a measured result.**
It needs thread pinning, realtime scheduling, and no allocation or logging on the
trigger branch, and it is pending rig validation. Do not present it as achieved.

---

## 11. Data and metadata

- Every session writes video, timestamps, trial records, and events
- Session metadata is schema v2, and embeds the full system configuration that produced
  it
- Hardware state is recorded as both configured and runtime
- A validation command checks a finished session against a rule set

Lineage strip:
- reach-training: per-session YAML plus loose output files
- auto-trainer: project and session structure with a metadata schema
- reachAQ: schema v2 with configuration embedding, alignment and trial-summary
  artifacts, and `reachaq-validate-session`

Notes: The embedded configuration means a session can be interpreted years later without
the rig notes.

---

## 12. Demo

Assets: `demo_substitution.png`, `demo_frames.png`

- No mouse today, so the camera frames come from a recorded session
- Exactly one stage is substituted. Everything downstream is the real pipeline.
- Pose inference, NI-DAQ streaming, pellet control, protocols, and ports are all live

Notes — run order:

1. Launch with `--demo`, or toggle **Tools → Demo Mode** while idle, to show both routes.
2. Show the live camera panels with the pose overlay. Real inference, on played frames.
3. Show the NI-DAQ live signal stream. Genuinely live.
4. Drive the pellet device manually. Genuinely live.
5. Open the protocol table; show cue timing and trigger profiles. Genuinely live.
6. Start a recording session, let trials run, show trial records accumulating.
7. Stop, and show the written session directory including its `DEMO` marker file.
8. State explicitly which single element was pre-recorded, and why.

Point at the red banner in the status bar. It is there so nobody mistakes a demo session
for data.

---

## 13. Current status and limitations

Validated on the rig:
- Camera acquisition, recording, and session output
- Pellet delivery over CAN, with firmware compatibility checking
- NI-DAQ signal streaming and port configuration

Not yet validated on the rig:
- The 5 ms Tier 1 stim-loop budget, as a tail figure
- Every protocol timing, precheck, and cue behavior added in the current work, which
  needs TTL and video proof before experimental use

Deferred:
- Offline curation, RFID, and SoftMouse automation
- Tunnel and head-fix behavior inherited from auto-trainer

Notes: Be honest here. This audience will run experiments on this system, and a
surprise later costs far more than a caveat now.

---

## 14. Backup — getting started

- Install: `linux-install-instructions.md`, and `tools/install/reachaq-linux-install.sh`
- Run: `python -m reachAQ.app -c ~/Autotrainer/system_configuration.yaml`
- Demo mode: `docs/acquisition/demo-mode.md`
- Session recording and synchronization:
  `docs/acquisition/session-recording-and-synchronization.md`
- Trials and protocols: `docs/acquisition/session-trials-protocols.md`

Notes: Leave this up while taking questions.
