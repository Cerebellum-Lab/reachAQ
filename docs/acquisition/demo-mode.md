# Demo mode

Demo mode plays pre-recorded video through the real acquisition pipeline. It exists so
the system can be demonstrated when no animal is present.

Only the camera frames are substituted. Pose inference runs on the played frames, and
NI-DAQ streaming, pellet device control, laser control, protocol editing, and port
configuration are all live and real.

## Requirements

Demo mode runs on the rig. Live inference requires a working CUDA runtime and refuses
to start without one, so a machine with no GPU will show the video but no pose.

## Setup

Stage a recorded session as demo media and write the spec in one step:

    python scripts/stage_demo_media.py ~/Documents/rawdatalocal/<session-dir>

That copies each camera's first video part to `~/Autotrainer/demo/`, reads the frame
rate off the source, writes `~/Autotrainer/demo/demo_sources.yaml`, and validates the
result through the same loader the application uses. Add `--dry-run` to see what it
would stage, `--cameras left right` to restrict it, and `--force` to replace files that
already exist.

To do it by hand instead, copy `tools/hardware/reachaq_demo_sources.example.yaml` to
`~/Autotrainer/demo/demo_sources.yaml` and point it at real session video. Match `fps`
to the rate the source session was recorded at; the shipped rig records at 150 FPS.

Media files are never committed.

## Checking readiness before a demo

Two things can be wrong without crashing and without announcing themselves: the
configured pose model may not fire on the chosen video, and the rig calibration may not
match the geometry that video was recorded under. Either one produces a demo that plays
video and quietly never closes the loop, or a correct-looking 2D overlay over wrong 3D.

Run the check on the rig, well before the meeting:

    python scripts/verify_demo_readiness.py

It reports video geometry, the calibration's own `image_shape` and which session it was
built from, the effective confidence threshold for the configured backend and backbone,
and the per-body-part detection rate over the demo video. It exits non-zero if anything
fails.

`--skip-pose` runs the video and calibration checks only and needs no GPU. `--frames N`
sets how many frames to score. `--model` overrides the model named in the system
configuration.

Note that the reach cameras can have different crops: a `christie2P` session records
left at 256x258 and right at 260x258. The calibration's per-camera `image_shape` has to
match each of them, which is exactly what this check compares.

## Running

Start in demo mode:

    python -m reachAQ.app -c ~/Autotrainer/system_configuration.yaml --demo

With a specific spec:

    python -m reachAQ.app -c ~/Autotrainer/system_configuration.yaml --demo /path/to/spec.yaml

Or toggle it at runtime from **Tools → Demo Mode**. The toggle is available only while
the session is idle, because the configuration cannot be reloaded mid-recording.

`--demo` and `--random-cameras` are mutually exclusive.

## While demo mode is active

A red banner appears in the status bar and the window title gains a `DEMO MODE` suffix.
Both stay visible until demo mode is switched off.

## Demo session data

Demo sessions record normally. Every demo session is tagged twice:

- a `demoMode` block in the session metadata, recording the source videos and playback
  rate;
- a `DEMO` marker file in the session directory.

Session identifiers are unchanged. **Do not analyze demo sessions.**

## Camera names

Valid names are `left`, `right`, `camera3`, `camera4`, `camera5`, and `camera6`.
`stimCam` is accepted as an alias for `camera3`, matching the shipped rig
configuration.

A reach camera with no video in the spec is disabled for the run, rather than left
pointing at a physical camera the demo is not driving.

## Frame synchronization

Playback cameras start together on a shared barrier. This matters: hardware-synchronized
cameras guarantee that frame *N* on left matches frame *N* on right, but independently
started playback processes do not, and live 3D triangulation pairs by frame id. Without
the barrier the 2D overlay looks correct while the 3D is silently wrong.

The barrier covers the triangulated reach group only. The independent stim camera is
Tier 1, is never triangulated, and starts before that group with its own first-frame
wait, so including it would deadlock its own startup.

The wait is placed after each capture process publishes `RUNNING`, not before. Cameras
are prepared sequentially and each is waited on for `RUNNING`, so blocking any earlier
would deadlock the next camera's preparation.
