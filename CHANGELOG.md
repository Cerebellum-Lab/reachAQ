# Changelog

All notable changes to reachAQ are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Releases before 3.0.0 are not listed.

## [4.0.1] - 2026-11-25

### Fixed

- The installer's one-CUDA-runtime check reads the environment `conda run` runs in
  (`CONDA_PREFIX`), so its plugin-installer test no longer fails on a rig that has the
  DeepLabCut plugin installed.

## [4.0.0] - 2026-11-25

The project is renamed from Autotrainer to reachAQ throughout. There are no
compatibility aliases: every v3 name below stops working. Upgrade with the
installer, which migrates the settings; see "Upgrading from v3" in
[INSTALL.md](INSTALL.md).

### Changed (breaking)

- One distribution, `reachaq`, built from the top-level `pyproject.toml`, replaces
  `auto-trainer` and its seven sub-distributions (`auto-trainer-core`, `-video`,
  `-device`, `-inference`, `-behavior`, `-model`, `-pyside`) and
  `auto-trainer-tools`. `setup.py`, the sub-project `pyproject.toml` files and
  `auto_trainer_projects_order.txt` are gone.
- The code lives in one tree, `src/reachaq/`: `autotrainer.<pkg>` is now
  `reachaq.<pkg>`, `tools.acquisition` is `reachaq.app` (start it with
  `python -m reachaq.app`), and the other `tools.*` packages are `reachaq.hardware`,
  `.latency`, `.pellet_delivery`, `.platform_support`, `.session_validation`,
  `.simulation` and `.softmouse_sync`. Each package's tests are in `tests/<pkg>`.
  The `reachAQ/` launcher package is removed.
- The pose plugins are `reachaq-pose-deeplabcut` and `reachaq-pose-yolo`
  (`reachaq.pose_plugins.*`), registered in the `reachaq.pose_backends`
  entry-point group, which replaces `autotrainer.pose_backends`.
- Console scripts: `auto-trainer-headless` is `reachaq-headless`;
  `auto-trainer-local` is dropped (use `reachaq`).
- Environment variables: every `AUTOTRAINER_*` variable is `REACHAQ_*`. The
  application's SocketCAN settings now share the `REACHAQ_CAN_` prefix with the CAN
  bring-up script.
- Logger names start with `reachaq` instead of `autotrainer`.
- Settings: `~/Autotrainer` is now `~/reachAQ`; the preferences file
  `~/.config/Colorado/Auto Trainer.conf` is now `~/.config/reachAQ/reachAQ.conf`
  (QSettings organisation and application `reachAQ`); the running-status file is
  `~/.config/reachAQ/running_status.env`. `reachaq` and `reachaq-headless`
  refuse to start while only `~/Autotrainer` exists and `REACHAQ_CONFIG_DIR` is
  unset.
- Pose h5 files written before 2025-06-25 no longer load until converted:
  run `reachaq-convert-pose-h5 --check <dir>`, then `--in-place` or `--output-dir`.
  reachaq keeps no `autotrainer` module alias.
- The window title reads `"{serial} - reachAQ - Acquisition v{version}"`.

### Added

- `reachaq-migrate-settings` moves a v3 installation's settings: it renames
  `~/Autotrainer`, rewrites the old directory in the live yaml files there
  (keeping each original) and in `~/.config/reachaq/launcher.conf`, and copies the
  QSettings files. `--dry-run` shows the
  plan; `--reverse` restores the v3 layout byte for byte.
- `reachaq-convert-pose-h5` rewrites the pickled `SceneElement` labels of
  pre-2025-06-25 pose files as plain strings (`--check`, `--in-place` with
  `FILE.bak`, `--output-dir`). Converted files also load in v3.
- The installer upgrades a v3 install: it runs the settings migration,
  uninstalls every `auto-trainer*` distribution, restores the pose plugins v3
  registered, and replaces installed SoftMouse user units that still name a v3
  module.
- The installer stops at the first failed step and names it, checks the platform
  list (Ubuntu 22.04, NVIDIA driver 570.26 or later), uses the Spinnaker SDK and
  binding the customer downloads (3.2, no vendored FLIR wheels) and checks that the
  environment carries one CUDA runtime per engine.
- ChArUco calibration boards are the default (`board:` in the calibration
  settings; the plain checkerboard stays available), with partial board views, a
  board generator (`reachaq.calibration.make_charuco_board`) and a per-view fit in
  which board poses follow the stage. The per-camera reprojection grade uses each
  camera's own refit; stage accuracy is reported separately and only warns.

## [3.2.3] - 2026-11-24

### Fixed

- The pellet stage is stopped (ESTOP, all axes) when a stepper move or home is not
  acknowledged within its expected time, when the board reports a motion command
  busy, when a scout waypoint is not reached, when a capture or calibration fails
  or ends while the stage moves, and when the application disconnects or closes
  mid-move. A timed-out move is no longer re-sent. After a stop, the stage must be
  homed before it moves again; a blocking alert says why, and the stop is recorded
  in the session metadata.

### Added

- An always-enabled STOP button for the pellet stage.
- `pellet.profile_ack_deadlines` in `motor_config.yaml` (default on): per-move
  acknowledgement deadlines from each move's distance, speed and acceleration.
  Off falls back to a fixed 3 s timeout, which still stops the stage.

## [3.2.2] - 2026-11-24

### Added

- `kill -USR1 <pid>` prints every thread's stack from the running GUI, for diagnosing a freeze.

## [3.2.1] - 2026-11-24

### Added

- Pellet firmware 2.4.0 is accepted (config/pellet-firmware-compatibility.yaml); rollout pending.

### Fixed

- Lasers: a failed open still drives every configured laser safe; `close()` resets the command even
  when the shutter write fails; a laser Hold closes its shutter even when the restore fails; protocol
  lasers stay armed for the whole trial, up to 120 s; a stimulus trigger after the laser was released
  is refused; a timed-out Test stim cancels its armed output.
- Protocols: a Direct NI First Reach laser starts on First Reach, not its phase, and one no reach
  triggered is not a hardware error; a Stop during trial preparation halts its later hardware steps;
  the cycle's end waits for its laser without the executor lock; a row's Lock timing and post-clear
  delay reach the cue policy; recompiling the selected experiment reselects its new revision; a
  failed SEND is reported even when the operation has no recipe.
- The cue gate answers pellet presence from live pose, or from a camera ROI for Tone 2
  (`reachaq.video.pellet_roi`; a lock left held cannot wedge it, and a misfit ROI reads unknown).
- Analysis: a pellet seen again after a dropout is not counted as consumed.

## [3.2.0] - 2026-11-24

The last release before the rename; module and file names here are the v3 ones.

### Added

- Multi-camera, multi-resolution calibration library (`autotrainer.inference.multires_calibration`).
- Operator alerts (banner, alert log and data-loss dialogs), a first-run step for a new install or an
  unusable data folder, and `reachaq-diagnostics`, which exports a diagnostic bundle with secrets
  redacted (docs/acquisition/operator-alerts-and-diagnostics.md).

### Fixed

- Live pose follows the live queue's cameras and their order, pinned at the app and when a monitor
  is reused.
- Stage travel limits: positions, pellet positions and move lists outside the travel are refused
  before anything moves, checked against the base the board moves from; a refused SET is rolled
  back, no pellet is presented after a refused base position, and the intersession shift is applied
  to every axis or none.
- Closing or stopping acquisition never deletes a session unasked; a second close while a session
  saves keeps the window open; a failed or retried start stops what it started; cameras and the pose
  monitor brought back by a retry are supervised again; storage-limit and source-failure stops end
  a young session at once.

## [3.1.3] - 2026-11-19

### Fixed

- SoftMouse: every field the Animals list returns is written.

## [3.1.2] - 2026-11-19

### Changed

- Animal metadata setup asks only for the physical tag and RFID columns.

## [3.1.1] - 2026-11-19

### Added

- The reachAQ camera icon for the window and the desktop launcher.

## [3.1.0] - 2026-11-19

### Added

- A plain-torch keypoint pose backend and trainer, with a DeepLabCut dataset
  import, coverage-matched evaluation, and stoppable, resumable training.
- The pose training window, whose runs pause while a session records.
- A pose backend plugin contract and discovery; DeepLabCut and YOLO are opt-in
  plugin distributions, installed on demand and restored by the installer on update.
- Animal metadata read through pluggable sources (SoftMouse, spreadsheet,
  database, shared publication).
- A `--simulate` profile with a generated rig configuration, simulated cameras,
  NI stream, laser and STIM lines, and simulated-session provenance in validation.
- reachAQ training plans in the acquisition app, and a local NDJSON event API on an
  ephemeral port with a per-instance discovery file.
- NI delivery calibration (configuration v59), measured with a loopback tool.
- The NI laser runs in its own process, which drives it safe when the app goes,
  and latches FAILED when that process fails.
- `REACHAQ_CONFIG_DIR` names the settings directory.
- Pellet firmware 2.3.1: a refused command is stopped, reported, and ends the trial
  without presenting a pellet; a stepper driver fault is shown and blocks Record.

## [3.0.1] - 2026-11-10

### Added

- The board clock chain: per-board clock models and sequence trackers, board time
  fitted to NI time through claps and barcodes, recorded in `alignment.json`.

### Fixed

- READY after Stop is faster: 15.5 s instead of 89.8 s after a 45-minute session,
  2.3 s after a 5-minute NI-on one. The NI copies end 1 s past the session end.
- Stimulation mode: the stimCam stays out of live inference and its detector
  survives the spawn into its capture process.
- Tone durations, the board delay and the pre-reveal wait after the STIM pulse are
  timed to the millisecond.
- Validation warns on unscored pellet attempts instead of failing the session, and
  judges each board's sequence in its own order.

## [3.0.0] - 2026-11-06

First release of the reachAQ 3 line. It does not run against pellet firmware
v2.4.0.

[Unreleased]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.2.2...HEAD
[3.2.2]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.2.1...v3.2.2
[3.2.1]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.2.0...v3.2.1
[3.2.0]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.1.3...v3.2.0
[3.1.3]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.1.2...v3.1.3
[3.1.2]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.1.1...v3.1.2
[3.1.1]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.0.1...v3.1.0
[3.0.1]: https://github.com/Cerebellum-Lab/reachAQ/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/Cerebellum-Lab/reachAQ/releases/tag/v3.0.0
