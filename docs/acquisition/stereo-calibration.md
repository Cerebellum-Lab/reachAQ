# Multi-resolution stereo calibration

One calibration session a month gives a rig a 3D calibration for every
supported camera preset. The presets share one field of view, so a single
capture serves all of them as long as the cameras have not moved since.

Part 1 is the operator routine. Part 2 is the reference: files, presets, the
quality report, exports, and the limits of what has been measured. Part 3 is for
developers. For the session metadata that names the calibration a recording
used, see
[`alignment.json` and final metadata](session-recording-and-synchronization.md#alignmentjson-and-final-metadata).

---

# Part 1: the monthly routine

## Before you start

- **System Mode** is Idle and acquisition is stopped. **Make 3D calibration** is
  unavailable otherwise.
- A valid coordinate system (DCS) is loaded. The capture places the board
  through the DCS mapping, so a DCS that is out of date is fine; with none, the
  capture stops with "Please first calibrate diamond-triangle by starting the
  acquisition".
- The left camera is configured at binning 4, the live preset. A binning that
  is not a multiple of 4 stops the capture before the stage moves.
- The capture records left and right for now (the calibration dialog's choice of
  2 to 6 cameras comes later). Each camera's own configured geometry gives its
  presets, and `waypoints.yaml` records each camera's name, serial and camera id.

## Steps

1. **Mount the board on the motor.** It is the 4 mm board with 6 x 8 inner
   corners. Mount it so it moves with the stage and both cameras see it at the
   start position. The board is only translated, never tilted, and its
   orientation on the stage does not need to be measured: the fit solves for it.
2. Choose **Tools → Make 3D calibration**.
3. Wait while the stage runs. Both cameras record the full sensor (1440 x 1080,
   binning 1, gain 1.0 dB, gamma off, 4500 µs exposure, 40 fps) while the stage
   homes, goes to the start position, and then visits **37 positions**. Waypoint
   0 is the start and the other 36 are single-axis moves; one move repeats the
   previous position, so 36 are distinct. At each position the app commands the
   move, waits for the motor's acknowledgement, **settles** for as long as the
   move takes at the motor's configured speed (below), checks the position the
   motor reports, and **pauses 0.5 s** with the board still. On christielab10
   the settles and pauses add about 37 s in all. The cameras then stop and their
   own configurations are restored.

   The settle after each move is `distance / v_max + v_max / a_max + 0.25 s`, at
   least 0.25 s, with `v_max` and `a_max` the moving axis's maximum velocity and
   acceleration as the app wrote them to the motors at connect (from
   `~/Autotrainer/motor_config.yaml`). An axis without a usable configuration
   settles 1.0 s, and so does waypoint 0 when the motor has reported no position
   before it. On christielab10 (120 mm/s and 600 mm/s² on every axis) the settles
   run from 0.45 s to 0.59 s, the longest for the 17 mm x move. Each settle is
   logged at debug level.
4. The app finds the board at each pause, fits the cameras jointly, grades the
   fit, and shows the **report** (below). A camera that saw too little of the
   board is left out of the fit and is **2D only** (below).
5. **Redo the coordinate system.** Every new calibration makes the DCS out of
   date (below).

The waypoints are DCS coordinates (x -9.4 to 10, y 10 to 35, z -35 to -22 mm),
converted to motor coordinates through the loaded DCS. A coordinate that would be
below the stage's lower limit is commanded at the limit and logged; see
[Known rig facts and limits](#known-rig-facts-and-limits).

## The report

The report starts with every camera that could not be calibrated, then lists
failures, warnings and passes, each with its value and limits, then notes. The
thresholds are in [The quality report](#the-quality-report).

### A camera that saw too little: 2D only

Each camera must meet the per-camera rule before it is fitted:

1. at least 12 board views kept;
2. the views' commanded positions vary along each of the stage's x, y and z;
3. at least 8 of its views were also seen by another calibrated camera;
4. it is connected to the other calibrated cameras through shared views (two
   cameras are connected when they saw the board at the same position). Only the
   largest connected group is calibrated.

A camera that fails is **not calibrated**: the other cameras are fitted without
it, it records 2D only, and the report's first lines name it and the reason, for
example

```
NOT CALIBRATED (2D only): camera4: too few board views: 4 of 12 needed
```

The reasons are `too few board views: n of 12 needed`, `board views do not span
all 3 stage axes`, `too few views shared with another camera: n of 8 needed` and
`not connected to the other cameras through shared views`. Rule 3 is applied
until nothing changes, so a camera whose shared views are only with cameras that
were left out is left out too. A 2D-only camera makes the verdict at least warn.
Its views, rejections and coverage image are still in the report.

If fewer than 2 cameras meet the rule, the calibration is refused, nothing is
written, and **3D calibration failed** names each camera and its reason.

| Verdict | Meaning | What happens |
| --- | --- | --- |
| pass | Every metric is within its warn limit | Installed at once; the report is shown |
| warn | At least one metric is beyond its warn limit, none beyond its fail limit | Installed at once; the report is shown. Read the warnings; nothing blocks |
| fail | At least one metric is beyond its fail limit | Not installed. A dialog offers **Keep previous** (the default) or **Install anyway…** |

- **Keep previous** changes nothing: the active calibration and the DCS stay as
  they are. The failed calibration remains in the capture's session folder as
  `4mm_6r_8c_4x.new` with its report, so you can see why. Correct the cause and
  run the capture again.
- **Install anyway…** asks a second time, defaulting to No. On Yes the override
  is recorded as `confirmed`, `by` (the operating-system user) and `at` (local
  time). It is recorded as a line in the library's `decisions.jsonl`
  (`event: override`, with the record under `detail`); the calibration itself is
  never edited, because sessions name it by id (see Calibration library).
- If the capture or the fit itself raises (for example, a camera's pauses do not
  match the waypoints, fewer than 2 cameras meet the per-camera rule, or the fit
  does not converge), nothing is installed and
  **3D calibration failed** shows the reason. The capture folder stays in the
  session folder.

An install takes effect at once: live 3D reloads the new calibration without a
restart. If that reload fails (an I/O fault, say), the calibration is still
installed and the app says "3D calibration installed, but it could not be
loaded: <reason>. Restart the app." Live 3D is then off, never left on the
previous calibration's parameters: the installed calibration is the active one,
so a DCS measured against the previous one is out of date and Record is blocked,
and a session recorded meanwhile carries `live3dDisabledReason`, which offline
3D refuses. The next configuration load or a restart loads the installed entry
again. Any explicit reload that fails (the DCS run's too) leaves live 3D off the
same way.

## After a new calibration: redo the coordinate system

A 3D calibration changes the frame that every DCS value was measured in. When
the active calibration is not the one the DCS was measured against:

- The app says so once: "Coordinate system out of date: it was measured against
  a different 3D calibration. Redo Tools → Calibrate Coordinate System before
  running sessions."
- **Record is blocked** with "Coordinate system out of date: it was measured
  against a different 3D calibration. Redo Tools → Calibrate Coordinate
  System".
- Everything that uses DCS-space coordinates is held off as if no DCS were
  loaded: motor moves given in DCS (refused with "DCS move refused:" and the
  message above), drift monitoring, the pellet checks, the pellet shift. Live 2D pose and raw 3D keep running. The DCS files are left in
  place, not deleted; they count as out of date because the calibration id they
  recorded differs from the active one.
- **Make 3D calibration still works**, so a capture you are not happy with can
  be repeated before the DCS is redone.

To redo it, set System Mode to Running, move the pellet spoon to a typical
delivery position with the Set and Send buttons, and choose **Tools → Calibrate
Coordinate System**. DCS records the id of the calibration it was measured
against in `diamond_triangle_offset.yaml` and writes fresh `camera_offsets.pkl`
for the active calibration: beside a library entry, as
`calibrations/.offsets/<id>/camera_offsets.pkl`, because an entry is never
written to; inside the folder for a calibration folder given explicitly (the
simulator's). That clears the out-of-date state and
unblocks Record. A DCS saved before calibration ids existed has none, so it is
out of date against any format-2 calibration. A legacy calibration has no id: a
DCS without one is accepted with it, but a DCS that records an id was measured
in a format-2 calibration's frame and is out of date under a legacy one, for
example after rolling back to a legacy backup.

## The monthly reminder

When the active calibration is 30 days old or more, or there is none, or its
date cannot be read, the app suggests a new one. It checks once per run, at the
first configuration load:

- a non-modal **Calibration due** box gives the date of the last calibration
  ("never" if there is none, "unknown (could not be read)" if the date is
  unreadable) and names **Tools → Make 3D calibration**;
- the status bar shows **3D calibration due (last: YYYY-MM-DD)**. It stays until
  a new calibration is installed. Keeping the previous calibration after a fail,
  or a failed install, leaves it up.

The reminder never starts a calibration; the operator does, when the rig is
ready. The threshold is the Qt setting `calibration/reminder_days` (an integer,
default 30), read as `UserPreferences.calibration_reminder_days`. There is no
field for it in the Preferences dialog. A value of 0 or less counts every
calibration as due; there is no off switch. A threshold crossed mid-run is not
noticed until the next launch.

---

# Part 2: reference

## What gets written and where

The capture creates a session folder in the normal data location. In it:

- `4mm_6r_8c_4x/` is the capture: `source_videos/left.mp4`,
  `source_videos/right.mp4`, a copy of each camera's timestamps file as
  `source_videos/left_timestamps.txt` and `right_timestamps.txt`, and
  `waypoints.yaml` (the commanded and reported motor positions, which waypoints
  were verified, the times, and the camera settings).
- `4mm_6r_8c_4x.new/` is the new calibration, at format 3. It stays after an
  install as the session's record, and when a fail is kept out.

An install adds `.new` to the **calibration library** (next section) as a new
entry, `<configuration location>/calibrations/<id>/` (default `~/Autotrainer`),
and makes it the active entry; the app loads it from there. If the active
pointer cannot be written, the entry is removed again and the previous
calibration stays active, so the install can be retried. A `.new` folder holding
a `camera_offsets.pkl` is refused.

The `4mm_6r_8c_4x` folder and its dated backups `4mm_6r_8c_4x_<YYYYMMDD_HHMMSS>`
from before the library are no longer written. They stay in place, read only,
for the sessions recorded with them and for an older app version, which still
loads `4mm_6r_8c_4x` and so sees the calibration that was active when the
library took over.

## Calibration library

Every calibration is kept as an immutable entry under
`<configuration location>/calibrations/` (`~/Autotrainer/calibrations/`):

```
calibrations/
  <id>/               one calibration, at format 3 (or a copy of an older one)
  active.yaml         the one mutable pointer: id, by (who), at (when)
  decisions.jsonl     append-only, one JSON object per line: event, id, at, ...
  settings.yaml       library-wide switches (only the file is named here)
  .derived/           anipose files derived on demand; never inside an entry
```

- **An entry never changes.** A session records the calibration's id, so the
  entry it names must still be the one it was recorded with. An entry is copied
  to `<id>.partial` and renamed into place, so a power cut leaves no entry that
  loads, and nothing writes inside it afterwards. Adding an id that is already
  there is refused.
- **What happens to an entry afterwards is a decision.** Making it active and
  installing it over a failed quality check (**Install anyway…**) are lines in
  `decisions.jsonl`, for example
  `{"event": "set_active", "id": "…", "by": "ben", "at": "2026-11-20T14:05:06"}`.
  The log is only ever appended to.
- **`active.yaml` is replaced whole** (temp file, then rename), so a power cut
  leaves the old pointer or the new one. A missing or corrupt `active.yaml`, or
  one that names an entry that is not there, means there is no active
  calibration: the library reports none and logs why; it never raises.
- **Older calibrations are imported, not moved.** `import_existing` copies the
  `4mm_6r_8c_4x` folder and every dated backup into the library. A format-2
  folder keeps its id. A legacy folder (no `calibration.yaml`) gets a new id,
  its `stereo_params.pickle` time as its date, and a generated
  `calibration.yaml` marked `format: legacy`. It sets the active pointer to the
  active folder's entry when there is none yet, skips anything already in the
  library (so it can run again), and leaves the old folders as they were.
- **Finding a calibration.** A session's metadata id is looked up in the library
  first, then among the old folders. A session with no id is resolved by time:
  the entry the log shows as active then, or, for a time before the log begins,
  the old folders' dated backups as before.
- **The app loads the active entry.** At start, and whenever no calibration
  folder is given, it reads `active.yaml`. When the library has no `active.yaml`
  yet, the app first runs `import_existing`, so a rig keeps its calibration on
  its first start with the library. A missing, corrupt or dangling pointer, or
  an entry that cannot be read (a half-written file after a power cut), never
  stops the app: it records 2D, and `live3dDisabledReason` says why, for example
  "no active 3D calibration: …/calibrations/active.yaml does not exist" or
  "could not load …/calibrations/<id>: …".
- **Cameras are matched by serial.** A configured camera is the calibration's
  camera with the same serial (`host` of a Spinnaker camera), so renaming a
  camera keeps its calibration. A camera with no serial on either side
  (`random` or `playback`, or a format-2 or legacy calibration, which recorded
  none) is matched by name. Each configured camera is either calibrated or 2D
  only with a reason: `not in the active calibration`, `swapped: calibrated
  serial X, now Y` (a new body under a calibrated camera's name), the
  calibration's own reason for a camera it could not calibrate, or `no active 3D
  calibration`. The reasons are recorded per camera in the session metadata.
- **Live 3D is still the left-right pair.** It runs when the configured left and
  right are the calibration's left and right, both calibrated, at the same
  geometry, and the calibration holds that pair at it; the preset comes through
  `load_preset_set(...).stereo`. Otherwise live 3D is off with the reason, for
  example "live 3D needs left and right calibrated as left and right: left:
  swapped: calibrated serial 21000001, now 21000099". 2D pose is unaffected.
- **Live 3D is a setting** (**Preferences → Behavior → Live 3D**), on by default, so
  nothing changes unless you switch it off. It is stored as `live3d: true` or
  `false` in `calibrations/settings.yaml`; a missing, empty or corrupt file
  means on, and the file is replaced whole (temp file, then rename), keeping any
  other key in it. Switching it reinstalls the pose algorithm at once, and only
  while the session is Ready.
  - **Off**, 2D pose keeps running. The drift monitor and the DCS are
    unavailable, with the reason "live 3D is off": the loaded DCS configuration
    is kept but not used (as when it is out of date), a move given in DCS is
    refused with that reason, and animals' saved pellet positions in DCS are
    kept, not zeroed. Preferences shows "Drift monitor and DCS need live 3D"
    beside the switch.
  - Nothing switches it on by itself: not a calibration, not an install.
  - The session metadata records it as `calibration3d.live3d`:
    `{enabled, disabledReason}`. With live 3D off, `live3dDisabledReason` is
    "live 3D is off" too. Offline 3D refuses a session with a
    `live3dDisabledReason` (its 2D points' preset is not recorded), so a session
    recorded with live 3D off cannot be triangulated offline for now.

A format-3 folder, for any 2 to 6 cameras:

```
4mm_6r_8c_4x/
  calibration.yaml          the index, written last: format, id, created, board,
                            cameras (name, serial, camera_id, calibrated, reason),
                            preset_sets (name, and each calibrated camera's binning,
                            size and ROI offset), verdict, override, motor_scale,
                            capture (settings, statistics, and capture_folder: the
                            capture's absolute path)
  sensor/fit.yaml           the full fit in sensor pixels: K, dist, rvec, tvec per
                            calibrated camera, board rotation, motor scale, RMS,
                            parameter std
  calibration.toml          the anipose export of every calibrated camera, in sensor pixels
  presets/<set>/            one per preset set (see Presets), for example presets/live/
    calibration.toml        the anipose export, each camera at its geometry in the set
    README.txt
    camera_matrix/stereo_params.pickle           } only when left and right are both
    camera_matrix/left_intrinsic_params.pickle   } calibrated and share a geometry in
    camera_matrix/right_intrinsic_params.pickle  } this set: today's left-right files,
    frame.yaml              the 3D frame         } read by live 3D and DeepLabCut
  quality_report.json       the report, with the thresholds it was graded with
  quality_report.txt        the operator's summary
  coverage_<camera>.png     one per capture camera, calibrated or not
  waypoints.yaml            copied from the capture
```

DCS's `camera_offsets.pkl` for a library entry is kept beside it, in
`calibrations/.offsets/<id>/`. An entry imported from an older folder carries
that folder's own `camera_offsets.pkl`, which is read when `.offsets` has none.

- A camera that saw too little is listed in `cameras` with `calibrated: false`
  and its reason. It is in no `calibration.toml` and no preset set: it is 2D
  only.
- Nothing inside the folder is written again by reading it. A geometry no
  stored set has (a camera recorded at another binning or ROI) gets its anipose
  file derived from `sensor/fit.yaml` on demand, into a cache outside the
  folder, `calibrations/.derived/<id>/<key>/calibration.toml` in the library.
- **Format 2** (calibrations made before 2026-11-20) still loads: left and right
  only, one folder per preset, `presets/<WxH_binB>/`, each with the
  `camera_matrix` files, `frame.yaml` and a two-camera `calibration.toml`, and no
  top-level `calibration.toml`. Its index reads as cameras left and right, both
  calibrated, with one preset set per preset.
- A folder without `calibration.yaml` at format 2 or 3 is a **legacy**
  calibration (`camera_matrix/stereo_params.pickle` at the top) and is loaded
  the old way. It has no id; its date is its `stereo_params.pickle`
  modification time.
- `calibration.yaml` is written last, into a `.partial` folder that is renamed
  into place, so a half-written folder never loads.
- **`camera_offsets.pkl` is never carried over.** The install refuses a new
  folder that has one, and nothing copies the previous calibration's to the new
  one. Offsets are measured by DCS against a particular calibration, so the
  previous ones are wrong in the new frame. They stay with the previous entry.
- **The capture videos are not copied** into the calibration, nor into its
  install and backups. They stay in the session's capture folder, which
  `calibration.yaml` names as `capture.capture_folder` (absolute), so a
  calibration can be recomputed later as long as that session folder is kept;
  see Part 3.

## Presets

A preset set gives every calibrated camera one geometry. Per-camera sets come
from `geometry.presets_for_cameras` of each camera's live geometry: `live` (the
cameras as they record) and `bin1`, `bin2` and `bin4`, each camera's own live
window at that binning. A binning is left out when some camera's window does
not fall on whole pixels at it, or when it would repeat `live`. The capture
passes each camera's own geometry, and the sets are built after the fit from
the calibrated cameras only (`pipeline.calibrated_preset_sets`), so a camera
left 2D only costs no other camera a binning. The writer also drops a set that
repeats an earlier one's geometries once its uncalibrated cameras are out.
Older calibrations, and a capture given only presets, have every camera share
each preset of `geometry.presets_from_base`, with each set named after its
preset, for example `presets/256x256_bin4/`.

The presets are `geometry.presets_from_base` of the live camera geometry, with
binnings 4, 2 and 1. For the base geometry of 256 x 256 at binning 4 and ROI
offset (52, 6), which is the SpinCam default and the rig's:

| Preset (folder) | Binning | Size | ROI offset (binned px) | Sensor window (px) |
| --- | --- | --- | --- | --- |
| `256x256_bin4` | 4 | 256 x 256 | (52, 6) | 1024 x 1024 at (208, 24) |
| `512x512_bin2` | 2 | 512 x 512 | (104, 12) | same |
| `1024x1024_bin1` | 1 | 1024 x 1024 | (208, 24) | same |

Each preset holds intrinsics in that preset's exact pixels and the same
extrinsics. A different base gives different presets; the base binning must be
a multiple of 4.

**Live inference** picks the preset by the left and right cameras'
configurations, which must give the same geometry. Live frames reach inference
pooled to the base capture whatever the capture-binning preset, so the base
geometry (binning, size, ROI offset, as saved, with SpinCam's defaults for
anything left out) is the pixel space live 3D works in. The preset whose
geometry equals it is loaded. If there is none, or left and right differ, live
3D stays off, the operator is told why, and 2D pose keeps running. Another
preset is never substituted, because its pixels are not the frames' pixels.

**Offline analysis** resolves the calibration a session used from its metadata,
`hardware.configured.calibration3d`:

- `id` finds the calibration under `--calib-root` (default `~/Autotrainer`),
  whether it is the active folder or a dated backup. With no id, the session's
  start time picks the calibration that was active then.
- `preset` names the preset to read. A session with no `calibration3d`, or one
  recorded on a legacy calibration, is read at the base preset (the largest
  binning).
- A session recorded with live 3D off (`live3dDisabledReason`), an id not found,
  an ambiguous match or no match is refused with "pass --calib-dir".
- A session older than every kept calibration is refused too, with
  "pass --calib-dir", the session's time and when the calibration it would have
  landed on was made: the calibration it was recorded with is gone, and the
  oldest kept one is no stand-in. By time, a calibration made after the session
  is never picked. A session that names its calibration by id is not affected.
  On christielab10 this means sessions before 2026-07-07 (the legacy
  calibration's date) need `--calib-dir`.
- `--calib-dir` overrides all of it.

reachAQ itself triangulates only at the base preset: live frames and the
recorded live 2D files are in that space. The 512 and 1024 presets are for
external tools (see Exports).

## The quality report

Each metric has a (warn, fail) pair. For most, the status is fail when the value
is above the fail limit, else warn when above the warn limit. For `views_kept`,
`covisible_views` and the two coverage metrics, a smaller value is worse. The
verdict is the worst status, and at least warn when a camera is 2D only. A value
that cannot be determined is infinite and takes the worst status its limits
allow, so an undetermined quantity cannot pass. The report records the thresholds
it was graded with.

Only the calibrated cameras are graded. A pair metric is graded for every pair
of them that saw the board at the same position, and its camera is the pair,
for example `left-right`.

| Metric | Scope | Warn | Fail | Meaning |
| --- | --- | --- | --- | --- |
| `views_kept` | camera | < 20 | < 10 | Distinct positions where this camera's board detection was kept (at most 36) |
| `covisible_views` | camera | < 8 | < 4 | Its views that another calibrated camera also saw. The fit does not need them; they expose stage geometry the model leaves out (axis skew) |
| `reprojection_rms_px` | camera | > 0.5 | > 1.0 | RMS distance between detected corners and the fit's projection, sensor px |
| `preset_reprojection_rms_px` | camera, per preset | none | none | Informational. The same RMS in that preset's pixels (sensor RMS / binning); the note names the preset |
| `coverage_roi_fraction` | camera | < 0.5 | < 0.25 | Fraction of the base preset's sensor window inside the union of the detected-corner hulls |
| `coverage_frame_fraction` | camera | none | none | Informational. The same coverage as a fraction of the full 1440 x 1080 sensor frame |
| `fx_fy_ratio_dev` | camera | > 0.01 | > 0.03 | \|fx/fy - 1\|; a large value is a sign of a poorly constrained fit |
| `focal_rel_std` | camera | > 0.005 | > 0.02 | One-sigma uncertainty of fx, relative to fx; fails if the capture does not determine it |
| `focal_pair_rel` | pair | > 0.10 | none | \|fx a - fx b\| over their mean |
| `stereo_reprojection_rms_px` | capture | none | none | Informational. RMS of the joint fit over every corner of every calibrated camera, sensor px |
| `baseline_mm` | pair | none | none | Informational. Distance between the two camera centres, mm |
| `principal_point_offset_frac` | camera | > 0.2 | > 0.4 | Distance of (cx, cy) from the window centre, each axis as a fraction of the window half-side |
| `max_distortion_shift_px` | camera | > 60 | > 150 | Largest displacement the distortion model makes along the window border, sensor px |
| `epipolar_error_px` | pair | > 0.5 | > 1.0 | Mean symmetric epipolar distance, sensor px, over positions both cameras saw |
| `square_size_rel_error` | pair | > 0.01 | > 0.03 | \|mean triangulated square - 4 mm\| / 4 mm |
| `loo_triangulation_px` | camera, 3 or more cameras | > 0.5 | > 1.0 | Median distance, sensor px, between this camera's corners and the points the other calibrated cameras triangulate, projected into it, over positions it and at least two others saw |
| `motor_scale_dev` | capture | > 0.01 | none | \|motor scale - 1\|: true travel (from the board's squares) against commanded travel |
| `waypoints_unverified` | capture | > 0 | none | Positions where the motor's reported position did not come within 0.05 mm of the command in time. Verified means the motor reported the commanded position, not a traced arrival |
| `waypoints_unreached` | capture | none | > 0 | Positions where the reported position was more than 0.1 mm from the commanded one on some axis. The fit places the board at the commanded position, so any such view is in the wrong place. A position with no finite reported position is not counted |
| `cross_check_delta_px` | binned preset | > 1.5 | none | Largest difference in fx, fy, cx or cy between a direct fit in the preset's own pooled frames and the mapped sensor fit, preset px. Informational: the direct fit has a noise floor near 0.5 px |
| `cross_check_corner_offset_px` | binned preset | > 0.1 | > 0.25 | Mean offset between corners detected in the preset's pooled frames and the sensor corners mapped into it, preset px. Tests the pixel-centre mapping itself |
| `camera_rotation_change_deg` | camera, vs previous | > 0.5 | none | Rotation relative to the motor axes since the previous calibration; above the limit the report says the camera moved |
| `baseline_change_mm` | pair, vs previous | > 1.0 | none | Change of the vector between the two camera centres since the previous calibration, for every pair both calibrations have |
| `focal_change_rel` | camera, vs previous | > 0.02 | none | Relative change of fx since the previous calibration |

- An informational metric has no limits: it is recorded so months can be
  compared and never changes the verdict.
- The change metrics appear only when the previous calibration is a format-2
  or format-3 folder whose fit can be read. Otherwise the report notes "no
  comparable previous calibration". They compare the cameras both calibrations
  have, so a format-2 (left and right) previous is compared on left and right,
  and the report notes each newly added camera as not in the previous one.
- A binning-1 preset is a crop of the frames the sensor fit used, so it has no
  cross-check.
- Notes list what the metrics do not: views rejected per camera and the reason
  (`not found`, `inconsistent`), a camera whose corner order was reversed to
  match the others, each unreached waypoint with the position the motor
  reported, and the still frames per waypoint (shortest, where, median).
- `coverage_<cam>.png` shows the full sensor, the covered area, the ROI
  rectangle and every detected corner.

## Exports

Every preset set folder is an anipose calibration, and, when it holds left and
right at one geometry, a DeepLabCut 3D `camera_matrix` folder. All pixel values
are in the set's pixels, each camera's own, so use a set's files only with
video each camera recorded at its binning and ROI. `README.txt` in the folder
lists the geometries.

- **DeepLabCut 3D.** `camera_matrix/stereo_params.pickle` holds one key,
  `"left-right"`, with the keys and shapes of the pre-existing file
  (`cameraMatrix1/2`, `distCoeffs1/2`, `R`, `T`, `E`, `F`, `R1/R2`, `P1/P2`,
  `roi1/roi2`, `Q`, `image_shape`, `rot_cor`), so existing loaders read it
  unchanged. `T` is in mm and `rot_cor` is 0.0: new calibrations rotate no
  pixels. `{left,right}_intrinsic_params.pickle` hold `{cam: {"mtx", "dist"}}`.
  DLC keys the pair by the project's camera names, so the DLC project's
  `config.yaml` must set `camera_names: ["left", "right"]`. Triangulated points
  are in the rectified left-camera frame, in mm.
- **anipose.** `calibration.toml` is in aniposelib's `CameraGroup` format and
  loads with aniposelib 0.8.0's `CameraGroup.load`. It has one table per
  calibrated camera, `cam_0` to at most `cam_5` in capture order, each with
  name (the camera's name, as its video files carry it), size `[w, h]`, matrix,
  five distortion coefficients, rotation (Rodrigues, motor frame to camera) and
  translation in mm. The camera poses are in the motor frame. `[metadata]`
  holds `adjusted = false`, `error` (the mean of the cameras' RMS, each in its
  own pixels), `calibration_id`, `cameras` (`[{name, serial}]`, with no serial
  key for a camera that has none), `units = "mm"` and `frame = "motor"`.
  A format-2 folder's file has only left and right and only `adjusted` and
  `error` in its metadata.
- **Frame transform.** `frame.yaml` holds the 3 x 3 `transform` M, `units: mm`
  and the per-axis `signs`. Live and offline 3D compute
  `oriented = (raw - camera_offsets) @ M`, where `raw` is the triangulation
  with P1 and P2. M turns the rectified left-camera frame into the motor axes in
  millimetres, with the signs of the DCS motor-to-diamond map, so
  `diamond_to_motor` is unchanged. It replaces the earlier reorientation from
  hand-entered camera angles and the square-size scaling. The origin comes from
  DCS, through `camera_offsets.pkl`.

## Known rig facts and limits

- **Motor x stops at 0.** Measured on christielab10 on 2026-10-26: a move to x
  -1.722 left the stage at 0.000. The firmware stops there; `_move_axis` does not
  clamp, `HardwareModel._set_axis` does. `plan_waypoints` therefore commands any
  motor coordinate below 0 at 0, on every axis, and logs which waypoints were
  clamped. On that rig's DCS, 12 of the 37 waypoints were affected (DCS x -9.4
  and -9.0). The fit uses the clamped positions, which are where the stage is, so
  the low-x sweep is shorter by up to 1.7 mm there; watch `coverage_roi_fraction`.
  The upper travel limit is unknown; `waypoints_unreached` fails any waypoint
  that does not arrive.
- **`last_position` reports once per move, before the acknowledgement.** On the
  same check it updated about 0.10 s after the command, to the target (error at
  most 0.007 mm), and the acknowledgement followed 0.20 to 0.30 s after the
  command. "Verified" therefore means the motor reported the commanded
  position, not that the arrival was traced; the report's `waypoints_unverified`
  line says so. Whether the acknowledgement means the move ended was not shown
  for long moves, so the settle covers the whole move as if it meant only
  "received" (see the routine above). `still_frames` in `calibration.yaml`
  records how long the board held still at each waypoint, per camera (the
  shortest, the waypoint it was at, and the median, in frames), and the report
  notes it; a waypoint needs 8.
- **Older code cannot read a format-2 calibration folder.** Code from before
  this branch has no loader for `calibration.yaml`, and the DCS file written
  against a format-2 calibration has a `calibration_id` field it does not
  accept. Install a format-2 calibration only on a rig running this branch or
  later.
- **Format-2 code cannot read a format-3 calibration folder.** It takes a
  `calibration.yaml` at another format for a legacy folder, which then fails to
  load. Install a format-3 calibration only on a rig whose code reads format 3.
- **A crash during an install.** The entry is copied as `<id>.partial` and
  renamed into place, then `active.yaml` is replaced whole, so a crash leaves
  the previous calibration active, or the new one, never half of either. A
  leftover `<id>.partial` is removed by the next write of that id.
- **Camera movement is checked against the previous calibration only.** Both
  cameras shifting by the same vector cannot be told from the board being
  remounted, and the change metrics do not see it. The DCS, redone after every
  calibration, measures the origin.
- **The anipose file has been checked without aniposelib.** It is tested by
  reading it back and repeating `CameraGroup.triangulate`'s arithmetic
  (`cv2.undistortPoints`, then a DLT per camera pose); aniposelib was not
  installed. The DeepLabCut pickle is tested against DLC's own reader.
- **The fit holds k3 at 0** (see Part 3).

---

# Part 3: developer notes

## Where the maths lives

`auto-trainer-inference/src/autotrainer/inference/multires_calibration/`:

| Module | Holds |
| --- | --- |
| `geometry.py` | `PresetGeometry`, `presets_from_base`, the sensor-to-preset mapping |
| `extract.py` | stationary segments, board detection, per-waypoint observations |
| `fit.py` | the joint motor-frame fit (`fit_motor_frame`) |
| `report.py` | `build_report`, `DEFAULT_THRESHOLDS`, coverage |
| `products.py` | per-preset stereo parameters, the frame transform, the anipose export |
| `store.py` | the folder format, loading, resolving by id or time |
| `library.py` | the calibration library: entries, the active pointer, decisions, import |
| `pipeline.py` | `calibrate_capture`, which ties them together |

Capture and install live in `tools/acquisition`: `model/handle_3d_calibration.py`
(waypoints, pauses, `capture_settings`), `model/calibration_install.py`,
`model/dcs_state.py` (VALID, STALE or MISSING), `model/calibration_reminder.py`,
`model/calibration_geometry.py` (the live preset from the camera configuration)
and `view/main_window.py` (`on_3d_calibrate`, `_finish_3d_calib`, the reminder
and the status-bar label).

## Pixel-centre mapping

The fit is done once, in full-sensor binning-1 pixels (1440 x 1080), and each
preset is derived from it. Pixel i is centred on coordinate i, and a binned
pixel averages b sensor pixels, so for binning b and ROI offset (ox, oy) in
binned pixels:

```
u_b = (u_s + 0.5) / b - 0.5 - ox        v_b = (v_s + 0.5) / b - 0.5 - oy
fx_b = fx_s / b                          fy_b = fy_s / b
cx_b = (cx_s + 0.5) / b - 0.5 - ox       cy_b = (cy_s + 0.5) / b - 0.5 - oy
```

Distortion, R and T are unchanged: the distortion acts on normalised
coordinates, before the intrinsic matrix. Rectification is recomputed at the
preset's (width, height). Treating a binned pixel as a point at its corner
(the pixel-edge convention) is off by (b - 1) / 2b per axis, 0.375 px at binning
4, which is what `cross_check_corner_offset_px` would catch.

## The fit and its gauge

The board never tilts, so each camera's views are parallel planes, and a planar
calibration cannot separate focal length from distance. The commanded motor
positions can: all corners together form one non-planar target,

```
X_w,j = R(board_rvec) P_j + board_tvec + sigma * m_w
X_cam = R(rvec_c) X_w + tvec_c        (mm)
```

with `P_j` the board corner, `m_w` the commanded position of waypoint w and
`sigma` the motor scale, which the fit measures against the board's squares.
Every calibrated camera's intrinsics and pose are fitted jointly by least
squares, 2 to 6 cameras, each from the waypoints it saw.

- **Gauge.** `board_tvec` is held at zero. Moving the board origin by d and
  every camera by -R_c d changes no projection, so the motor frame's origin is a
  choice, not a measurement; here it is board corner 0 at motor position 0.
  Fitting it too would leave the normal equations singular. The DCS sets the live
  origin later. This is why the change metrics compare the baseline between the
  cameras and each camera's rotation against the motor axes, never a camera's
  own position: the board is remounted every month, so a lone camera position
  would read "moved" every time.
- **k3 is held at 0.** The board covers only the middle of the sensor, and with
  k3 free the fit pushed it to about -8 +/- 6 on the synthetic rig, moving the
  distortion at the corners of the 1024 window by 2.7 px while fitting the
  covered area no better.
- Parameter standard deviations come from the Jacobian. A direction below 1/1e9
  of the largest eigenvalue counts as undetermined, and its standard deviation
  is infinite.
- The planar start is tried first and kept if it converges with every camera at
  or below 1.0 px RMS. Otherwise the fit is also started from a grid of focal
  lengths, and of all the starts that converged the lowest cost is kept. A start
  that has not converged in 400 evaluations is abandoned; `RuntimeError` is
  raised only when every start fails.

Extraction takes each distinct waypoint's frames from its logged dwell: the
frames whose wall-clock time (column 0 of the camera's timestamps file) lies
between the waypoint's logged arrival and leave, from the first of a run of equal
positions to the last. The arrival is stamped after the settle, so these are the
paused board's frames whatever the image shows. A capture without timestamps
files (made before 2026-11-20) is split into stationary segments by frame-to-frame
motion instead, and segment i matched to distinct waypoint i by order. Motion
alone failed the first real capture: a dark image's flicker split dwells and
moves it barely saw merged them. `WaypointAlignmentError` is raised when a
camera's frame times do not number its frames, a dwell holds fewer than 8 frames,
a motion-segmented camera's segment count differs from the number of distinct
waypoints, or two cameras' dwells start more than 3 frames apart. The
calibration's `capture.segments_from` records which way was used. At each
segment the board is detected in the 5 middle frames and the per-corner median is
the observation. A camera is dropped from a waypoint as `not found` if no frame
has the board, or as `inconsistent` if any corner's detections spread by more
than 1.0 px (RMS from their mean).

The frame transform is `M = (diag(signs) @ R_left^T @ R1^T)^T`, with `R_left`
the left camera's rotation and `R1` the rectification rotation. The signs are
measured from `DiamondTriangleOffsetConfig.motor_to_diamond`, the path the live
drift monitor uses, not copied from its flips and not taken from the test-only
`motor_to_inference` helpers, which give the opposite signs.

## Re-running a calibration offline

`calibrate_capture` takes a capture folder and writes a new calibration. The
capture folder needs `waypoints.yaml` and `source_videos/left.mp4` and
`right.mp4`, and for extraction by the logged dwells, both timestamps files: a session's `4mm_6r_8c_4x/`, which an installed calibration names
as `capture.capture_folder` in its `calibration.yaml`. The calibration folder
itself holds no videos.

```python
from pathlib import Path

from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from autotrainer.inference.multires_calibration.pipeline import calibrate_capture

base = PresetGeometry(binning=4, width=256, height=256, offset_x=52, offset_y=6)
index, report = calibrate_capture(
    Path("<session>/4mm_6r_8c_4x"),
    output_dir=Path("recalibrated/4mm_6r_8c_4x.new"),  # must not exist
    presets=presets_from_base(base),
    previous_dir=None,        # a format-2 or 3 folder, to get the change metrics
    refine="subpix",          # or "legacy"
)
print(report.to_text())
```

- `refine` is the corner refinement. `"subpix"` (the default, and what the app
  uses) is `cv2.cornerSubPix` on the original grey frame. `"legacy"` is the
  older Canny and line-fit refinement from `calibration_FLIR`, on the
  contrast-stretched, gamma-2 image. The one used is recorded in
  `calibration.yaml` as `capture.corner_refinement`.
- The base must match the camera geometry the capture's presets were meant for;
  the app reads it from the left camera's saved configuration through
  `live_inference_geometry`.
- Every preset with binning above 1 is cross-checked: the frames the detection
  used are pooled by that binning (area averaging) and cropped to the ROI, the
  board is detected again, and the preset is fitted directly, starting from the
  mapped sensor fit. A direct fit that fails gives an infinite
  `cross_check_delta_px` and a note, which warns; it does not abort.
- `calibrate_capture` grades with `DEFAULT_THRESHOLDS`. `build_report` accepts
  other thresholds, but the pipeline does not pass them: to tune a limit, change
  the constant. Reports carry the limits they were graded with, so months stay
  comparable after a change.
- Errors are not caught: `WaypointAlignmentError`, `RuntimeError` and
  `ValueError` from the fit, and `FileExistsError` for an existing output folder,
  reach the caller. The app shows them in **3D calibration failed** and posts the
  `calibration3dFailed` API event; any other outcome posts
  `calibration3dCompleted`.
- A synthetic rig for capture-free tests, with known K, dist, R and T, is in
  `auto-trainer-inference/tests/multires_synthetic.py`.
