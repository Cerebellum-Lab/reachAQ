# Offline 3D

After a session's 2D processing, intersession analysis triangulates its pose in
3D from every recorded camera that the session's calibration holds, not only
left and right. It needs a format-3 calibration, the kind the calibration
library holds. A legacy or format-2 calibration keeps today's left-right 3D
only. For how the calibration is found, see
[Presets](stereo-calibration.md#presets) ("Offline analysis")
in the stereo calibration guide.

Offline 3D is an extra product. If it fails, the error is logged and the
session's 2D processing and reach segmentation carry on.

## Inputs

- **The calibration.** It is resolved from the session's
  `hardware.configured.calibration3d.id`. The library is checked first, then
  older folders by id or by time.
- **The cameras.** These come from the session metadata's
  `hardware.configured.cameras`. Each recorded camera is matched to the
  calibration's cameras by serial, so a renamed camera keeps its calibration. A
  camera without a serial, on either side, is matched by name. A camera is left
  out, with its reason, when:
  - its serial differs from the calibration's camera of the same name:
    "swapped: calibrated serial X, now Y";
  - the calibration left it 2D only, giving the calibration's reason, for
    example "too few board views: 4 of 12 needed";
  - the session recorded it as not in the calibration (`inCalibration` false);
  - the calibration does not hold it: "not in the session's calibration".
- **The 2D pose.** Each camera's `*_<camera>_raw2D_live.h5` in the session
  folder, as live pose writes it. Frames from different cameras are paired by
  the files' `frame_idx`. A frame that a camera did not pose counts as not seen
  by that camera.
- **The pixel space.** Each camera's 2D points are in the pixels of its
  configured base capture: binning, width, height and ROI offset from the
  session metadata's `configuration.cameras`. This is read the same way as the
  app reads it for live 3D. The calibration's anipose TOML for exactly those
  geometries is used. When no stored file has them, the TOML is derived into the
  library's `.derived` cache. Nothing is written inside the calibration entry.

With live 3D switched off ("live 3D is off"), a session is processed as usual.
Switching live 3D off does not change the cameras or their pixels.

## Method

1. Each point whose likelihood reaches the confidence cutoff (`min_confidence`,
   0.5 by default) is undistorted with its camera's matrix and distortion.
2. It is triangulated by N-view DLT, with `P = [R|t]` in the calibration's motor
   frame.
3. A point seen by fewer than 2 cameras is NaN.

The session is read and processed 20 000 frames at a time, so memory does not
grow with the session's length. A 400 000-frame session of 6 cameras and 16
keypoints peaked at 691 MiB resident on christielab10, against 661 MiB for
100 000 frames; about 180 MiB of that is the libraries loaded before it starts.

## Outputs

These files are written into the session folder. Each is written under a
temporary name and renamed into place when it is complete, so a failed run
leaves no partial file and an earlier result as it was. The datasets are
chunked and gzip-compressed.

`points3d.h5`:

| dataset | shape | |
|---|---|---|
| `points3d` | (frames, keypoints, 3) | mm, motor frame; NaN where fewer than 2 cameras saw the point |
| `node_names` | (keypoints,) | the body parts |
| `camera_names` | (cameras,) | the calibration's names for the cameras used |
| `frame_idx` | (frames,) | the recording-relative frame index of each row |

The file's attributes are `calibration_id` and `units = "mm"`.

`reprojections.h5`:

| dataset | shape | |
|---|---|---|
| `reprojections` | (cameras, frames, keypoints, 2) | each 3D point projected into each camera, in px |
| `reprojection_error` | (cameras, frames, keypoints) | the distance from the camera's own point, in px; NaN where that point was not used |
| `node_names`, `camera_names`, `frame_idx` | | as above |

`offline3d_report.json` records the following:

- `calibrationId`;
- `cameras`: the recorded name of each camera used, mapped to its calibration
  name;
- `excluded`: each camera left out, with its reason;
- `twoDOnlyReason`;
- `poseConsistency`.

When fewer than 2 cameras are left, the session is 2D only. No 3D files are
written, and `twoDOnlyReason` says why, for example "fewer than 2 calibrated
cameras: left (right: swapped: calibrated serial 21000002, now 21000099)".

## Pose-consistency check

The check is optional and off by default. It looks for a camera that has
probably moved since the calibration. Recalibrating by hand is still the normal
way to deal with a moved camera.

- **With 3 or more cameras**, each camera is checked by leave-one-out. Each of
  its confident points is compared with the projection of the point
  triangulated from the other cameras. A camera whose median residual is over
  the limit is reported as "possibly moved". Its report entry and the log show
  the residual and the limit, for example "possibly moved: camera4, median
  residual 12.0 px over 5000 observations (limit 6.0)".
- **With exactly 2 cameras**, the residual is the same for both cameras. It is
  the sum of the two cameras' reprojection errors, which is about how far apart
  the two cameras' views are. A pair over the limit is reported as "pair
  inconsistent", because the check cannot tell which camera moved.
- A camera with too few frames or observations is reported as "not enough
  data".
- The residuals are kept as histograms, 400 bins per decade, so the median and
  p95 are read to within 0.6 % of their value. The confidence cutoff applies to
  the hundredth.

The verdicts are logged and written to `poseConsistency` in
`offline3d_report.json`.

### Measured behaviour

In synthetic sessions with no noise, one camera's 2D points were shifted by
12 px. The table gives the median residuals.

| cameras | shifted camera | the others |
|---|---|---|
| 6 | 12.0 px, possibly moved | 2.9 to 3.8 px, ok |
| 3 | 12.0 px, possibly moved | 8.6 and 8.9 px, also possibly moved |
| 2 | 12.6 px for the pair, pair inconsistent | |

With 3 cameras, each of the other cameras is checked against a point that the
moved camera helped to place. A single moved camera can therefore push every
camera over the limit. The moved camera then has the largest median.

### Settings

The settings are in `calibrations/settings.yaml`, under
`checks.pose_consistency`. Offline analysis reads this file and never writes
to it. The acquisition app sets them in **Preferences → Behavior → Moved-camera
checks → Pose consistency**: a switch (`enabled`) and a field per limit, median
limit (px, 0.5 to 100), p95 limit (px, 1 to 200, empty for off, saved as
`null`), minimum frames (1 to 1000000), minimum observations (1 to 10000000)
and confidence cutoff (0.0 to 1.0). Each field shows the stored value, or the
default where it is unset or invalid, and Preferences reads the file each time
it opens. An edit saves that one key and keeps every other; an entry outside
its range is refused beside the fields and not saved. Per-camera overrides:
edit checks.pose_consistency.per_camera in calibrations/settings.yaml.

```yaml
checks:
  pose_consistency:
    enabled: false        # run the check
    median_px: 6.0        # median residual limit, in the camera's live pixels
    p95_px: null          # optional 95th-percentile limit, off by default
    min_frames: 300       # minimum frames, about 2 s at 150 Hz
    min_observations: 1000  # minimum keypoint observations
    min_confidence: 0.5   # likelihood cutoff for points that are used
    per_camera:           # optional; overrides for one camera, by its calibration name
      camera4: {median_px: 8.0}
```

- The limits are global. A `per_camera` entry overrides only the keys it gives.
- The global `min_confidence` is also the cutoff that the triangulation itself
  uses.
- A missing or invalid value falls back to its default, and a warning is logged.
- The limits are explained in the design (spec section 8(b)):
  - A consistent camera's residual is about 1 to 2 px.
  - A 1° knock moves a camera's image by about 12 px.
  - 6 px therefore catches a knock of 1° with margin, and a knock of 0.5° at
    the edge.
- The p95 limit and the minimums are placeholders, to be tuned on the first
  real N-camera sessions.

## Camera offsets

The centred left-right 3D of intersession analysis uses the camera offsets that
DCS measured. A session that recorded the offsets its live 3D used
(`calibration3d.offsetsFile`, name and sha256) is reprocessed with that file
while it is still kept with the same sha256, even when DCS was redone since.
Otherwise one warning names the recorded file, why it is not used, and the
file used instead, which is found, for a library entry, in this order:

1. the library's `.offsets/<key>/`, `<key>` being the oldest id with the
   entry's fit: the newest `camera_offsets-<UTC time>.pkl` by name, else a
   plain `camera_offsets.pkl`; when that folder has neither, the entry's own
   `.offsets/<id>/` the same way (`library.current_offsets_path`, the same
   choice live 3D makes);
2. the entry's own `camera_offsets.pkl`, which an imported legacy folder may
   have brought with it.

When there are none, offsets are generated for that run only and are never
saved. Offline analysis writes nothing inside an entry, and nothing in
`.offsets`, which belongs to DCS. A legacy folder outside the library keeps
its old behaviour: generated offsets are saved in it.
