# auto-trainer-pose-yolo

YOLO-pose (Ultralytics) as an optional reachAQ pose backend. reachAQ itself no
longer depends on ultralytics: without this plugin a YOLO model directory is
refused at pre-validation with a message naming this distribution.

**Licence: AGPL-3.0-only.** This distribution depends on `ultralytics`, which
is AGPL-3.0, so it is licensed the same way, unlike the rest of reachAQ. CI never
installs it, and reachAQ's installer only when asked
(`tools/install/reachaq-plugin-install.sh yolo`) or, on an update, into an
environment that already has it or `ultralytics`. Install it only where the
AGPL's terms are acceptable; the supported alternative is to retrain the model
as a keypoint model (`python -m autotrainer.inference.keypoint.train`).

The plugin registers itself in the `autotrainer.pose_backends` entry-point
group and claims any model directory that contains a `yolo_pose.yaml` sidecar.
YOLO weights are torch and run on the torch runtime only.

## Install

Into the environment reachAQ runs in, from the checkout:

```bash
python -m pip install -e plugins/yolo
```

It depends on `auto-trainer` (the installed application) and
`ultralytics >= 8.4, < 9`.

## Contents

- `src/autotrainer/pose_plugins/yolo/` - the plugin (`plugin.py`,
  `YOLO_PLUGIN`, with the measured confidence threshold) and the model
  (`YoloPoseModel`).
- `tests/` - run with the plugin installed: `pytest plugins/yolo/tests`.
