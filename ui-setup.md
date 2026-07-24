# Acquisition UI and pellet motor setup

## Working motor configuration

The tested, working `steps_per_revolution` value for each pellet stepper on
this rig is **24.0**, not 48.0.

Use the following values in the active `motor_config.yaml`:

| Axis | `steps_per_revolution` | `microsteps` | `flip_limit_orientation` |
|---|---:|---:|---:|
| X | 24.0 | 8 | 1 |
| Y | 24.0 | 8 | 0 |
| Z | 24.0 | 8 | 1 |

Keep the established speed, acceleration, and homing settings unless the
physical motor or rail changes:

```yaml
max_vel: 120
max_acc: 600
home_vel: 30
```

## Home-relative UI coordinates

The Hardware Control UI presents every pellet axis as **0.0 to 35.0 mm**:

- `0.0 mm` is the home end of the rail.
- `35.0 mm` is the far end of the rail.
- Negative target values are not accepted by the UI.

`flip_limit_orientation` controls which electrical direction the firmware uses
to find the limit switch. It does not reverse the board coordinate: firmware
sets the position to `0.0` when homing reaches that switch. The UI therefore
uses the board coordinate directly for every axis, regardless of the flip
setting.

## Existing move configuration compatibility

Values in `~/Autotrainer/move_config.yaml` remain motor coordinates, so an
existing step such as:

```yaml
- type: x
  value: 25
```

continues to command motor X to position `25`, exactly as it did before this UI
change.

## Position tracker

The first tracker line shows the current board-reported motor position:

```text
• LIVE •   X 2.3 | Y 25.1 | Z 0.7 mm
```

It changes to `STALE` or `DISCONNECTED` when current feedback is not reliable.
The second line shows the board-reported saved/send position:

```text
• SET •   X 2.3 | Y 25.1 | Z 0.7 mm
```

Both lines use the same home-relative 0.0–35.0 mm orientation as the Set
controls.
