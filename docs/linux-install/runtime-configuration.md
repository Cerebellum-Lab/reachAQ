# Runtime configuration, launch, and verification

Complete the portable install and each applicable hardware guide before using a
physical subsystem.

reachAQ ships no Spinnaker software: the SDK and its Python binding are
installed from the customer's own download (see the
[FLIR Spinnaker guide](flir-spinnaker.md)). Windows hardware operation is not
qualified by this repository; the platform-neutral interfaces are kept to
avoid an architectural rewrite if that deployment is needed later.

## Runtime paths

```bash
export REACHAQ_REPO="$HOME/Documents/reachAQ"
export REACHAQ_ENV="reachaq"
export REACHAQ_CONFIG="$HOME/reachAQ/system_configuration.yaml"
export REACHAQ_DATA="$HOME/Documents/rawdatalocal"
mkdir -p "$(dirname "$REACHAQ_CONFIG")" "$REACHAQ_DATA"
```

Seed a configuration only when one does not already exist:

```bash
if [ ! -f "$REACHAQ_CONFIG" ]; then
  cp "$REACHAQ_REPO/src/reachaq/hardware/reachaq_system_configuration.example.yaml" \
    "$REACHAQ_CONFIG"
fi
```

## Configuration checklist

| Section | Required decision |
|---|---|
| `persistence.outputLocation` | Writable local acquisition directory |
| `cameras` | Only physically present cameras; correct scheme/serial/shape/FPS |
| `hardware.*Enabled` | Rig-level CAN, pellet-controller, NI-DAQ, and RFID switches; enable only connected, validated subsystems |
| `hardware.rfidDevice` | Stable `/dev/serial/by-id/...` path for the USB RFID reader |
| `inference.poseModelLocation` | Existing compatible model directory |
| `laser.backend` | `nidaq` for validated hardware; otherwise `null`/`disabled` |
| `laser.channels` | Real NI-DAQ aliases and wired channel roles |
| `laser.channels[].boardStimLine` | `2` or `3`: the board line wired to that laser's trigger input. Unset makes the board STIM route unavailable for that laser |
| `laser.channels[].boardTriggerPulseUs` | `100`..`5000000`; default `1000`. Width of the board STIM pulse that starts that laser's waveform |
| `laser.channels[].triggerMonitorInput` | Optional. The input the board STIM line is wired back into, an analog input (`aiN`) or a port0 line (`port0/lineN`), never a PFI terminal or a port1/port2 line (the PFI pins, which the stream cannot clock). Recorded as `laserN_trigger` and plotted on the laser tab's Board trigger graph. Set it as **trigger readback input** in Edit DAQ Ports |
| `laser.pmtShutterOutput`, `laser.channels[].triggerOutput`, `laser.channels[].timingTriggerOutput` | Optional. Clocked digital output lines that run with a pulse's waveform: each a line on a board that can run clocked digital output (on christielab10 a port0 line on the 6221, never the 6713). Each runs on the analog output's own sample clock with no start trigger (an M Series board's clocked digital output takes none): named on the output's own board, or, for a line on another board, carried on `laser.pulseClockLine` for the pulse. The PMT shutter runs only for a profile with a PMT open lead or close lag above zero; with `pmtShutterOutput` unset those margins are ignored, the pulse fires without them, and the log says so once per laser. None is set on christielab10, and this path has not yet been run on its hardware |
| `laser.backplaneClockLine` | Default `PXI_Trig1`. The bare PXI_Trig line (no board; `PXI_Trig0` to `PXI_Trig7`) the shared sample clock, and the calibration ramp's, are driven onto. It must differ from `laser.pulseClockLine`, from every `laser.channels[].triggerSource` line, from a PXI_Trig `laser.channels[].triggerRouteSource` and from every `laser.triggerListenerInputs` line, compared by line name whatever the board or case; a PFI never clashes. A configuration that clashes is refused as it loads, naming the field and the line. christielab10: clock on `PXI_Trig1`, triggers and trigger inputs on `PXI_Trig0` and `PXI_Trig2` |
| `laser.pulseClockLine` | Default `PXI_Trig3`. A free, bare PXI_Trig line (`PXI_Trig0` to `PXI_Trig7`) every pulse drives its analog output's own sample clock onto, for a clocked digital line on another board. The same rules as `backplaneClockLine`, and different from it. Optional: a file without it takes the default. A file saved with it is refused by an older build; delete the line to go back. christielab10 leaves `PXI_Trig3` free for it |
| `nidaqPorts`, `nidaqStream` | Real device identity, supported channel types, timing policy, and independent plot selection |
| `nidaqPorts.timing.sampleClockExportTerminal`, `nidaqPorts.timing.startTriggerExportTerminal` | Unset unless the stream's inputs span two boards in backplane mode, outside a multidevice task that merges them; then `sampleClockExportTerminal` is required, and `startTriggerExportTerminal` too when the master has an analog input, with no default. Each a PXI_Trig line on the timing master (`/PXI1Slot5/PXI_Trig4`, or bare `PXI_Trig4`): the master exports its sample clock onto the first and, when it has an analog input, its start trigger onto the second, and the other board's input tasks read each by its own name. A required one unset, not a PXI_Trig line, or on another board, and the stream start is refused, naming the field. They must differ from each other, from `laser.backplaneClockLine` and `laser.pulseClockLine`, from every `laser.channels[].triggerSource` line and PXI_Trig `triggerRouteSource`, and from every `laser.triggerListenerInputs` line; Run and the stream's own start, Idle's included, refuse a clash, naming both, and the DAQ Monitor lists it. The DAQ Monitor window's own stream is configured without the laser, so it refuses only one export line against the other. christielab10: both unset, as every input is on the 6221. Not yet verified on hardware |

Use **Edit → Edit DAQ Ports** while idle to discover supported channel types and
prevent duplicate assignments. The digital roles (`tone1`, `tone2`, `tone3R`,
`tone3L`, `camFrames`, `barcode`) must be port0 lines on the input card, the
only lines the stream's clocked digital-input task can sample; a port1 or port2
line (a PFI pin) is refused, naming the field. Saving from the dialog also records stable
product/serial identities and validates timing-master capability. All mapped and
custom NI-DAQ inputs are recorded when NI-DAQ is enabled; plotting only the
subset in `displayChannels` does not change persistence. The dialog edits only
the port roles it shows (analog output, diode input, shutter output, command
copy input, trigger readback input); saving it keeps every other laser field as
it was, including `boardStimLine` and `boardTriggerPulseUs`. It refuses, before
it closes, a trigger readback input that another role uses, that is a laser
output line, or that is not an analog input or port0 line. On
christielab10, laser 1 uses
`boardStimLine: 3` and laser 2 `boardStimLine: 2`, both with
`boardTriggerPulseUs: 1000`.

The same rig-level switches can be changed while idle under **File → Hardware**.
The multi-selection submenu stays open while checkable entries are changed and
closes when the operator clicks outside it. Each choice is saved to the system
YAML immediately; enabling one or more runtime hardware domains schedules one
hardware refresh after the submenu closes. **RFID Serial Device…** changes the
stable reader path from the same menu. These controls are unavailable after
acquisition leaves Idle.

Settings are saved to the file reachAQ loaded: the one named with `-c`, or the
Preferences default when none is named. They are saved only when the operator
changes a stored setting while idle, at the moment of the change: leaving
camera edit mode, a Camera binning choice, **Edit DAQ Ports** OK, **File →
Hardware**, an NI-DAQ plot selection, and each Preferences or Behavior edit
(data location, inference model, live inference, session and pellet options).
A Preferences or Behavior edit is written half a second after the last change,
so stepping a spin box or typing a path saves once; exiting or loading another
configuration writes one still waiting.
Starting, recording, stopping and exiting write nothing to the file, so what a
run changes on its own (a stim session's camera mode, a `--live-inference`
override, the protocol's progress) is not carried into the next start; each
session's metadata records the configuration it ran with. Turning live
inference on or off in Preferences replaces a `--live-inference` override and
is saved. A `-c` file that does not exist is refused rather than created with
defaults; with no `-c`, a missing default file is still created on first start.
Nothing is saved when the last load failed or did not complete, so
a broken file is not overwritten with a half-applied one; the log says
`Not saving configuration: the last configuration load did not complete`. A run
started with `--random-cameras` is never saved either.

See [Session recording, synchronization, and hardware isolation](../acquisition/session-recording-and-synchronization.md)
before commissioning Record/Stop/Abort or a multi-device timing topology.

## Safe first launch

The GUI defaults to idle. This command also disables inference for the run, so
no GPU preflight is required when the operator later selects Running:

```bash
cd "$REACHAQ_REPO"
conda run --no-capture-output -n "$REACHAQ_ENV" python -m reachaq.app \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

Use `--start-mode running` only when immediate acquisition startup is
intentional. Headless mode has no idle operator control, so it starts
acquisition by default:

```bash
conda run --no-capture-output -n "$REACHAQ_ENV" reachaq-headless \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

When CAN hardware is enabled, source the validated rig environment first:

```bash
set -a
source "$REACHAQ_REPO/src/reachaq/hardware/reachaq_hardware.env.example"
set +a
```

The boot service and application use different variable namespaces:

- `/etc/default/reachaq-can` supplies `REACHAQ_CAN_*` values to
  `reachaq-can.service`, which configures and brings up the Linux interface.
- The launching process supplies `REACHAQ_CAN_*` values to reachAQ, which
  opens the already-active interface.

The service does not export its values into user applications. Source the
application environment in the same shell that launches reachAQ, or configure
equivalent values in the desktop/service launcher. The service interface and
application channel must match:

```bash
. /etc/default/reachaq-can
test "$REACHAQ_CAN_CHANNEL" = "$REACHAQ_CAN_INTERFACE"
systemctl is-enabled reachaq-can.service
systemctl is-active reachaq-can.service
ip -details link show "$REACHAQ_CAN_CHANNEL"
```

The retired Anschutz hardware profile has been removed. ReachAQ targets the
pellet board through the selected JerryCAN, SocketCAN, or emulation transport;
transport selection does not change the pellet-board command model. Use
`REACHAQ_CAN_TRANSPORT=emulation` only for software testing.

## Pellet motor coordinates

The validated rig configuration uses `steps_per_revolution: 24.0` and
`microsteps: 8` for X, Y, and Z. Limit orientation is X=1, Y=0, Z=1. Retain the
established `max_vel: 120`, `max_acc: 600`, and `home_vel: 30` unless the
physical motor or rail changes.

The Hardware Control panel presents board coordinates from 0.0 mm at the home
switch to 35.0 mm at the far end. `flip_limit_orientation` changes the
electrical homing direction, not this coordinate convention. Existing
`move_config.yaml` values remain board coordinates; for example an X value of
25 still commands X to 25 mm. The LIVE line reports current feedback and its
freshness, while SET reports the saved send position in the same coordinates.

## Motor travel limits

`pellet.x`, `pellet.y` and `pellet.z` in `~/reachAQ/motor_config.yaml` can
each set `min_pos` and `max_pos`, the stage's travel in board millimetres.
christielab10 uses `min_pos: 0` and `max_pos: 35` on all three axes. Either key
can be left out: that side is then not limited. A value that is not a finite
number, or `min_pos` above `max_pos`, stops the connect with an error naming
the axis (`pellet-x`, `pellet-y` or `pellet-z`). The keys stay on the host;
nothing new is written to the board.

ReachAQ checks each stage target against the travel before it sends anything:

- **SET and MOVE** of x, y or z, absolute, relative or given in DCS
  coordinates. That covers the animal's base position, protocol presentations,
  the Hardware Control buttons, the intersession shift and every calibration
  and scout waypoint. SET keeps its clamp of a target below 0 to 0, and the
  check runs after it. On an axis with travel, a relative SET is sent to the
  board as the absolute target that was checked, so the board cannot add it
  to a different base. A target within 1e-6 mm of a limit counts as inside.
- **The animal's base position and a protocol presentation** are checked on
  all three axes before any is SET. One axis out of range sends none of them,
  so the stage never mixes the new position with the previous one. The base
  position logs `Not sending <animal>'s pellet position ...: it lies outside
  the configured travel` and shows a "Pellet position refused" error. The
  previous animal's position is still SET on the board, so **no pellet is
  presented** (`Pellet presentation refused: ...` in the log) until the board
  has accepted a SET of each of X, Y and Z: an animal whose position is inside
  the travel is selected, or the operator sets all three in Hardware Control.
  A SET the board refuses, or never acknowledges, leaves its axis blocked. A
  protocol target or the intersession shift does not lift the block. A
  protocol presentation stops with an error. While the animal has a protocol,
  X, Y and Z set in Hardware Control apply to this session only and are not
  saved to the animal; the dialog says so. To keep a new position, select
  **Manual pellet control** in Protocol, set X, Y and Z, then select the
  protocol again.
- **The intersession shift** is checked on every axis it shifts before any is
  SET. One shifted axis out of range shifts none of them and logs `Not
  applying the pellet send_position shift ...`. A refused shift is not
  recorded as applied: the last processed shift (the `applied_shift` of later
  pellet-cycle records) keeps its previous value, the `intertrialPelletShift`
  event carries `applied: false` and the `refusal` reason, and a tongue-eaten
  shift's y-limit is not saved to the animal. A shift whose SETs fail with an
  error (for example a relative SET before any SET since the connect) is
  posted the same way, with the error as the reason. `applied` is true only
  when every shifted axis was sent to the board. A shift that is disabled (in
  Preferences or by a training-plan phase), that has no pellet device, or
  whose SET was not queued is posted with `applied: false` and that reason,
  and is not recorded as the last processed shift.
- **Changed with this:** a tongue-eaten y-limit is saved to the animal only
  when its shift was applied. While the intersession shift is disabled, a
  tongue-eaten result no longer sets the animal's y-limit.
- **The `move_config.yaml` sequences** (`load_pellet`, `send_pellet`,
  `cover_pellet`, `release_pellet`, `move_retract`, or their built-in
  defaults). Every stepper step is checked before the sequence starts, as
  the steps before it leave the stage. One step out of range refuses the
  whole sequence. As the board command does, a step's target is first held
  to 0 and 15 motor turns (36.5 mm), so a retract that runs into 0 still
  passes. Steps relative to the send position, and `predefined: send`, are
  checked from both the requested send position and the one the board
  reports, when they differ. A sequence with `predefined: send` is refused
  while the send position of an axis with travel is not known at all (no SET
  since the connect and none reported).
- **A motor drift correction** that would put the saved send position outside
  the travel.

A target outside the travel is **refused, not clamped**: nothing is sent, the
command is not queued, and the log has an error such as

```
Axis-X: move to 36.000 refused: outside the configured travel 0-35 mm
```

An unset side prints as `-inf` or `inf`. A refused sequence also logs
`Pellet sequence load_pellet refused: a step lies outside the configured
travel`. Without `min_pos` and `max_pos`, behaviour is as before.

When the board refuses a SET on an axis with travel, the host puts that axis's
requested send position back to the one the board reports, so later checks use
the base the board really has. It does so silently (the animal's saved position
is not changed), and not when a newer SET of that axis was queued since or the
refusal's outcome is unknown (`-EBUSY` on a resend). Without travel, a refused
SET leaves the requested position as before.

Right after an in-range SET, a `send_pellet` can be refused once if the
send position the board still reports is outside the travel (left there by a
maintenance tool, for example). It clears with the board's next status.

Not checked:

- **Homing** (`send_home`, `send_to_limits` and home steps in a sequence). The
  firmware stops at the switch.
- **A target the host does not know**: a relative sequence step from a
  position the board has not reported. The host sends nothing for it. In the
  moment after a connect, before the first status reaches the host, a
  `send_*_rel` step is not checked either, while the board side may already
  hold a send position. Neither built-in sequence uses `send_*_rel`.
- **Turning drift correction off**. That sends a SET of 0 relative to the
  send position the board last reported, not the last requested SET. Right
  after a new SET it can therefore bring back the previous send position. It
  never saves a position the board has not held already.
- **The maintenance tools that bypass the acquisition app**: the pellet
  delivery tool (`src/reachaq/pellet_delivery`), `scripts/can_console.py` and
  `src/reachaq/hardware/validate_can_hardware.py`. They command the board directly
  and do not read the travel.

The 3D calibration scout searches the configured travel (see
[Automatic move list](../acquisition/stereo-calibration.md#automatic-move-list)).

For the current JerryCAN board, expect `can0`, `mtu 72`, CAN FD,
1 Mbit/s arbitration, and 5 Mbit/s data. See the
[PEAK/SocketCAN guide](peak-socketcan.md) for installation, termination,
permissions, reset behavior, and safe validation.

## Pellet stage stop (ESTOP)

Pellet firmware 2.3.1 on a v1.2 board has no motion timeout and no travel
clamp of its own, and nothing on the board stops a stepper when the host goes
away. A limit-switch stop in the middle of a move can leave the axis's step
timer running: the axis keeps moving, and the board answers every later move
with `-EBUSY`. The one command that stops it is ESTOP, a single broadcast
frame with no acknowledgement. ReachAQ sends it (since v3.2.3):

- when a stepper-motion frame is not acknowledged in time: a MOVE, a home or
  `SEND_TO_FIXED_XYZ`, standalone or a step of a sequence. That frame is **not
  re-sent** any more (a re-sent move is what the running axis answers with
  `-EBUSY`). A sequence's servo, delay and tone frames and a lost SET are still
  re-sent, and a lost stimulus-pulse acknowledgement is still skipped;
- when the board answers `-EBUSY` to a move, a home, a sequence or
  `SEND_TO_FIXED_XYZ`. A busy SET (which moves nothing) is logged and sent again
  every 0.25 s, up to 12 times, before it is refused;
- when a command's waiter gives up on a move that is still awaiting its
  acknowledgement (3 s by default, the 30 s give-up);
- when a calibration capture or scout fails, before its cameras stop, and at
  its teardown if a move is still in flight; when a scout waypoint is not
  reached;
- when the CAN connection is closed while a move is in flight: a disconnect,
  a CAN safety shutdown or recovery, a failed connect, closing the app. An
  idle stage is never stopped by a disconnect;
- when the operator presses **STOP** in Hardware Control. STOP is enabled
  whenever the board is connected, also while a command runs.

Each ESTOP is logged at ERROR with its reason (`Pellet ESTOP sent: ...`) and
written to the session's metadata as `recording.estops` (`wallTime`,
`perfTime`, `reason`, `axes`, `operator`, `sent`). The command it stopped ends
as `stopped`, never acknowledged.

After an ESTOP every axis needs a home, the LIVE position shows `—`, and every
move, send, load, retract, cover, release and automatic home is refused; SETs
still go out. The alert "Pellet stage stopped: <reason>. Home before continuing
(Hardware Control ▸ Home); the session-end home also clears it." stays until the
stage is homed (a STOP the operator pressed says so instead). The board stays latched in E-stop and refuses every move and
home with `-EPERM` until it is released. **Hardware Control ▸ Home** releases
it and then homes; Refresh Hardware (a connect) does the same, since the latch
outlives the app. The session-end home is the one automatic home that may run
after a stop: it releases the latch, homes, and on its acknowledgement clears the
stop (logged). The pellet machine's re-homes and an automatic CAN recovery do not
release a stop this app made: they are refused until an operator Home or a
reconnect.

The maintenance tools (`src/reachaq/pellet_delivery`, `scripts/can_console.py`,
`src/reachaq/hardware/validate_can_hardware.py`) do not send ESTOP.

## Software-only camera launch

`--random-cameras` changes configured camera sources only in memory for that
run; it does not overwrite physical camera serials on close:

```bash
conda run --no-capture-output -n "$REACHAQ_ENV" python -m reachaq.app \
  --random-cameras \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

For a fully separate software configuration, use
`src/reachaq/hardware/reachaq_random_camera_configuration.example.yaml` with a
separate Preferences configuration directory.

## Desktop launcher

The complete portable installer installs this launcher automatically. Install
or refresh only the GNOME desktop launcher without rerunning the complete host
installer with:

```bash
tools/install/install-reachaq-desktop.sh
```

The installer places a trusted `reachAQ.desktop` entry, with the reachAQ camera
icon, on the operator's desktop and in the application menu. It starts in Idle through the
configured `reachaq` Conda environment and opens a terminal containing live
logs. A per-user lock rejects accidental duplicate GUI launches. Startup
failures remain visible until Enter is pressed.

The default installed files are:

```text
<XDG desktop directory>/reachAQ.desktop
~/.local/share/applications/reachaq.desktop
~/.local/share/icons/hicolor/scalable/apps/reachaq.svg
~/.local/share/icons/hicolor/<N>x<N>/apps/reachaq.png   (N = 16 to 512)
~/.local/bin/reachaq-launcher
~/.local/bin/reachaq
~/.local/bin/reachaq-sync
~/.config/reachaq/launcher.conf
```

The installer replaces an obsolete `ReachAQ-startup` shell fragment in that
same desktop directory.
Launcher paths are stored in `~/.config/reachaq/launcher.conf`. Optional
`REACHAQ_*` and `REACHAQ_*` runtime values can be placed as literal
`KEY=VALUE` lines in `~/.config/reachaq/runtime.env`; the launcher parses this
file as data and does not source it as shell code. Re-run the launcher installer
after moving the repository, changing the Conda environment, or changing the
system-configuration path.

## Terminal commands

The same installer puts two commands in `~/.local/bin`. Both read
`launcher.conf`, so they act on the checkout the desktop icon runs.

```bash
reachaq
```

Starts reachAQ from any shell without activating Conda first. It takes the
same per-user lock as the desktop launcher, so the two cannot open the
hardware at once, but does not pause on exit: in a terminal the error is
already on screen. Arguments pass through to the application. When the
`reachaq` Conda environment is active its own `reachaq` console script comes
first on `PATH` instead; it starts the same checkout.

The application also holds a lock of its own,
`$XDG_RUNTIME_DIR/reachaq/app.lock`, however it is started: through either
launcher, the environment's console script, `python -m reachaq.app`, or
`reachaq-headless`. A second start exits with status 3 and names the
running instance's PID. `-h` still works while one runs.

```bash
reachaq-sync
reachaq-sync devel
```

Fetches, then fast-forwards every local branch that tracks a remote, or only
the branches named. A named branch missing locally is created from the
remote. It never merges, rebases, resets or discards: a branch that is ahead
or has diverged is reported and left alone, and a checkout with modified
tracked files is refused before anything is fetched. Untracked files such as
`planning.md` do not block it. If the running branch's packaging files
changed, it prints the reinstall command rather than running it.

Run `reachaq-sync` before flashing new pellet firmware: the application only
accepts firmware its checkout lists. See
[pellet firmware compatibility](../acquisition/pellet-firmware-compatibility.md).

On a first install `~/.local/bin` joins `PATH` at the next login.

## Environment variables

These are read from the process environment rather than
`system_configuration.yaml`. `SystemConfiguration.version` is pinned at 58 and
rejects any other value, so adding schema fields would stop deployed rigs
loading their configuration. They should move into the schema when a version
bump past 58 is acceptable.

| Variable | Default | Effect |
|---|---|---|
| `REACHAQ_CONFIG_DIR` | `~/reachAQ` | The per-computer settings directory: `system_configuration.yaml`, the motor and move files, `animals/`, the 3D calibration and the hardware tools' records. Every default that names one of those files is built from it (`reachaq.core.config_directory`). Read at start-up. The installer uses it when `REACHAQ_INSTALL_CONFIG_DIR` is unset. Put it in `runtime.env` so the launcher exports it. |
| `REACHAQ_STIM_RT_PRIORITY` | `80` | SCHED_FIFO priority for the stim capture loop. `0`, `off` or `false` disables it. See the [latency tuning guide](latency-tuning.md). |
| `REACHAQ_POSE_BACKEND` | `tensorflow` where installed, else `torch` | Selects the engine a DeepLabCut model runs on: `tensorflow` or `torch`. It applies to DeepLabCut models only, which need the optional DeepLabCut plugin (`tensorflow` also needs the plugin's TensorFlow engine, which its installer adds). A keypoint or YOLO model ignores it and always runs on `torch`. For a model no installed plugin can run, or no model, it still selects the GPU runtime the live-inference preflight probes, so the two cannot disagree. |
| `REACHAQ_POSE_CONFIDENCE_THRESHOLD` | per engine | Confidence a keypoint needs to count as present and to enter 3D triangulation. The default comes from the pose backend that runs the model, because every engine reports on its own scale: 0.1 for keypoint and YOLO models, and for DeepLabCut 0.9 on `tensorflow` (which saturates its likelihood at 1.0) and 0.6 on `torch`, with measured per-backbone refinements. The DeepLabCut PyTorch default is provisional and should be re-derived on held-out frames. |
| `REACHAQ_INFERENCE_ROI` | unset | `x,y,width,height`. Crops live frames to this window before inference and shifts the returned coordinates back into full-frame space. Unset means full frame. |
| `REACHAQ_CAN_FRAME_LOG` | unset | `1` turns python-can's per-frame log lines back on: every received frame, and the socketcan "tx" line for every sent one. Unset, python-can's logger is held at WARNING, because those lines were about 270 log records a second on the CAN reader thread; python-can's few open and close INFO/DEBUG lines go quiet with them, and its warnings still pass. The application's own CAN lines, such as "CAN reader throughput", and the kernel receive stamps in `device.csv` are unaffected. Applies to `reachaq`, `reachaq` and `reachaq-headless`; `can_console.py` and the pellet-delivery tool still log every frame. Read at start-up, so a diagnostic run needs a restart. |

`REACHAQ_POSE_BACKEND=torch` requires a shuffle trained with the PyTorch
engine; a project trained under TensorFlow has no `pytorch_config.yaml` and the
model load fails with a message saying so.

## Verification

Portable CLI/import checks, the pose-engine GPU checks and the focused
non-hardware suite are part of the tracked installer. The supported
repair/verification workflow reruns every category:

```bash
tools/install/reachaq-linux-install.sh
```

On a rig whose root-owned setup is already in place, skip those steps; they are
reported as `SKIP`:

```bash
REACHAQ_INSTALL_SYSTEM=0 tools/install/reachaq-linux-install.sh
```

## Diagnose startup waits

Hardware initialization milestones always go to the application log and
launching terminal. Search for the last unmatched record:

```text
HARDWARE INIT | START
HARDWARE INIT | READY
HARDWARE INIT | SKIP
HARDWARE INIT | FAILED
```

Each camera, CAN, NI-DAQ, laser, and GPU operation includes elapsed timing. The
last `START` without a terminal state identifies the current wait.

Initialization results are independent. A failed NI-DAQ, laser, CAN, camera, or
inference domain does not tear down unrelated healthy domains. Instead, the
failed required source is listed as a recording blocker. The refresh icon at the
far right of the Hardware Status title bar refreshes discovery while idle and
retries failed domains while Running and Ready.

The documented launch commands use `conda run --no-capture-output`. Do not omit
that option: default `conda run` captures the stream and can hide every terminal
record until reachAQ exits.

## Common failures

| Symptom | Next check |
|---|---|
| Git LFS test asset error | `git -C "$REACHAQ_REPO" lfs pull` |
| Qt xcb plugin error | `libxcb-cursor0`, `libxkbcommon-x11-0`, and display environment |
| Cameras absent or no first frame | Applicable [FLIR guide](flir-spinnaker.md), effective trigger-node diagnostics, physical trigger/power/ground path |
| NI devices absent | [NI-DAQ/PXI guide](ni-daq-pxi.md), starting at PCI/USB enumeration |
| Status bar: *NI-DAQ inputs are not acquired: ... Fix it in Edit → Edit DAQ Ports*; Hardware Status shows NI-DAQ blocked | A line in `nidaqPorts` or a laser input that the stream cannot acquire - a duplicate pin, a port1/port2 line (a PFI pin) on a digital role, a PFI trigger readback. Everything else loaded; the NI-DAQ stream has no channels until it is fixed. Open **Edit → Edit DAQ Ports**: the bad value is on its field, named in the status line, and OK stays disabled until it is changed. A valid save starts the stream. The file keeps your values meanwhile, even on exit |
| PXI off and camera timeout occur together | Treat as correlated until wiring is checked; NI-DAQ software initialization does not trigger cameras |
| CAN interface down | [PEAK/SocketCAN guide](peak-socketcan.md), bitrate and termination |
| CAN reset asks for a password | Activate the `reachaq` login group by logging out/in, then verify the narrow `sudo -n` permission |
| CAN RX continues but startup ACK times out | Preserve the application log; this is an application startup/ACK path issue rather than proof of a dead bus |
| Log warns `CAN error notice on can0 ... controller tx-warning` (or `tx-passive`, `back-to-active`) | Informational. The CAN controller's error counters rose and fell, as they do after any stretch without acknowledgement such as a pellet-board reboot; nothing was lost and the connection stays up. Repeats within a second are counted as `N more suppressed`. Only bus-off or a controller receive overflow stops the connection, reported as `CAN error frame ... on can0` |
| Pellet controller connection fails after a firmware update | The checkout does not accept that firmware; see [pellet firmware compatibility](../acquisition/pellet-firmware-compatibility.md#if-startup-refuses-the-board) |
| Start exits at once: `reachAQ is already running for this user (pid N)` | Another instance holds the hardware. Close it, or find it with `ps -p N` |
| Hardware Status warns `NI-DAQ wiring is unverified` | The named channels have not been confirmed against the cables; run the wiring test from **Tools → DAQ Monitor** ([NI-DAQ guide](ni-daq-pxi.md#check-the-wiring)) |
| TensorFlow or PyTorch reports no GPU | [NVIDIA guide troubleshooting](nvidia-inference.md#troubleshooting) or `--no-live-inference` |
| Output permission error | Configured data directory ownership and write permission |

## Current workstation reference

Last documentation update 2026-09-24. Entries marked (verified) were re-checked
on the workstation on that date; the hardware entries are from 2026-09-09.

- Dell Precision 3660 Tower; Ubuntu 22.04.5 LTS, x86_64; kernel
  `6.8.0-136-generic`, booted with `intel_iommu=off` so the NI boards can DMA
  ([NI-DAQ guide](ni-daq-pxi.md#iommu-and-dma-boot-with-intel_iommuoff)). 12th Gen Core i9-12900: 8 P-cores on CPU 0-15 at
  5000-5100 MHz, 8 E-cores on CPU 16-23 at 3800 MHz.
- Conda environment `reachaq`, built by the portable installer on 2026-09-24
  (verified): Python 3.10.21; torch 2.7.1+cu128 (CUDA 12.8, cuDNN 9.7);
  TensorFlow 2.12.1 with its CUDA 11.8 / cuDNN 8.6 in
  `lib/reachaq-tensorflow-cuda11`; DeepLabCut 3.0.2; ultralytics 8.4 (both engines were
  installed then; today no engine is installed by default, see the
  [pose plugin installer](nvidia-inference.md)); numpy
  1.26.4; PySide6 6.6.3.1; nidaqmx 1.6.0; spinnaker-python 3.2.0.62. Both
  engines pass their GPU checks, alone and in one process; the live YOLO model
  runs at about 4 ms/frame and the DeepLabCut TensorFlow model
  `mouseGYM-christie-2025-09-18` at about 53 ms/frame.
- Conda environment `reachaq-py38-20260924`: the previous Python 3.8.20
  environment, archived by the installer, kept as a fallback. Its TensorFlow
  has no GPU (its cuDNN 8 was overwritten), so DeepLabCut TensorFlow models run
  on the CPU there; YOLO models still use the GPU.
- Spinnaker system runtime 3.2.0.57 (verified); Python wheel 3.2.0.62, which
  lists both cameras against it (verified). reachAQ bundled that wheel until
  2026-11; the installer now keeps an installed 3.2 binding and otherwise takes
  one from the FLIR download.
- NI-DAQmx 26.3.1 (verified) and PXI Platform Services 26.3; configured
  PXI-6713 output alias `PXI1Slot4` and PXI-6221 sampled-input alias
  `PXI1Slot5`. Kernel modules `nitiork`, `niwfrk` and `nixsrk` are built
  through DKMS.
- PEAK PCIe adapter on `peak_pciefd`, exposing `can0` and `can1`.
- Quadro T1000, driver 595.84, CUDA 13.2 (verified). **Live inference works**;
  the earlier `nouveau` entry is obsolete. The driver is installed as
  `linux-modules-nvidia-595-open-<kernel>`, Canonical's prebuilt per-kernel
  modules rather than DKMS, which constrains any kernel change. The T1000 uses
  the TU117 die and has no tensor cores despite reporting compute capability
  7.5, so FP16 measures 3-4x slower than FP32 on it.
- Local output `/home/christielab10/Documents/rawdatalocal`.
- The installer's focused suite in `reachaq`, Python 3.10: 228 passed
  (verified). The full suite was last run on the Python 3.8 environment
  (2026-09-09): 1164 passed, 35 skipped. Run it with `ulimit -n` at its default
  1024; the suite no longer exhausts the descriptor limit.
