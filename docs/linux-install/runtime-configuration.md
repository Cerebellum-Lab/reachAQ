# Runtime configuration, launch, and verification

Complete the portable install and each applicable hardware guide before using a
physical subsystem.

Bundled Spinnaker Python artifacts are organized under `vendor/spinnaker` by
operating system and architecture. `tools.platform_support` resolves Linux
x86-64, Linux aarch64, and the retained Windows x86-64 artifact from the shared
manifest. Windows hardware operation is not qualified by this repository; the
artifact and platform-neutral interfaces are retained to avoid an architectural
rewrite if that deployment is needed later.

## Runtime paths

```bash
export REACHAQ_REPO="$HOME/Documents/reachAQ"
export REACHAQ_ENV="reachaq"
export REACHAQ_CONFIG="$HOME/Autotrainer/system_configuration.yaml"
export REACHAQ_DATA="$HOME/Documents/rawdatalocal"
mkdir -p "$(dirname "$REACHAQ_CONFIG")" "$REACHAQ_DATA"
```

Seed a configuration only when one does not already exist:

```bash
if [ ! -f "$REACHAQ_CONFIG" ]; then
  cp "$REACHAQ_REPO/tools/hardware/reachaq_system_configuration.example.yaml" \
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
| `nidaqPorts`, `nidaqStream` | Real device identity, supported channel types, timing policy, and independent plot selection |

Use **Edit → Edit DAQ Ports** while idle to discover supported channel types and
prevent duplicate assignments. Saving from the dialog also records stable
product/serial identities and validates timing-master capability. All mapped and
custom NI-DAQ inputs are recorded when NI-DAQ is enabled; plotting only the
subset in `displayChannels` does not change persistence.

The same rig-level switches can be changed while idle under **File → Hardware**.
The multi-selection submenu stays open while checkable entries are changed and
closes when the operator clicks outside it. Each choice is saved to the system
YAML immediately; enabling one or more runtime hardware domains schedules one
hardware refresh after the submenu closes. **RFID Serial Device…** changes the
stable reader path from the same menu. These controls are unavailable after
acquisition leaves Idle.

See [Session recording, synchronization, and hardware isolation](../acquisition/session-recording-and-synchronization.md)
before commissioning Record/Stop/Abort or a multi-device timing topology.

## Safe first launch

The GUI defaults to idle. This command also disables inference for the run, so
no GPU preflight is required when the operator later selects Running:

```bash
cd "$REACHAQ_REPO"
conda run --no-capture-output -n "$REACHAQ_ENV" python -m reachAQ.app \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

Use `--start-mode running` only when immediate acquisition startup is
intentional. Headless mode has no idle operator control, so it starts
acquisition by default:

```bash
conda run --no-capture-output -n "$REACHAQ_ENV" auto-trainer-headless \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

When CAN hardware is enabled, source the validated rig environment first:

```bash
set -a
source "$REACHAQ_REPO/tools/hardware/reachaq_hardware.env.example"
set +a
```

The boot service and application use different variable namespaces:

- `/etc/default/reachaq-can` supplies `REACHAQ_CAN_*` values to
  `reachaq-can.service`, which configures and brings up the Linux interface.
- The launching process supplies `AUTOTRAINER_CAN_*` values to reachAQ, which
  opens the already-active interface.

The service does not export its values into user applications. Source the
application environment in the same shell that launches reachAQ, or configure
equivalent values in the desktop/service launcher. The service interface and
application channel must match:

```bash
. /etc/default/reachaq-can
test "$AUTOTRAINER_CAN_CHANNEL" = "$REACHAQ_CAN_INTERFACE"
systemctl is-enabled reachaq-can.service
systemctl is-active reachaq-can.service
ip -details link show "$AUTOTRAINER_CAN_CHANNEL"
```

The retired Anschutz hardware profile has been removed. ReachAQ targets the
pellet board through the selected JerryCAN, SocketCAN, or emulation transport;
transport selection does not change the pellet-board command model. Use
`AUTOTRAINER_CAN_TRANSPORT=emulation` only for software testing.

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

For the current JerryCAN board, expect `can0`, `mtu 72`, CAN FD,
1 Mbit/s arbitration, and 5 Mbit/s data. See the
[PEAK/SocketCAN guide](peak-socketcan.md) for installation, termination,
permissions, reset behavior, and safe validation.

## Software-only camera launch

`--random-cameras` changes configured camera sources only in memory for that
run; it does not overwrite physical camera serials on close:

```bash
conda run --no-capture-output -n "$REACHAQ_ENV" python -m reachAQ.app \
  --random-cameras \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

For a fully separate software configuration, use
`tools/hardware/reachaq_random_camera_configuration.example.yaml` with a
separate Preferences configuration directory.

## Desktop launcher

The complete portable installer installs this launcher automatically. Install
or refresh only the GNOME desktop launcher without rerunning the complete host
installer with:

```bash
tools/install/install-reachaq-desktop.sh
```

The installer places a trusted mouse-icon `reachAQ.desktop` entry on the
operator's desktop and in the application menu. It starts in Idle through the
configured `reachaq` Conda environment and opens a terminal containing live
logs. A per-user lock rejects accidental duplicate GUI launches. Startup
failures remain visible until Enter is pressed.

The default installed files are:

```text
<XDG desktop directory>/reachAQ.desktop
~/.local/share/applications/reachaq.desktop
~/.local/share/icons/hicolor/192x192/apps/reachaq.png
~/.local/bin/reachaq-launcher
~/.config/reachaq/launcher.conf
```

The installer replaces an obsolete `ReachAQ-startup` shell fragment in that
same desktop directory.
Launcher paths are stored in `~/.config/reachaq/launcher.conf`. Optional
`AUTOTRAINER_*` and `REACHAQ_*` runtime values can be placed as literal
`KEY=VALUE` lines in `~/.config/reachaq/runtime.env`; the launcher parses this
file as data and does not source it as shell code. Re-run the launcher installer
after moving the repository, changing the Conda environment, or changing the
system-configuration path.

## Environment variables

These are read from the process environment rather than
`system_configuration.yaml`. `SystemConfiguration.version` is pinned at 57 and
rejects any other value, so adding schema fields would stop deployed rigs
loading their configuration. They should move into the schema when a version
bump past 57 is acceptable.

| Variable | Default | Effect |
|---|---|---|
| `REACHAQ_STIM_RT_PRIORITY` | `80` | SCHED_FIFO priority for the stim capture loop. `0`, `off` or `false` disables it. See the [latency tuning guide](latency-tuning.md). |
| `REACHAQ_POSE_BACKEND` | `tensorflow` | Selects the DeepLabCut engine: `tensorflow` or `torch`. Also selects which GPU runtime the live-inference preflight probes, so the two cannot disagree. |
| `REACHAQ_INFERENCE_ROI` | unset | `x,y,width,height`. Crops live frames to this window before inference and shifts the returned coordinates back into full-frame space. Unset means full frame. |

`REACHAQ_POSE_BACKEND=torch` requires a shuffle trained with the PyTorch
engine; a project trained under TensorFlow has no `pytorch_config.yaml` and the
model load fails with a message saying so.

## Verification

Portable CLI/import checks and the focused non-hardware suite are part of the
tracked installer. The supported repair/verification workflow reruns every
category:

```bash
tools/install/reachaq-linux-install.sh
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
| PXI off and camera timeout occur together | Treat as correlated until wiring is checked; NI-DAQ software initialization does not trigger cameras |
| CAN interface down | [PEAK/SocketCAN guide](peak-socketcan.md), bitrate and termination |
| CAN reset asks for a password | Activate the `reachaq` login group by logging out/in, then verify the narrow `sudo -n` permission |
| CAN RX continues but startup ACK times out | Preserve the application log; this is an application startup/ACK path issue rather than proof of a dead bus |
| TensorFlow reports no GPU | [NVIDIA guide](nvidia-inference.md) or `--no-live-inference` |
| Output permission error | Configured data directory ownership and write permission |

## Current workstation reference

Last documentation update 2026-09-09. Entries marked (verified) were re-checked
on the workstation on that date.

- Dell Precision 3660 Tower; Ubuntu 22.04.5 LTS, x86_64; kernel
  `6.8.0-136-generic` (verified). 12th Gen Core i9-12900: 8 P-cores on CPU
  0-15 at 5000-5100 MHz, 8 E-cores on CPU 16-23 at 3800 MHz (verified).
- Conda environment `/home/christielab10/anaconda3/envs/reachaq`, Python
  3.8.20, TensorFlow **2.12.1** (verified). Not 2.13: that release pairs with a
  keras which moved `legacy_tf_layers`, which breaks `tf_slim` batch norm and
  therefore every DeepLabCut model load.
- Conda environment `reachaq-test310`, Python 3.10.21, for running the test
  suite on the version the packages now require. It has torch 2.14.0 and
  DeepLabCut 3.0.1 and deliberately no TensorFlow. It cannot run the three
  tests that import `PySpin`, because the vendored Spinnaker wheel is cp38.
- Spinnaker system runtime 3.2.0.57 (verified); bundled Python wheel 3.2.0.62.
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
- Full software suite on this workstation, Python 3.8: 1164 passed, 35 skipped
  (verified). Run it with `ulimit -n` at its default 1024; the suite no longer
  exhausts the descriptor limit.
