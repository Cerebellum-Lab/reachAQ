# Runtime configuration, launch, and verification

Complete the portable install and each applicable hardware guide before using a
physical subsystem.

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
| `hardware.*Enabled` | Enable only connected, validated subsystems |
| `inference.poseModelLocation` | Existing compatible model directory |
| `laser.backend` | `nidaq` for validated hardware; otherwise `null`/`disabled` |
| `laser.channels` | Real NI-DAQ aliases and wired channel roles |
| `nidaqPorts`, `nidaqStream` | Real device alias and supported channel types |

Use **Edit → Edit DAQ Ports** while idle to discover supported channel types and
prevent duplicate assignments.

## Safe first launch

The GUI defaults to idle. This command also disables inference for the run, so
no GPU preflight is required when the operator later selects Running:

```bash
cd "$REACHAQ_REPO"
conda run --no-capture-output -n "$REACHAQ_ENV" python -m reachAQ.app \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

Use `--start-mode acquiring` only when immediate acquisition startup is
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

The hardware profile defaults to `alogus` independently of which Python CAN
library is installed. The rig environment sets
`AUTOTRAINER_HARDWARE_VERSION=alogus` explicitly. A legacy Anschutz runtime
must opt in with `AUTOTRAINER_HARDWARE_VERSION=anschutz`; unknown values are
rejected rather than silently selecting another profile. Emulation is selected
separately with `AUTOTRAINER_CAN_TRANSPORT=emulation` and still uses the Alogus
profile.

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

The documented launch commands use `conda run --no-capture-output`. Do not omit
that option: default `conda run` captures the stream and can hide every terminal
record until reachAQ exits.

## Common failures

| Symptom | Next check |
|---|---|
| Git LFS test asset error | `git -C "$REACHAQ_REPO" lfs pull` |
| Qt xcb plugin error | `libxcb-cursor0`, `libxkbcommon-x11-0`, and display environment |
| Cameras absent | Applicable [FLIR guide](flir-spinnaker.md) or USB enumeration |
| NI devices absent | [NI-DAQ/PXI guide](ni-daq-pxi.md), starting at PCI/USB enumeration |
| CAN interface down | [PEAK/SocketCAN guide](peak-socketcan.md), bitrate and termination |
| CAN reset asks for a password | Activate the `reachaq` login group by logging out/in, then verify the narrow `sudo -n` permission |
| CAN RX continues but startup ACK times out | Preserve the application log; this is an application startup/ACK path issue rather than proof of a dead bus |
| TensorFlow reports no GPU | [NVIDIA guide](nvidia-inference.md) or `--no-live-inference` |
| Output permission error | Configured data directory ownership and write permission |

## Current workstation reference

Last checked 2026-07-21:

- Dell Precision 3660 Tower; Ubuntu 22.04.5 LTS, x86_64; kernel
  `6.8.0-124-generic`.
- Conda environment `/home/christielab10/anaconda3/envs/reachaq`, Python 3.8,
  TensorFlow 2.13.1.
- Spinnaker system runtime 3.2.0.57; bundled Python wheel 3.2.0.62.
- NI-DAQmx 26.3.1 and PXI Platform Services 26.3; PXI-6713 alias
  `PXI1Slot4`.
- PEAK PCIe adapter on `peak_pciefd`, exposing `can0` and `can1`.
- Quadro T1000 present but using `nouveau`; live inference unavailable.
- Local output `/home/christielab10/Documents/rawdatalocal`.
- Focused non-hardware/installer verification: 52 passed.
