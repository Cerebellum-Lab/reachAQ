# reachAQ Linux installation

This is the entry point for Ubuntu reachAQ setup. Portable, repeatable host
commands live in a tracked installer; vendor drivers and rig-specific settings
are separated into focused guides.

Supported platforms are listed in
[supported platforms](docs/linux-install/supported-platforms.md) (today Ubuntu
22.04 x86_64, NVIDIA driver 570.26+, compute capability 7.5-12.0, Spinnaker
3.2). The installer checks the host first and refuses one that is not on the
list unless `REACHAQ_INSTALL_UNSUPPORTED=1` is set.

## Installation map

Complete only the hardware categories present on the target rig.

| Stage | Scope | Action |
|---|---|---|
| 1 | Repository | Clone the checkout |
| 2 | Portable host setup | Run `tools/install/reachaq-linux-install.sh` |
| 3a | FLIR cameras | [Spinnaker guide](docs/linux-install/flir-spinnaker.md) |
| 3b | NI DAQ/PXI | [NI-DAQmx and PXI/MXI guide](docs/linux-install/ni-daq-pxi.md), including the `intel_iommu=off` boot parameter Intel hosts need for NI DMA |
| 3c | Pellet CAN | [PEAK SocketCAN guide](docs/linux-install/peak-socketcan.md) |
| 3d | Live inference | [NVIDIA guide](docs/linux-install/nvidia-inference.md) |
| 3e | SoftMouse/RFID | [SoftMouse/RFID guide](src/reachaq/softmouse_sync/CONFIGURATION_GUIDE.md) |
| 3f | Closed-loop stim timing | [Latency tuning guide](docs/linux-install/latency-tuning.md) |
| 4 | Rig configuration and launch | [Runtime guide](docs/linux-install/runtime-configuration.md) |

The portable installer does **not** install NI, PEAK out-of-tree, or
NVIDIA kernel drivers and does not select camera serials, DAQ channels, CAN
bitrate, or an inference model. It does install everything that lives in the
Python environment: PyTorch and its CUDA user-space libraries, the pose plugins
the environment already had, and the Spinnaker SDK and Python binding from the
operator's own FLIR download (reachAQ ships no FLIR software). It installs the
portable `can-utils` and `iproute2` packages, but it does not install or enable
`reachaq-can.service`; that service is intentionally a separate, reviewed
hardware step.

## 1. Clone the repository

Install Git first if the host does not have it, then clone the project:

```bash
sudo apt update
sudo apt install -y git
mkdir -p "$HOME/Documents"
git clone <REACHAQ_REPOSITORY_URL> "$HOME/Documents/reachAQ"
cd "$HOME/Documents/reachAQ"
```

For an existing checkout:

```bash
cd "$HOME/Documents/reachAQ"
git status --short --branch
```

## 2. Run the portable installer

The script installs general Ubuntu/Qt/build packages, creates the Python 3.10
conda environment, installs Python dependencies and the editable project,
installs PyTorch, reinstalls the pose plugins the environment already had,
initializes Git LFS, installs the Spinnaker SDK and Python binding from the
operator's download, creates runtime directories, configures the operator's `dialout`
membership for USB serial access, and runs generic verification. It also
installs the Linux Secret Service/keyring components needed to store SoftMouse
credentials securely.

Run the installer:

```bash
cd "$HOME/Documents/reachAQ"
./tools/install/reachaq-linux-install.sh
```

For FLIR cameras, first save the Spinnaker download in `~/Downloads`; see the
[FLIR Spinnaker guide](docs/linux-install/flir-spinnaker.md).

The script accepts no options and runs every portable category. It
automatically installs Miniconda when Conda is absent, includes test
dependencies, runs the focused tests, installs PyTorch's CUDA user-space
libraries, and verifies PyTorch on the GPU. No pose plugin is installed by
default; `./tools/install/reachaq-plugin-install.sh deeplabcut` (DeepLabCut,
with TensorFlow, its CUDA libraries and the GPU checks for both engines) or
`./tools/install/reachaq-plugin-install.sh yolo` (YOLO-pose, AGPL-3.0) adds one
to the existing environment at any time. The
CUDA libraries remain inside the Conda environment; the script does not change
the kernel driver or install a system-wide CUDA toolkit.

### Installer behavior

- It is safe to re-run: existing directories are reused, and an environment
  already on Python 3.10 is updated in place.
- An environment on another Python version cannot be upgraded in place, so it
  is kept, renamed `reachaq-py<version>-<date>`, and a fresh one is built. Run
  the archive with
  `conda run --no-capture-output -n <archive> python -m reachaq.app`.
- The pose plugins the environment had are reinstalled from the checkout, into
  a rebuilt environment too. An environment from before the plugins existed
  gets the DeepLabCut plugin if it has `deeplabcut` installed and the YOLO
  plugin if it has `ultralytics`. To drop a plugin for good, uninstall both its
  distribution and its engine package (for example
  `pip uninstall reachaq-pose-yolo ultralytics`) before rerunning the
  installer, otherwise it restores the plugin.
- `REACHAQ_INSTALL_SYSTEM=0` skips the steps that need root (apt packages,
  groups, limits, systemd, FLIR camera access) and reports them as `SKIP`. Use
  it to rerun on a rig that already has them, for example after `reachaq-sync`.
  An installed Spinnaker SDK is still checked; one that is missing stops the
  install, because installing it needs root.
- Run it as the operator, not with `sudo`: it asks for sudo itself, and under
  sudo every per-user path would be root's. It refuses to run under `sudo`.
- `REACHAQ_INSTALL_SPINNAKER=0` leaves out the Spinnaker steps on a host
  without FLIR cameras; `REACHAQ_SPINNAKER_DOWNLOADS` names the folder holding
  the Spinnaker download (default `~/Downloads`).
- The first failed step stops the install. The summary printed at the end
  names that step and its failed command, and the output just above it says
  why; nothing after it has run. The exit status is nonzero.
- Every step is recorded as `PASS`, `FAIL`, `SKIP` (left out on purpose, with
  the reason), or `WARN` (worth knowing, did not stop the install, such as an
  unsupported platform accepted with `REACHAQ_INSTALL_UNSUPPORTED=1`).
- Re-running the same no-argument command is the supported repair and
  verification workflow.
- If the installer adds the operator to `dialout`, log out and back in before
  using the RFID reader, then rerun the installer to verify the active login's
  serial permissions.
- The same applies to the `reachaq-rt` group used for closed-loop stim timing:
  `rtprio` limits are applied at login, so log out and back in before the stim
  loop can request real-time priority. Until then it runs at normal priority
  and logs a warning; acquisition is unaffected. See the
  [latency tuning guide](docs/linux-install/latency-tuning.md).
- The installer enables `reachaq-cpu-governor.service`, which sets the
  `performance` CPU governor at boot, and, for FLIR cameras,
  `reachaq-usbfs-memory.service`, which sets the USB-FS buffer to 1000 MB.
  These are the systemd units the installer enables; `reachaq-can.service`
  remains a separate reviewed step.
- The report separately verifies SoftMouse imports/keyring availability, RFID
  imports/device permissions, and the tracked nightly systemd units.
- The generic import check verifies module file paths against the current
  checkout. This catches stale editable installs that still reference a removed
  worktree.

## 3. Install only applicable hardware support

### FLIR/Spinnaker cameras

The portable installer installs the SDK, udev rule, operator groups, USB-FS
memory and Python binding from the operator's own download, and lists the
cameras it can see. The [FLIR Spinnaker guide](docs/linux-install/flir-spinnaker.md)
says what to download and what each step does.

### NI-DAQmx and PXI/MXI

Follow the [NI-DAQ/PXI guide](docs/linux-install/ni-daq-pxi.md) for vendor
repositories, kernel modules, PXI services, cold-boot enumeration, device
aliases, and channel limitations.

### PEAK CAN / pellet controller

Follow the [PEAK SocketCAN guide](docs/linux-install/peak-socketcan.md) to verify
the adapter and bitrate, install and enable the tracked boot service, verify
that `can0` is automatically `UP` after reboot, install the narrowly privileged
reset helper, and test pellet-board discovery.

That guide installs these root-owned system artifacts:

- `/usr/local/sbin/reachaq-bring-up-can`
- `/usr/local/sbin/reachaq-bring-down-can`
- `/usr/local/sbin/reachaq-reset-can`
- `/etc/default/reachaq-can`
- `/etc/systemd/system/reachaq-can.service`
- `/etc/tmpfiles.d/reachaq-can.conf`
- `/etc/sudoers.d/reachaq-can-reset`

Adding the operator to the `reachaq` group requires a full logout/login before
the application can use its non-interactive safety reset. The service
configuration and application environment are separate; make
`REACHAQ_CAN_INTERFACE` and `REACHAQ_CAN_CHANNEL` select the same channel.

### NVIDIA live inference

Live inference is optional. Follow the
[NVIDIA guide](docs/linux-install/nvidia-inference.md) to replace `nouveau`
and verify `nvidia-smi`; the installer then adds both engines' CUDA libraries
and runs the GPU checks. Otherwise launch with `--no-live-inference`.

## 4. Configure and launch

The [runtime guide](docs/linux-install/runtime-configuration.md) covers:

- configuration and output paths;
- camera, DAQ, laser, CAN, and inference settings;
- safe idle GUI and headless launch commands;
- the terminal-visible desktop launcher installed by the portable installer;
- software-only random cameras;
- startup timing logs and troubleshooting.

Start it with `reachaq`, or the desktop icon. A safe first launch after
configuration, without live inference:

```bash
conda run --no-capture-output -n reachaq python -m reachaq.app \
  --no-live-inference \
  -c "$HOME/reachAQ/system_configuration.yaml"
```

`--no-capture-output` makes application logs stream to the launching terminal.
Omitting it causes Conda to hold stdout/stderr until the application exits.

The GUI opens idle by default. No acquisition hardware starts until the
operator selects Running.

## Completion checklist

- [ ] Portable installer report has no `FAIL` entry and no unexplained `WARN`.
- [ ] Only physically present hardware categories were installed.
- [ ] FLIR cameras, if used, appear as `spinnaker://` sources.
- [ ] NI devices, if used, appear in `nilsdev` and Python `nidaqmx` discovery.
- [ ] CAN interfaces, if used, are `UP` with the confirmed bitrate and board.
- [ ] `reachaq-can.service` is `enabled` and `active`, and `can0` comes back
  `UP` after a reboot without launching reachAQ.
- [ ] `id -nG` lists `reachaq`, and the reset-helper `sudo -n -l` check
  succeeds without prompting.
- [ ] Root-owned installed CAN files match the reviewed repository versions;
  `/etc/default/reachaq-can` retains any intentional rig-specific values.
- [ ] The application `REACHAQ_CAN_CHANNEL` matches service
  `REACHAQ_CAN_INTERFACE`.
- [ ] If inference is enabled, the report passes the PyTorch GPU check, and,
  with the DeepLabCut plugin, the TensorFlow and both-engines GPU checks.
- [ ] Configuration contains real serials, aliases, channels, paths, and model.
- [ ] Output directory exists and is writable by the operator.
- [ ] GUI launches idle and hardware initialization emits `HARDWARE INIT`
  progress records when Running is selected.
- [ ] The mouse-icon `reachAQ.desktop` entry opens a terminal with live logs,
  starts reachAQ in Idle, and refuses a duplicate GUI launch.
- [ ] SoftMouse runtime and systemd-unit checks pass in the installer report.
- [ ] The acquisition operator belongs to `dialout`; any attached RFID reader
  under `/dev/serial/by-id/` is readable and writable after a fresh login.
- [ ] On each machine allowed to publish, the one-time SoftMouse credential
  setup and a manual sync pass; the timer is enabled on exactly one machine.

## Safety boundary

The portable installer deliberately avoids actions that require knowledge of a
specific rig. Review each hardware guide before commands that can change kernel
drivers, boot services, bus state, analog outputs, shutters, or motor position.
