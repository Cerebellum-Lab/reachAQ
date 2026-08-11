# Installation

This repository still contains the broader Autotrainer modules, but the current
Linux reachAQ bring-up path is documented in:

- [linux-install-instructions.md](linux-install-instructions.md)

Use that guide for a machine that needs to run the reachAQ application with
Spinnaker cameras, NI-DAQmx/PXI hardware, PEAK SocketCAN, inference, and local
data output.

## Portable installation

From an existing checkout, run the tracked portable installer:

```bash
cd "$HOME/Documents/reachAQ"
./tools/install/reachaq-linux-install.sh
```

The installer accepts no options. It attempts every portable category in one
run: Ubuntu packages, automatic Miniconda bootstrap when needed, the complete
Python environment, TensorFlow-compatible CUDA libraries, Git LFS, CLI/import
verification, GPU preflight, and the focused non-hardware suite. Individual
failures do not stop later categories; a complete report is printed at the end.

The portable dependency set includes the complete Qt runtime used by the GUI,
the Linux Secret Service/keyring used for SoftMouse credentials, Requests and
OpenPyXL for metadata publication/import, and pyserial plus `dialout` group
membership for the USB RFID reader. If the installer adds your account to
`dialout`, log out and back in once, reconnect the reader, and rerun the same
installer command so its serial-permission check runs in the new login session.

Install the NVIDIA kernel driver first so the automatic TensorFlow GPU preflight
can pass. See the [NVIDIA/TensorFlow guide](docs/linux-install/nvidia-inference.md)
for supported versions and diagnostics.

Vendor drivers and rig configuration are intentionally separate. Use the
[Linux installation map](linux-install-instructions.md) to select only the
FLIR, NI/PXI, PEAK CAN, and NVIDIA guides applicable to the target system.
SoftMouse credentials and the optional single-machine nightly timer are also a
post-install setup because credentials cannot be entered safely by an
unattended installer. Follow the
[SoftMouse/RFID checklist](tools/softmouse_sync/CONFIGURATION_GUIDE.md).

Start the GUI with:

```bash
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

Keep `--no-capture-output` in long-running launch commands. Without it, Conda
buffers stdout/stderr and terminal logs may not appear until the app exits.

The GUI starts idle by default. Cameras, NI-DAQ, CAN, and live inference do not
start until the operator selects Running. Use `--start-mode running` only
when immediate acquisition startup is intentional.

For a software-only camera smoke test:

```bash
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  --random-cameras \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

Headless mode remains available through the following command. Unlike the GUI,
headless mode starts acquisition by default because it has no operator control
that can start an idle process later.

```bash
conda run --no-capture-output -n reachaq auto-trainer-headless \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

Omit `--no-live-inference` to honor the saved configuration, or use
`--live-inference` to enable it for one run. Live inference requires a working
NVIDIA driver and TensorFlow GPU runtime and intentionally does not use CPU
fallback.

## Legacy Notes

Older Autotrainer deployments used environment names such as
`auto-trainer-1`, Jetson-specific dependency pins, and direct module entry
points like `python -m tools.acquisition.gui`. Those may still be useful for
legacy rigs, but they are not the current reachAQ Dell/Ubuntu setup. Prefer the
Linux guide above when setting up a new reachAQ Linux machine.
