# reachAQ Linux installation

This is the entry point for Ubuntu reachAQ setup. Portable, repeatable host
commands live in a tracked installer; vendor drivers and rig-specific settings
are separated into focused guides.

Supported reference platform: Ubuntu 22.04 x86_64. Other distributions and
Jetson/aarch64 require matching vendor packages.

## Installation map

Complete only the hardware categories present on the target rig.

| Stage | Scope | Action |
|---|---|---|
| 1 | Repository | Clone the checkout |
| 2 | Portable host setup | Run `tools/install/reachaq-linux-install.sh` |
| 3a | FLIR cameras | [Spinnaker guide](docs/linux-install/flir-spinnaker.md) |
| 3b | NI DAQ/PXI | [NI-DAQmx and PXI/MXI guide](docs/linux-install/ni-daq-pxi.md) |
| 3c | Pellet CAN | [PEAK SocketCAN guide](docs/linux-install/peak-socketcan.md) |
| 3d | Live inference | [NVIDIA/TensorFlow guide](docs/linux-install/nvidia-inference.md) |
| 4 | Rig configuration and launch | [Runtime guide](docs/linux-install/runtime-configuration.md) |

The portable installer does **not** install FLIR, NI, PEAK out-of-tree, or
NVIDIA drivers and does not select camera serials, DAQ channels, CAN bitrate, or
an inference model.

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

The script installs general Ubuntu/Qt/build packages, creates the conda
environment, installs Python dependencies and the editable project, initializes
Git LFS, creates runtime directories, and runs generic verification.

If conda is already installed:

```bash
cd "$HOME/Documents/reachAQ"
./tools/install/reachaq-linux-install.sh
```

If conda is not installed, explicitly allow the script to install Miniconda:

```bash
./tools/install/reachaq-linux-install.sh --install-miniconda
```

To include the focused non-hardware tests:

```bash
./tools/install/reachaq-linux-install.sh --run-tests
```

### Installer behavior

- It is safe to re-run: existing directories and conda environments are reused.
- It does not use `set -e`; one failed command does not stop later categories.
- Every step is recorded as `PASS`, `FAIL`, `SKIP`, or `PLAN`.
- A complete summary is always printed at the end.
- The final process exit is nonzero if any step failed, after all eligible steps
  have run.
- `--dry-run` prints the plan without changing the host.

Common options:

| Option | Effect |
|---|---|
| `--repo PATH` | Use another checkout |
| `--env NAME` | Use another conda environment name |
| `--python VERSION` | Select the environment Python version |
| `--config-dir PATH` | Select the configuration directory |
| `--data-dir PATH` | Select the acquisition output directory |
| `--install-miniconda` | Bootstrap Miniconda only when conda is absent |
| `--run-tests` | Run the focused 46-test non-hardware/installer suite |
| `--without-test-deps` | Omit the Python test extra |
| `--skip-system-packages` | Skip apt update/install |
| `--skip-python-env` | Reuse but do not modify the conda environment |
| `--skip-git-lfs` | Skip Git LFS initialization/pull |
| `--skip-verification` | Skip CLI/import verification |
| `--dry-run` | Report planned commands only |

Full option help:

```bash
./tools/install/reachaq-linux-install.sh --help
```

## 3. Install only applicable hardware support

### FLIR/Spinnaker cameras

Follow the [FLIR Spinnaker guide](docs/linux-install/flir-spinnaker.md) to match
the vendor SDK, CPython wheel, architecture, udev rules, and operator groups.

### NI-DAQmx and PXI/MXI

Follow the [NI-DAQ/PXI guide](docs/linux-install/ni-daq-pxi.md) for vendor
repositories, kernel modules, PXI services, cold-boot enumeration, device
aliases, and channel limitations.

### PEAK CAN / pellet controller

Follow the [PEAK SocketCAN guide](docs/linux-install/peak-socketcan.md) to verify
the adapter, bitrate, bus state, tracked boot service, and pellet-board
discovery.

### NVIDIA live inference

Live inference is optional. Follow the
[NVIDIA/TensorFlow guide](docs/linux-install/nvidia-inference.md) to replace
`nouveau`, verify `nvidia-smi`, align CUDA/cuDNN with TensorFlow, and run the
reachAQ preflight. Otherwise launch with `--no-live-inference`.

## 4. Configure and launch

The [runtime guide](docs/linux-install/runtime-configuration.md) covers:

- configuration and output paths;
- camera, DAQ, laser, CAN, and inference settings;
- safe idle GUI and headless launch commands;
- software-only random cameras;
- startup timing logs and troubleshooting.

Safe first GUI launch after configuration:

```bash
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

`--no-capture-output` makes application logs stream to the launching terminal.
Omitting it causes Conda to hold stdout/stderr until the application exits.

The GUI opens idle by default. No acquisition hardware starts until the
operator selects Running.

## Completion checklist

- [ ] Portable installer report has no unexplained `FAIL` entries.
- [ ] Only physically present hardware categories were installed.
- [ ] FLIR cameras, if used, appear as `spinnaker://` sources.
- [ ] NI devices, if used, appear in `nilsdev` and Python `nidaqmx` discovery.
- [ ] CAN interfaces, if used, are `UP` with the confirmed bitrate and board.
- [ ] TensorFlow, if inference is enabled, reports a GPU and passes preflight.
- [ ] Configuration contains real serials, aliases, channels, paths, and model.
- [ ] Output directory exists and is writable by the operator.
- [ ] GUI launches idle and hardware initialization emits `HARDWARE INIT`
  progress records when Running is selected.

## Safety boundary

The portable installer deliberately avoids actions that require knowledge of a
specific rig. Review each hardware guide before commands that can change kernel
drivers, boot services, bus state, analog outputs, shutters, or motor position.
