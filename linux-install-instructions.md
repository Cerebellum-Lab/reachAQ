# reachAQ Linux install instructions

This guide is for setting up a Linux machine that can start and run the
reachAQ application with cameras, NI-DAQ hardware, CAN hardware, inference, and
local acquisition output. It includes the system drivers that may already be
installed on an existing rig, because those are required even when the Python
environment is correct.

The commands are written for Ubuntu 22.04 on x86_64. Other Linux distributions
can work, but you must choose matching NI, FLIR/Spinnaker, NVIDIA, and Python
packages for that OS and CPU architecture.

## Choose Install Paths

Set these values in your shell before following the commands. Adjust them for
the target machine.

```bash
export REACHAQ_REPO="$HOME/Documents/reachAQ"
export REACHAQ_ENV="reachaq-py38"
export REACHAQ_CONFIG="$HOME/Autotrainer/system_configuration.yaml"
export REACHAQ_DATA="$HOME/Documents/rawdatalocal"
```

Use Python 3.8 unless you also have a Spinnaker Python wheel built for another
Python version. The bundled Spinnaker wheels are `cp38` wheels.

## System Packages

Install core Linux packages, GUI/Qt dependencies, CAN tools, Git LFS, and build
helpers:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  dkms \
  expat \
  ffmpeg \
  git \
  git-lfs \
  iproute2 \
  pkg-config \
  v4l-utils \
  wget \
  can-utils \
  libegl1 \
  libgl1 \
  libopenal1 \
  libxcb-cursor0 \
  libxkbcommon-x11-0
```

Add the operator user to hardware-access groups used by serial/CAN/USB/camera
drivers. Log out and back in, or reboot, after changing groups.

```bash
sudo usermod -a -G dialout,plugdev "$USER"
groups
```

Spinnaker may also create a `flirimaging` group during its install. If it
exists, add the user to it too:

```bash
getent group flirimaging && sudo usermod -a -G flirimaging "$USER"
```

## Repository

Clone the repo and fetch Git LFS assets:

```bash
mkdir -p "$(dirname "$REACHAQ_REPO")"
git clone <REACHAQ_REPOSITORY_URL> "$REACHAQ_REPO"
cd "$REACHAQ_REPO"
git lfs install
git lfs pull
```

If the repo is already cloned, update the checkout and LFS assets instead:

```bash
cd "$REACHAQ_REPO"
git status --short --branch
git lfs install
git lfs pull
```

## Conda Installer

Install Miniconda, Anaconda, or another conda-compatible distribution if the
machine does not already have `conda`. This installer URL is for x86_64; use
the matching installer on aarch64/Jetson:

```bash
command -v conda || {
  cd "$HOME/Downloads"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
}
```

Open a new shell after installing conda, or source the conda profile script for
the install location you chose:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda --version
```

## Conda Environment

Create the runtime environment and install the Python package in editable mode:

```bash
conda create -y -n "$REACHAQ_ENV" python=3.8
conda run -n "$REACHAQ_ENV" python -m pip install --upgrade pip setuptools wheel build
cd "$REACHAQ_REPO"
conda run -n "$REACHAQ_ENV" python -m pip install -r requirements.txt
conda run -n "$REACHAQ_ENV" python -m pip install -e '.[test]'
```

Install Git LFS inside the env as well, so env-local commands work:

```bash
conda install -y -n "$REACHAQ_ENV" -c conda-forge git-lfs
conda run -n "$REACHAQ_ENV" git lfs install --local
conda run -n "$REACHAQ_ENV" git lfs pull
```

Expected important Python packages include:

- `nidaqmx`
- `python-can`
- `PySide6`
- `opencv-python`
- `deeplabcut`
- `tensorflow`
- `torch`
- `spinnaker-python`, installed separately below to match the system SDK

## FLIR Spinnaker Cameras

Install the Teledyne FLIR Spinnaker SDK system runtime before installing the
Python binding. The system SDK and Python wheel must have the same Spinnaker
version.

On a new x86_64 Ubuntu 22.04 machine, use the same SDK version as the wheel you
intend to install. This repo currently includes Spinnaker Python 3.2.0.62 wheels
under `library/`:

```bash
ls "$REACHAQ_REPO"/library/spinnaker_python-*-linux_*.whl
```

Install the Spinnaker SDK from the FLIR download bundle. The exact package names
vary by SDK release, so use the SDK's README/install script when available. A
typical manual install flow is:

```bash
cd "$HOME/Downloads/Spinnaker-<VERSION>-Linux"
sudo apt install ./*.deb
sudo ldconfig
sudo reboot
```

After reboot, verify the system SDK and udev rules:

```bash
ldconfig -p | grep -i spinnaker
ls /etc/udev/rules.d/*spinnaker* 2>/dev/null
groups
```

Install the matching Python wheel. For the bundled 3.2.0.62 x86_64 wheel:

```bash
conda run -n "$REACHAQ_ENV" python -m pip install \
  "$REACHAQ_REPO/library/spinnaker_python-3.2.0.62-cp38-cp38-linux_x86_64.whl"
```

For Jetson/aarch64, use the matching aarch64 wheel and the matching aarch64
system SDK:

```bash
conda run -n "$REACHAQ_ENV" python -m pip install \
  "$REACHAQ_REPO/library/spinnaker_python-3.2.0.62-cp38-cp38-linux_aarch64.whl"
```

Verify Python can import PySpin:

```bash
conda run -n "$REACHAQ_ENV" python -c "import PySpin; print('PySpin import ok')"
```

If cameras are not discovered, confirm camera power/cables, group membership,
udev rules, and reboot after driver changes.

## NI-DAQmx

Install NI Linux Device Drivers / NI-DAQmx for the target Ubuntu release and the
actual DAQ hardware model. Raw USB/PCI visibility is not enough for reachAQ; the
device must be visible through NI-DAQmx.

For the NI 2026 Q2 Ubuntu 22.04 repository bundle, the local install flow is:

```bash
cd "$HOME/Downloads/NILinux2026Q2DeviceDrivers"
sudo apt install ./ni-ubuntu2204-drivers-2026Q2.deb
sudo apt update
sudo apt install -y dkms expat libopenal1
apt-cache search ni-daqmx
sudo apt install -y ni-daqmx ni-hwcfg-utility
sudo dkms autoinstall
sudo reboot
```

After reboot, verify the NI services and kernel modules:

```bash
systemctl --no-pager --plain list-units 'ni*' 'nidaq*' 'nipal*' 'nim*'
dkms status | grep -Ei 'ni|nidaq|nipal|nistc|nic'
lsmod | grep -Ei '^(ni|nidaq|nipal|nimbus|nistc|nic|nix|nim)'
```

Verify DAQ visibility:

```bash
lsusb | grep -Ei 'national|instruments' || true
lspci -nn | grep -Ei 'national|instruments|daq' || true
lsni -v
nilsdev
nidaqmxconfig --export /tmp/reachaq-nidaqmx-export.ini
sed -n '1,120p' /tmp/reachaq-nidaqmx-export.ini
```

Verify the same from Python:

```bash
conda run -n "$REACHAQ_ENV" python - <<'PY'
import nidaqmx
system = nidaqmx.system.System.local()
devices = tuple(system.devices)
print("device_count", len(devices))
for device in devices:
    print(device.name, device.product_type, device.product_num, device.serial_num)
PY
```

The configured DAQ must appear through `nilsdev` or
`nidaqmx.system.System.local().devices` before reachAQ can use channels such as
`Dev1/ao0`, `Dev1/ai0`, and `Dev1/port0/line0`.

NI currently documents kernel/IOMMU caveats for some drivers on Linux kernel 6.8
and newer. If NI hardware is absent after install and reboot, check NI's current
Ubuntu driver page for the target driver release.

## CAN / PEAK SocketCAN

reachAQ uses `python-can`. On Linux, the preferred path is SocketCAN, with PEAK
PCIe/USB adapters exposed as `can0`, `can1`, and so on.

Load the PEAK kernel driver and inspect CAN interfaces:

```bash
sudo modprobe peak_pciefd || sudo modprobe peak_usb
ip -details link show type can
lspci -nnk | grep -A3 -Ei 'peak|pcan|can' || true
lsusb | grep -Ei 'peak|pcan' || true
```

Bring up the bus after confirming the bench bitrate:

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

Load the reachAQ CAN environment variables:

```bash
cd "$REACHAQ_REPO"
set -a
source tools/hardware/reachaq_hardware.env.example
set +a
```

Validate with can-utils and the repo helper:

```bash
candump can0
conda run -n "$REACHAQ_ENV" python tools/hardware/validate_can_hardware.py \
  --transport socketcan \
  --channel can0 \
  --action discover
```

Use PEAK's out-of-tree PCAN-Linux package only when the in-kernel SocketCAN
driver is missing, too old for the adapter, or the rig intentionally uses PCAN
Basic instead of SocketCAN.

## Optional NVIDIA GPU

The application can start without a CUDA-capable GPU, but inference is usually
much faster with NVIDIA drivers installed. Install the NVIDIA driver/CUDA stack
that matches the target machine and the TensorFlow/Torch wheels in the Python
environment. After installation:

```bash
nvidia-smi
conda run -n "$REACHAQ_ENV" python - <<'PY'
import tensorflow as tf
print("tensorflow", tf.__version__)
print("gpus", tf.config.list_physical_devices("GPU"))
PY
```

On Jetson/aarch64, use NVIDIA's JetPack-compatible TensorFlow/Torch packages.
The repo metadata pins Jetson-specific package builds separately from x86_64.

## Runtime Configuration

Create the local app config and acquisition output directory:

```bash
mkdir -p "$HOME/Autotrainer" "$REACHAQ_DATA"
cd "$REACHAQ_REPO"
if [ ! -f "$REACHAQ_CONFIG" ]; then
  cp tools/hardware/reachaq_system_configuration.example.yaml "$REACHAQ_CONFIG"
fi
```

Edit `~/Autotrainer/system_configuration.yaml` for the target machine:

- `persistence.outputLocation`: set to a writable data directory, commonly
  `$HOME/Documents/rawdatalocal`.
- `inference.poseModelLocation`: set to the local DeepLabCut/reach model
  directory. The directory must exist and contain the model files expected by
  the configured inference workflow.
- `cameras`: configure only cameras physically present on the rig. Camera slots
  3-6 appear in the UI only when explicitly configured.
- `cameras[*].scheme`: usually `spinnaker` for FLIR cameras.
- `cameras[*].path`: set to the actual camera identifier used by the video
  layer, such as the configured camera name or serial mapping for the rig.
- `hardware.canEnabled`, `hardware.pelletControllerEnabled`, and related
  hardware flags: enable only connected subsystems.
- `laser.backend`: use `nidaq` for NI-DAQ laser control, `null` or `disabled`
  for a software-only smoke test.
- `laser.channels`, `nidaqPorts`, and `nidaqStream`: replace every `Dev1/...`
  channel with the actual NI-DAQ device alias and channel map.

For this workstation's requested local data path, the persistence section is:

```yaml
persistence: !PersistenceConfiguration
  outputLocation: /home/christielab10/Documents/rawdatalocal
```

On another machine, prefer:

```yaml
persistence: !PersistenceConfiguration
  outputLocation: /home/<USER>/Documents/rawdatalocal
```

Validate laser channel configuration only after the real DAQ hardware and wiring
are confirmed:

```bash
conda run -n "$REACHAQ_ENV" python tools/hardware/validate_laser_hardware.py \
  --config "$REACHAQ_CONFIG" \
  --action connect
```

## Launch

Load CAN environment variables before running with CAN hardware enabled:

```bash
cd "$REACHAQ_REPO"
set -a
source tools/hardware/reachaq_hardware.env.example
set +a
```

Start the GUI in idle mode while validating drivers and bench wiring:

```bash
conda run -n "$REACHAQ_ENV" python -m reachAQ.app \
  --start-mode idle \
  -c "$REACHAQ_CONFIG"
```

Equivalent installed entry point:

```bash
conda run -n "$REACHAQ_ENV" reachaq --start-mode idle -c "$REACHAQ_CONFIG"
```

Headless mode:

```bash
conda run -n "$REACHAQ_ENV" auto-trainer-headless \
  --start-mode idle \
  -c "$REACHAQ_CONFIG"
```

The default start mode may begin acquisition immediately, so use
`--start-mode idle` during bring-up.

Optional convenience symlinks in the repo root:

```bash
cd "$REACHAQ_REPO"
CONDA_BASE="$(conda info --base)"
ln -sf "$CONDA_BASE/envs/$REACHAQ_ENV/bin/reachaq" run-reachaq
ln -sf "$CONDA_BASE/envs/$REACHAQ_ENV/bin/auto-trainer-local" run-auto-trainer-local
ln -sf "$CONDA_BASE/envs/$REACHAQ_ENV/bin/auto-trainer-headless" run-auto-trainer-headless
```

You can also use the absolute conda env path directly, for example
`/home/<USER>/anaconda3/envs/reachaq-py38/bin/reachaq`.

## Verification

Run these checks after install:

```bash
cd "$REACHAQ_REPO"
conda run -n "$REACHAQ_ENV" python --version
conda run -n "$REACHAQ_ENV" python -m pip check
conda run -n "$REACHAQ_ENV" reachaq -h
conda run -n "$REACHAQ_ENV" auto-trainer-headless -h
conda run -n "$REACHAQ_ENV" python -c "import autotrainer.core, autotrainer.device, autotrainer.video, PySide6, cv2, can, nidaqmx; print('imports ok')"
conda run -n "$REACHAQ_ENV" python -c "import PySpin; print('PySpin import ok')"
```

Run the focused non-hardware tests:

```bash
conda run -n "$REACHAQ_ENV" python -m pytest \
  auto-trainer-device/tests/can_transport_test.py \
  auto-trainer-device/tests/laser_test.py \
  tests/behavior_model_test.py::TestEmergency \
  tests/autotrainer_headless_test.py::test_cli_help \
  tests/autotrainer_headless_test.py::test_load_config \
  tests/autotrainer_headless_test.py::test_load_config_extra_reach_camera_slot \
  -q
```

## Troubleshooting

- If tests fail with `invalid load key, 'v'`, Git LFS assets were not fetched.
  Run `git lfs pull` from the repo.
- If Qt reports an xcb platform plugin error, reinstall the GUI dependency set,
  especially `libxcb-cursor0` and `libxkbcommon-x11-0`.
- If PySpin imports but cameras are absent, verify camera power/cables,
  Spinnaker udev rules, `flirimaging` membership, and reboot after driver
  changes.
- If NI devices are absent, confirm the exact DAQ model is supported by the
  installed NI-DAQmx Linux driver and that it appears in `nilsdev`.
- If CAN defaults to another backend, source
  `tools/hardware/reachaq_hardware.env.example` or export the SocketCAN
  variables manually.
- If `can0` exists but shows `DOWN` or `STOPPED`, bring it up with the correct
  bitrate and confirm the bus is wired/terminated before expecting traffic.
- If the app cannot write data, create the configured output directory and
  confirm it is writable by the operator user.
- If TensorFlow logs that CUDA/TensorRT is missing, the app can still start in
  CPU mode, but inference may be slower.

## Current Workstation Reference

The machine used to build these notes was checked on 2026-07-07:

- OS: Ubuntu 22.04.5 LTS, x86_64.
- Kernel: `6.8.0-124-generic`.
- Repo: `/home/christielab10/Documents/reachAQ`, branch `devel`.
- Conda env: `/home/christielab10/anaconda3/envs/reachaq-py38`.
- Spinnaker SDK runtime: 3.2.0.57 under `/opt/spinnaker`.
- NI-DAQmx: 26.3.1 from the NI 2026 Q2 Ubuntu 22.04 repo.
- CAN: PEAK PCIe card detected by `peak_pciefd`, exposing `can0` and `can1`.
- Data output: `/home/christielab10/Documents/rawdatalocal`.
- Focused verification result: `21 passed`.

## Upstream References

- NI Ubuntu driver install:
  https://www.ni.com/docs/en-US/bundle/ni-platform-on-linux-desktop/page/installing-ni-products-ubuntu.html
- NI-DAQmx Python docs:
  https://nidaqmx-python.readthedocs.io/en/latest/
- PEAK Linux / SocketCAN driver notes:
  https://www.peak-system.com/fileadmin/media/linux/index.php
- PEAK SocketCAN driver implementation notes:
  https://www.peak-system.com/fileadmin/media/linux/can-implementation.php
- Teledyne FLIR Spinnaker SDK:
  https://prep.flir.com/products/spinnaker-sdk/
- GitHub Git LFS install notes:
  https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
