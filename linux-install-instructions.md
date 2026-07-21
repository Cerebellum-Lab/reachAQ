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
export REACHAQ_ENV="reachaq"
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
sudo apt install -y ni-daqmx ni-pxiplatformservices ni-qpxi ni-hwcfg-utility
sudo dkms autoinstall
sudo reboot
```

For PXI/PXIe chassis over MXI, install NI-DAQmx plus PXI Platform Services.
`ni-pxiplatformservices` provides the PXI resource-manager stack, MXI support,
and `nipxiconfig`. `ni-qpxi` adds PXI query support used by NI System
Configuration. `ni-visa` is optional for VISA/instrument workflows, but reachAQ's
NI output path uses NI-DAQmx device names through `nidaqmx`, not VISA:

```bash
sudo apt install -y ni-daqmx ni-pxiplatformservices ni-qpxi ni-hwcfg-utility

# Optional, only if you also need NI-VISA resource visibility/tools:
sudo apt install -y ni-visa ni-visa-passport-pxi
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
lspci -nn | grep -Ei '1093|national|instruments|daq|6713|pxi|plx|10b5' || true
lsni -v
nipxiconfig --list-system --verbose
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
`<DEVICE>/ao0`, `<DEVICE>/ai0`, and `<DEVICE>/port0/line0`.

NI currently documents kernel/IOMMU caveats for some drivers on Linux kernel 6.8
and newer. If NI hardware is absent after install and reboot, check NI's current
Ubuntu driver page for the target driver release.

### PXIe-1073 / NI 6713 Over MXI

The PXIe-1073 chassis with an NI 6713 analog-output card should be treated as a
PCI/PXI device path, not as USB hardware. Bring it up in this order:

1. Power down the computer and PXIe-1073.
2. Seat the NI 6713 firmly in the PXIe-1073.
3. Connect the MXI cable between the computer-side NI MXI interface and the
   PXIe-1073.
4. Power on the PXIe-1073 first and wait for the chassis/MXI link LEDs to
   settle.
5. Power on the computer and boot Linux with the chassis already on.
6. Run the verification commands below before starting reachAQ.

```bash
lspci -tv
lspci -nnk | grep -A4 -Ei '1093|national|6713|10b5|plx'
journalctl -k -b --no-pager | grep -Ei 'pxi|mxi|8361|1073|6713|1093|plx|bus number' | tail -200
lsni -v
nipxiconfig --list-system --verbose
nilsdev
```

Expected progression:

- `lspci` should show the MXI/PLX bridge chain and an NI endpoint for the 6713
  with vendor ID `1093`.
- `lsni -v` should show the MXI link/chassis instead of only the host computer.
- `nipxiconfig --list-system --verbose` should report PXI/PXIe system
  information.
- `nilsdev` should list a DAQmx device before reachAQ can use channels such as
  `PXI1Slot4/ao0`.

If Linux shows only the MXI bridge or PLX switch and no NI `1093` endpoint,
the chassis/card has not enumerated at the PCIe layer yet. Do a cold boot with
the PXIe chassis powered on before the PC. If `journalctl` reports
`No bus number available for hot-added bridge`, the chassis was hot-added or the
BIOS did not reserve enough downstream PCIe bus numbers. Check BIOS/firmware for
PCIe options such as Above 4G Decoding, PCIe hotplug/pre-boot enumeration, and
ASPM/power-management settings, then cold boot again. Also try a different PCIe
slot for the MXI host card and verify the MXI cable/link LEDs.

If the hardware appears in `lspci` but not `nilsdev`, rebuild DKMS and restart
NI services or reboot:

```bash
sudo dkms autoinstall
sudo systemctl restart nipal nidevldu nidrum nimxssvr ni-pxipf-nipxirm-bind nipxicmsd
sudo reboot
```

Once `nilsdev` lists the NI 6713, update
`~/Autotrainer/system_configuration.yaml` so `laser.channels`,
`nidaqPorts.deviceName`, and any `nidaqStream` channels use that device alias.
The NI 6713 provides analog output, TTL-compatible digital I/O, and timing I/O;
it does not provide analog input. If the reachAQ laser configuration uses
`diodeInput` or `commandCopyInput` analog feedback channels, add a supported
NI-DAQ analog-input device or disable/adjust that feedback path before expecting
full laser readback behavior.

Confirmed on this workstation, the NI 6713 appears as `PXI1Slot4`:

```text
nilsdev: PXI1Slot4
ProductType: PXI-6713
AO: PXI1Slot4/ao0 through PXI1Slot4/ao7
AI: none
DIO: PXI1Slot4/port0/line0 through PXI1Slot4/port0/line7
Counters: PXI1Slot4/ctr0, PXI1Slot4/ctr1, PXI1Slot4/freqout
```

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

The current PEAK PCIe card on this workstation is recognized as PCI device
`001c:0013` with kernel driver `peak_pciefd` and exposes two SocketCAN
interfaces: `can0` and `can1`.

Bring up the bus after confirming the bench bitrate:

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

To bring up both current PEAK PCIe channels manually:

```bash
for dev in can0 can1; do
  sudo ip link set "$dev" down || true
  sudo ip link set "$dev" type can bitrate 1000000 restart-ms 100
  sudo ip link set "$dev" txqueuelen 1000
  sudo ip link set "$dev" up
done
ip -details -brief link show type can
```

### CAN Interfaces On Boot

Install the repo-provided systemd oneshot service so Linux brings `can0` and
`can1` up on every boot:

```bash
cd "$REACHAQ_REPO"
sudo install -m 0755 tools/hardware/reachaq-bring-up-can.sh /usr/local/sbin/reachaq-bring-up-can
sudo install -m 0644 tools/hardware/reachaq-can.default /etc/default/reachaq-can
sudo install -m 0644 tools/hardware/reachaq-can.service /etc/systemd/system/reachaq-can.service
sudo systemctl daemon-reload
sudo systemctl enable --now reachaq-can.service
```

The default service settings are:

```bash
REACHAQ_CAN_INTERFACES="can0 can1"
REACHAQ_CAN_BITRATE=1000000
REACHAQ_CAN_RESTART_MS=100
REACHAQ_CAN_TXQUEUELEN=1000
REACHAQ_CAN_DRIVER=peak_pciefd
REACHAQ_CAN_FD=false
```

Edit `/etc/default/reachaq-can` if another rig uses a different interface list,
bitrate, driver, or CAN-FD settings. Do not configure the same CAN interfaces
through another boot mechanism at the same time.

Verify the boot service immediately after installing:

```bash
systemctl --no-pager status reachaq-can.service
ip -details -brief link show type can
journalctl -u reachaq-can.service -b --no-pager
```

Expected interface state:

```text
can0 UP ...
can1 UP ...
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

Optional PEAK terminal monitor:

```bash
codename="$(lsb_release -cs)"
wget -q "http://www.peak-system.com/debian/dists/${codename}/peak-system.sources" \
  -O /tmp/peak-system.sources
wget -q http://www.peak-system.com/debian/peak-system-public-key.asc \
  -O /tmp/peak-system-public-key.asc
sudo install -m 0644 /tmp/peak-system.sources /etc/apt/sources.list.d/peak-system.sources
sudo install -m 0644 /tmp/peak-system-public-key.asc /etc/apt/trusted.gpg.d/peak-system-public-key.asc
sudo apt-get update
sudo apt-get install -y pcanview-ncurses
```

The Debian package is named `pcanview-ncurses`, but it installs the executable
as `/usr/bin/pcanview`:

```bash
dpkg -L pcanview-ncurses | grep /usr/bin
pcanview
```

Important: `pcanview` expects PEAK's `/dev/pcanx` character-device driver path,
not Linux SocketCAN interfaces such as `can0` and `can1`. On the current
reachAQ workstation the PEAK card is handled by the in-kernel SocketCAN driver
`peak_pciefd`, so `can0`/`can1` should be monitored with `candump`,
`cansniffer`, `ip -details link`, or the repo helper instead:

```bash
candump can0
cansniffer can0
```

Only use `pcanview` if the rig intentionally uses PEAK's out-of-tree PCAN driver
and `/dev/pcan*` devices exist:

```bash
ls -l /dev/pcan*
pcanview /dev/pcan0
```

Use PEAK's out-of-tree PCAN-Linux package only when the in-kernel SocketCAN
driver is missing, too old for the adapter, or the rig intentionally uses PCAN
Basic instead of SocketCAN.

## Optional NVIDIA GPU

The application can run without a CUDA-capable GPU when live inference is
disabled. Live inference requires the proprietary NVIDIA driver and a working
TensorFlow GPU runtime; it does not fall back to CPU. Install the NVIDIA
driver/CUDA stack that matches the target machine and the TensorFlow wheel in
the Python environment. After installation:

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
- `laser.channels`, `nidaqPorts`, and `nidaqStream`: replace every placeholder
  channel with the actual NI-DAQ device alias and channel map.
  The in-app DAQ port editor uses NI-DAQmx discovery to list only channel types
  supported by the selected device and rejects duplicate assignments.

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

### Software-Only Random Cameras

To start the app without physical camera hardware, keep your normal system
configuration but add `--random-cameras`. This changes camera sources in memory
for the current run only and does not overwrite the hardware camera config on
close:

```bash
conda run -n "$REACHAQ_ENV" python -m reachAQ.app \
  --random-cameras \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

The override converts configured reach/web camera entries to `random://`
sources. If the config has no reach camera entries at all, it creates enabled
random `left` and `right` reach cameras for that run.

For a fully software-only config file, keep it in a separate configuration
directory with a matching preferences file. This matters because the app saves
configuration back to the preferences configuration directory on close:

```bash
mkdir -p "$HOME/Autotrainer-random/animals"
cp "$REACHAQ_REPO/tools/hardware/reachaq_random_camera_configuration.example.yaml" \
  "$HOME/Autotrainer-random/system_configuration.yaml"
python - <<'PY'
from pathlib import Path
root = Path.home() / "Autotrainer-random"
config = root / "system_configuration.yaml"
text = config.read_text()
text = text.replace("/tmp/reachaq-random-data", str(Path.home() / "Documents/rawdatalocal"))
config.write_text(text)
(root / "settings.ini").write_text(
    "[system]\n"
    f"configuration_location={root.as_posix()}\n"
    f"animal_location={(root / 'animals').as_posix()}\n"
)
PY
conda run -n "$REACHAQ_ENV" python -m reachAQ.app \
  --no-live-inference \
  --preferences-file "$HOME/Autotrainer-random/settings.ini" \
  -c "$HOME/Autotrainer-random/system_configuration.yaml"
```

## Launch

Load CAN environment variables before running with CAN hardware enabled:

```bash
cd "$REACHAQ_REPO"
set -a
source tools/hardware/reachaq_hardware.env.example
set +a
```

Start the GUI while validating drivers and bench wiring. The GUI defaults to
idle, and this example explicitly disables live inference for the run:

```bash
conda run -n "$REACHAQ_ENV" python -m reachAQ.app \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

Equivalent installed entry point:

```bash
conda run -n "$REACHAQ_ENV" reachaq --no-live-inference -c "$REACHAQ_CONFIG"
```

Headless mode:

```bash
conda run -n "$REACHAQ_ENV" auto-trainer-headless \
  --no-live-inference \
  -c "$REACHAQ_CONFIG"
```

The GUI default is idle. Headless mode still starts acquisition immediately by
default because it has no UI start control. Use `--start-mode acquiring` to
request immediate GUI startup, and use `--live-inference` or
`--no-live-inference` to override the saved inference setting for one run.

Optional convenience symlinks in the repo root:

```bash
cd "$REACHAQ_REPO"
CONDA_BASE="$(conda info --base)"
ln -sf "$CONDA_BASE/envs/$REACHAQ_ENV/bin/reachaq" run-reachaq
ln -sf "$CONDA_BASE/envs/$REACHAQ_ENV/bin/auto-trainer-local" run-auto-trainer-local
ln -sf "$CONDA_BASE/envs/$REACHAQ_ENV/bin/auto-trainer-headless" run-auto-trainer-headless
```

You can also use the absolute conda env path directly, for example
`/home/<USER>/anaconda3/envs/reachaq/bin/reachaq`.

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
  tests/autotrainer_headless_test.py::test_load_config_random_camera_override \
  tests/autotrainer_headless_test.py::test_load_config_random_camera_override_adds_default_reach_cameras \
  tests/nidaq_port_configuration_dialog_test.py \
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
- If TensorFlow cannot see a CUDA GPU, disable live inference in Preferences or
  launch with `--no-live-inference`. Live inference intentionally refuses to
  run on CPU.

## Current Workstation Reference

The machine used to build these notes was checked on 2026-07-17:

- OS: Ubuntu 22.04.5 LTS, x86_64.
- Kernel: `6.8.0-124-generic`.
- Repo: `/home/christielab10/Documents/reachAQ`, branch `devel`.
- Conda env: `/home/christielab10/anaconda3/envs/reachaq`.
- Spinnaker SDK runtime: 3.2.0.57 under `/opt/spinnaker`.
- NI-DAQmx: 26.3.1 from the NI 2026 Q2 Ubuntu 22.04 repo.
- PXI Platform Services: 26.3 from the NI 2026 Q2 Ubuntu 22.04 repo.
- PXIe/MXI observation on 2026-07-17: `lsni -v` saw `MXI1` as an NI PCIe-8361
  and `PXIChassis1` as an NI PXIe-1073. `lspci` saw the NI PXI-6713 endpoint
  `1093:2b80` using kernel driver `niwf`, and `nilsdev` listed `PXI1Slot4`.
- CAN: PEAK PCIe card detected by `peak_pciefd`, exposing `can0` and `can1`.
- Data output: `/home/christielab10/Documents/rawdatalocal`.
- Focused verification result: `8 passed`.

## Upstream References

- NI Ubuntu driver install:
  https://www.ni.com/docs/en-US/bundle/ni-platform-on-linux-desktop/page/installing-ni-products-ubuntu.html
- NI supported Linux driver packages:
  https://www.ni.com/docs/en-US/bundle/ni-platform-on-linux-desktop/page/supported-drivers-for-linux-distributions.html
- NI-DAQmx Python docs:
  https://nidaqmx-python.readthedocs.io/en/latest/
- NI-DAQmx for Linux readme, including supported NI 6713 AO devices:
  https://www.ni.com/pdf/manuals/378678b.html
- NI PXI Platform Services Linux readme:
  https://www.ni.com/pdf/manuals/378682a.html
- NI MXI-Express troubleshooting guide:
  https://knowledge.ni.com/KnowledgeArticleDetails?id=kA03q000000x0MKCAY
- PEAK Linux / SocketCAN driver notes:
  https://www.peak-system.com/fileadmin/media/linux/index.php
- PEAK SocketCAN driver implementation notes:
  https://www.peak-system.com/fileadmin/media/linux/can-implementation.php
- Teledyne FLIR Spinnaker SDK:
  https://prep.flir.com/products/spinnaker-sdk/
- GitHub Git LFS install notes:
  https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
