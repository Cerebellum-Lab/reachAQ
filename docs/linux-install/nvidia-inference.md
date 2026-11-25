# NVIDIA GPU and live inference setup

Live inference is optional, but when enabled it requires a CUDA-capable NVIDIA
GPU and the NVIDIA driver. reachAQ intentionally does not fall back to CPU
inference.

The portable installer never changes the NVIDIA kernel driver. Everything
above the driver - the pose engines and their CUDA user-space libraries -
goes into the reachAQ Conda environment: PyTorch through the no-argument
installer, TensorFlow through the DeepLabCut plugin's installer.

## 1. Install the Ubuntu-recommended NVIDIA driver

```bash
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

Do not hard-code a driver package copied from another host. Ubuntu may recommend
an NVIDIA open-kernel package on supported GPUs; that is distinct from the
non-CUDA `nouveau` driver. reachAQ's PyTorch is a CUDA 12.8 build, so use a
current driver; christielab10 runs 595.84 (CUDA 13.2).

## 2. Verify the kernel and userspace driver

```bash
lspci -nnk | grep -A3 -i nvidia
nvidia-smi
ls -l /dev/nvidia* 2>/dev/null
```

Required result: the active kernel driver is `nvidia`, `nvidia-smi` lists the
GPU, and NVIDIA device nodes exist.

## 3. Install the pose engines

Once `nvidia-smi` succeeds, run the installer:

```bash
./tools/install/reachaq-linux-install.sh
```

It installs PyTorch. TensorFlow, the other engine DeepLabCut 3 can run, comes
with the optional DeepLabCut plugin, installed into the same Python 3.10
environment with:

```bash
./tools/install/reachaq-plugin-install.sh deeplabcut
```

| Engine | Version | CUDA / cuDNN | Runs |
|---|---|---|---|
| PyTorch | 2.7, CUDA 12.8 build | 12.8 / 9 | Keypoint models (the live default), DeepLabCut PyTorch models |
| TensorFlow | 2.12 | 11.8 / 8.6 | DeepLabCut TensorFlow models |

PyTorch comes with the base install, and the built-in keypoint backend runs on
it. No optional pose plugin is installed by default, and TensorFlow arrives
only with the DeepLabCut plugin. A keypoint model always runs on PyTorch,
and so does a YOLO model. A DeepLabCut model runs on the engine
`REACHAQ_POSE_BACKEND` selects, or, unset, on TensorFlow where it is installed
and PyTorch otherwise; the variable applies to DeepLabCut models only. See the
[runtime guide](runtime-configuration.md#environment-variables). Plugin
authors: [plugins/README.md](../../plugins/README.md).

### Keypoint models

A keypoint model is a directory holding `weights.pt` and `keypoint_pose.yaml`.
Point `poseModelLocation` at the directory; the sidecar is how the backend is
recognised. Operators label, train and deploy one inside the app with
**Tools → Pose Training…** ([guide](../acquisition/pose-training.md)). From a
terminal, train one from an exported dataset with:

```bash
python -m reachaq.inference.keypoint.train DATASET OUTPUT --bodyparts NAMES
```

`DATASET` holds `images/{train,val}/*.png` and `labels/{train,val}/*.txt` (one
line per frame: class, box, then `x y v` per part as fractions of the frame,
`v` 0 for unlabelled). `NAMES` lists the part names in label order, one per
line, or is a YAML file with a `bodyparts` key. The 623-frame reach dataset
trains in about 3.5 minutes (300 epochs) on an RTX 5060 Ti.

### YOLO (Ultralytics) models

reachAQ no longer installs `ultralytics`: it is AGPL-3.0. A YOLO model
directory (`yolo_pose.yaml`) is refused at start-up with a message saying so.
Retrain it as a keypoint model, or install the optional AGPL plugin
(`./tools/install/reachaq-plugin-install.sh yolo`, distribution
`reachaq-pose-yolo`) to keep
running it.

The PyTorch build covers compute capability 7.5 through 12.0: T1000, RTX A2000
and RTX 5060 Ti.

### One CUDA runtime

The base install carries one CUDA user-space runtime: PyTorch's CUDA 12.8, as
`nvidia-*-cu12` wheels in `site-packages/nvidia`. The DeepLabCut plugin adds
the only other one allowed, TensorFlow's CUDA 11.8 and cuDNN 8.6, in
`<env>/lib/reachaq-tensorflow-cuda11` (a `README-reachaq.txt` there says so),
reached only through the environment's `LD_LIBRARY_PATH`. The installer's
`Verify one CUDA runtime per engine` step (`tools/install/cuda_runtime_check.py`)
fails on anything else: `nvidia-*` wheels of another CUDA major in
site-packages, `LD_LIBRARY_PATH` pointing into PyTorch's wheels, or the
TensorFlow runtime left behind without the plugin. Its message names what to
remove.

To remove the DeepLabCut plugin and its runtime:

```bash
conda run -n reachaq python -m pip uninstall -y reachaq-pose-deeplabcut deeplabcut tensorflow
rm -rf "$(conda run -n reachaq python -c 'import sys; print(sys.prefix)')/lib/reachaq-tensorflow-cuda11"
conda env config vars list -n reachaq   # LD_LIBRARY_PATH: keep only your own entries
conda env config vars unset -n reachaq LD_LIBRARY_PATH   # if it held only TensorFlow's
```

### Keep these if you change the environment by hand

With the DeepLabCut plugin, the plugin installer sets the environment up so
the two engines can share it. Rerun it rather than upgrading pose packages with
`pip`; if you must, keep
these:

- **TensorFlow's CUDA libraries stay in their own directory,**
  `<env>/lib/reachaq-tensorflow-cuda11`, reached through the environment's
  `LD_LIBRARY_PATH`. Do not `pip install` the `nvidia-*-cu11` packages into the
  environment itself: they share `site-packages/nvidia` with PyTorch's, and one
  set overwrites the other, leaving TensorFlow on the CPU without an error.
- **PyTorch stays a CUDA 12.8 build (2.7.x).** DeepLabCut imports PyTorch as
  soon as it is imported, so DeepLabCut TensorFlow models always run with
  PyTorch in the same process. The CUDA 13 builds make TensorFlow abort there
  ("stack smashing detected").
- **numpy stays at 1.26.** TensorFlow 2.12 declares numpy 1.24.3 or older and
  typing-extensions below 4.6; reachAQ's plotting, HDF5 and augmentation
  libraries need newer, and TensorFlow 2.12 works with them. `pip check`
  therefore reports those two TensorFlow requirements; the installer expects
  exactly those and nothing else.

## 4. Verify

The installer report's `Verify PyTorch GPU preflight` step is the acceptance
check for PyTorch; with the DeepLabCut plugin, the plugin installer's report
adds TensorFlow on the GPU and both engines in one process. To
repeat the reachAQ preflight by hand:

```bash
conda run -n reachaq python - <<'PY'
from reachaq.inference import detect_gpu_runtime

print(detect_gpu_runtime(required_backend="torch"))
print(detect_gpu_runtime(required_backend="tensorflow"))
PY
```

When live inference is enabled, the preflight for the engine the configured
model uses runs before cameras, CAN, laser, or NI-DAQ acquisition hardware
starts. A failure leaves the GUI idle.

## Launch choices

```bash
# Honor the saved inference setting
conda run --no-capture-output -n reachaq python -m reachaq.app \
  -c "$HOME/reachAQ/system_configuration.yaml"

# Disable inference for one run
conda run --no-capture-output -n reachaq python -m reachaq.app \
  --no-live-inference \
  -c "$HOME/reachAQ/system_configuration.yaml"

# Require inference for one run
conda run --no-capture-output -n reachaq python -m reachaq.app \
  --live-inference \
  -c "$HOME/reachAQ/system_configuration.yaml"
```

The command-line overrides are not persisted. Use the **Live inference** switch
in Preferences to change the saved setting.

## Troubleshooting

- `nvidia-smi` fails: fix the kernel driver first; the user-space installer
  cannot repair `nouveau`, a missing module, or absent `/dev/nvidia*` devices.
- TensorFlow lists no GPU (`[]`) while PyTorch works: its CUDA 11.8 libraries
  are not on the library path. Rerun
  `./tools/install/reachaq-plugin-install.sh deeplabcut` and confirm the saved path
  with `conda env config vars list -n reachaq`; it should name
  `reachaq-tensorflow-cuda11`. A plain `python` from the environment's `bin`
  directory, without `conda run` or `conda activate`, does not get that path.
- A DeepLabCut TensorFlow model aborts the process on load: PyTorch is not the
  CUDA 12.8 build. Rerun the installer.
- The `TF-TRT Warning: Could not find TensorRT` message is not a failure for the
  standard TensorFlow CUDA path. The reachAQ preflight and GPU checks are the
  acceptance tests.

[tensorflow-builds]: https://www.tensorflow.org/install/source#gpu
[pytorch-install]: https://pytorch.org/get-started/locally/
[nvidia-driver]: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html

References: [TensorFlow tested build configurations][tensorflow-builds],
[PyTorch installation][pytorch-install], and
[NVIDIA Ubuntu driver installation][nvidia-driver].
