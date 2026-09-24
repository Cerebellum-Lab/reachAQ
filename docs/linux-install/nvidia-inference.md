# NVIDIA GPU and live inference setup

Live inference is optional, but when enabled it requires a CUDA-capable NVIDIA
GPU and the NVIDIA driver. reachAQ intentionally does not fall back to CPU
inference.

The portable installer never changes the NVIDIA kernel driver. Everything
above the driver - both pose engines and their CUDA user-space libraries -
goes into the reachAQ Conda environment, installed and checked by the same
no-argument installer.

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

It installs both engines DeepLabCut 3 can run, into one Python 3.10
environment:

| Engine | Version | CUDA / cuDNN | Runs |
|---|---|---|---|
| PyTorch | 2.7, CUDA 12.8 build | 12.8 / 9 | YOLO models, DeepLabCut PyTorch models |
| TensorFlow | 2.12 | 11.8 / 8.6 | DeepLabCut TensorFlow models |

A YOLO model always runs on PyTorch. A DeepLabCut model runs on the engine
`REACHAQ_POSE_BACKEND` selects (`tensorflow` by default); see the
[runtime guide](runtime-configuration.md#environment-variables).

Three things make the two engines safe together. Each was a failure on a real
rig before it was fixed, and the installer checks each one:

- **Separate CUDA libraries.** Both engines' CUDA libraries arrive as
  `nvidia-*` Python packages that install into the same `site-packages/nvidia`
  directory, so the later install overwrote the earlier: TensorFlow lost
  `libcudnn.so.8` and fell back to the CPU without an error. TensorFlow's CUDA
  11.8 / cuDNN 8.6 now go in `<env>/lib/reachaq-tensorflow-cuda11`, reached
  only through the environment's `LD_LIBRARY_PATH`.
- **A CUDA 12.8 PyTorch.** DeepLabCut 3 imports PyTorch as soon as it is
  imported, so its TensorFlow engine always shares a process with PyTorch.
  With PyTorch's CUDA 13 build that aborted TensorFlow ("stack smashing
  detected"); with the CUDA 12.8 build both run. The installer takes PyTorch
  from the PyTorch cu128 index for that reason.
- **Newer numpy than TensorFlow 2.12 asks for.** TensorFlow 2.12 caps numpy at
  1.24.3 and typing-extensions below 4.6, but the plotting, HDF5 and
  augmentation libraries need newer. TensorFlow 2.12 runs GPU convolutions and
  DeepLabCut models with numpy 1.26, so the installer keeps 1.26, and its
  dependency check allows exactly those two TensorFlow caps.

The PyTorch build covers compute capability 7.5 through 12.0: T1000, RTX A2000
and RTX 5060 Ti.

## 4. Verify

The installer report's `Pose engine GPU runtimes` category is the acceptance
check: TensorFlow on the GPU, PyTorch on the GPU, and both in one process. To
repeat the reachAQ preflight by hand:

```bash
conda run -n reachaq python - <<'PY'
from autotrainer.inference import detect_gpu_runtime

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
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  -c "$HOME/Autotrainer/system_configuration.yaml"

# Disable inference for one run
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"

# Require inference for one run
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  --live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

The command-line overrides are not persisted. Use the **Live inference** switch
in Preferences to change the saved setting.

## Troubleshooting

- `nvidia-smi` fails: fix the kernel driver first; the user-space installer
  cannot repair `nouveau`, a missing module, or absent `/dev/nvidia*` devices.
- TensorFlow lists no GPU (`[]`) while PyTorch works: its CUDA 11.8 libraries
  are not on the library path. Rerun the installer and confirm the saved path
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
