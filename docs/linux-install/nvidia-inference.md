# NVIDIA GPU and live inference setup

Live inference is optional, but when enabled it requires a CUDA-capable NVIDIA
GPU, the NVIDIA CUDA driver stack, and a TensorFlow runtime that reports a GPU.
reachAQ intentionally does not fall back to CPU inference.

The portable installer excludes GPU setup because driver and CUDA/cuDNN
selection depends on the GPU, OS, kernel, and installed TensorFlow version.

## 1. Install the Ubuntu-recommended NVIDIA driver

```bash
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

Do not hard-code a driver package copied from another host. Ubuntu may recommend
an NVIDIA open-kernel package on supported GPUs; that is distinct from the
non-CUDA `nouveau` driver.

## 2. Verify the kernel and userspace driver

```bash
lspci -nnk | grep -A3 -i nvidia
nvidia-smi
ls -l /dev/nvidia* 2>/dev/null
```

Required result: the active kernel driver is `nvidia`, `nvidia-smi` lists the
GPU, and NVIDIA device nodes exist.

## 3. Verify TensorFlow in the reachAQ environment

```bash
conda run -n reachaq python - <<'PY'
import tensorflow as tf

print("tensorflow", tf.__version__)
print("gpus", tf.config.list_physical_devices("GPU"))
PY
```

Match CUDA and cuDNN to the reported TensorFlow version using the upstream
[tested build configurations](https://www.tensorflow.org/install/source#gpu).
For example, TensorFlow 2.13 was tested with CUDA 11.8 and cuDNN 8.6.

If `nvidia-smi` succeeds but TensorFlow returns `[]`, the kernel driver is no
longer the primary problem; align the environment's CUDA/cuDNN libraries before
enabling inference.

## 4. Run the reachAQ preflight directly

```bash
conda run -n reachaq python - <<'PY'
from autotrainer.inference import detect_gpu_runtime

print(detect_gpu_runtime(required_backend="tensorflow"))
PY
```

When live inference is enabled, the same preflight runs before cameras, CAN,
laser, or NI-DAQ acquisition hardware starts. A failure leaves the GUI idle.

## Launch choices

```bash
# Honor the saved inference setting
conda run -n reachaq python -m reachAQ.app \
  -c "$HOME/Autotrainer/system_configuration.yaml"

# Disable inference for one run
conda run -n reachaq python -m reachAQ.app \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"

# Require inference for one run
conda run -n reachaq python -m reachAQ.app \
  --live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

The command-line overrides are not persisted. Use the **Live inference** switch
in Preferences to change the saved setting.

## Current workstation observation

On 2026-07-21, the Dell Precision 3660 contained an NVIDIA Quadro T1000, but it
was using `nouveau`; `nvidia-smi` and `/dev/nvidia*` were unavailable, and both
TensorFlow and PyTorch reported no GPU. Keep live inference disabled until the
checks above pass.

Reference: [NVIDIA Ubuntu driver installation](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html).
