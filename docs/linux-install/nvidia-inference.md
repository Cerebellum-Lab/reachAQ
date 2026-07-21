# NVIDIA GPU and live inference setup

Live inference is optional, but when enabled it requires a CUDA-capable NVIDIA
GPU, the NVIDIA CUDA driver stack, and a TensorFlow runtime that reports a GPU.
reachAQ intentionally does not fall back to CPU inference.

The portable installer never changes the NVIDIA kernel driver. On supported
x86_64 TensorFlow versions, its opt-in GPU step installs matching CUDA/cuDNN
user-space libraries into the reachAQ Conda environment.

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

## 3. Install the compatible TensorFlow runtime

The upstream [TensorFlow tested build configurations][tensorflow-builds]
define these combinations:

| TensorFlow | Python | CUDA | cuDNN | reachAQ automated install |
|---|---|---|---|---|
| 2.12.x | 3.8-3.11 | 11.8 | 8.6 | Supported |
| 2.13.x | 3.8-3.11 | 11.8 | 8.6 | Supported |
| 2.14.x | 3.9-3.11 | 11.8 | 8.7 | Not supported by the current Python 3.8 environment |

The default reachAQ environment currently resolves TensorFlow 2.13.x. Once
`nvidia-smi` succeeds, run the no-argument installer. It installs CUDA 11.8 and
cuDNN 8.6 into that environment and performs the GPU preflight:

```bash
./tools/install/reachaq-linux-install.sh
```

The installer uses NVIDIA's versioned `nvidia-*-cu11` Python packages. They can
coexist with the `nvidia-*-cu12` packages required by PyTorch, and the installer
persists the library search path only in the selected Conda environment. It
refuses unrecognized TensorFlow versions instead of guessing a CUDA/cuDNN pair.
The [TensorFlow pip guide][tensorflow-pip] documents the equivalent GPU package
and verification approach for current TensorFlow releases.

## 4. Verify TensorFlow in the reachAQ environment

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
longer the primary problem. Rerun the no-argument installer, then check its
`TensorFlow GPU runtime` report.

## 5. Run the reachAQ preflight directly

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
- TensorFlow reports `Could not find cuda drivers` or `Cannot dlopen some GPU
  libraries`: rerun the installer GPU option and confirm its saved
  `LD_LIBRARY_PATH` with `conda env config vars list -n reachaq`.
- The installer rejects the TensorFlow version: use the upstream compatibility
  table rather than forcing the CUDA 11.8 package set onto another release.
- The `TF-TRT Warning: Could not find TensorRT` message is not a failure for the
  standard TensorFlow CUDA path. The reachAQ preflight and GPU calculation are
  the acceptance checks.

[tensorflow-builds]: https://www.tensorflow.org/install/source#gpu
[tensorflow-pip]: https://www.tensorflow.org/install/pip
[nvidia-driver]: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html

References: [TensorFlow tested build configurations][tensorflow-builds],
[TensorFlow pip installation][tensorflow-pip], and
[NVIDIA Ubuntu driver installation][nvidia-driver].
