# Supported platforms

`tools/install/reachaq-linux-install.sh` checks the host against
[`tools/install/supported-platforms.json`](../../tools/install/supported-platforms.json)
before it changes anything. On a host that is not on the list it stops with
what it found and what is supported. `REACHAQ_INSTALL_UNSUPPORTED=1` installs
anyway; that is untested, and the report carries a WARN saying so.

The list is data: supporting another platform means adding it to that file
once the installer has run on it end to end, not editing the installer.

| Item | Supported | Why |
|---|---|---|
| Operating system | Ubuntu 22.04 LTS | The reference rig (christielab10, Ubuntu 22.04.5, kernel 6.8). The Spinnaker SDK package is chosen for this version too. |
| Architecture | x86_64 | The PyTorch CUDA build, the NI-DAQmx and Spinnaker packages used are x86_64. Jetson/aarch64 is not supported. |
| NVIDIA driver | 570.26 or newer | PyTorch 2.7 on CUDA 12.8 needs it. The installer does not install the driver; see the [NVIDIA guide](nvidia-inference.md). |
| NVIDIA GPU | compute capability 7.5 to 12.0 | The kernels the CUDA 12.8 PyTorch build carries: T1000 (7.5), RTX A2000 (8.6), RTX 5060 Ti (12.0) are known to work. Every GPU in the host is checked. |
| FLIR Spinnaker SDK | 3.2 | The release the reference rig runs (3.2.0.57). The Python binding must have the SDK's major.minor. See the [FLIR guide](flir-spinnaker.md). |

Checked by `tools/install/platform_check.py` (operating system from
`/etc/os-release`, `uname -m`, and `nvidia-smi`), and for Spinnaker by
`tools/install/spinnaker_download.py`. Both run with the host's `python3`,
before the reachAQ environment exists.

Not covered by the check, and documented in their own guides: NI-DAQmx and
PXI/MXI hardware, PEAK CAN adapters, and camera models.

## Still open

- The pilot site's Ubuntu version is not known yet. When it is, add it to the
  list after a full installer run on it, together with the matching Spinnaker
  SDK package (`spinnaker-<version>-amd64-pkg-<ubuntu>.tar.gz`).
- The distribution path (how a customer gets reachAQ itself, rather than
  `git clone`) is not decided yet.
