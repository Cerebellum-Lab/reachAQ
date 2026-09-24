# Installation

reachAQ installs on Ubuntu (x86_64) with one command. Vendor drivers - NVIDIA,
FLIR Spinnaker, NI-DAQmx, PEAK CAN - are separate; the
[Linux installation map](linux-install-instructions.md) says which apply to a
given rig and links each guide.

## Install

```bash
cd "$HOME/Documents/reachAQ"
./tools/install/reachaq-linux-install.sh
```

That builds everything into a Python 3.10 Conda environment named `reachaq`:

- the application, its Ubuntu and Qt packages, and Git LFS assets;
- both pose engines DeepLabCut 3 runs, PyTorch and TensorFlow, and the YOLO
  runtime, each verified on the GPU and together in one process;
- the Spinnaker camera binding, when the Spinnaker SDK is installed;
- closed-loop latency tuning, the `dialout` group for the RFID reader, and
  the desktop icon plus the `reachaq` and `reachaq-sync` commands;
- a verification pass and the focused non-hardware tests.

It takes no options. Every step is attempted, failures do not stop later
ones, and a report at the end lists each step as PASS, FAIL or SKIP.

Install the NVIDIA driver first, so the GPU checks can pass; see the
[NVIDIA guide](docs/linux-install/nvidia-inference.md). If the installer adds
you to `dialout` or the real-time group, log out and back in once and run it
again, so the checks that depend on those groups run in the new session.

## Start

```bash
reachaq
```

Or double-click the reachAQ icon. Either starts in Idle; cameras, NI-DAQ, CAN
and live inference start when you choose Running. Only one reachAQ runs at a
time per user.

## Update

```bash
reachaq-sync
```

That fast-forwards the checkout. If it says packaging files changed, run the
installer again. Rerunning is always safe:

- an environment already on Python 3.10 is updated in place;
- one on another Python is kept, renamed `reachaq-py<version>-<date>`, and a
  fresh `reachaq` is built beside it;
- `REACHAQ_INSTALL_SYSTEM=0 ./tools/install/reachaq-linux-install.sh` skips
  the steps that need root, for a rig that already has them.

To run an archived environment instead:

```bash
conda run --no-capture-output -n reachaq-py38-20260924 python -m reachAQ.app
```

## Other ways to run it

Keep `--no-capture-output` in long-running `conda run` commands; without it,
Conda buffers the logs until the application exits.

A software-only smoke test, with random cameras and no live inference:

```bash
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  --random-cameras --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

Headless mode starts acquisition immediately, because it has no control that
could start an idle process later:

```bash
conda run --no-capture-output -n reachaq auto-trainer-headless \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

Omit `--no-live-inference` to honour the saved configuration, or use
`--live-inference` to enable it for one run. Live inference requires the GPU
and deliberately does not fall back to the CPU.

## After installing

- SoftMouse credentials and its optional nightly timer cannot be entered by an
  unattended installer; follow the
  [SoftMouse/RFID checklist](tools/softmouse_sync/CONFIGURATION_GUIDE.md).
- The latency tuning grants the stim loop real-time priority and sets the
  `performance` CPU governor at boot; both are measured, reversible, and
  described in the [latency tuning guide](docs/linux-install/latency-tuning.md).
- Launcher paths live in `~/.config/reachaq/launcher.conf`; the
  [runtime guide](docs/linux-install/runtime-configuration.md) covers it, the
  terminal commands, and the environment variables reachAQ reads.

## Legacy notes

Older Autotrainer deployments used environment names such as
`auto-trainer-1`, Jetson-specific pins, and entry points like
`python -m tools.acquisition.gui`. They may still help on legacy rigs, but are
not the current reachAQ Ubuntu setup.
