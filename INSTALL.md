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
- PyTorch for the built-in keypoint models, verified on the GPU, and the pose
  plugins the environment already had (see below);
- the Spinnaker camera binding, when the Spinnaker SDK is installed;
- closed-loop latency tuning, the `dialout` group for the RFID reader, and
  the desktop icon plus the `reachaq` and `reachaq-sync` commands;
- a verification pass and the focused non-hardware tests.

It takes no options. Every step is attempted, failures do not stop later
ones, and a report at the end lists each step as PASS, FAIL or SKIP.

No pose plugin is installed by default. Add one to the existing environment at
any time, with the same kind of report:

```bash
./tools/install/reachaq-plugin-install.sh deeplabcut   # DeepLabCut, with its TensorFlow engine
./tools/install/reachaq-plugin-install.sh yolo         # YOLO-pose; installs AGPL-3.0 ultralytics
```

The DeepLabCut plugin also installs TensorFlow's CUDA runtime and checks
TensorFlow on the GPU and both engines in one process.

`REACHAQ_POSE_BACKEND` (`tensorflow` or `torch`) applies to DeepLabCut models only; a
keypoint or YOLO model always runs on PyTorch. To write a plugin for another
engine, see [plugins/README.md](plugins/README.md).

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
  the steps that need root, for a rig that already has them;
- the pose plugins the environment had are reinstalled from the checkout, into
  the rebuilt environment too. An environment from before the plugins existed
  gets the DeepLabCut plugin if it has `deeplabcut` installed and the YOLO
  plugin if it has `ultralytics`. To drop a plugin for good, uninstall both its
  distribution and its engine package (for example
  `pip uninstall reachaq-pose-yolo ultralytics`) before rerunning the
  installer, otherwise it restores the plugin. If the installer cannot tell which plugins
  the environment has, it reports a FAIL for that step rather than restoring none.

To run an archived environment instead:

```bash
conda run --no-capture-output -n reachaq-py38-20260924 python -m reachaq.app
```

## Upgrading from v3

reachAQ 4.0 renamed the project. The eight `auto-trainer*` distributions are
now one, `reachaq`; the code moved from `autotrainer.*` and `tools.*` to
`reachaq.*` (`python -m reachaq.app`); `AUTOTRAINER_*` environment variables
are `REACHAQ_*`; and the settings moved. There are no compatibility aliases:
v3 names stop working.

Pull the new checkout with `reachaq-sync`, then run the installer once:

```bash
./tools/install/reachaq-linux-install.sh
```

Before anything else it runs `reachaq-migrate-settings`, which

- renames `~/Autotrainer` to `~/reachAQ` (one filesystem, so nothing is copied);
- rewrites the old directory in the live yaml files there (DeepLabCut project
  `config.yaml` files, `demo/demo_sources.yaml`, and any other `*.yaml` that names
  it), keeping each original as `FILE.bak-reachaq-migration`. Backups, other
  file types and the calibration library are not changed;
- copies `~/.config/Colorado/Auto Trainer.conf` to
  `~/.config/reachAQ/reachAQ.conf`, and the pellet tool's
  `PelletDeliveryUI.conf` beside it. The originals stay where they are.
- rewrites `~/.config/reachaq/launcher.conf` the same way (kept as
  `launcher.conf.bak-reachaq-migration`), so the launcher still finds the
  settings if the install stops before it regenerates that file.

The installer then uninstalls every `auto-trainer*` distribution (including
`auto-trainer-api` and `auto-trainer-training` where a rig still has them and
the v3 `auto-trainer-pose-*` plugins), installs `reachaq`, reinstalls the pose
plugins the environment had, and replaces installed SoftMouse user units that
still start a v3 module (the old unit is kept as `.bak`). Recorded sessions are
not changed.

To see what the migration would do first, or to run it by hand:

```bash
python3 src/reachaq/core/settings_migration.py --dry-run
```

`reachaq` and `reachaq-headless` refuse to start while only `~/Autotrainer`
exists, naming the command. `REACHAQ_CONFIG_DIR`, if set, names the settings
directory outright, and then nothing is migrated or refused.

**Pose files from before 2025-06-25.** Their column labels are pickled as
`autotrainer.inference.pose_elements.SceneElement`, which no longer exists, so
reachaq cannot read them. Convert them once before reprocessing:

```bash
reachaq-convert-pose-h5 --check /path/to/sessions        # lists them; exit 1 if any need converting
reachaq-convert-pose-h5 --in-place /path/to/sessions     # converts, keeping FILE.bak
reachaq-convert-pose-h5 --output-dir OUT /path/to/sessions  # or writes converted copies
```

The converter stores the labels as plain strings, the form reachAQ has written
since 2025-06-25, so converted files still load in v3. Delete the `.bak` files
once a converted session reprocesses. Sessions recorded since then need nothing.

**Rolling back.** Undo the settings move, then check out the v3 release and run
its installer:

```bash
reachaq-migrate-settings --reverse
```

It restores `~/Autotrainer` and every rewritten file byte for byte and removes
the copied QSettings files; a copy changed since the migration is kept and
named in a warning (v3 reads the original in `~/.config/Colorado`).

## Other ways to run it

Keep `--no-capture-output` in long-running `conda run` commands; without it,
Conda buffers the logs until the application exits.

A software-only smoke test, with random cameras and no live inference:

```bash
conda run --no-capture-output -n reachaq python -m reachaq.app \
  --random-cameras --no-live-inference \
  -c "$HOME/reachAQ/system_configuration.yaml"
```

Headless mode starts acquisition immediately, because it has no control that
could start an idle process later:

```bash
conda run --no-capture-output -n reachaq reachaq-headless \
  --no-live-inference \
  -c "$HOME/reachAQ/system_configuration.yaml"
```

Omit `--no-live-inference` to honour the saved configuration, or use
`--live-inference` to enable it for one run. Live inference requires the GPU
and deliberately does not fall back to the CPU.

## After installing

- SoftMouse credentials and its optional nightly timer cannot be entered by an
  unattended installer; follow the
  [SoftMouse/RFID checklist](src/reachaq/softmouse_sync/CONFIGURATION_GUIDE.md).
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
