# reachAQ


* [Overview](#overview)
* [Installation instructions](INSTALL.md)
* [reachAQ Linux install guide](linux-install-instructions.md)
* [Applications](#applications)
  * [Acquisition](#applications)
  * [Pellet Delivery Test](#applications)
* [Scripts](#scripts)
* [Additional Tools](#additional-tools)
* [Testing](#testing)
* [Modules](#modules)
  * [reachaq.core](#reachaqcore)
  * [reachaq.video](#reachaqvideo)
  * [reachaq.device](#reachaqdevice)
  * [reachaq.inference](#reachaqinference)
  * [reachaq.behavior](#reachaqbehavior)
  * [reachaq.model](#reachaqmodel)
  * [reachaq.pyside](#reachaqpyside)
* [Code Guidelines](#code-guidelines)

## Overview
This repository contains reachAQ, installed as one distribution, `reachaq`, from `pyproject.toml`. Its
packages live under `src/reachaq/` (`reachaq.core`, `.video`, `.device`, `.inference`, `.behavior`, `.model`,
`.pyside`, the acquisition application `reachaq.app`, and the hardware and analysis tools beside it). The pose
backends that bring heavy or AGPL-licensed engines are separate plugin distributions under
[`plugins/`](plugins/README.md). Keep the lower-level packages loosely coupled, as listed under [Modules](#modules),
when managing dependencies or refactoring common code. reachAQ grew out of upstream Autotrainer; see
[CHANGELOG.md](CHANGELOG.md) for the 4.0 rename.

For the current reachAQ Dell/Ubuntu runtime, use
[linux-install-instructions.md](linux-install-instructions.md). The expected
environment name is `reachaq`, the GUI entry point is `python -m reachaq.app`,
and local acquisition data should be configured under
`$HOME/Documents/rawdatalocal`.

Install, start and update in three commands - see [INSTALL.md](INSTALL.md):

```bash
./tools/install/reachaq-linux-install.sh   # install, or repair; Python 3.10, PyTorch
reachaq                                    # start
reachaq-sync                               # update the checkout
```

Portable host setup is automated by
[`tools/install/reachaq-linux-install.sh`](tools/install/reachaq-linux-install.sh).
Vendor drivers and rig-specific hardware configuration remain in separate
guides selected from the Linux installation map. The portable installer also
installs the terminal-visible mouse-icon desktop launcher, verifies the
SoftMouse HTTPS/spreadsheet/keyring dependencies and RFID serial dependency and
permissions, and installs the tracked publisher systemd units. Setting up each
user's animal metadata source (SoftMouse, a spreadsheet, a database or a shared
publication) and enabling an optional nightly publisher remain explicit
post-install steps in the
[animal metadata and RFID checklist](src/reachaq/softmouse_sync/CONFIGURATION_GUIDE.md).

## Applications

Applications currently use PySide6 for the user interface.  To the extent possible, this UI layer is isolated
from the core logic of the applications for two reasons:
* The core acquisition application must be able to run in a headless mode, i.e., not simply "hiding" the application window.
* PySide6 may not be a suitable UI framework in the future.  Replacement should be as straightforward as possible.

**Current Applications**

* Acquisition Application
  * The local user interface for integrated camera, pellet delivery, NI-DAQ,
    laser, and pose-inference modules
  * `conda run --no-capture-output -n reachaq python -m reachaq.app -c ~/reachAQ/system_configuration.yaml`
    * The GUI starts idle by default. Use `--start-mode running` only when immediate startup is intentional.
    * Use `--no-live-inference` or `--live-inference` to override the saved inference setting for one run.
    * Hardware startup progress is written to the log and launching terminal as `HARDWARE INIT` records.
    * The installed `reachAQ.desktop` mouse icon opens the same application in a terminal; `reachaq` starts it from any shell.
    * Only one instance runs per user, however it is started: a second start exits with status 3 and names the running PID.
    * [Detailed Instructions](src/reachaq/app/README.md)
    * [Session recording, synchronization, persistence, and hardware isolation](docs/acquisition/session-recording-and-synchronization.md)
    * [Recording sessions, pellet trials, protocols, and schema migration](docs/acquisition/session-trials-protocols.md)
    * [Training plans](docs/acquisition/training-plans.md)
    * [Local API: events and commands over TCP](docs/acquisition/local-api.md)
    * [Session validation command and rule coverage](docs/acquisition/session-validation.md)
    * [Pellet firmware compatibility and rollout](docs/acquisition/pellet-firmware-compatibility.md)
    * [Operator alerts, first-run setup and the diagnostic bundle](docs/acquisition/operator-alerts-and-diagnostics.md)
    * Use `--random-cameras` to start with software-generated frames when no physical cameras are configured.
    * Use `--simulate` to run with no hardware at all: [simulation mode](docs/acquisition/simulation-mode.md) and its [backend interfaces](docs/acquisition/backend-interfaces.md).
  * Headless implementation for command line only
    * `reachaq-headless -c ~/reachAQ/system_configuration.yaml`
    * Headless mode continues to start acquisition by default because it has no UI start control.
* Pellet Delivery Test Application
  * Standalone UI for interfacing with the pellet delivery system
  * `python -m reachaq.pellet_delivery.gui`

## Scripts

The `scripts` directory contains focused diagnostics and offline processing
utilities. Run them from the repository root in the configured environment.

* `scripts/acquire_image.py` - capture and display one frame from a camera URL.
* `scripts/capture.py` - preview a camera for a selected frame count and
  optionally record images/video.
* `scripts/list_cameras.py` - list random, Spinnaker, and playback camera URL
  forms visible to the video layer.
* `scripts/can_console.py` - interactive command interface for supported CAN
  hardware.
* `scripts/can_measure_counts.py` - decoded message-rate diagnostic using the
  same environment-selected CAN transport as reachAQ.
* `plugins/deeplabcut/scripts/load_dlc_model.py` - load a DeepLabCut model and
  print its body-part metadata. Needs the DeepLabCut plugin installed.
* `plugins/deeplabcut/scripts/run_dlc_model.py` - run a DeepLabCut model against
  paired recorded videos. Needs the DeepLabCut plugin installed.

## Additional Tools

Current hardware bring-up tools live under `src/reachaq/hardware`:

* `src/reachaq/hardware/validate_can_hardware.py` - discover the pellet CAN board,
  request status/configuration, or perform explicitly enabled motion tests.
  It exits 1, naming the code, when the board refuses a command (any ACK
  error other than 0 and -11). Exit 0 means only that no refusal arrived
  within the listen window (`--listen-seconds`, default 2.0 s); a slow
  command such as a home can outlast it.
* `src/reachaq/hardware/validate_laser_hardware.py` - validate configured NI-DAQ laser
  channels one controlled operation at a time.
* `src/reachaq/hardware/verify_nidaq_wiring.py` - drive each board output and laser
  command in turn and record which NI-DAQ line followed; the same test as
  **Tools → DAQ Monitor → Wiring test**. See
  [Check the wiring](docs/linux-install/ni-daq-pxi.md#check-the-wiring).
* `src/reachaq/hardware/qualify_pellet_firmware.py` - collect the rig evidence needed
  to qualify a new pellet firmware version. See
  [pellet firmware compatibility](docs/acquisition/pellet-firmware-compatibility.md#qualifying-a-new-firmware-version).
* `src/reachaq/hardware/reachaq-bring-up-can.sh` and the accompanying systemd files -
  configure and bring up the selected reachAQ SocketCAN interface (`can0` by
  default) automatically at boot.
* `src/reachaq/hardware/reachaq-bring-down-can.sh` - service stop action that brings
  down only the configured application CAN channel.
* `src/reachaq/hardware/reachaq-reset-can.sh` and
  `src/reachaq/hardware/reachaq-can-reset.sudoers` - root-owned, narrowly permitted
  recovery helper used only after a confirmed CAN-domain failure. Ordinary
  application close only closes its own socket. The helper serializes and
  debounces service restart, refuses to reset a channel owned by another
  reachAQ process, and is not a physical emergency stop or board power cycle.
* [`docs/acquisition/animal-metadata-sources.md`](docs/acquisition/animal-metadata-sources.md) -
  set up where animal metadata comes from (SoftMouse, spreadsheet, database,
  shared publication, or a lab's own source), credentials, refresh, offline
  behaviour and the optional nightly publisher.
* [`src/reachaq/softmouse_sync/README.md`](src/reachaq/softmouse_sync/README.md) - the
  SoftMouse source, USB RFID readers and metadata ownership.

See [linux-install-instructions.md](linux-install-instructions.md) for the
installation map and [the PEAK SocketCAN guide](docs/linux-install/peak-socketcan.md)
for service installation, permissions, termination, commissioning, updates,
and current invocation examples.


## Testing

PyTest testing is supported and configured via `./conftest.py`.

Each package's tests are in `tests/<package>` (`tests/core`, `tests/device`, ...); the tests of the applications
and of anything that combines several packages are directly in `tests/`.

Tests that are longer or require additional configuration are marked as `@pytest.mark.functional` and are not run
by default.

Tests that require the physical pellet board are marked as `@pytest.mark.canbus` and are not run by default.

The portable installer already includes the test dependencies. To add them to
another environment, from the repository root:

```bash
conda run -n reachaq python -m pip install -e '.[test]'
```

You also need git LFS installed & enabled in your clone repo:

1. install with: `sudo apt-get install git-lfs  # or yum or brew eventually`
2. enable in current clone repo with: `git lfs install --local`
3. fetch the repository's binary test assets with: `git lfs pull`

Now, to run *all* default tests from the root directory:

`pytest` (or `pytest -v` for more verbose output of the logs)

Run all tests, including functional, from the root directory:

`pytest --functional`

To limit testing to an individual namespace package, issue still from the base repo main root directory:

`pytest ./tests/device ./tests/video`

for instance.

To list all test cases, from repo base/root dir:

`pytest --collect-only --quiet`

Then you can use any of the outputted lines (1 per test case) as arg to pytest to execute that specific test case.
You can also give many at once.

## Modules

### reachaq.core

[Core](src/reachaq/core/README.md) is base module for functions and objects that used across most or all modules and applications.

**Depends on**
* None

### reachaq.video
[Video](src/reachaq/video/README.md) implements the camera interfaces for all supported cameras.

**Depends on**
* Core


### reachaq.device

[Device](src/reachaq/device/README.md) implements the shared interfaces for
pellet-board, NI-DAQ, laser, and related non-camera hardware. The primary
purpose is to provide a consistent interface to applications.


**Depends on**
* Core

### reachaq.inference

[Inference](src/reachaq/inference/README.md) implements pose inference and any other low-level machine learning elements.
Its primary purpose is to provide a implementation-agnostic interface to inference, such as the current dependency
on DeepLapCut.

**Depends on**
* Core

### reachaq.behavior
[Behavior](src/reachaq/behavior/README.md) implements the behavior training state machine, algorithm, and real-time analysis.

**Depends on**
* Core

### reachaq.pyside
[PySide](src/reachaq/pyside/README.md) provides convenience classes on top of PySide.

**Depends on**
* Inference

### reachaq.model

[Model](src/reachaq/model/README.md) is a collection of models and providers that simplify common elements of
reachAQ applications.  Generally, common code that bridges across multiple reachAQ packages is contained
here, rather than creating additional hard-coupling between the lower-level modules.

**Depends on**
* Core
* Device

## Code Guidelines

_Note that the original code came from a different structure is not yet fully consistent.  The following
guidelines are in place for future additions and changes to help with and improve consistency._

* Style generally follows PEP8.  This is the default in most editors or lint tools.
  * An exception is made for `reachaq.pyside`.  Classes derived from PySide follow PySide conventions.
* `reachaq` is a namespace package (there is no `src/reachaq/__init__.py`), so the pose plugins add
  `reachaq.pose_plugins.*` from their own distributions.
* Code, particularly in modules, should be as platform-agnostic as possible despite having a current reachAQ target of Ubuntu 22.04 on x86_64.
  * Fallback support does not need to match the target platform behavior where is can not (e.g. CUDA), but allow the code to run as correctly as possible.
  * This is primarily for automated testing in other environments such as GitHub Actions.
  * Secondarily, it allows for development off-hardware when not available or not practical.
* Modules should provide a well-defined interface to the rest of the system and not expose implementation details unless absolutely necessary.
  * There are multiple scripts and applications that use the functionality in the modules.
  * Exposure to implementation details has a cascading effect of requiring more frequent updates to consumers that generally don't care. 
* Public interfaces to modules generally define a Protocol [1] for objects that fall into certain categories
  * Objects that have multiple implementations, such as the different hardware implementations
  * Objects that are likely to be mocked in automated testing.
    * Particularly needed for environments where hardware, inference models, or other unique elements are not present.  One environment is GitHub Actions that run automated testing for Pull Requests.
* `pip` and `pyproject.toml` are currently used.
  * `pyproject.toml` in modules should be kept up to date if possible.
* Versions in `pyproject.toml` generally need team-wide notification to update.
* Docstrings should be in the "Google" style.
  * A lot of existing docstrings are in "reStructuredText" (Sphinx) style, which may be confusing.
  * Documentation generation can be assumed to be using `mkdocs`.
    * Note: there is no place configured to privately publish the documentation at this time, so modules have not been initialized with a `mkdocs` project yet.  This will change.


[1] This is primarily to enhance static type checking and code analysis in general.  Protocols were chosen
over Python ABCs or other options to allow as much flexibility as possible or needed in the implementation.
