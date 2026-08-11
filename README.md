# Autotrainer / reachAQ


* [Overview](#overview)
* [Installation instructions](INSTALL.md)
* [reachAQ Linux install guide](linux-install-instructions.md)
* [Applications](#applications)
  * [Acquisition](#acquisition-application)
  * [Pellet Delivery Test](#pellet-delivery-test-application)
* [Scripts](#scripts)
* [Additional Tools](#additional-tools)
* [Testing](#testing)
* [Modules](#modules)
  * [autotrainer.core](#autotrainercore)
  * [autotrainer.video](#autotrainervideo)
  * [autotrainer.device](#autotrainerdevice)
  * [autotrainer.inference](#autotrainerinference)
  * [autotrainer.behavior](#autotrainerbehavior)
  * [autotrainer.model](#autotrainermodel)
  * [autotrainer.pyside](#autotrainerpyside)
* [Code Guidelines](#code-guidelines)

## Overview
This repository contains the Autotrainer modules and reachAQ applications that are primarily used on-device.  A monorepo
is used here primarily as a development convenience.  Individual modules and applications can and are installed
in on devices independently.  This loose-coupling should be assumed when managing dependencies or refactoring
common code.

For the current reachAQ Dell/Ubuntu runtime, use
[linux-install-instructions.md](linux-install-instructions.md). The expected
environment name is `reachaq`, the GUI entry point is `python -m reachAQ.app`,
and local acquisition data should be configured under
`$HOME/Documents/rawdatalocal`.

Portable host setup is automated by
[`tools/install/reachaq-linux-install.sh`](tools/install/reachaq-linux-install.sh).
Vendor drivers and rig-specific hardware configuration remain in separate
guides selected from the Linux installation map.

## Applications

Applications currently use PySide6 for the user interface.  To the extent possible, this UI layer is isolated
from the core logic of the applications for two reasons:
* The core acquisition application must be able to run in a headless mode, i.e., not simply "hiding" the application window.
* PySide6 may not be a suitable UI framework in the future.  Replacement should be as straightforward as possible.

**Current Applications**

* Acquisition Application
  * The local user interface for integrated camera, pellet delivery, NI-DAQ,
    laser, and pose-inference modules
  * `python -m reachAQ.app -c ~/Autotrainer/system_configuration.yaml`
    * The GUI starts idle by default. Use `--start-mode running` only when immediate startup is intentional.
    * Use `--no-live-inference` or `--live-inference` to override the saved inference setting for one run.
    * Hardware startup progress is written to the log and launching terminal as `HARDWARE INIT` records.
    * [Detailed Instructions](tools/acquisition/README.md)
    * [Session recording, synchronization, persistence, and hardware isolation](docs/acquisition/session-recording-and-synchronization.md)
    * [Recording sessions, pellet trials, protocols, and schema migration](docs/acquisition/session-trials-protocols.md)
    * Use `--random-cameras` to start with software-generated frames when no physical cameras are configured.
  * Headless implementation for command line only
    * `auto-trainer-headless -c ~/Autotrainer/system_configuration.yaml`
    * Headless mode continues to start acquisition by default because it has no UI start control.
* Pellet Delivery Test Application
  * Standalone UI for interfacing with the pellet delivery system
  * `python -m tools.pellet_delivery.gui`

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
* `scripts/load_dlc_model.py` - load a DeepLabCut model and print its body-part
  metadata.
* `scripts/run_dlc_model.py` - run a DeepLabCut model against paired recorded
  videos.

## Additional Tools

Current hardware bring-up tools live under `tools/hardware`:

* `tools/hardware/validate_can_hardware.py` - discover the pellet CAN board,
  request status/configuration, or perform explicitly enabled motion tests.
* `tools/hardware/validate_laser_hardware.py` - validate configured NI-DAQ laser
  channels one controlled operation at a time.
* `tools/hardware/reachaq-bring-up-can.sh` and the accompanying systemd files -
  configure and bring up the selected reachAQ SocketCAN interface (`can0` by
  default) automatically at boot.
* `tools/hardware/reachaq-bring-down-can.sh` - service stop action that brings
  down only the configured application CAN channel.
* `tools/hardware/reachaq-reset-can.sh` and
  `tools/hardware/reachaq-can-reset.sudoers` - root-owned, narrowly permitted
  recovery helper used only after a confirmed CAN-domain failure. Ordinary
  application close only closes its own socket. The helper serializes and
  debounces service restart, refuses to reset a channel owned by another
  reachAQ process, and is not a physical emergency stop or board power cycle.

See [linux-install-instructions.md](linux-install-instructions.md) for the
installation map and [the PEAK SocketCAN guide](docs/linux-install/peak-socketcan.md)
for service installation, permissions, termination, commissioning, updates,
and current invocation examples.


## Testing

PyTest testing is supported and configured via `./conftest.py`.

Individual namespace packages (*e.g,* `auto-trainer-core`) contain a `tests` directory.  There are also unit and functional
tests for high-level functionality in the applications and that combine elements of multiple packages.

Tests that are longer or require additional configuration are marked as `@pytest.mark.functional` and are not run
by default.

Tests that require the physical pellet board are marked as `@pytest.mark.canbus` and are not run by default.

The test dependencies are optional. Install them from the repository root with:

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

`pytest ./auto-trainer-device/tests ./auto-trainer-video/tests`

for instance. Alternatively you can `cd` into the subdir, and execute `pytest ./tests` too.

To list all test cases, from repo base/root dir:

`pytest --collect-only --quiet`

Then you can use any of the outputted lines (1 per test case) as arg to pytest to execute that specific test case.
You can also give many at once.

## Modules

### autotrainer.core

[Core](auto-trainer-core/README.md) is base module for functions and objects that used across most or all modules and applications.

**Autotrainer Dependencies**
* None

### autotrainer.video
[Video](auto-trainer-video/README.md) implements the camera interfaces for all supported cameras.

**Autotrainer Dependencies**
* Core


### autotrainer.device

[Device](auto-trainer-device/README.md) implements the shared interfaces for
pellet-board, NI-DAQ, laser, and related non-camera hardware. The primary
purpose is to provide a consistent interface to applications.


**Autotrainer Dependencies**
* Core

### autotrainer.inference

[Inference](auto-trainer-inference/README.md) implements pose inference and any other low-level machine learning elements.
Its primary purpose is to provide a implementation-agnostic interface to inference, such as the current dependency
on DeepLapCut.

**Autotrainer Dependencies**
* Core

### autotrainer.behavior
[Behavior](auto-trainer-behavior/README.md) implements the behavior training state machine, algorithm, and real-time analysis.

**Autotrainer Dependencies**
* Core

### autotrainer.pyside
[PySide](auto-trainer-pyside/README.md) provides convenience classes on top of PySide.

**Autotrainer Dependencies**
* Inference

### autotrainer.model

[Model](auto-trainer-model/README.md) is a collection of models and providers that simplify common elements of
Autotrainer applications.  Generally, common code that bridges across multiple Autotrainer modules is contained
here, rather than creating additional hard-coupling between the lower-level modules.

**Autotrainer Dependencies**
* Core
* Device

## Code Guidelines

_Note that the original code came from a different structure is not yet fully consistent.  The following
guidelines are in place for future additions and changes to help with and improve consistency._

* Style generally follows PEP8.  This is the default in most editors or lint tools.
  * An exception is made for `autotrainer.pyside`.  Classes derived from PySide follow PySide conventions.
* All modules are defined as namespace packages to allow for separation of modules under the same `autotrainer` namespace.
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
  * There are several dependencies whose version traces back to the specific environment that is currently required on the Jetson.
* Docstrings should be in the "Google" style.
  * A lot of existing docstrings are in "reStructuredText" (Sphinx) style, which may be confusing.
  * Documentation generation can be assumed to be using `mkdocs`.
    * Note: there is no place configured to privately publish the documentation at this time, so modules have not been initialized with a `mkdocs` project yet.  This will change.


[1] This is primarily to enhance static type checking and code analysis in general.  Protocols were chosen
over Python ABCs or other options to allow as much flexibility as possible or needed in the implementation.
