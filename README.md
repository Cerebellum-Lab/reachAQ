# Autotrainer / reachAQ


* [Overview](#overview)
* [Installation instructions](INSTALL.md)
* [reachAQ Linux install guide](linux-install-instructions.md)
* [Applications](#applications)
  * [Acquisition](#acquisition-application)
  * [Tunnel Test](#tunnel-test-application)
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

## Applications

Applications currently use PySide6 for the user interface.  To the extent possible, this UI layer is isolated
from the core logic of the applications for two reasons:
* The core acquisition application must be able to run in a headless mode, i.e., not simply "hiding" the application window.
* PySide6 may not be a suitable UI framework in the future.  Replacement should be as straightforward as possible.

**Current Applications**

* Acquisition Application
  * The local user interface for integrated camera, head fix, pellet delivery, and pose inference modules
  * `python -m reachAQ.app --start-mode idle -c ~/Autotrainer/system_configuration.yaml`
    * [Detailed Instructions](tools/acquisition/README.md)
    * Use `--random-cameras` to start with software-generated frames when no physical cameras are configured.
  * Headless implementation for command line only
    * `auto-trainer-headless --start-mode idle -c ~/Autotrainer/system_configuration.yaml`
* Tunnel Test Application
  * Standalone UI for interfacing with the tunnel hardware components
  * `python -m tools.head_fix.gui`
* Pellet Delivery Test Application
  * Standalone UI for interfacing with the pellet delivery system
  * `python -m tools.pellet_delivery.gui`

## Scripts

Most of the content in `scripts` are lightweight utilities to determine if various components of the system are working as expected.

* acquire_image.py
  * Captures a single image frame from the camera specified by the `cameraurl`.
* can_console.py
  * A command line interface to the Alogus hardware.
* capture_camera.py
  * Captures 150 frames from the camera specified by the `cameraurl` argument to the location specified by `output`.
* head_fix_console.py
  * A command line interface to the Anshutz tunnel unit.  Will log data stream to a csv file.  Supports subset of device commands.
* list_cameras.py
  * List all cameras available in the system.
* load_dlc_model.py
  * Validates loading of a DLC model with the network module
* pellet_delivery_console.py
  * A command line interface to the Anshutz pellet delivery unit.  Supports a subset of device commands.
* run_dlc_model.py
  * Sends two saved files through a DLC model with the network module.

## Additional Tools

* auto-trainer-device\tools\head_fix_server.py
  * Mock server for the Anshutz tunnel unit for testing without the physical device
  * `python auto-trainer-device\tools\head_fix_server.py`
    * specify the serial port, `/dev/ttyACM1`, `COM4`, etc...
    * specify measurement update frequency `-f 100` for 100 Hz
    * specify random data vs. fixed `-r`
    * specify firmware version to report `-v 3.0`
  * whether set to random or fixed data, use the `s`, `d`, `a`, `t`, or `h` commands followed by the value to change those measurement values
* auto-trainer-device\tools\pellet_server.py
  * Mock server for the Anshutz pellet delivery unit for testing without the physical device
  * `python auto-trainer-device\tools\pellet_server.py`
    * specify the serial port, `/dev/ttyACM1`, `COM4`, etc...
    * specify firmware version to report `-v 3


## Testing

PyTest testing is supported and configured via `./conftest.py`.

Individual namespace packages (*e.g,* `auto-trainer-core`) contain a `tests` directory.  There are also unit and functional
tests for high-level functionality in the applications and that combine elements of multiple packages.

Tests that are longer or require additional configuration are marked as `@pytest.mark.functional` and are not run
by default.

Tests that require the Alogus hardware are marked as `@pytest.mark.canbus` and are not run by default.

PyTest is not installed with via the default installation. To enable testing use

`pip install -e .[test]`

You also need git LFS installed & enabled in your clone repo:

1. install with: `sudo apt-get install git-lfs  # or yum or brew eventually`
2. enable in current clone repo with: `git lfs --install`
3. `git lfs pull` and `git checkout` or similar command might be needed to fetch the current binary files.

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

[Device](auto-trainer-device/README.md) implements the hardware interfaces for most non-camera hardware for both existing
and legacy hardware.  The primary purpose is to provide a consistent interface to the hardware for applications.


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
