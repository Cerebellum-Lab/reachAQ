# FLIR Spinnaker camera setup

reachAQ does not ship FLIR software. On a host with Teledyne FLIR/Point Grey
cameras, the customer downloads the Spinnaker SDK for Linux from Teledyne
FLIR, and `tools/install/reachaq-linux-install.sh` installs the SDK packages
and the Spinnaker Python binding (PySpin) from that download, configures
camera access and verifies the cameras. Supported releases are listed in
[supported platforms](supported-platforms.md): Spinnaker 3.2, whose binding
matches the reachAQ environment's Python 3.10.

On a host without FLIR cameras, run the installer with
`REACHAQ_INSTALL_SPINNAKER=0`; the Spinnaker steps are then reported as SKIP.

## 1. Download

1. Sign in (or create a free account) at
   <https://www.teledynevisionsolutions.com/products/spinnaker-sdk/>.
2. Under *Spinnaker SDK All Versions for Download*, take the Linux Ubuntu
   release of Spinnaker 3.2 (the reference rig runs 3.2.0.57). It is one zip,
   `Spinnaker-3.2.<build>-Linux.zip`, holding the SDK for each Ubuntu version
   and the Python bindings.
3. Save it in `~/Downloads`, or anywhere and set
   `REACHAQ_SPINNAKER_DOWNLOADS=<that folder>` for the installer.

The installer accepts the zip, the folder it unpacks to, or just the two files
it needs from it:

| File | Used for |
|---|---|
| `spinnaker-<version>-amd64-pkg-22.04.tar.gz` | the SDK `.deb` packages for Ubuntu 22.04 |
| `spinnaker_python-<version>-cp310-cp310-linux_x86_64.tar.gz` (or the `.whl` inside it) | PySpin for Python 3.10 |

Other files in the zip (other Ubuntu versions, ARM builds, Python 3.8) are
ignored.

## 2. Run the installer

```bash
./tools/install/reachaq-linux-install.sh
```

Its Spinnaker steps:

| Step | What it does |
|---|---|
| Install Spinnaker SDK | Unless a supported SDK is already installed, unpacks the SDK packages from the download and installs them with `apt-get`. The FLIR packages show the **FLIR Spinnaker SDK licence agreement** in the terminal; reading and accepting it is the operator's decision, and declining stops the install. The `libspinnaker` package also reports the install to FLIR (Google Analytics); that is FLIR's packaging, not reachAQ's. |
| Configure FLIR camera access | Adds the operator to `flirimaging` and `plugdev`, writes `/etc/udev/rules.d/40-flir-spinnaker.rules` and reloads udev, and enables `reachaq-usbfs-memory.service`, which sets the USB-FS buffer to 1000 MB at boot (the kernel's 16 MB starves USB3 cameras). This is what FLIR's interactive `install_spinnaker.sh` offers, without its prompts; its GenTL and example `PATH` settings are not needed and are left out. |
| Install Spinnaker Python binding | Installs PySpin from the download into the `reachaq` environment, preferring the build of the installed SDK, else the newest of the same major.minor. A binding of the SDK's major.minor that is already installed is kept, so reruns need no download. |
| Verify Spinnaker cameras | Imports PySpin, prints the SDK version it loaded and the serials of the attached cameras. |

If it stops at one of these steps, the output above the report says what is
missing. The usual causes:

- no download where it looked: it names the folder searched and what to
  download;
- an installed SDK that is not on the supported list: remove it with the
  `remove_spinnaker.sh` from that release's own download, then rerun;
- the licence was declined, or the installer was not run in a terminal, so the
  licence could not be shown: rerun it from a terminal;
- `REACHAQ_INSTALL_SYSTEM=0` with no SDK installed: installing it needs root.

After the operator is added to `flirimaging`, log out and back in (or reboot)
before using the cameras, then rerun the installer so the camera check runs in
the new session.

## 3. Verify the complete camera path by hand

```bash
dpkg-query -W libspinnaker
cat /etc/udev/rules.d/40-flir-spinnaker.rules
cat /sys/module/usbcore/parameters/usbfs_memory_mb
id -nG | tr ' ' '\n' | grep -x flirimaging
conda run -n reachaq python -c "import PySpin; print('PySpin import ok')"
conda run -n reachaq python "$HOME/Documents/reachAQ/scripts/list_cameras.py"
```

Expected result: each connected FLIR camera appears with a
`spinnaker://<SERIAL>` URL.

## Troubleshooting

| Symptom | Check |
|---|---|
| `import PySpin` fails | System SDK and Python wheel versions/architectures |
| Import succeeds but no cameras appear | Power, USB/GigE cabling, udev rules, and `flirimaging` membership |
| Permission denied | Log out/in after group changes; re-run the SDK udev setup |
| Camera opens with unexpected behavior | Camera firmware and SDK compatibility; configured serial and URL properties |
| One synchronized camera never produces a frame | Primary/secondary roles and the physical GPIO trigger path |

Upstream SDK: <https://www.teledynevisionsolutions.com/products/spinnaker-sdk/>

## Work in progress: right-camera preview timeout

Status: **open investigation as of 2026-07-23**. No camera code or
configuration workaround has been applied yet.

The current bench failure has this signature:

```text
detected watchdog camera.right timed out: 5.6
Error during capture loop: Failed capture a frame in time
```

This is not initially a preview-widget failure. The right camera process calls
Spinnaker repeatedly for an image, but receives no complete image for 15
seconds. Because the capture loop is blocked during that interval, its
5-second watchdog expires first. With no captured frame in the shared image
queue, the right preview has nothing to display.

### Evidence from the 2026-07-23 runs

- Both configured cameras are discovered as Blackfly S BFS-U3-16S2M USB3
  cameras at 5 Gbit/s.
- The configured left camera, serial `24152533`, starts as primary, reports its
  first frame, and finishes at approximately 150 FPS.
- The configured right camera, serial `24152513`, starts as secondary but
  never reports a first frame. The same result occurs in logs `008`, `009`,
  and `010`.
- A secondary Spinnaker camera is configured by reachAQ to wait for a hardware
  frame trigger on `Line3`. The primary drives its synchronization signal from
  `Line1`. Therefore, discovering and opening the right camera is not enough:
  a valid electrical trigger path must also be present.
- The kernel reset the left USB device once during initialization and the
  right device twice. The application also logged an intentional
  begin/end-acquisition reinitialization for the right camera. Since the left
  reset and then captured normally, a reset message alone does not prove a
  USB fault; the extra right-camera reset remains relevant if the independent
  camera test below also fails.
- CAN initialization occurs after camera capture starts. Pellet-board CAN
  traffic, homing, and a later RGB command all succeeded while the camera
  fault persisted. The available evidence does not identify CAN as the cause.

### First isolation test

Close reachAQ completely, then back up the active configuration:

```bash
cp "$HOME/reachAQ/system_configuration.yaml" \
  "$HOME/reachAQ/system_configuration.before-right-camera-test.yaml"
```

In **Edit Camera Settings**, temporarily:

1. Disable `left`.
2. Keep `right` enabled.
3. Set `right` as the primary camera.
4. Start acquisition and check for a right-camera preview and a
   `captured first frame` entry.
5. Restore the original configuration after the test.

Interpret the result as follows:

| Result | Most likely fault area | Next action |
|---|---|---|
| Right works alone as primary | Camera and USB image transport are functional; the synchronization trigger is missing or invalid | Inspect the primary-to-secondary GPIO trigger cable, connector orientation, common reference, required pull-up, and Line1/Line3 electrical levels |
| Right still produces no frame | Right camera, USB cable/port/power, camera state, firmware, or Spinnaker runtime | Test the right camera alone in SpinView with trigger mode off, then try a known-good USB3 cable/port |
| Right works alone but fails only when both cameras run | Trigger path or shared USB controller/power issue | Verify the trigger electrically, then test the cameras on separate USB host controllers or with appropriate external camera power |

Do not increase the watchdog timeout as a fix. It would delay the fatal report
but would not make the missing camera frames appear.

### Trigger-path checks

The reachAQ configuration must contain exactly one enabled primary reach
camera. For the normal bench configuration, `left` is primary and `right` is
secondary:

```yaml
# left
params:
  primary: true

# right
params:
  primary: false
```

The BFS-U3-16S2 GPIO output used as `Line1` is an opto-isolated, open-drain
output and requires the correct external electrical circuit; the secondary
`Line3` input and its ground/reference must be connected as required by the
camera manual. Do not assume that a continuity-only cable is sufficient.
Confirm the actual trigger waveform at the secondary input with an
oscilloscope while the primary is capturing.

Model-specific references:

- [BFS-U3-16S2 input/output control](https://softwareservices.flir.com/BFS-U3-16S2/latest/40-Installation/InputOutputControl.htm)
- [BFS-U3-16S2 power requirements](https://softwareservices.flir.com/BFS-U3-16S2/latest/40-Installation/Power.htm)

### USB and power checks

Run these commands immediately after reproducing the problem:

```bash
lsusb
lsusb -t
journalctl -k -b --since "-5 min" --no-pager |
  grep -Ei 'usb|xhci|reset|disconnect|over-current|bandwidth'
```

For each FLIR device, confirm `5000M` operation rather than a USB2 fallback.
Use short, secured USB3 cables and test the failing camera with a known-good
cable. The two current cameras share one xHCI controller. If independent
free-running tests pass but two-camera capture remains unreliable, move one
camera to a port backed by a different USB host controller. FLIR also notes
that insufficient USB power or long cables can cause intermittent operation;
use supported external GPIO power if the host cannot provide stable power.

Before using SpinView, close reachAQ and confirm that no stale acquisition
process still owns a camera:

```bash
pgrep -af 'reachAQ|run_acquisition|PySpin'
```

Test one camera at a time in SpinView with trigger mode disabled. A successful
free-running right-camera test is strong evidence that the unresolved fault is
the reachAQ synchronization wiring or trigger configuration, not the sensor or
preview UI.

### Follow-up software diagnostics

If the hardware isolation tests do not resolve the issue, the next software
change should preserve the first `PySpin.SpinnakerException` code and message
instead of replacing 15 seconds of retries with only
`Failed capture a frame in time`. It should also report the camera serial,
primary/secondary role, frame count, and trigger settings in the fatal error.
That instrumentation is diagnostic work and is not yet implemented.
