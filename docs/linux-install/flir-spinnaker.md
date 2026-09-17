# FLIR Spinnaker camera setup

Use this guide only on a reachAQ host with Teledyne FLIR/Point Grey cameras.
The portable installer does not install vendor SDKs because the SDK, wheel,
architecture, and camera firmware must be compatible.

## Inputs

Set the checkout and environment names used below:

```bash
export REACHAQ_REPO="$HOME/Documents/reachAQ"
export REACHAQ_ENV="reachaq"
```

reachAQ includes Spinnaker Python 3.2.0.62 wheels for CPython 3.8 and 3.10.
Pick the one matching the interpreter of the target environment: the wheels
are ABI-locked, so a cp38 wheel will not import on 3.10 and renaming it does
not help. `vendor/spinnaker/manifest.json` lists what is bundled.

The DeepLabCut 3.x PyTorch engine requires Python 3.10, so the inference
environment needs the cp310 wheel; the cp38 wheels remain for the older
TensorFlow environment and the Jetson image:

```bash
find "$REACHAQ_REPO/vendor/spinnaker/linux" -name 'spinnaker_python-*.whl'
```

## 1. Install the matching system SDK

Download the Linux SDK from Teledyne FLIR. Match its release to the Python
wheel whenever possible; do not mix an arbitrary wheel and system runtime.
Follow the SDK bundle's README/install script. A typical downloaded Debian
package bundle uses:

```bash
cd "$HOME/Downloads/Spinnaker-<VERSION>-Linux"
sudo apt install ./*.deb
sudo ldconfig
sudo reboot
```

The package names vary by SDK version, so this vendor step is intentionally not
part of the portable installer.

## 2. Configure operator permissions

After the SDK creates its camera-access group, add the operator and then log out
and back in (or reboot):

```bash
getent group flirimaging
sudo usermod -a -G flirimaging,plugdev "$USER"
groups
```

If `flirimaging` does not exist, consult the installed SDK's udev instructions
instead of creating an unrelated group manually.

## 3. Install the matching Python wheel

Let the interpreter choose its own wheel, so a 3.8 and a 3.10 environment can
be provisioned with the same command:

```bash
conda run -n "$REACHAQ_ENV" python - <<'PY'
import json, pathlib, platform, subprocess, sys
root = pathlib.Path("$REACHAQ_REPO/vendor/spinnaker")
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
manifest = json.loads((root / "manifest.json").read_text())
match = [a for a in manifest["artifacts"]
         if a["system"] == "linux"
         and a["machine"] == platform.machine()
         and a["python"] == tag]
if not match:
    raise SystemExit(f"no bundled Spinnaker wheel for {tag} on {platform.machine()}")
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       str(root / match[0]["path"])])
PY
```

Or name the wheel directly. x86_64 Ubuntu, CPython 3.10:

```bash
conda run -n "$REACHAQ_ENV" python -m pip install \
  "$REACHAQ_REPO/vendor/spinnaker/linux/x86_64/spinnaker_python-3.2.0.62-cp310-cp310-linux_x86_64.whl"
```

Jetson/aarch64 requires the corresponding aarch64 system SDK and wheel:

```bash
conda run -n "$REACHAQ_ENV" python -m pip install \
  "$REACHAQ_REPO/vendor/spinnaker/linux/aarch64/spinnaker_python-3.2.0.62-cp310-cp310-linux_aarch64.whl"
```

There is no cp310 wheel for Windows here: the 3.2.0.62 Windows bundle is a
separate download from the Linux one, and only its cp38 wheel was vendored.

## 4. Verify the complete camera path

```bash
ldconfig -p | grep -i spinnaker
ls /etc/udev/rules.d/*spinnaker* 2>/dev/null
conda run -n "$REACHAQ_ENV" python -c "import PySpin; print('PySpin import ok')"
conda run -n "$REACHAQ_ENV" python "$REACHAQ_REPO/scripts/list_cameras.py"
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

Upstream SDK: <https://prep.flir.com/products/spinnaker-sdk/>

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
cp "$HOME/Autotrainer/system_configuration.yaml" \
  "$HOME/Autotrainer/system_configuration.before-right-camera-test.yaml"
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
