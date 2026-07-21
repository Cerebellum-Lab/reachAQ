# Acquisition UI

The acquisition application is the reachAQ operator UI for camera acquisition,
hardware status, pellet delivery, NI-DAQ port mapping, laser controls, behavior,
and inference.

## Launch

Use the reachAQ entry point from the configured conda environment:

```bash
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

The `--no-capture-output` option is required for live terminal logging when the
application is launched through Conda.

The GUI starts idle by default, so camera and DAQ configuration remain editable
until Start is selected. Use `--start-mode acquiring` only when immediate
startup is intentional. The saved live-inference setting can be overridden for
one run with `--live-inference` or `--no-live-inference`.

For software-only camera testing:

```bash
conda run --no-capture-output -n reachaq python -m reachAQ.app \
  --random-cameras \
  --no-live-inference \
  -c "$HOME/Autotrainer/system_configuration.yaml"
```

The random-camera override is in-memory for that run and does not overwrite the
configured physical camera serials when the app closes.

## Startup and Live Inference

Selecting Running first shows `Starting acquisition...` and temporarily
disables camera and DAQ editing. If live inference is enabled, its NVIDIA driver
and TensorFlow GPU preflight runs before cameras, CAN, laser, or NI-DAQ hardware
is started. A failed preflight leaves the application idle and reports that no
capture hardware was started.

The **Live inference** switch in Preferences controls the saved setting. The
`--live-inference` and `--no-live-inference` command-line flags override that
setting for one run without changing it; other configuration edits made during
the run can still be saved. Live inference intentionally refuses CPU fallback.

## Configurations

Configuration files load preset values for cameras, devices, NI-DAQ channels,
laser controls, behavior, inference, and output. On Linux reachAQ rigs, the
default local config is usually:

```text
~/Autotrainer/system_configuration.yaml
```

The app saves configuration back to the preferences configuration directory. For
alternate software-only configs, use a separate preferences file and config
directory so test settings do not overwrite the bench config.

## Cameras

Reach cameras are configured as `CameraConfiguration` entries. The left and
right cameras are the normal two-camera reachAQ setup. Cameras 3-6 appear in
the UI only when explicitly present in the configuration.

reachAQ does not require a webcam. Leave the `web` camera absent or disabled
unless a rig intentionally configures it.

For Spinnaker cameras, put the camera serial or configured Spinnaker identifier
in `host`:

```yaml
- !CameraConfiguration
  id: 0
  name: left
  isEnabled: true
  isRecordEnabled: true
  recordMode: 1
  recordPrebufferDuration: 1.0
  scheme: spinnaker
  host: '24152533'
  port: 0
  path: ''
  params:
    fps: 150
    width: 256
    height: 256
    hbin: 4
    vbin: 4
    exposure: 175
    primary: 'yes'
```

The matching right camera should use its own serial and `primary: 'no'`.

## NI-DAQ Ports

The DAQ port editor discovers devices through NI-DAQmx. The channel-source
selector controls which device contributes new choices; assignments already
selected from other devices remain visible and selected while the source
changes. The editor also prevents duplicate channel assignments across roles.
If the selected source has no channels of a required type and that role has no
retained assignment, the role is disabled instead of allowing an invalid choice.

Discovery runs in a background worker. While it is active, the UI displays
`Discovering NI-DAQ devices...`; the editor opens when discovery completes or
shows the discovery error if no usable device is returned.

The Analysis card contains **Stream** and **Signals** tabs. Signals lists camera
frame, barcode, Tone 1, Tone 2, Tone 3 right, and Tone 3 left inputs. Laser
signals are intentionally excluded from this card. A checkbox becomes
selectable only after that signal has a physical assignment in **Edit → Edit
DAQ Ports** and NI-DAQ is enabled. Stop the stream before changing main Analysis
selections; every checkbox change is immediately saved in
`nidaqStream.channels`. Start Stream and Clear are disabled when NI-DAQ is
disabled, and Start Stream also requires at least one selected Analysis channel.

NI-DAQmx task creation and reads run in an isolated worker process. The Analysis
card stays responsive while the worker starts, and a native driver crash such as
`SIGSEGV` is shown as a stream error instead of terminating reachAQ. A worker
that does not become ready within 10 seconds is stopped and reported as a
startup timeout. Hardware initialization milestones from the worker are relayed
to both the application log and terminal.

On the current PXIe-1073 / PXI-6713 setup, NI-DAQmx reports:

```text
Device: PXI1Slot4
AO: PXI1Slot4/ao0 through PXI1Slot4/ao7
AI: none
DIO: PXI1Slot4/port0/line0 through PXI1Slot4/port0/line7
```

The 6713 can provide analog outputs and digital I/O, but it cannot provide
analog input readback. Add a supported NI analog-input card if laser diode or
command-copy feedback channels are required.

Each Laser Control tab includes an Output Stream area with nested **Stream** and
**Signals** tabs. Stream contains the graph and its independent Start/Pause and
Clear controls; Signals contains only that laser's diode-feedback and
command-copy display options. **Start DAQ Inputs** starts the shared input worker;
the button clearly labels its shared stop action while it is running. These
selections also persist immediately in `nidaqStream.channels`, while remaining
absent from the main Analysis selector and plot. Manual/internal and externally
triggered pulse operations append their command waveform. Selected measured
inputs from the shared NI-DAQ stream are added to the corresponding laser
graph. Calibration always resumes and clears the associated graph, then
displays every returned command, diode, and command-copy ramp point without
applying the Analysis rolling-window trim.

Every stream option and curve uses the same high-contrast color assignment:
the first displayed signal is blue, the second green, followed by orange,
purple, red, teal, magenta, and blue-gray. A compact matching legend appears
below the main Analysis graph and below every laser Output Stream graph.

## Output

Set acquisition output in the persistence section:

```yaml
persistence: !PersistenceConfiguration
  outputLocation: /home/<USER>/Documents/rawdatalocal
```

Create the directory before running acquisition:

```bash
mkdir -p "$HOME/Documents/rawdatalocal"
```

## Reference

### Toolbar

* System Mode - select Idle or Running. During transitions it explicitly shows
  Starting or Stopping acquisition.
* Hardware Refresh - scan camera sources, NI-DAQ devices, CAN adapter, and
  pellet delivery board while idle.
* Notes and Subject - set acquisition notes and select the current animal.
* Training Mode and Protocol - select the active training workflow when the
  protocol UI is enabled.
* Preferences - configure live inference and other application preferences.

The Hardware Status table uses `Enabled`, `Devices`, and `Info` columns. The
Info column preserves the latest Hardware Refresh discovery result and appends
the current binding, stream, or connection state. Discovery results remain
visible even when that hardware category is disabled.

### Menus

* File -> Quit - close the application through its controlled shutdown path.
* Edit -> Edit Camera Settings - enable or disable editable camera fields while
  idle.
* Edit -> Edit DAQ Ports - discover NI-DAQ devices and configure named channel
  roles while idle.
* Tools -> Calibrate Coordinate System / Make 3D calibration - run the available
  calibration workflows.
* View -> Diagnostics / Debug - development-mode panels shown only when the
  application is launched with development options.

### Camera Control

* Video Capture - starts capture only when the camera is enabled.
* Record Mode - `Continuous` records the full duration, `Trigger` records around trigger events.
* Video Recording - writes frames to video files.
* Image Capture - captures still images at a configured interval.

## Startup Diagnostics

Hardware initialization milestones are written to the normal log file and are
always echoed to the launching terminal, even when the ordinary console log
level is Warning. Search for records beginning with:

```text
HARDWARE INIT | START
HARDWARE INIT | READY
HARDWARE INIT | SKIP
HARDWARE INIT | FAILED
```

These records cover the GPU preflight, camera discovery and child processes,
CAN/pellet controller, NI-DAQ discovery and tasks, and laser channels. Each slow
operation records elapsed time so the final emitted `START` line identifies the
initialization step that is still waiting.

See [../../linux-install-instructions.md](../../linux-install-instructions.md)
and [../hardware/reachaq_system_configuration.example.yaml](../hardware/reachaq_system_configuration.example.yaml)
for the current Linux hardware setup and example config.
