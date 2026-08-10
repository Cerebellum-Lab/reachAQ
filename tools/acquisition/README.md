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
startup is intentional. The application window opens maximized by default while
retaining its normal title bar and restore control. After restoring, it remains
freely resizable in both width and height—even below child-panel size hints.
Overflowing toolbar actions remain accessible from the toolbar menu, and the
status-bar corner has a resize grip. The saved live-inference setting can be
overridden for one run with `--live-inference` or `--no-live-inference`.

All visible acquisition panel boundaries are draggable. This includes the
camera panels, the vertical camera/analysis/hardware sections,
Behavior/Analysis, Hardware Control/Status, and the main workspace/right-side
tabs. Split positions are stored in the user preferences file and restored on
the next launch. Splitter handles use deferred resize so high-rate graphs do
not repaint continuously while a handle is being dragged.

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
and TensorFlow GPU preflight is reported as the live-inference domain. A failed
preflight prevents inference from starting but does not prevent independent
camera, CAN, laser, or NI-DAQ initialization. Each domain reports its own Ready,
Blocked, Failed, or Disabled state. Record remains unavailable until every
configured source required for persistence is Ready.

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

The current production schema is system-configuration version 57. Older and
newer versions, unknown fields, retired load-cell/tunnel fields, and obsolete
NI recording controls are rejected instead of being silently migrated. Start
from the maintained example at
[`tools/hardware/reachaq_system_configuration.example.yaml`](../hardware/reachaq_system_configuration.example.yaml)
when converting a rig configuration.

## Cameras

Reach cameras are configured as `CameraConfiguration` entries. The left and
right cameras are the normal two-camera reachAQ setup. A third `stimCam`
(`id: 3`) is added as an optional, disabled camera. While disabled it is not
shown in the normal camera grid or Hardware Status and is not opened or required
at startup. **Edit → Edit Camera Settings** always shows the `stimCam` settings
panel so it can be enabled or disabled there. Leaving camera-edit mode saves the
selection and rebuilds the normal camera grid: enabled creates the third,
far-right panel; disabled removes it. Cameras 4-6 likewise appear only while
enabled.

Enable `stimCam` by setting its `CameraConfiguration.isEnabled` value to
`true`. Its source can then be selected from the far-right camera panel. Keep
it `false` on rigs without the camera; the placeholder random source is never
opened while the camera is disabled.

For Spinnaker cameras, put the camera serial or configured Spinnaker identifier
in `host`. The Edit Camera Settings selector keeps that configured binding under
the logical camera name (`left`, `right`, or `stimCam`); discovered
`Spinnaker <serial>` entries are intentionally excluded from the selector so a
camera cannot be silently rebound to another configured position. Hardware
Status still lists every discovered serial for connection diagnostics.

```yaml
- !CameraConfiguration
  id: 0
  name: left
  isEnabled: true
  isRecordEnabled: true
  recordMode: 1
  # Manual recordings begin at the first frame after Record; prebuffer is disabled.
  recordPrebufferDuration: 0
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

The two-camera Spinnaker path is hardware synchronized: the primary camera
drives a GPIO signal and the secondary waits for that signal before producing
frames. A discovered secondary camera can therefore be connected yet show no
preview when the trigger path is absent or invalid. See
[FLIR Spinnaker camera setup](../../docs/linux-install/flir-spinnaker.md#work-in-progress-right-camera-preview-timeout)
for the current right-camera timeout investigation and isolation procedure.

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
selections; every checkbox change is immediately saved as the
`nidaqStream.displayChannels` plot selection. It never changes the complete
acquisition channel plan. Start Stream and Clear are disabled when NI-DAQ is
disabled, and Start Stream also requires at least one selected Analysis channel.
Streams remain stopped when reachAQ opens. They can be started manually while
idle; configured streams start automatically when acquisition becomes active
and stop again when acquisition stops.

NI-DAQmx task creation and reads run in an isolated worker process. The Analysis
card stays responsive while the worker starts, and a native driver crash such as
`SIGSEGV` is shown as a stream error instead of terminating reachAQ. A worker
that does not become ready within 10 seconds is stopped and reported as a
startup timeout. Hardware initialization milestones from the worker are relayed
to both the application log and terminal.

The Analysis graph is visualization-only, but its checkbox selection does not
control acquisition or persistence. While NI-DAQ is enabled, every mapped and
custom acquisition channel is sampled continuously and saved to
`streams/nidaq.h5` during a recording, even when it is not plotted. Incoming
blocks, fixed-size NumPy circular buffers, and
pixel-aligned peak-envelope reduction run in a dedicated plot-data process. The
process publishes alternating fixed-size `float32` shared-memory buffers; no
plot arrays are serialized through a multiprocessing queue. The GUI timer uses
the refresh rate reported by the screen containing the application, consumes
only the newest completed shared buffer, and performs the final Qt curve draw.
Qt widgets themselves remain in the GUI process. Each curve contains exactly
one minimum/maximum pair per physical graph pixel so narrow TTL activity remains
visible without repainting every raw sample.
Curves are always solid lines; digital signals are
distinguished by color and label rather than dots or dashes. Controls below
each graph set its visible time window in seconds and its minimum/maximum
vertical range in volts.

The default 10 kHz hardware sample rate provides ten samples across each half
cycle of a 500 Hz square wave. At runtime, the NI read chunk is derived from the
active screen rate (`sample rate / refresh rate`), so one new block is normally
available per display refresh. The configured `readChunkSize` remains a
backward-compatible fallback rather than the live display cadence. Digital-only tasks use an NI counter output as
their sample clock instead of software-timed per-sample reads. A TTL must be
connected to the exact physical channel named in `nidaqStream.channels`.

On the current PXIe-1073 / PXI-6713 setup, NI-DAQmx reports:

```text
Device: PXI1Slot4
AO: PXI1Slot4/ao0 through PXI1Slot4/ao7
AI: none
DIO: PXI1Slot4/port0/line0 through PXI1Slot4/port0/line7
```

The 6713 can provide analog outputs and digital I/O, but it cannot provide
analog input readback. This rig also contains a PXI-6221 (`PXI1Slot5`) with 16
analog inputs, hardware-clocked digital input support, and two counters. Prefer
the 6221 for buffered input streams and verify that `nidaqPorts` and
`nidaqStream.channels` identify the same wired terminals.

Each Laser Control channel uses compact **Pulse**, **Calibration**, and **Output**
tabs so its controls remain usable when the right-side panel is narrow. The
Output area has nested **Stream** and **Signals** tabs. Stream contains the graph
and its independent Start/Stop and Clear controls; Signals contains only that
laser's diode-feedback and command-copy display options. Physical NI-DAQ paths
are shown in signal tooltips instead of widening the panel. Plots and controls
shrink with the panel; use the main splitter to give them more room when desired.
**Start DAQ Inputs** starts the shared input worker;
the button clearly labels its shared stop action while it is running. These
selections also persist immediately in `nidaqStream.displayChannels`, while remaining
absent from the main Analysis selector and plot. Manual/internal and externally
triggered pulse operations append their command waveform. Selected measured
inputs from the shared NI-DAQ stream are added to the corresponding laser
graph. Calibration explicitly starts and clears the associated graph. Each live
stream graph has its own editable time window and voltage limits.

Every stream option and curve uses the same high-contrast color assignment:
the first displayed signal is blue, the second green, followed by orange,
purple, red, teal, magenta, and blue-gray. A compact matching legend appears
below the main Analysis graph and below every laser Output Stream graph.
Every legend swatch and plotted curve is solid.

All Analysis and laser graphs take their physical pixel width and height from
the containing panel. Dragging a horizontal or vertical splitter therefore
grows or shrinks the graph itself; no graph keeps a fixed pixel height or forces
its panel wider. This is independent of the editable seconds and voltage ranges.

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
* Notes and Subject - set acquisition notes and select the current animal.
* Protocol - select a protocol, or select **Manual pellet control**. Automatic
  pellet cycles and automatic protocol advance are independent settings.
* Preferences - configure live inference and other application preferences.

Hardware Status uses one collapsible subpanel per category. Subpanels start
collapsed and show only the category plus `Enabled` or `Disabled`; green,
amber, red, blue-gray, and gray headers indicate ready, warning, failure, idle,
and disabled states. Expand a category to see its scan and binding details. A
scan runs once when the application opens; Hardware Refresh repeats it on
demand. A successful scan leaves the pellet controller connected so
acquisition can reuse the initialized session. The result is a stable scan
snapshot and is not cleared or rewritten when acquisition starts. `CAN Adapter`
reports the physical PCIe device,
kernel driver, and Linux CAN interfaces separately from the `Pellet Controller`
application session. Pellet Controller is green when its session is connected
and red when initialization fails or its connection is lost. Compact vertical
detail lines show each PXI slot and card
model/product number, the configured CAN backend/interface, and the detected
GPU model, memory, and driver.

The Hardware Refresh icon is at the far right of the Hardware Status title bar.
While idle it repeats discovery and refreshes bindings. While System Mode is
Running and no recording or analysis is active, it retries failed subsystems
without restarting healthy domains.

### Menus

* File -> Quit - close the application through its controlled shutdown path.
* Edit -> Edit Camera Settings - enable or disable editable camera fields while
  idle.
* Edit -> Edit DAQ Ports - discover NI-DAQ devices and configure named channel
  roles while idle.
* Tools -> Calibrate Coordinate System / Make 3D calibration - run the available
  calibration workflows.
* View -> Logging - show or hide the application log. Errors are reported here,
  in the status bar, in the launching terminal, and in the log file instead of
  being rendered inside individual control panels.
* View -> Debug - development-mode panel shown only when the application is
  launched with development options.

### Camera Control

* Video Capture - starts capture only when the camera is enabled.
* Video Recording - includes the camera in manually started sessions.
* Image Capture - captures still images at a configured interval.

Normal acquisition always uses triggered recording. Selecting System Mode
`Running` starts acquisition and preview without writing session video. Use
`Record` in the Behavior panel to initialize a session and begin writing,
`Stop` to retain it and run analysis, or `Abort` to discard the entire session.
Record remains unavailable until analysis for the stopped session finishes.
Presented, Reaches, Success, and Consumed are session-only counts: they reset on
Record, remain visible after Stop, and reset to zero after Abort. The old System
state display, day/total counters, load-cell UI, and load-cell recording triggers
have been removed.

Session-aligned auxiliary files are stored beneath the matching `sessionNNN`
directory:

```text
streams/nidaq.h5
streams/device.csv
streams/laser.csv
streams/trials.jsonl
streams/trial_summary.json
streams/alignment.json
logs/session.log
```

`alignment.json` records the primary-camera start/end boundary and the first
and last saved timestamps and offsets for every auxiliary stream. It also
records NI-DAQ topology, camera/NI edge matching, device-tone/NI confirmation,
enabled-source paths and counts, gaps, overruns, failures, and session
completeness. Final JSON/YAML metadata must contain the same finite canonical
camera boundary before a session is reported as fully saved.

See [Session recording, synchronization, and hardware isolation](../../docs/acquisition/session-recording-and-synchronization.md)
for the complete state, persistence, timing, metadata, failure, and verification
contract.

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
CAN adapter/controller, NI-DAQ discovery and tasks, and laser channels. Each slow
operation records elapsed time so the final emitted `START` line identifies the
initialization step that is still waiting.

Hardware Status categories are collapsed by default. Expanded categories use
independently scrollable, fixed-width tables with aligned columns and a bold,
underlined header. Camera serials, NI card models and named port/stream routes,
CAN driver/interface selection, GPU driver/memory, and laser channel/timing
bindings are shown per device; only actionable scan warnings are added below
those rows. Camera rows are limited to configured reach and optional stimCam
roles; synthetic and playback test sources are omitted from hardware discovery.

Some NI devices, including M-Series static digital I/O, return DAQmx status
`-200303` because their digital lines have no internal hardware sample clock.
The analysis stream automatically falls back to software-timed reads for a
digital-only task and logs a warning. This mode is appropriate for operator
visualization, but its timing is approximate and it can miss pulses shorter than
the polling interval; use a routed hardware sample clock when edge-complete
recording is required.

See [../../linux-install-instructions.md](../../linux-install-instructions.md)
and [../hardware/reachaq_system_configuration.example.yaml](../hardware/reachaq_system_configuration.example.yaml)
for the current Linux hardware setup and example config.

See [Recording sessions, pellet trials, protocols, and schema migration](../../docs/acquisition/session-trials-protocols.md)
for trial numbering, retries, automatic stop behavior, protocol progress, and
animal v4-to-v5 migration.
