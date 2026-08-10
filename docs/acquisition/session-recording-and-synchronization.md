# Session recording, synchronization, and hardware isolation

This document defines the reachAQ recording contract implemented by the
acquisition application. It is the operator and developer reference for manual
session control, recording readiness, persisted streams, timing metadata,
NI-DAQ synchronization, and degraded hardware operation.

## Implemented change inventory

- Replaced load-cell and automatic recording triggers with explicit Record,
  Stop, and Abort controls.
- Replaced Behavior System-state and day/total count UI with four compact
  session counts.
- Moved Hardware Refresh to the Hardware Status title bar and made it retry only
  failed domains while acquisition is running.
- Separated camera, reach synchronization, inference, CAN/pellet, NI-DAQ, laser,
  session-log, and offline-analysis lifecycle state and failure propagation.
- Preserved independent camera preview while requiring all enabled reach cameras
  to form a valid synchronization topology before Record.
- Made every enabled persistence source part of the recording-readiness and
  writer-finalization contract, independent of plot visibility.
- Replaced block-arrival NI timestamps with a continuous sample-index timeline
  and added camera-edge and tone-confirmation correlation.
- Added portable NI device identity, capability and terminal discovery,
  validated timing-master selection, multi-device route planning, and
  slave-before-master startup.
- Added continuous decoded device/CAN event capture and exact event-buffer
  overrun reporting.
- Added session-scoped NI-DAQ, device, laser, and log files plus a finalized
  source manifest.
- Unified camera, stream, JSON, YAML, and API metadata around one canonical
  session boundary and rejected incomplete finalization.
- Added scoped watchdog/runtime-loss handling and stale-worker-generation
  rejection so old callbacks cannot revive failed or aborted state.
- Preserved live inference computation and frame-queue behavior while making
  its startup and failure state explicit.

## Acquisition and recording are separate

Selecting System Mode **Running** starts enabled camera acquisition and preview,
the configured live-inference pipeline, CAN/pellet communication, the NI-DAQ
input stream, laser control, and session-output validation. It does not create a
trial or write session video frames.

The Behavior panel owns the session controls:

| Control | Available state | Result |
|---|---|---|
| **Record** | System Mode is Running, recording state is Ready, and every required source is Ready | Creates the trial and begins the existing camera/video-writing path |
| **Stop** | Recording | Selects the final synchronized camera boundary, closes all writers, retains the trial, and runs offline analysis |
| **Abort** | Arming or Recording | Closes writers, cancels analysis for that trial, deletes the entire trial directory, and resets session counts |

The recording state progresses through `Ready`, `Arming`, `Recording`,
`Stopping`, and `Analyzing`. Record stays disabled until analysis for a stopped
trial finishes. Analysis is never started for an aborted trial.

The four Behavior counters are session-scoped: Presented, Reaches, Success, and
Consumed. They reset when Record is pressed, remain visible after Stop and
analysis, and return to zero after Abort. The previous System state controls and
day/total pellet counters are not part of the UI or current persistence model.
Legacy animal JSON remains loadable, but new recording behavior does not update
day or lifetime count fields.

The load-cell acquisition, tare, configuration, UI, and automatic recording
triggers have been removed. Older YAML tags and fields are accepted only as
load-and-drop compatibility data. They do not create a load-cell runtime.
reachAQ emergency stop/resume and alarm-driven LED overrides are disabled; the
remaining non-alarm analysis detectors continue to run where configured.

## Required sources and failure domains

Each runtime domain has an independent state, generation, initialization,
cleanup, and retry path. A failure does not stop unrelated healthy hardware.

| Domain | Required for Record when | Important dependency |
|---|---|---|
| Camera | Preview is enabled and the camera is record-enabled, or it participates in a multi-camera reach synchronization topology | Enabled reach secondaries depend on the reach primary trigger |
| Reach synchronization | A record-enabled reach camera exists, or multiple reach cameras are enabled | All enabled reach cameras must form one valid primary-to-secondary topology |
| Live inference and pose | Live inference is configured | Requires ready synchronized reach cameras |
| CAN/pellet device | The configured hardware requires the connection | Independent of cameras and NI-DAQ |
| NI-DAQ stream | NI-DAQ hardware and the input stream are enabled | Its timing plan must be valid for aligned recording |
| Laser | The laser backend is enabled | Feedback inputs may share the NI-DAQ acquisition task |
| Session logs | Always | The session output location must be writable |

With one enabled reach camera, reachAQ temporarily uses a standalone/free-run
role for preview. With multiple enabled reach cameras, exactly one effective
primary drives the trigger-dependent secondaries. A secondary failure leaves a
healthy primary previewing, but blocks reach synchronization, inference, and
Record. A primary failure marks its dependent secondaries Blocked while
unrelated cameras and hardware continue.

The refresh icon is at the far right of the **Hardware Status** title bar. While
idle it performs discovery and refreshes bindings. While System Mode is Running
and the session state is Ready, it retries failed domains without restarting
healthy domains. It is disabled during recording, stopping, abort cleanup, and
analysis. Stale callbacks from an older worker generation cannot overwrite the
state of a newer retry.

If a required source is unavailable, Record is disabled and its tooltip lists
the exact blockers. If a required source is lost during Recording, only the
active trial is aborted; unrelated acquisition domains remain running. Explicit
System Mode stop still attempts to stop every domain even when one cleanup
operation fails.

## Canonical session boundary

Record reuses the existing camera acquisition and video-writing implementation.
The trigger changed; frame acquisition, synchronized-frame selection, the
depth-one live-inference queue, inference computation, and video encoding did
not.

The first frame actually recorded by the effective primary reach camera defines
one immutable `SessionBoundary`:

- primary camera and frame ID;
- `time.perf_counter()` start time;
- wall-clock start time;
- camera-provided time;
- final primary-camera performance and wall times after Stop;
- matched NI-DAQ sample index when a `cam_frames` input is available.

Synchronized reach cameras retain the shared camera frame-index behavior. Pose
data follows recorded frame-index categories. An unsynchronized record-enabled
camera uses its first written frame at or after the canonical start and reports
its boundary offset; it cannot claim the same physical exposure edge without a
hardware trigger.

Non-camera sources run continuously into timestamped rolling buffers. Once the
primary camera commits the boundary, each writer keeps its first datum at or
after the start and its last datum at or before the stop boundary. This avoids
losing early auxiliary samples while the application waits for the camera's
first recorded-frame notification.

The same finite boundary is required in:

- `streams/alignment.json`;
- final JSON and YAML `start_record_timestamp` and `sessionBoundary`;
- stream slicing and camera-timing merge;
- the API/session metadata path.

Final metadata is serialized to temporary JSON and YAML files before either
destination is replaced. A missing alignment file, non-finite boundary,
boundary mismatch, primary timing mismatch, missing enabled source, writer
failure, acquisition gap, or buffer overrun prevents the trial from being
reported as fully saved.

## Persisted session files

Every session-owned file is below the existing `trialNNN` directory. Exact
camera, pose, and metadata names continue to follow the existing project naming
conventions.

```text
trialNNN/
├── <camera videos and camera timing files>
├── <live/offline pose and analysis files>
├── <trial metadata>.json
├── <trial metadata>.yaml
├── streams/
│   ├── nidaq.h5
│   ├── device.csv
│   ├── laser.csv
│   └── alignment.json
└── logs/
    └── session.log
```

Files for disabled sources are not required. A camera with `isEnabled: true`
and `isRecordEnabled: false` previews but does not create session video.

### `nidaq.h5`

The HDF5 file contains all configured acquisition channels, sample indices,
sample-index-derived performance timestamps, wall-time anchoring, channel
definitions, and timing-plan metadata. Display selection never removes a
channel from acquisition or persistence. At 10 kHz, the nominal timestamp
resolution is 0.1 ms.

The primary datasets are `sample_index`, `perf_time`, `offset_seconds`,
`wall_time`, `epoch`, `values`, `block_observation_perf_time`, and
`block_observation_wall_time`. `values` is channel-by-sample and follows the
`channel_names` attribute. File attributes include the recording boundaries,
sample rate, epoch, gap/overrun counts, source clock, and serialized timing
plan. Block-observation times are diagnostics only; `perf_time` is derived from
the sample-index epoch.

The NI-DAQ epoch is anchored once. Later block delivery time does not reconstruct
each block independently, so host scheduling jitter cannot create backward
timestamp corrections. Sample indices must be continuous and timestamps must be
strictly increasing. Gaps and overruns are recorded explicitly.

### `device.csv`

This is a general decoded device/CAN ledger, not a pressure-only measurement
file. It contains timestamped inbound messages, outbound commands, command
acknowledgements, decoded target/kind/payload data, context identifiers, device
timestamps, and device indices. Events buffer continuously before Record and
are sliced at the canonical boundary. Raw byte-for-byte CAN frame capture is not
part of this format.

Pellet-board tone events are stored here. Electrical tone confirmations wired to
NI-DAQ are separate signals in `nidaq.h5`; `alignment.json` correlates the two.

### `laser.csv`

The laser event ledger records commands, output/state changes, source, channel,
command voltage, measured diode/command-copy values when available, and generic
named output values. Physical diode and command-copy inputs are also persisted
continuously in `nidaq.h5` when configured.

### `session.log`

The session log contains only messages within the canonical trial boundary,
with offsets from the start. The acquisition-wide diagnostic log remains
outside the trial and is retained after Abort so hardware failures can still be
diagnosed.

### `alignment.json` and final metadata

`alignment.json` contains:

- `canonicalBoundary`;
- first/last offsets for auxiliary streams;
- the requested and resolved `nidaqTiming` topology;
- `cameraNidaqAlignment`, including matched edge/sample, signed offset,
  resolution, confidence, and ambiguity;
- `toneConfirmation`, including matched and unmatched device events and NI-DAQ
  edges;
- `enabledSources`, including source role/binding, runtime state, actual paths,
  sample/frame count, first/last offsets, gap and overrun counts, failure, and
  `persistenceStatus`;
- `deviceEventOverruns`, `sessionComplete`, and `incompleteReasons`.

Final trial JSON/YAML additionally records `sessionCounts`,
`sessionDataComplete`, `sessionDataErrors`, `hardwareConfigured`,
`hardwareRuntimeAtRecord`, `hardwareRuntime`, recording state, and analysis
duration.

Camera/NI matching uses the nearest `cam_frames` transition within half the
observed frame period and records whether it was rising or falling. The current
camera output is a square wave whose alternating peaks and troughs each identify
one frame. Alignment reports `host_estimated` when the line is not configured,
and `unmatched` when no plausible transition exists. Tone matching pairs
decoded per-line stimulus rises with unclaimed NI-DAQ rises within 250 ms.
Generic `PLAY_TONE` commands remain explicitly unmatched when the decoded
command does not identify a physical confirmation line.

## NI-DAQ acquisition and synchronization

`nidaqPorts` assigns physical inputs to semantic roles: camera frames, barcode,
and tone lines. Configured laser diode and command-copy feedback inputs are added
to the same acquisition plan. Existing `nidaqStream.channels` not claimed by a
named role remain custom acquired inputs. `displayChannels` controls only which
curves are plotted.

The DAQ Ports dialog discovers device identity and capabilities, including
model, serial, bus/chassis, AI/AO/DI/DO channels, counters, terminals, maximum
rates, timed AO support, and digital-trigger support. It rejects unavailable or
duplicate channels, incompatible sample rates, invalid timing terminals, and a
manual master that cannot drive the active topology before starting tasks.
Configuration cannot be edited while acquisition is active.

Saved `deviceIdentities` let a logical device resolve by product and serial when
NI-DAQmx changes its runtime alias. An absent, replaced, ambiguous, or
serial-mismatched device is never silently rebound.

`nidaqPorts.timing.syncMode` has four modes:

| Mode | Behavior |
|---|---|
| `auto` | Uses one device when possible, otherwise a discovered common PXI backplane, otherwise requires explicit external routes |
| `backplane` | Requires active devices on a discoverable common PXI chassis |
| `external` | Requires explicit start-trigger and sample-clock terminal sources |
| `independent` | Uses independent clocks; diagnostic-only when hardware synchronization is required |

The optional timing master is a device/task role, not a channel flag. Automatic
selection prefers the device owning the canonical sampled inputs, including
`cam_frames` and analog inputs. With one sampled device, no slave topology is
created. With multiple synchronized devices, slave tasks are armed before the
master and the resolved topology is written to session metadata.

For compatible PXI cards, the plan uses `PXI_CLK10` as the shared reference, a
shared backplane start trigger, and a routed master sample clock. A future
hardware-timed laser AO task is supported as a synchronized slave: common
reference when available, shared start trigger, sample-clock route when
required, and slave-before-master task ordering. The present laser output mode
remains on-demand unless `laser.hardwareTimed` is explicitly enabled and the
discovered hardware supports the requested topology.

No card model, PXI slot, alias, or route is a universal default. The current rig
uses a PXI-6221 input device as the natural acquisition master and a PXI-6713
for laser analog output, but every rig must resolve and validate its own devices
and routes through discovery.

## Operator verification

1. Start with the PXI chassis powered and select System Mode Running.
2. Confirm all enabled domains are Ready and both reach previews are live.
3. Record for 30–60 seconds while generating several tone, laser, and CAN events.
   Plot only one NI-DAQ signal during this test.
4. Stop and wait for analysis. Confirm every enabled source exists in the source
   manifest, all configured NI-DAQ channels are present, and the session is
   complete.
5. Confirm reach cameras share the committed frame boundaries; NI sample indices
   are continuous; NI timestamps are strictly increasing; and camera/NI and
   tone/NI correlations are present and unambiguous where the corresponding
   physical lines are wired.
6. Confirm JSON, YAML, and `alignment.json` contain the same finite start
   boundary. Repeat three times and compare camera/NI offset stability.
7. Start a trial and Abort. Confirm the trial directory is removed, counts are
   zero, analysis does not run, and healthy previews remain active.
8. Stop System Mode, power off only the PXI chassis, and start again. Cameras
   should report their own results and continue preview if their physical
   trigger path is available; NI-DAQ/laser should fail independently and block
   Record. Power the chassis on and use the Hardware Status refresh/retry button;
   healthy cameras must not restart.

If cameras do not receive frames only while the chassis is off, inspect and
document the actual trigger, power, and ground wiring. NI-DAQ software
initialization does not itself trigger a camera. Camera timeout diagnostics
preserve the first PySpin exception, incomplete-image status, configured role,
serial, and effective trigger-node values; a later watchdog timeout must not be
treated as the root cause.

## Software verification

From the repository root:

```bash
conda run -n reachaq python -m pytest -q \
  tests/nidaq_channel_plan_test.py \
  tests/nidaq_sample_timeline_test.py \
  tests/nidaq_timing_test.py \
  tests/nidaq_port_configuration_dialog_test.py \
  tests/session_data_recorder_test.py \
  tests/subsystem_status_test.py \
  tests/test_app_model.py

conda run -n reachaq python -m pytest -q
```

Software tests validate the state machine, persistence and metadata schemas,
synthetic timing topologies, failure isolation, and queue behavior. They do not
replace the physical wiring and chassis acceptance steps above.
