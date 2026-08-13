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
  session-log, and intertrial-analysis lifecycle state and failure propagation.
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
- Added explicit acquisition, recording-session, pellet-cycle, pellet
  automation, presence, shift-recommendation, and coordinate-validation
  controllers while retaining the proven camera writer, inference, pellet
  state-machine, calibration, and transform implementations.

## Acquisition and recording are separate

Selecting System Mode **Running** starts enabled camera acquisition and preview,
the configured live-inference pipeline, CAN/pellet communication, the NI-DAQ
input stream, laser control, and session-output validation. It does not create a
session directory or write session video frames.

The Behavior panel owns the session controls:

| Control | Available state | Result |
|---|---|---|
| **Record** | System Mode is Running, recording state is Ready, and every required source is Ready | Creates a session and begins the existing camera/video-writing path |
| **Stop** | Recording | Selects the final synchronized camera boundary, closes all writers, retains the session, and drains already-closed pellet-trial analyses |
| **Abort** | Arming, Recording, Stopping, or Analyzing | Cancels the intertrial-analysis generation, waits for writers to close, deletes the entire session directory, and resets session counts |

The recording state progresses through `Ready`, `Arming`, `Recording`,
`Stopping`, and `Analyzing`. Record stays disabled until all already-closed
pellet windows and stopped-session finalization finish. Intertrial analysis uses
the existing live inference outputs while Recording; it does not switch the
shared pipeline to offline video inference. Abort invalidates the session
generation, so queued/running results cannot publish after deletion.

The four Behavior counters are session-scoped: Presented, Reaches, Success, and
Consumed. They reset when Record is pressed, remain visible after Stop and
analysis, and return to zero after Abort. The previous System state controls and
day/total pellet counters are not part of the UI or current persistence model.
Animal v4, v5, and v6 JSON is accepted only for one-way migration to v7. Each
migration archives the original bytes. V4 migration preserves identity, pellet
coordinates, limits, and selected protocol, resets non-convertible
recording-based protocol progress, and removes day/lifetime count fields.

Subject and protocol selection are locked from Arming through analysis, while
Notes remains editable. Stop writes the current Notes value; later edits update
the stopped session's JSON/YAML metadata atomically until the next Record clears
the field. The ordered Trial Protocol table, row locking, and placeholder-field
scope are defined in
[Recording sessions, pellet trials, protocols, and schema migration](session-trials-protocols.md).

The load-cell acquisition, tare, configuration, UI, and automatic recording
triggers have been removed. System configuration v57 rejects obsolete schemas
and fields rather than creating a compatibility runtime. Alarm, emergency,
tunnel, head-fix, magnet, and webcam/top-camera behavior is removed. Pellet
presence, pellet misplacement, watchdog liveness, structured errors, and safe
hardware shutdown remain.

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
the exact blockers. Subsystem-specific policy applies if a source is lost
during Recording. NI or secondary-camera loss preserves the remaining streams
and marks the session incomplete. Primary-camera/writer loss requests normal
Stop and preserves partial data. CAN loss fails the in-flight operation, pauses
new pellet cycles, and attempts bounded recovery while other writers continue.
No source-loss path automatically deletes a retained session.

CAN teardown follows the same isolation rule. Ordinary Stop, Close, or
application exit closes only the process-owned device worker and socket; a
generic fatal callback from another domain cannot reset the shared kernel
interface. A confirmed CAN reader/acknowledgement failure marks CAN Failed,
marks an active session's CAN stream incomplete, and runs a bounded recovery
sequence without stopping camera, NI-DAQ, laser, or log acquisition. Recovery
closes the failed socket, reopens the transport,
rediscovers the pellet board, requests firmware, reloads motor/move
configuration, and restarts status streaming before publishing Ready. An
in-flight motor operation is finalized as unknown and is never replayed across
the reconnect boundary.

Explicit System Mode stop still attempts to stop every domain even when one cleanup
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
- final JSON and YAML `boundary`;
- stream slicing and camera-timing merge;
- the API/session metadata path.

Small critical outputs use a session-local `.staging/<generation>/` area,
file/directory `fsync` where supported, and atomic replacement. JSON and YAML
are serialized from the same in-memory object, and metadata, alignment, trial
files, and the manifest share one generation ID. The authoritative metadata
reference is published last. Large video and pose stores are not duplicated in
staging. A missing alignment file, non-finite boundary,
boundary mismatch, primary timing mismatch, missing enabled source, writer
failure, acquisition gap, or buffer overrun prevents the session from being
reported as fully saved.

The recorder retains both its immutable finalization snapshot and the associated
project identity until publication succeeds. A later Record attempt first retries
that exact generation and republishes its authoritative metadata; it cannot start a
new session while the previous one remains incomplete. Successful generations remove
only now-empty staging directories. Failed staged files are never recursively cleaned
and remain available for diagnosis and retry.

### Camera frame and closed-video validation

Camera timing rows are keyed by the vendor's actual integer frame ID, not row
position. Finalization checks uniqueness and strict monotonicity, merges enabled
synchronized cameras onto the primary frame-ID timeline, and explicitly records
missing, duplicate, out-of-order, early, and late IDs. These synchronization
diagnostics never delete a stopped session.

After all enabled writers acknowledge closure, ReachAQ validates camera videos in
parallel. Each `ffprobe` count has a 30-second deadline; a timeout becomes a retained
source failure rather than entering an unbounded decoder fallback. ReachAQ compares
the decoded frame count with both the writer-reported count and timestamp rows. A
count mismatch is a structured warning containing the camera role/serial and
all three counts. The first writer exception and total writer-error count are
preserved. An unreadable or zero-frame video marks the source and session
incomplete, but every file and diagnostic is retained.

Stop also starts a 30-second writer-acknowledgement watchdog. Expiry reports a
persistent lifecycle/data diagnostic but does not delete files, disconnect hardware,
or issue movement. The session remains in Stopping and can still complete when a late
acknowledgement arrives. Abort remains available; deletion waits until the recorder
and camera writers have actually closed.

## Persisted session files

Every session-owned file is below the `sessionNNN` directory. Exact
camera, pose, and metadata names continue to follow the existing project naming
conventions.

```text
sessionNNN/
├── <camera videos and camera timing files>
├── <live/offline pose and analysis files>
├── <session metadata>.json
├── <session metadata>.yaml
├── streams/
│   ├── nidaq.h5
│   ├── device.csv
│   ├── laser.csv
│   ├── tracking/
│   │   └── trial_<id>_attempt_<id>.json
│   ├── trials.jsonl
│   ├── trial_summary.json
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

Session NI data is appended incrementally to HDF5 through a bounded writer; the
live polling path reuses a preallocated scratch buffer rather than retaining an
unbounded list of chunks. Diagnostics preserve the first/last collection
exception, exception count, gaps, overruns, epochs, sample/time range, and
expected session boundary. Recovered gaps and isolated copy errors are warnings.
Zero samples, worker/persistence failure, or an effectively absent stream is
critical when NI is enabled. Start/end coverage shortfall up to five seconds is
a warning; more than five seconds marks the NI source incomplete while
preserving partial samples.

### `device.csv`

This is a general decoded device/CAN ledger, not a pressure-only measurement
file. It contains timestamped inbound messages, outbound commands, command
acknowledgements, decoded target/kind/payload data, context identifiers, device
timestamps, and device indices. Events buffer continuously before Record and
are sliced at the canonical boundary. Raw byte-for-byte CAN frame capture is not
part of this format.

Pellet-board tone events are stored here. Electrical tone confirmations wired to
NI-DAQ are separate signals in `nidaq.h5`; `alignment.json` correlates the two.
An embedded tone step inside a compound pellet sequence is also written as an
outbound `PLAY_TONE` row with the parent operation context. The pellet board's
immediate `TONE_STATUS` report is retained separately from the older, periodic
`STIMULUS_INPUTS` GPIO report. Their raw receipt times are preserved; post-hoc
alignment never overwrites transport timing.

### `trials.jsonl` and `trial_summary.json`

`trials.jsonl` is the authoritative pellet-delivery attempt ledger for the
session. It separates logical trial ID, physical attempt ID, operation/context
ID, send request, successful presentation acknowledgement, terminal outcome,
retry policy, and explicit hardware errors. Hardware, command, transport, and
acknowledgement failures remain visible but never count toward trial or protocol
progress. `trial_summary.json` stores derived counts for the configured trial
count basis. See
[Recording sessions, pellet trials, protocols, and schema migration](session-trials-protocols.md)
for the full accounting contract.

Each `streams/tracking/` JSON preserves one immutable tone-2-to-cycle-completion
live tracking slice, source frame identities, coverage/missing-frame
diagnostics, synchronous pellet-state evidence, and resulting analysis metrics.
It supports stopped-session verification/repair without reopening video or
performing novel inference.

### `laser.csv`

The laser event ledger records commands, output/state changes, source, channel,
command voltage, measured diode/command-copy values when available, and generic
named output values. Physical diode and command-copy inputs are also persisted
continuously in `nidaq.h5` when configured.

### `session.log`

The session log contains only messages within the canonical session boundary,
with offsets from the start. The acquisition-wide diagnostic log remains
outside the session and is retained after Abort so hardware failures can still be
diagnosed. Low-level `can.bus` frame dumps below Warning are excluded from the
session log because the decoded events are already persisted in `device.csv`;
CAN warnings and errors remain in both logs.

### `alignment.json` and final metadata

`alignment.json` contains:

- `canonicalBoundary`;
- first/last offsets for auxiliary streams;
- the requested and resolved `nidaqTiming` topology;
- `cameraNidaqAlignment`, including matched edge/sample, signed offset,
  resolution, confidence, and ambiguity;
- `toneConfirmation`, including canonical NI pulse onsets, grouped outbound
  command/immediate status/periodic GPIO observations, unmatched valid events
  and pulses, and separately classified short-pulse artifacts;
- `enabledSources`, including source role/binding, runtime state, actual paths,
  sample/frame count, first/last offsets, gap and overrun counts, failure, and
  `persistenceStatus`;
- `deviceEventOverruns`, `sessionComplete`, and `incompleteReasons`.

Final session JSON/YAML uses `metadataSchemaVersion: 2`. It records the animal
snapshot, `counts`, recording completion, stop policy/result, compact hardware
state at Record, and only hardware entries that changed by finalization. The
`artifacts` section references the authoritative `alignment.json`,
and `trial_summary.json` by path and SHA-256 instead of embedding those records
again. The complete `system_configuration.yaml` key/value snapshot remains
embedded under `configuration` so each recording is independently reproducible.
`boundary.durationSeconds` is the final camera-bounded duration. Non-finite
values are normalized to JSON/YAML null; finalized metadata is never written
with non-standard `NaN` tokens.

The pinned `auto-trainer-api` 0.11.0 lifecycle supports explicit session and
pellet-attempt events. ReachAQ's status schema contains no alarm, emergency,
tunnel, head-fix, or magnet placeholders. Session events identify the continuous
recording boundary; trial events identify pellet attempts, never recordings.
Session metadata, analysis outputs, and the pellet-trial ledger remain the
authoritative persisted records.

## Controller ownership

The acquisition application uses explicit responsibility boundaries:

- `AcquisitionController` owns acquisition start/stop flags, independent
  subsystem states, blockers, retries, and derived reach-synchronization
  readiness.
- `RecordingSessionController` owns Ready/Arming/Recording/Stopping/Analyzing/
  Aborting state, generation/session identity, canonical boundary, writer
  completeness, enabled-source results, and analysis timing. Delayed callbacks
  must match both generation and session ID. Record, Stop, and Abort enter through
  one re-entrant command boundary, so simultaneous UI/API commands cannot allocate
  or close the same session twice. Abort atomically invalidates its generation and
  returns to Ready before observers may start another session.
- `PelletCycleController` owns the session ledger, send/acknowledgement/error
  mutation paths, attempt closure, per-trial result application, protocol outcomes,
  lifecycle publication, persistence updates, and the authoritative count
  projection.
- `IntertrialAnalysisCoordinator` owns the bounded worker queue, cancellation
  generation, immutable requests/results, and measured throughput estimate.
- `PelletAutomationController` owns the established load/send/retract/cover/
  release state machine; `TrialProtocolRunner` owns protocol progress.
- `PelletPresenceTracker`, `ShiftRecommendationController`, and
  `CoordinateModel` own presence history, shift recommendation/application
  state, and live diamond-coordinate validation respectively.
- `PelletMisplacedDetector`, `WatchdogMonitor`, and core diamond/triangle
  calibration/transform objects retain their existing focused ownership.

`AppModel` routes UI, process, hardware, and analysis events between these
owners. It does not implement camera acquisition, video encoding, inference
calculation, or the mutable state owned by those controllers.

Physical pellet attempt IDs and their protocol settings are immutable after SEND.
Late analysis may attach diagnostics and results, but cannot reorder attempts or
change a row already used by a subsequent SEND. Analysis-dependent retry/protocol
effects are applied only when the result can still control the next physical trial.
The future-row edit check and SEND snapshot share one lock, and composite ledger,
public-event, protocol, and persistence mutations are serialized by the pellet-cycle
controller.

Camera/NI matching uses the nearest `cam_frames` transition within half the
observed frame period and records whether it was rising or falling. The current
camera output is a square wave whose alternating peaks and troughs each identify
one frame. Alignment reports `host_estimated` when the line is not configured,
and `unmatched` when no plausible transition exists. For configured Tone 1 and
Tone 2 inputs, the NI pulse onset is the canonical physical time. Pulses shorter
than 2 ms are classified as electrical artifacts and do not become tone events.
Both valid pulses and artifacts are clipped to the camera Record/Stop boundary,
so rolling-buffer activity outside the saved session is not reported as an
unmatched session edge.

Tone matching groups the embedded outbound `PLAY_TONE`, immediate pellet-board
`TONE_STATUS`, and legacy periodic `STIMULUS_INPUTS` observation for the same
valid NI pulse within 250 ms. `eventPerfTime` and per-observation latency retain
the raw host/CAN timing, while `alignedEventPerfTime` is the physical NI onset
used to compare the event with camera frames. A command whose frequency does
not identify Tone 1 or Tone 2 remains explicitly unmatched.

During recording, a validated NI Tone 2 pulse opens the live tracking window at
that exact sampled onset. This removes CAN polling latency from the behavioral
boundary. If the configured NI tone stream is unavailable, the immediate CAN
`TONE_STATUS` is the fallback; the periodic GPIO status is used only for legacy
firmware that supplies neither source. NI pulse validation and the live callback
carry state across acquisition blocks so a pulse is emitted once, only after its
duration is known to be at least 2 ms.

The pellet board supplies the first two decoded confirmation lines: physical
`STIM0` is `tone1` and physical `STIM1` is `tone2`. `STIM2` and `STIM3` remain
unassigned.

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

Digital-only multi-device plans use a routed counter sample clock and never
claim an AI start trigger when no AI task exists. The resolved descriptor names
the actual clock producer, consumers, start/reference/sample routes, and output
tasks. Hardware-timed finite laser output is synchronized only when its task
applies that descriptor and is armed before the master. On-demand laser pulses
remain explicitly independent.

## Storage, discovery, and safe hardware boundaries

Record preflight performs a real write, flush, `fsync`, and removal probe on the
target session filesystem. It reports free bytes and projected maximum recording
minutes from configured camera/NI demand; configured duration is capped at two
hours. Recording monitors free space and observed write rate at low cadence,
warns once below 30 and 10 projected minutes, and requests normal Stop below one
minute. Low space never invokes Abort.

Camera discovery runs in a disposable process with a deadline. SDK/import/load
failure is reported separately from a successful discovery that found zero
cameras and includes the first exception, backend/stage, and elapsed time.
Discovery failure affects only camera state. Animal JSON loading is likewise
isolated per file: valid animals remain available, malformed/unsupported files
are listed as skipped without modification, and duplicate UUIDs remain a hard
conflict.

Pellet-board connection and CAN recovery apply acknowledged motor configuration,
home X/Y/Z, then detach both pellet servos. Connection never attaches/releases
the cover or pellet mechanism. Normal LOAD/COVER/RELEASE operations retain their
on-demand servo behavior. Stop, automatic Stop, and Abort request home after the
stream boundary when the board is Ready; failure is logged but cannot prevent
save/deletion.

CAN recovery retains the first diagnostic failure, but only transport and
acknowledgement domains own recovery. Every run has a generation; a stale run
cannot publish Ready or replace a reader still alive. Recovery never replays a
non-idempotent in-flight motor command. Each command retains an explicit pending,
acknowledged, failed, unknown, cancelled, or timed-out outcome; transport loss marks
an in-flight non-idempotent operation unknown rather than treating disappearance from
the pending map as success. Generic unhandled exceptions do not
reset CAN, command motion, stop an active recording, or disconnect unrelated
hardware.

Preferences/configuration is locked outside session `Ready`, including an
already-open dialog. SoftMouse/RFID cache refreshes share one model-owned,
debounced background path; refresh requested during a session is deferred once,
and all widget updates return through the Qt thread. The event manager preserves
its public plugin/API contract while bounding its queue, producer wait, and
shutdown, and reports pending/failed delivery instead of hanging indefinitely.
Configuration mutations and subject/protocol selection also share the lifecycle
command boundary, preventing a Record preflight/allocation race while the public
state is still Ready. Future inactive protocol rows remain independently editable.

The desktop launcher locks through a user-owned mode-0700 runtime directory and
does not truncate a predictable file under `/tmp`. Calibration cleanup accepts
only resolved, non-symlink generated child directories under the selected
calibration root. Repeated Ctrl-C targets only reachAQ and explicitly registered
child processes; it never signals a presumed process group.

Event saturation retains exact-once session/trial lifecycle events in a bounded
publisher outbox. If shutdown times out inside a plugin, the existing manager remains
the singleton until its worker actually exits; reachAQ never creates a second manager
that could reorder delivery against the live worker.

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
7. Start a session and Abort. Confirm the session directory is removed, counts are
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
