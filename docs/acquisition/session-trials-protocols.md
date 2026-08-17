# Recording sessions, pellet trials, protocols, and schema migration

This document defines the current reachAQ behavioral model. It is the reference
for operators, configuration authors, API clients, and developers migrating
from the former one-recording-per-reach AutoTrainer workflow.

## Terms and lifecycle

A **recording session** is one continuous Record-to-Stop interval. Cameras,
pose, NI-DAQ, decoded device/CAN traffic, laser events, pellet-trial events, and
session logs share one canonical camera boundary. A session may contain zero,
one, or many reaches and pellet deliveries.

A **pellet trial** is one logical pellet-delivery cycle inside a recording
session. A physical send attempt is not automatically a new logical trial. The
attempt policy determines whether a retry stays under the same trial number,
receives a new number, or remains unnumbered until presentation succeeds.

Acquisition **Running** initializes enabled hardware, camera preview, and live
inference without opening session writers. **Record** resets session counts and
arms all enabled writers. **Stop** closes one synchronized boundary, retains the
session, and drains analysis already queued for closed pellet trials. **Abort**
cancels queued/running trial analysis, closes writers, deletes the session
directory, and clears the four counts. Record remains disabled while stopped
session finalization is draining.

Automatic pellet cycles are permitted only while a session is actively
recording. Manual pellet-board controls remain available under their explicit
hardware and motion-safety gates. Neither manual nor automatic pellet control
depends on tunnel, head-fix, gate, fan, or magnet hardware.

## Independent operator choices

The former global training-mode selector is removed. These settings are
independent:

- selected protocol, or **Manual pellet control**;
- **Automatic pellet cycles**;
- **Automatic protocol advance**;
- live per-trial trajectory feedback, shift recommendation, and optional
  application;
- optional automatic session stop conditions.

Selecting a protocol attaches it to the active animal. With no selected
protocol, pellet operation is manual unless automatic cycles are explicitly
enabled. Enabling automatic protocol advance does not itself enable pellet
cycles or start recording.

## Ordered trial protocol editor

The **Trial Protocol** tab presents one row per planned logical pellet trial.
Protocols are reusable, atomically saved resources selected per session; they
are not owned by an animal. Each row is typed and executable: pellet cycle,
base/fixed/reach-derived position, center/left/right lane, cover policy, named
tone, named finite laser pulse train, assignment, trigger, and retry policy.
Malformed or unresolved rows cannot become active.

Only future rows are editable. A row becomes highlighted and read-only while
its trial is active, then remains read-only once the logical trial completes.
A motor, transport, command, or acknowledgement error does not complete or
consume that row: under retry-within-trial accounting it returns to Future and
the next attempt keeps the same logical trial number.

Before SEND, the action executor reserves the future row and freezes its protocol
revision, resolves the animal base/lane/shift to one absolute DCS target and one
motor target, waits for motor and cover acknowledgements, and arms any laser
task. A failure here is persisted as a preparation error but creates no pellet
attempt. SEND acceptance creates the physical attempt and binds its CAN context.

Every send attempt receives the immutable row, compiled recipe, resolved target,
stimulus draw, profile revisions, and action lifecycle in `streams/trials.jsonl`.
Session JSON/YAML stores both the compact authoring hierarchy and expanded
schedule. Later edits cannot change an active/completed attempt.

Authoring precedence is protocol defaults, epoch, block, bulk selection, then
individual trial. Epochs may be noncontiguous but cannot overlap; blocks are
ordered, nonoverlapping subsets of one epoch. The editor provides multi-row
selection, fill/repeat, copy/paste, randomized preview, and revision Undo/Redo.

Automatic positioning is mutually exclusive with fixed XYZ. `Legacy batch`
matches the retained nonoverlapping autotrainer window; `Sliding last X`
recomputes after every eligible reach once X results exist. Both use the retained
diamond/triangle coordinate owner and resolve an absolute target, so retries
cannot move twice. Before X reaches, the calibrated lane baseline is retained.

Tone phases are Before SEND, Embedded in board sequence, Pellet presentation,
and Retract. Laser recipes freeze either Hardware STIM3 or Direct NI software
start. Scheduled pre-reveal stimulation is emitted and delayed by the pellet
board before cover release; Direct NI is rejected for that phase. First Reach
uses the 900 Hz stim-camera transition detector, not pose inference. ROI1/ROI2
remain schema/UI framework only and are rejected as runnable.

Subject selection is locked from session Arming through analysis. Session Notes
are saved when Stop closes the writers but remain editable afterward; subsequent
edits atomically replace the stopped session's JSON/YAML metadata. Beginning the
next recording finalizes the previous notes and clears the Notes field.

## Trial and attempt accounting

Every pellet send dispatch creates an operation ID and an attempt record. A
successful board acknowledgement records the presentation boundary. Queueing a
command is not treated as presentation.

The user-facing failed-attempt choices are:

| UI label | Configuration value | Behavior |
|---|---|---|
| Retry within the same trial | `retry_within_trial` | Default. Retries use labels such as `10.1`, `10.2`, and `10.3`. |
| Count every attempt as a new trial | `every_attempt_is_trial` | Each behavioral attempt consumes a logical trial number. |
| Count only successful pellet presentations | `successful_presentations_only` | An attempt receives a public trial number only after the board acknowledges presentation. |

Retry settings are either **Reuse the original trial settings** (`reuse`) or
**Choose new settings for the retry** (`resample`). The policy is recorded with
the attempt so analysis does not have to infer it later.

Motor, command, transport, and acknowledgement failures are always explicit
hardware errors. They receive an operation ID and error details, but never
increment any trial-limit or protocol-progress count. Under the default policy,
a later successful retry retains the reserved logical trial and advances its
attempt suffix.

`pellet_missing` requires direct live evidence that the pellet was absent.
A presented pellet with no detected reach is `no_reach`, not
`pellet_missing`. If tracking is unavailable, presence/misplacement is
`unknown`; ReachAQ does not invent an outcome to activate a retry rule.

The trial-limit count basis is independently configurable:

| UI label | Configuration value |
|---|---|
| Trials started | `started` |
| Pellets presented | `presented` |
| Trials completed | `completed` |
| Scored trials | `scored` |

Configured behavioral outcomes may be included or excluded from completed and
scored counts through the plain-language **Counted outcomes** checkboxes in
Preferences. Hardware errors cannot be included. Counts are derived from the
ledger rather than maintained as a second independent source of truth.

**Scored trials** is available only when live intertrial analysis is enabled.
Each score is finalized from the just-completed pellet trial while the recording
continues. An automatic scored-trial limit therefore reacts only to finalized
results; pending, skipped, unknown, and unavailable results never fabricate a
score.

## Trial persistence

Each retained session contains:

```text
sessionNNN/
└── streams/
    ├── tracking/
    │   └── trial_<id>_attempt_<id>.json
    ├── trials.jsonl
    └── trial_summary.json
```

Each JSON Lines record includes session ID, operation ID, logical trial ID,
attempt ID and display label, send request timestamps, acknowledgement
timestamps, capture/finalization timestamps, outcome, hardware-error kind,
error text, retry settings, whether the logical trial completed, pellet
position, planned/applied shift, protocol/phase context, per-attempt
reach/success/consumption counts and reach-event indices, and associated
tone/laser references.
`trial_summary.json` contains physical-attempt, hardware-error, incomplete,
pending-analysis, started, presented, completed, scored, and configured-basis
counts. Both files use the same canonical performance/wall timebase as the
other session streams and appear in `alignment.json`'s enabled-source manifest.

A validated physical NI Tone 2 onset opens a trial's tracking window when that
input is configured and running; pellet-cycle completion closes it with no
post-trial margin. This makes the boundary share the sampled camera/NI timeline
instead of the delayed periodic CAN GPIO report. The pellet board's immediate
tone status is the fallback when NI confirmation is unavailable, with periodic
GPIO status retained only for legacy firmware. The immutable tracking slice is
persisted below `streams/tracking/` and sent to one bounded background worker.
The worker consumes existing live pose/tracking data; it does not reopen video
and does not run a second inference pass. Results update the ledger atomically
while cameras and every other session stream continue.

Stop marks an active, not-yet-completed physical attempt `incomplete`, drains
only already-closed windows, and performs a final repair from the stored
tracking JSON if needed. An attempt left without a usable result is explicitly
finalized as incomplete, never silently retained as pending. Abort cancels the
analysis generation and removes the tracking records with the rest of the
session.

The decoded device ledger at `streams/device.csv` remains separate and records
general inbound/outbound pellet-board traffic. The trial ledger consumes the
same acknowledged events for behavioral accounting; it does not replace the
decoded event stream.

## Protocol progress

ReachAQ owns the trial-boundary adapter in
`tools/acquisition/model/trial_protocol_runner.py`. The installed training
package still supplies protocol parsing, reusable pellet actions/predicates,
progress serialization, and phase navigation, but its recording-boundary
callbacks are detached. A qualifying pellet trial is evaluated exactly once at
the pellet-trial boundary, so one recording session can advance through many
trials without stopping camera writers.

Hardware-error, aborted, and incomplete attempts do not advance protocol
progress. Automatic advance/fallback occurs only when enabled. Reaching the
terminal phase marks the protocol complete and may request a graceful session
stop if **Stop when protocol finishes** is enabled.

The retained training package still accepts a nullable tunnel collaborator at
its generic attachment boundary. ReachAQ passes `None`; no tunnel runtime or
tunnel status is created.

## Automatic session stop

Three stop policies are optional:

- elapsed recording duration (`durationLimitSeconds`);
- configured trial count (`trialLimit` plus `trialCountBasis`);
- selected protocol completion (`stopOnProtocolComplete`).

When more than one is enabled, the first reached condition requests Stop and
records all conditions that were simultaneously true. No new automatic trial is
scheduled. If no trial is active, writers stop immediately. If a trial is
active, reachAQ lets it finish and recover to its terminal pellet-machine state
before selecting the final recording boundary.

The duration threshold is therefore a minimum, not an exact output-file
duration. Normal drain is not an error. Only expiration of
`stopDrainTimeoutSeconds` is an error; its default is 15 seconds. On timeout,
the active attempt is marked incomplete with an explicit error, best-effort
safe recovery runs, and the session stops. Manual Stop and Abort remain
available and the stop arbiter permits only one terminal path to win.

Final metadata records the configured limits, actual stop reason, final
duration, trial summary, protocol state, and incomplete state.

## Session counts and analysis

The Behavior panel displays only **Reaches**, **Presented**, **Success**, and
**Consumed**. All reset together at Record. They remain visible after Stop and
finalization, and reset to zero after Abort. Day and lifetime counters are not
persisted in the active animal schema. The ledger summary is the one
authoritative source for all four values.

Live inference continues during acquisition/recording using the existing
depth-one frame queue and does not control whether camera or NI data is written.
When live intertrial analysis is enabled, tone-2-to-cycle-completion slices are
analyzed on a separate bounded worker during the same recording. Pellet
presence and misplacement are finalized synchronously from live tracking at
cycle completion and never wait for that worker. Without healthy live tracking,
those fields are recorded as `unknown` and operation continues.

Intertrial analysis requires live inference. Record readiness rejects the
contradictory configuration where analysis is enabled but inference is disabled;
disable both to record and run manual/automatic pellet protocols without
analysis or scoring.

**Continue while analyzing** permits another SEND while advisory analysis is
pending. **Wait for analysis** blocks the next SEND while other writers remain
active. Any configured behavioral retry outcome forces waiting so the next row
and retry label are selected deterministically, except `pellet_missing`: direct
missing-pellet evidence selects its retry synchronously and never enters the
analysis wait queue. The UI reports pending work and a locally measured
analysis-seconds-per-tracking-second estimate. If required analysis fails, the
operator can retry the same immutable window or continue with an explicit
unavailable result and skipped retry decision. Stop and Abort are never blocked
by analysis failure.

## Shift and calibration behavior

Diamond/triangle calibration, DCS transforms, motor drift checks, recommended
XYZ shift calculation, display, and persistence are retained. A recommended
per-trial shift can be applied only to a future trial when enabled. A late
advisory result cannot modify an active/completed row or relabel a physical
attempt. Protocol actions may request predetermined motor shifts between trials
after the pellet machine reaches a safe state and command acknowledgement is
available.

## System configuration policy

The production system configuration schema is version 57. ReachAQ loads only
that version and rejects older/newer versions, unknown keys, retired
load-cell/tunnel/head-fix fields, obsolete NI recording controls, and historic
pellet-shift coordinate keys. There is no hidden compatibility runtime.

The `behavior.sessionControl` fields are:

```yaml
sessionControl: !SessionControlConfiguration
  automaticPelletCyclesEnabled: false
  automaticProtocolAdvanceEnabled: false
  intertrialAnalysisEnabled: false
  intertrialProgressionMode: continue
  behavioralRetryOutcomes: []
  attemptAssignment: retry_within_trial
  retrySettings: reuse
  trialCountBasis: completed
  countedTrialOutcomes: [success, failure, pellet_missing, no_reach]
  durationLimitSeconds: null
  trialLimit: null
  stopOnProtocolComplete: false
  stopDrainTimeoutSeconds: 15.0
```

NI-DAQ `displayChannels` controls plotting only. All configured channels are
acquired and persisted whenever NI-DAQ is enabled, regardless of plot
selection. The retired `recordToAcquisition` and stream `outputName` controls
are rejected.

## Animal JSON schema

Animal JSON v7 uses its immutable reachAQ UUID as the filename and identity. It
persists the editable local name and animal notes, pellet coordinate space and
position, target limit, protocol/trial progress, and the permanent
UUID-to-SoftMouse/RFID link with its metadata snapshot.

Animal v4, v5, and v6 files are accepted for one-way migration on their first
successful save. The original bytes are preserved beside the active file as
`<animal>.json.vN-backup`; v4 recording-count progress is reset because it
cannot be converted reliably to pellet-trial progress. V0-v3, files without an
ID, and unknown future versions are rejected clearly.

## Public API lifecycle

ReachAQ pins `auto-trainer-api` 0.11.0 and publishes its own status schema with
acquisition state, recording-session state, reach synchronization readiness,
independent subsystem states, animal identity/pellet coordinates, protocol
state, pellet-device state, and the four session counts. Retired alarm,
emergency, tunnel, head-fix, and magnet-shaped status placeholders are absent.

`sessionStarted`/`sessionEnded` describe the continuous Record-to-Stop boundary.
`trialStarted`, `trialCaptureEnded`, and `trialEnded` now describe one pellet
attempt and carry stable session, trial, attempt, and operation IDs. Publication
is exactly once per lifecycle edge. Abort balances any opened lifecycle for
diagnostics, marks the session aborted, cancels analysis, and does not retain
session data. The former recording-as-trial result/presence events are not
published.

The retained `trainingModeChanged` event is a protocol-selection projection
(manual, manual with protocol, or automatic protocol advance), not a global
hardware mode and not a placeholder for removed trainer hardware.
