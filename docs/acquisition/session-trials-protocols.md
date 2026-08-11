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
session, and runs post-session analysis. **Abort** cancels pending analysis,
closes writers, deletes the session directory, and clears the four counts.
Record remains disabled while post-session analysis is running.

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
- post-session shift recommendation and optional application;
- optional automatic session stop conditions.

Selecting a protocol attaches it to the active animal. With no selected
protocol, pellet operation is manual unless automatic cycles are explicitly
enabled. Enabling automatic protocol advance does not itself enable pellet
cycles or start recording.

## Ordered trial protocol editor

The **Trial Protocol** tab presents one row per planned logical pellet trial.
The initial 15 rows contain placeholder examples for pellet-delivery behavior,
XYZ shift, cover, tone, and laser choices. These fields define the extensible
row schema; delivery execution can consume additional row options as those
behaviors are implemented.

Only future rows are editable. A row becomes highlighted and read-only while
its trial is active, then remains read-only once the logical trial completes.
A motor, transport, command, or acknowledgement error does not complete or
consume that row: under retry-within-trial accounting it returns to Future and
the next attempt keeps the same logical trial number.

Every send attempt receives an immutable copy of its row under
`protocol_context.trial_row` in `streams/trials.jsonl`. Session JSON/YAML also
stores the ordered `protocolSchedule`, so later edits cannot change the settings
attributed to an already recorded attempt.

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

The trial-limit count basis is independently configurable:

| UI label | Configuration value |
|---|---|
| Trials started | `started` |
| Pellets presented | `presented` |
| Trials completed | `completed` |
| Scored trials (post-session analysis only) | `scored` |

Configured behavioral outcomes may be included or excluded from completed and
scored counts through the plain-language **Counted outcomes** checkboxes in
Preferences. Hardware errors cannot be included. Counts are derived from the
ledger rather than maintained as a second independent source of truth.

**Scored trials** is disabled as an active-session automatic-stop basis. ReachAQ
does not claim a scored result until post-session analysis has assigned reach
events to an attempt window, so using it to stop the recording would be
misleading. A saved ledger still reports its scored count after analysis.

## Trial persistence

Each retained session contains:

```text
sessionNNN/
└── streams/
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

During recording, attempts close as `pending_analysis` because a behavioral
outcome is not yet trustworthy. After Stop, analysis maps frame-indexed events
to the non-overlapping performance-time window of each attempt, finalizes every
pending attempt exactly once, deterministically applies the configured retry
numbering (`1.1`, `1.2`, and so on), and atomically replaces both ledger files.
An attempt left without an analysis result is finalized as explicit incomplete,
never silently retained as pending.

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
post-session analysis, and reset to zero after Abort. Day and lifetime counters
are not persisted in the active animal schema.

Live inference continues during acquisition/recording using the existing
depth-one frame queue and does not control whether camera or NI data is written.
Post-session analysis remains enabled where configured and may populate final
reach results and shift recommendations after Stop. Analysis is cancelled for
Abort and Record remains blocked until analysis completes.

The implementation still uses some internal `intersession` class/event names
from the retained inference library. In ReachAQ lifecycle terms these always
mean **post-session analysis**, not an interval between pellet trials.

## Shift and calibration behavior

Diamond/triangle calibration, DCS transforms, motor drift checks, recommended
XYZ shift calculation, display, and persistence are retained. A recommended
post-session shift can be applied to the next session when enabled. Protocol
actions may request predetermined motor shifts between trials after the pellet
machine reaches a safe state and command acknowledgement is available.

Analysis-derived online per-trial shifts are not implemented: the retained
analysis runs after recording and cannot affect an earlier trial in that same
session.

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
  attemptAssignment: retry_within_trial
  retrySettings: reuse
  trialCountBasis: completed
  countedTrialOutcomes: [success, failure, pellet_missing]
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
