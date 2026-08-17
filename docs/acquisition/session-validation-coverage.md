# Session-validator acceptance coverage

This map connects the stable `reachaq-validate-session` rule IDs to the tracked
acceptance checklist. The rule ID is the durable automation identifier; checklist
wording may be clarified without changing report consumers. A Pass proves only
the persisted-data claim in the **Automated evidence** column. It never certifies
the manual or physical remainder.

| Stable rule ID | Acceptance topics covered | Automated evidence | Classification / remaining observation |
|---|---|---|---|
| `session.layout` | stopped-session retention; required session artifacts | published session root, manifest, and authoritative metadata exist | Fully automated for retained sessions; Abort deletion is UI/lifecycle testing |
| `session.schema` | first-release schema; rejection of disposable pre-release sessions | manifest and metadata schema versions are supported | Fully automated |
| `session.generation` | one atomic metadata generation | metadata, manifest, alignment, source manifest, and trial summary share an identity | Fully automated |
| `session.manifest` | artifact existence, containment, size, and hashes | declared paths remain inside the session; files, sizes, uniqueness, and Full-profile hashes agree | Fully automated; power-loss injection remains manual |
| `metadata.mirror` | JSON/YAML equivalence and finite boundaries | authoritative JSON/YAML content agrees and canonical boundary values are finite | Fully automated |
| `sources.contract` | all enabled sources recorded independently of plot visibility; completeness | enabled required sources have artifacts, terminal status, and appropriate coverage; disabled/optional sources are classified separately | Partially automated: plot visibility and intentional enable/disable actions are UI checks |
| `camera.frames` | MP4 closure and recorded counts | Fast checks stored writer/timestamp/decoded summaries; Full decodes every MP4 and reconciles counts | Fully automated after Stop; deliberate writer-timeout behavior is fault injection |
| `camera.ledger` | actual frame identity and event/video indexing | frame IDs are unique/monotonic, missing frames are explicit, and video index mapping is coherent | Fully automated for persisted evidence; physical exposure synchronization requires a shared-pulse test |
| `nidaq.continuity` | NI sample continuity, duration, coverage, gaps, overruns | datasets, channel order, sample indices, reconstructed time, counts, and source health agree | Fully automated for persisted samples; card/wiring qualification is hardware-required |
| `nidaq.timing_graph` | selected NI topology and synchronization claim | task graph, exact preflight result, strategy, routes, device count, and timing quality are internally consistent | Partially automated: DAQmx resource behavior and shared-pulse skew require the configured rig |
| `events.alignment` | canonical slicing and general event timing | event timestamps/offsets fall inside the saved boundary and preserve method/confidence | Fully automated |
| `events.frames` | all-event association with acquired/recorded frames | wall time, recording offset, actual frame ID, video index, relation, method, and confidence reconcile with the ledger | Fully automated for persisted associations |
| `events.tones` | Tone 1/2 CAN, NI TTL, artifacts, and frame alignment | decoded commands/status, valid NI confirmations, in-boundary unmatched edges, artifact classification, and recorded-frame references agree | Partially automated: audible tone and electrical polarity/voltage are physical checks |
| `events.laser` | laser operation/event persistence | rows have typed events and recorded-frame associations | Partially automated: AO/command-copy/diode onset and safe output require wired hardware tests |
| `stim.evidence` | 900 Hz evidence completeness and one-shot detection | evidence schema, contiguous acquired IDs, gaps, ownership, arming, bounded clips, and manifest counts agree | Partially automated: sustained effective rate, ROI suitability, and shared-pulse uncertainty are physical/endurance checks |
| `events.board_time` | board timestamp/clock model/sequence evidence | raw host and board time remain distinct; boot, sequence, monotonicity, fit, uncertainty, transport delay, and selected confidence are coherent | Partially automated: firmware flash/version and NI comparison require the board |
| `trials.lifecycle` | attempts, retry identity, terminal outcome, and four counts | identities, labels, SEND/ack/final order, terminal outcomes, and recomputed summaries agree | Fully automated for recorded attempts; inducing each failure mode is manual/fault injection |
| `trials.protocol` | immutable row, compiled actions, automatic shift, and laser lifecycle | protocol/row/attempt identity, action ordering, absolute targets, policy/recommendation snapshots, and laser terminal evidence agree | Partially automated: actual motor/cover/tone/laser behavior must be observed or independently wired |

## Checklist areas that intentionally remain manual

The validator deliberately does not claim coverage for application-window/menu
behavior, control enablement, desktop launching, subject/notes editing, operator
workflow, visible plots, camera discovery/recovery, CAN ownership/reconnect,
privileged reset behavior, SoftMouse publication, protocol-editor usability, or
automatic Stop interaction. Those are UI, lifecycle, service, or fault-injection
tests.

The following require retained physical evidence and cannot be promoted to a
software-only Pass: camera exposure synchronization; NI topology skew; PFI/RTSI/
PXI wiring and voltage compatibility; audible tones; pellet movement, cover,
presence, and misplacement; 900 Hz sustained stim-camera throughput; STIM3 and
Direct-NI end-to-end latency; laser AO/command-copy/diode response and safe-state
cleanup; firmware flash/rollback; and multi-hour endurance.

The tracked checklist remains the release gate. Validator reports automate its
persisted-data portions and attach stable evidence IDs; they do not replace the
unchecked operator and hardware items.
