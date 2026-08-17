# Session validation

`reachaq-validate-session` is a read-only audit of a successfully stopped,
published first-release session. It never connects hardware, repairs files,
changes canonical metadata, or validates aborted/deleted pre-release sessions.

```bash
reachaq-validate-session /path/to/sessionNNN --fast
reachaq-validate-session /path/to/sessionNNN --full --json
reachaq-validate-session /path/to/sessionNNN --rule events.tones
```

Profiles:

- `--quick`: publication identity, critical parsing, required enabled sources,
  stored counts, and manifest values; automatically queued after successful
  Stop and atomically written to `validation/quick.json`.
- `--fast` (CLI default): schemas, merged frame ledger, summarized NI coverage,
  event/frame associations, board time, tone/laser/stim evidence, and trial/
  protocol lifecycle without decoding every frame or scanning every NI sample.
- `--full`: additionally rehashes every declared artifact, decodes every MP4,
  and scans all NI/stim sample indices and timelines.

The UI offers Fast and Full validation of the last retained stopped session.
Workers run outside Qt, expose progress/cancel, use reduced priority during
preview, and are disabled during recording/finalization/analysis. Cancel affects
only validation. Derived reports live under `validation/` and are excluded from
the canonical manifest to avoid circular hashes.

Exit codes are 0 for no failures, 1 for validation failure (or warning with
`--strict-warnings`), and 2 for a tool error that prevented a trustworthy rule.
Each result is Pass, Warning, Fail, Not applicable, or Tool error.

Current stable rule IDs:

| Area | Rules |
|---|---|
| Publication | `session.layout`, `session.schema`, `session.generation`, `session.manifest`, `metadata.mirror` |
| Sources/media | `sources.contract`, `camera.frames`, `camera.ledger` |
| NI | `nidaq.continuity`, `nidaq.timing_graph` |
| Events | `events.alignment`, `events.frames`, `events.tones`, `events.laser`, `events.board_time`, `stim.evidence` |
| Trials | `trials.lifecycle`, `trials.protocol` |

The rule-to-checklist proof boundary is maintained in
[session-validation-coverage.md](session-validation-coverage.md). In particular,
a validator Pass does not promote a physical wiring, stimulus, UI, recovery, or
endurance checklist item to complete.

Automated validation cannot prove cable polarity/voltage, physical laser onset,
motor/cover behavior, UI usability, sustained 900 Hz throughput, or topology
qualification. Those remain explicit items in the tracked acceptance TODO.
