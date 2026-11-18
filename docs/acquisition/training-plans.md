# Training plans

A training plan moves a subject through phases (for example close pellet, then
standard position) based on its pellet-trial results. It is separate from the
ordered trial protocol, which says what each trial does; a plan can run with or
without one. The engine is `auto-trainer-behavior/src/autotrainer/behavior/training/`.

## Files

Plans are JSON files in the loaded configuration's `training_plans/` directory,
beside `trial_protocols/`, one plan per file. A bad file is reported when the
configuration loads and leaves the plan list empty; fix it and load the
configuration again. `planId` must be unique in the directory and every
`phaseId` unique across all plans, because saved progress names phases by id.

An example is `auto-trainer-behavior/tests/fixtures/training_plans/reach-shaping.json`.
The full schema is in the docstring of `training/schema.py`. In short:

```json
{
  "schemaVersion": 1,
  "planId": "reach-shaping",
  "name": "Reach shaping",
  "phases": [
    {
      "phaseId": "shaping-close",
      "name": "Close pellet",
      "device": {"pelletDelivery": true, "pelletCover": false, "sendOffset": {"y": -1.0}},
      "advance": {"all": [{"metric": "trials", "op": ">=", "value": 20},
                          {"metric": "successRate", "op": ">=", "value": 0.6, "window": 20}]},
      "fallback": {"metric": "consecutiveFailures", "op": ">=", "value": 10},
      "trialActions": [
        {"action": "movePelletTarget", "when": "success", "dy": 0.25, "limits": {"y": [0, 1]}},
        {"action": "jitterPelletTarget", "x": 0.2},
        {"action": "tone", "frequencyHz": 4000, "durationMs": 200},
        {"action": "pause", "minSeconds": 3, "maxSeconds": 6}
      ]
    }
  ]
}
```

## Phases

- **device**: optional settings applied when the phase becomes current:
  `pelletDelivery`, `pelletCover`, `pelletShift` (between-trial pellet shift).
  A missing key leaves the configuration's value, also when an earlier phase
  set it. The configuration's values come back when the plan is detached
  (unless the operator changed the setting meanwhile), and saving the
  configuration while a plan is attached saves them, not the phase's. `sendOffset` is a DCS offset in mm from the subject's
  pellet target.
- **advance / fallback**: predicates over the current visit to the phase. A
  condition is `{"metric", "op", "value"}`; combine with `{"all": [...]}` or
  `{"any": [...]}`. Metrics: `trials`, `successes`, `failures`, `noReach`,
  `reaches`, `successfulReaches`, `pelletsPresented`, `pelletsConsumed`,
  `successRate` (optionally over the last `window` trials; false until that many
  have run), `consecutiveSuccesses`, `consecutiveFailures`, `timeInPhase`
  (seconds of trial time). Advance is checked before fallback.
- **trialActions**, evaluated for each SEND:
  - `movePelletTarget`: after each counted trial (`when`: `always`, `success` or
    `failure`), add `dx/dy/dz` to the phase's accumulated offset, clamped per
    axis by `limits`.
  - `jitterPelletTarget`: a uniform random offset within `±x/y/z` per trial,
    seeded from the session seed, trial and attempt, so a session is
    reproducible.
  - `tone`: played on the pellet board right before every SEND, once the SEND
    is allowed.
  - `pause`: holds automatic SENDs for a fixed (`seconds`) or random
    (`minSeconds`-`maxSeconds`) time after each counted trial. A forced manual
    SEND is not held.

Pellet-target offsets (`sendOffset`, move, jitter) are applied through an
ordered protocol's compiled trial, where they add to the base and lane target
and are recorded in the trial's position evidence (`training_offset_dcs`,
`training_directive`). A reach-derived automatic target is used as computed,
and its evidence says `training_offset_applied: false`.
Without an ordered protocol a SEND uses the current pellet position, and the
application logs a warning once.

## Counting and progress

A trial counts once its result is final. Hardware-error, aborted and incomplete
attempts never count. Pellet-missing and precheck failures count as trials
without a presented pellet. With **Automatic protocol advance** on, an advance
or fallback result changes phase at once; advancing from the last phase marks
the plan complete, which can stop the session if **Stop when protocol
finishes** is on. Without it, results are only reported and the operator moves
phases.

Counters reset each time a phase is entered (a visit). Lifetime totals per phase
and the number of visits are kept and shown in the progress panel.

Progress is saved in the animal file under `training.protocolProgress`, one
record per plan (`plan_id`, `progressVersion` 1, state, current phase, and each
phase's visit counters, attempts, lifetime totals and a content fingerprint). If
a phase's rules, device settings or actions have changed since the progress was
saved (renaming or rewording it does not count), loading reports
it as out of date; confirming resets that phase's progress and keeps the rest.
