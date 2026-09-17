# Protocol panel, protocol sets, and laser stim testing

Design for three operator-facing additions to the acquisition UI, plus the
firmware follow-up one of them depends on.

- Branch: `feature-dev`, cut from `devel` after
  `reach-training-protocol-features` merges into it. The merge and the push of
  `devel` are held until explicitly authorised; this document is written ahead
  of that and is committed onto `feature-dev` once the branch exists.
- Work happens in a dedicated worktree at `../reachAQ-feature`.
- Landing order: panel, then sets, then stim test.
- Each feature is cherry-picked onto `demo-mode-and-presentation` once it is
  verified on the rig.

## Problem

`ProtocolContent` presents 29 typed columns in a `QTableWidget` that lives in a
`QTabWidget` given 430 px of a 1610 px window. The protocol editor is not
usable at that width, and the Laser Control tab beside it has the same
constraint.

Separately, a protocol today is one flat document of N trials. There is no way
to save a reusable group of trials and compose several of them into an
experiment, so every experiment that shares a phase duplicates it.

Finally, the only way to fire a laser through the path an experiment actually
uses is to run a session. `_LaserChannelTab._run_pulse` drives the analog
output directly from the host, which does not exercise the board trigger, so
the hardware-triggered route cannot be tested on the bench.

## Non-goals

- No change to `TrialProtocolSchedule`, `TrialProtocolRunner`, session
  persistence, or session evidence. The compiler emits an ordinary
  `TrialProtocolDocument`, which is the whole point of composing by value.
- No shuffling or interleaving of the set order itself. Only trial order
  inside a set instance is shuffled.
- No live references from an experiment to a set. Set revisions are pinned at
  compose time.
- No `stim_line` parameter in the laser profile until firmware can honour it.

## Decisions

| Question | Decision |
| --- | --- |
| Panel expansion | True overlay in a top-level window, not a raised child widget |
| Panel scope | The whole right-side tab widget, Laser Control and Protocol together |
| Set model | Own file per set, composed by value into a flat document |
| Composition | Ordered concatenation, repeat count per entry, shuffle trials within an instance |
| Stim test placement | Per laser channel in Laser Control, firing a saved profile |
| Stim test path | Board pulse hardware-triggers the armed analog output |
| Stim lines | STIM2 and STIM3. STIM3 works today; STIM2 needs firmware |

## 1. Expandable, detachable panel

### Approach

One reparenting mechanism with three states:

- **Docked** - the tab widget sits in `_main_splitter` exactly as it does now.
- **Expanded** - reparented into a frameless `Qt.Tool` window sized over the
  main window's central area, tracking the main window's moves and resizes.
- **Detached** - reparented into an ordinary top-level window with a title bar,
  freely placed on any monitor, not tracking the main window.

A placeholder widget holds the splitter slot while the panel is away, so
restoring does not disturb saved splitter sizes.

### Why a top-level window rather than a raised child

The left side contains `QtGLImageView`, a `QOpenGLWidget`. Stacking a sibling
widget above a `QOpenGLWidget` depends on platform compositing and is the
classic case that works on a Windows development machine and fails on the
rig's Linux GL stack. A top-level window is composited by the window manager,
so the overlay is unconditional. It is also the same code path detachment
needs, so both states share one implementation.

### Components

- **New** `tools/acquisition/view/detachable_panel.py` - `DetachablePanelHost`,
  owning the hosted widget, the splitter slot, the placeholder, and the current
  state. Public surface: `expand()`, `collapse()`, `detach()`, `reattach()`,
  `state`. No dependency on the app model or on hardware, so it is testable
  offscreen in isolation.
- `tools/acquisition/view/main_content.py` - wrap `_create_right_side_tabs()`
  in the host; add expand and detach controls to the tab bar corner.
- `tools/acquisition/view/main_window.py` - forward `moveEvent` and
  `resizeEvent` to the host so the expanded overlay tracks the main window.
- `tools/acquisition/model/user_preferences.py` - persist panel state and
  detached geometry alongside the existing splitter state.

### Error handling

Restoring a detached geometry that lies outside every current screen falls back
to centring on the primary screen. A panel that was detached when the
application closed reopens detached; if that fails for any reason, it reopens
docked rather than not at all.

### Risk

Some Linux window managers place or flicker `Qt.Tool` windows oddly on move.
This is verified on the rig before the feature is called done. The fallback, if
tracking misbehaves, is a plain top-level window that does not follow the main
window.

## 2. Protocol sets and experiment composition

### Model

**New** `tools/acquisition/model/trial_protocol_set.py`:

```python
TrialProtocolSet(
    set_id, name, revision=1, trial_count=...,
    defaults: ProtocolPatch, bulk_overrides: Tuple[ProtocolScope, ...],
    trial_overrides: Tuple[TrialOverride, ...], description="",
    schema_version=SET_SCHEMA_VERSION,
)

ExperimentSetEntry(
    set_id, set_revision, repeat=1,
    shuffle_trials=False, shuffle_seed=None,
)

ExperimentComposition(
    experiment_id, name, revision=1,
    entries: Tuple[ExperimentSetEntry, ...], description="",
    schema_version=EXPERIMENT_SCHEMA_VERSION,
)

compile_experiment(composition, set_library) -> TrialProtocolDocument
```

A set reuses `ProtocolPatch`, `ProtocolScope`, and `TrialOverride` from
`trial_protocol_schedule.py` rather than introducing a parallel vocabulary, so
it inherits the same typed validation.

A set deliberately has no `epochs` and no `blocks`. Those slots belong to the
experiment level and are written by the compiler.

### Compilation

`TrialProtocolDocument.resolve()` applies patches in the order `defaults`,
`epochs`, `blocks`, `bulk_overrides`, `trial_overrides`, recording a per-field
source label. The compiler maps set structure onto that order so within-set
precedence survives unchanged:

| Set element | Becomes | Resolution position |
| --- | --- | --- |
| `defaults` | the instance's epoch patch | after document defaults |
| `bulk_overrides` | document `bulk_overrides`, name-prefixed | after epochs |
| `trial_overrides` | document `trial_overrides` | last |

The compiled document's own `defaults` stays empty, because different sets
disagree on almost every field.

Algorithm:

1. For each entry in order, resolve the set by `set_id` and require its stored
   revision to equal `set_revision`. A missing set or a revision mismatch is a
   hard error naming the entry; it never silently compiles against a different
   revision.
2. For each repeat instance `r` in `1..repeat`:
   1. Allocate a contiguous global trial-id range starting at the running
      counter, of length `set.trial_count`.
   2. Build a `local -> global` mapping over the set's local id space,
      `1..trial_count`. Unshuffled it is `base + local`. Shuffled it is
      `base + rank(local)` under a permutation drawn from a per-instance
      generator seeded with the triple `(shuffle_seed, entry_index, r)`, so
      repeats of the same entry differ from each other while the whole compile
      stays reproducible from the single seed stored on the entry. When
      `shuffle_trials` is set and `shuffle_seed` is `None`, the compiler draws
      a seed once and writes it back to the entry, so a compile is never
      irreproducible.
   3. Emit one epoch named `<set_id>` when `repeat == 1`, otherwise
      `<set_id>#<r>`, whose `trial_ids` are the allocated range and whose patch
      is the set's `defaults`.
   4. Remap every `bulk_overrides` scope's `trial_ids` and every
      `trial_overrides` entry's `trial_id` through the mapping, prefixing scope
      names with the instance name.
3. Set `trial_count` to the sum of `set.trial_count * repeat` across entries.
4. Return provenance: per-instance epoch name, `set_id`, `set_revision`,
   repeat index, resolved seed and trial ids, alongside the input composition
   with any drawn seed written back.

Provenance is returned rather than stamped into the document. Adding fields to
`TrialProtocolDocument` would require bumping `PROTOCOL_SCHEMA_VERSION`, which
the runner and session evidence both read, for a benefit the epoch names
already deliver. The compiled document carries a human-readable `description`
naming the sets and revisions it came from; the machine-readable record lives
in the experiment file.

Instance names must satisfy the existing identifier rule,
`^[a-z0-9][a-z0-9._-]{0,63}$`, so an instance is `<set_id>-<n>` rather than
`<set_id>#<n>`, and the ordinal counts appearances of that set across the whole
composition rather than within one entry — otherwise the same set used by two
separate entries would collide on a duplicate epoch name, which
`_validate_scopes` rejects. Generated names longer than 64 characters are
truncated with a short digest suffix.

Because every trial keeps a source label naming its epoch, the existing
`Sources` column reads `epoch:baseline#1` with no UI change, and session
evidence records exactly which set revisions produced the run.

### Validation

The compiler's output is constructed through `TrialProtocolDocument`, so
`_validate_scopes` checks it: epochs are non-overlapping and in range, trial
override ids are unique. Compilation of a valid set library cannot produce an
invalid document; if it ever does, the constructor raises rather than the
runner discovering it mid-session. A compile error is reported against the
offending entry, not the whole experiment.

### Storage

Mirrors the existing one-file-per-document pattern of
`TrialProtocolRepository`, which is left untouched:

- `<config>/trial_protocol_sets/<set_id>.json`
- `<config>/trial_experiments/<experiment_id>.json`

Two thin repositories, `TrialProtocolSetRepository` and
`ExperimentCompositionRepository`, shaped like the existing one: `documents`,
`errors`, `get`, `reload`, `save`, `duplicate`, `rename`, `import_file`,
`export_file`. Both are rebound on configuration load exactly where
`_trial_protocol_repository` is rebound in `AppModel`.

Compiled experiments are saved into the existing `trial_protocols/` directory
as ordinary protocol documents, so selecting one for a session needs no new
code path. The compiled document takes `protocol_id` from the experiment id and
`name` from the experiment name, and its `revision` increments on each compile
of the same experiment. Recompiling an experiment therefore replaces its
protocol in place rather than accumulating copies, and a session that has
already selected an earlier revision keeps the document it resolved.

### UI

The Protocol panel gains a left sidebar, usable because feature 1 has already
landed:

- a set library list, above
- an experiment builder list of ordered entries, each row showing set name,
  repeat count, and a shuffle toggle, with reorder and remove controls.

A set's trial content is authored **through the existing protocol editor**
rather than in a second editor: `Set from protocol` captures the protocol
currently selected for the session as a set, and `Open set` publishes a set
back as an ordinary protocol to revise and capture again. This avoids
duplicating the 29-column table's edit path, and it means a set can only ever
contain content the existing editor already validates. A protocol that uses
epochs or blocks is refused rather than flattened, because those are the older
protocol-local grouping and the compiler writes epochs itself.

A `Compile and save` action writes the compiled document into the protocol
library, from where it is selected for a session normally.

### Documentation

`docs/acquisition/session-trials-protocols.md` describes the protocol editor as
it ships. It gains a section on sets and composition when this feature lands.

## 3. Laser stim testing

### Approach

The hardware-triggered route already exists and is reused rather than
reimplemented. `LaserModel.prepare_pulse_profile` arms a
`LaserSynchronizedPulseTrain` with `trigger_source=profile.trigger_terminal`
for the `hardware_stim3` route, and `AppModel._trigger_protocol_stim3` fires
the board pulse through `hardware.pulse_stim3`. A bench test is arm, pulse,
release, against machinery that is already covered by tests.

**New** `AppModel.run_stim_bench_test(profile_id, channel_id) -> StimTestResult`:

1. Resolve the saved laser profile. Require `trigger_route` to be
   `HARDWARE_STIM3` and `trigger_terminal` to be set.
2. Build a bench recipe. `prepare_pulse_profile` uses the recipe only to build
   its `operation_context` provenance, so a recipe identifying itself as a
   bench test is honest and carries no session identity.
3. Arm the analog output on the trigger terminal.
4. Fire the board pulse for `profile.trigger_pulse_us`.
5. Await the laser terminal event, release the prepared profile, and return the
   measured arm-to-terminal interval.

### Guards

The test refuses, with a message naming the reason, when:

- a recording session is active, or a trial operation is prepared or active;
- the laser backend is not `nidaq`;
- the pellet firmware does not advertise `finite_stim3_pulse`, which
  `FirmwareCompatibilityPolicy` already reports;
- the selected channel has no hardware mapping.

The prepared profile is released in a `finally`, so a failed test never leaves
an armed analog output behind.

### UI

Each `_LaserChannelTab` gains one row: a combo listing saved laser profiles
targeting that channel, a `Test stim (hardware trigger)` button, and a result
label showing the measured interval or the refusal reason. The work runs
through the existing `_start_operation`, so it is already off the GUI thread
and already disabled during capture by `set_controls_enabled`.

## 4. Firmware follow-up

Committed work, tracked in `todo.md` when feature 3 lands.

STIM0 and STIM1 are not available. The board device tree assigns `gpiob 11` and
`gpiob 12` to the tone generator as the TTL confirmations the NI-DAQ records
(Tone 1 at 5 kHz, Tone 2 at 6 kHz), and `tone_generator.c` drives them
directly. Using them for stim would contend with tone confirmation on the same
pins.

The second timed line is therefore STIM2, `gpiob 13`.

In `reachAQ-hardware`, a separate repository checked out under `temp/`:

- extend the `GPIOPulse` command to accept STIM2 as well as STIM3;
- bump the pellet firmware version and advertise a new capability.

In this repository:

- add the capability to `CAPABILITY_BITS` in
  `tools/acquisition/model/firmware_compatibility.py`;
- add the new firmware version to `config/pellet-firmware-compatibility.yaml`;
- relax the `STIMULUS_4`-only check in `CanInterface.pulse_digital_output`
  behind that capability, and the matching check in `emulation_interface.py`;
- add line selection to the laser profile and to the stim test button.

Then flash the board and verify on the rig. Until the capability is reported,
behaviour is unchanged.

## Testing

Every test run, suite, and application launch happens on `christielab10`. The
local Windows checkout is for reading and editing only; it lacks the
dependencies and the hardware.

Per feature: push from the worktree, pull on the rig, run the touched suites
with `~/anaconda3/envs/reachaq/bin/python -m pytest`, launch the application on
the rig, then cherry-pick onto `demo-mode-and-presentation`.

- **Panel** - offscreen widget tests for `DetachablePanelHost` state
  transitions and placeholder behaviour, following the pattern of the existing
  UI tests. Visual confirmation on the rig that the overlay sits above the GL
  camera views and that the detached window survives a restart.
- **Sets** - unit tests for the compiler: ordering, repeat expansion, id
  remapping, shuffle reproducibility from a stored seed, precedence
  preservation against a hand-resolved expectation, and the revision-mismatch
  error. Round-trip tests for both repositories.
- **Stim test** - unit tests for each guard and for the release-on-failure
  path, against the existing laser and hardware fakes. Bench verification on
  the rig with the laser on a scope, confirming the analog output starts on the
  board trigger edge rather than on a host call.

## Open risks

- `Qt.Tool` overlay placement under the rig's window manager is unproven and is
  the first thing verified on the rig.
- The physical BNC-to-STIM mapping on the pellet board is not resolved from the
  schematic. It does not block any software work, but it must be confirmed
  before the firmware change is wired to a connector.
