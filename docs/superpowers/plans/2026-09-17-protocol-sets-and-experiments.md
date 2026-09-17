# Protocol Sets and Experiment Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save reusable groups of trials as sets, compose several sets into an experiment in a chosen order with repeats and optional within-set shuffling, and compile that experiment into an ordinary runnable protocol.

**Architecture:** A set is its own file holding trial content only. An experiment is an ordered list of pinned set references. A pure compiler flattens an experiment into a `TrialProtocolDocument`, emitting one **epoch** per set appearance so the existing resolution order, the Sources column, and session evidence all keep working with no change to the runner, the schedule, or session persistence.

**Tech Stack:** Python 3.8, PySide6, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-protocol-panel-sets-stim-test-design.md`

**Depends on:** `2026-09-17-detachable-protocol-panel.md`. Task 5's sidebar is unusable in a 430 px column, so land the panel first.

## Global Constraints

- **Python 3.8.20.** The rig runs `~/anaconda3/envs/reachaq/bin/python`, which is 3.8.20, even though `pyproject.toml` says `requires-python = ">= 3.10"`. Write 3.8-compatible code: `from __future__ import annotations` at the top of every new module, `typing.Optional` / `typing.Tuple` rather than `X | Y` and `tuple[...]`, no `match` statements, no `dict | dict`.
- **All test runs, suites and app launches happen on `christielab10`.** Never run pytest locally.
- Rig SSH: `ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10`. Repo at `~/Documents/reachAQ`.
- **Never kill rig processes by pattern-matching the interpreter path.** Kill by explicit PID.
- Branch `feature-dev`, worked in the worktree `../reachAQ-feature`.
- **Identifiers must match `^[a-z0-9][a-z0-9._-]{0,63}$`** — the rule in `trial_protocol_schedule._IDENTIFIER`. `#` is illegal and 64 characters is the hard ceiling. Every generated epoch and scope name goes through `_bounded_name`.
- **Do not change `PROTOCOL_SCHEMA_VERSION` or add fields to `TrialProtocolDocument`.** The runner and session evidence read those documents; machine-readable provenance lives in the experiment file instead.
- Test files are `tests/<name>_test.py`; UI tests set `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` before any Qt import with `# noqa: E402` on the imports after it.

---

### Task 1: The set document

**Files:**
- Create: `tools/acquisition/model/trial_protocol_set.py`
- Modify: `tools/acquisition/model/trial_protocol_schedule.py` (add one public alias after `_identifier`, around line 167)
- Test: `tests/trial_protocol_set_test.py`

**Interfaces:**
- Consumes: `ProtocolPatch`, `ProtocolScope`, `TrialOverride`, `DEFAULT_TRIAL_COUNT` from `trial_protocol_schedule`.
- Produces:
  - `normalize_identifier(value: str, *, field: str, allow_empty: bool = False) -> str` in `trial_protocol_schedule`
  - `SET_SCHEMA_VERSION = 1`
  - `TrialProtocolSet(set_id, name, revision=1, trial_count=DEFAULT_TRIAL_COUNT, defaults=ProtocolPatch(), bulk_overrides=(), trial_overrides=(), description="", schema_version=SET_SCHEMA_VERSION)`
  - `TrialProtocolSet.to_record() -> dict`, `TrialProtocolSet.from_record(record) -> TrialProtocolSet`

- [ ] **Step 1: Write the failing tests**

Create `tests/trial_protocol_set_test.py`:

```python
import pytest

from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
)
from tools.acquisition.model.trial_protocol_set import (
    SET_SCHEMA_VERSION,
    TrialProtocolSet,
)


def make_set(**overrides):
    values = dict(
        set_id="baseline",
        name="Baseline",
        trial_count=4,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )
    values.update(overrides)
    return TrialProtocolSet(**values)


def test_a_set_round_trips_through_its_record():
    original = make_set(
        bulk_overrides=(
            ProtocolScope.create("late", [3, 4], {"pre_reveal_ms": 50}),
        ),
        trial_overrides=(TrialOverride.create(2, {"enabled": False}),),
        description="Four warm-up trials",
    )

    restored = TrialProtocolSet.from_record(original.to_record())

    assert restored == original


def test_set_id_is_normalised():
    assert make_set(set_id="  BaseLine  ").set_id == "baseline"


def test_a_set_rejects_an_empty_name():
    with pytest.raises(ValueError, match="name cannot be empty"):
        make_set(name="   ")


def test_a_set_rejects_a_scope_outside_its_own_range():
    with pytest.raises(ValueError, match="out-of-range"):
        make_set(
            bulk_overrides=(ProtocolScope.create("late", [9], {"enabled": True}),)
        )


def test_a_set_rejects_a_scope_that_names_a_parent_epoch():
    with pytest.raises(ValueError, match="parent epoch"):
        make_set(
            bulk_overrides=(
                ProtocolScope.create(
                    "late", [1], {"enabled": True}, parent_epoch="phase"
                ),
            )
        )


def test_a_set_rejects_duplicate_trial_overrides():
    with pytest.raises(ValueError, match="Duplicate trial override"):
        make_set(
            trial_overrides=(
                TrialOverride.create(2, {"enabled": False}),
                TrialOverride.create(2, {"enabled": True}),
            )
        )


def test_a_set_refuses_an_unknown_schema():
    record = make_set().to_record()
    record["schema_version"] = SET_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="Unsupported set schema"):
        TrialProtocolSet.from_record(record)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/trial_protocol_set_test.py -v'
```

Expected: collection error, `ModuleNotFoundError: No module named 'tools.acquisition.model.trial_protocol_set'`.

- [ ] **Step 3: Expose the shared identifier rule**

In `tools/acquisition/model/trial_protocol_schedule.py`, immediately after the `_identifier` function, add:

```python
def normalize_identifier(
    value: str,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    """Public name for the identifier rule protocol documents already use."""
    return _identifier(value, field=field, allow_empty=allow_empty)
```

This is an alias, not a behaviour change. It exists so sibling modules do not import a private name.

- [ ] **Step 4: Write the set document**

Create `tools/acquisition/model/trial_protocol_set.py`:

```python
"""Reusable trial groups, composed into experiments by value.

A set holds trial content only. It deliberately has no epochs and no blocks:
those slots belong to the compiled experiment, which uses one epoch per set
appearance to carry provenance through the existing resolution order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from tools.acquisition.model.trial_protocol_schedule import (
    DEFAULT_TRIAL_COUNT,
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
    normalize_identifier,
)


SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrialProtocolSet:
    """One reusable, independently saved group of trials."""

    set_id: str
    name: str
    revision: int = 1
    trial_count: int = DEFAULT_TRIAL_COUNT
    defaults: ProtocolPatch = ProtocolPatch()
    bulk_overrides: Tuple[ProtocolScope, ...] = ()
    trial_overrides: Tuple[TrialOverride, ...] = ()
    description: str = ""
    schema_version: int = SET_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(
            self, "set_id", normalize_identifier(self.set_id, field="set_id")
        )
        if not str(self.name).strip():
            raise ValueError("Set name cannot be empty")
        if self.schema_version != SET_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported set schema {}; expected {}".format(
                    self.schema_version, SET_SCHEMA_VERSION
                )
            )
        if int(self.revision) < 1:
            raise ValueError("Set revision must be positive")
        if not 1 <= int(self.trial_count) <= 100_000:
            raise ValueError("trial_count must be between 1 and 100000")
        object.__setattr__(self, "bulk_overrides", tuple(self.bulk_overrides))
        object.__setattr__(self, "trial_overrides", tuple(self.trial_overrides))
        self._validate()

    def _validate(self) -> None:
        limit = set(range(1, int(self.trial_count) + 1))
        for scope in self.bulk_overrides:
            if scope.parent_epoch:
                raise ValueError(
                    "Set scope {} cannot name a parent epoch; epochs belong to "
                    "the compiled experiment".format(scope.name)
                )
            if not set(scope.trial_ids) <= limit:
                raise ValueError(
                    "Set scope {} contains out-of-range trials".format(scope.name)
                )
        seen = set()
        for override in self.trial_overrides:
            if override.trial_id not in limit:
                raise ValueError("Set trial override is outside the set range")
            if override.trial_id in seen:
                raise ValueError(
                    "Duplicate trial override for trial {}".format(override.trial_id)
                )
            seen.add(override.trial_id)

    def to_record(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "set_id": self.set_id,
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
            "trial_count": self.trial_count,
            "defaults": self.defaults.to_mapping(),
            "bulk_overrides": [item.to_record() for item in self.bulk_overrides],
            "trial_overrides": [item.to_record() for item in self.trial_overrides],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "TrialProtocolSet":
        stored = int(record.get("schema_version", -1))
        if stored != SET_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported set schema {}; expected {}".format(
                    stored, SET_SCHEMA_VERSION
                )
            )
        return cls(
            schema_version=SET_SCHEMA_VERSION,
            set_id=record["set_id"],
            name=record["name"],
            description=record.get("description", ""),
            revision=int(record.get("revision", 1)),
            trial_count=int(record.get("trial_count", DEFAULT_TRIAL_COUNT)),
            defaults=ProtocolPatch.from_mapping(record.get("defaults", {})),
            bulk_overrides=tuple(
                ProtocolScope.from_record(item)
                for item in record.get("bulk_overrides", ())
            ),
            trial_overrides=tuple(
                TrialOverride.create(item["trial_id"], item.get("values", {}))
                for item in record.get("trial_overrides", ())
            ),
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/trial_protocol_set_test.py -v'
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add tools/acquisition/model/trial_protocol_set.py tools/acquisition/model/trial_protocol_schedule.py tests/trial_protocol_set_test.py
git commit -m "feat(protocol): add a reusable trial set document"
```

---

### Task 2: The experiment composition

**Files:**
- Modify: `tools/acquisition/model/trial_protocol_set.py` (append)
- Test: `tests/experiment_composition_test.py`

**Interfaces:**
- Consumes: `normalize_identifier` from Task 1.
- Produces:
  - `EXPERIMENT_SCHEMA_VERSION = 1`, `MAX_SET_REPEAT = 100`
  - `ExperimentSetEntry(set_id, set_revision, repeat=1, shuffle_trials=False, shuffle_seed=None)` with `to_record()` / `from_record()`
  - `ExperimentComposition(experiment_id, name, revision=1, entries=(), description="", schema_version=EXPERIMENT_SCHEMA_VERSION)` with `to_record()` / `from_record()`

- [ ] **Step 1: Write the failing tests**

Create `tests/experiment_composition_test.py`:

```python
import pytest

from tools.acquisition.model.trial_protocol_set import (
    MAX_SET_REPEAT,
    ExperimentComposition,
    ExperimentSetEntry,
)


def make_composition(**overrides):
    values = dict(
        experiment_id="day3",
        name="Day 3",
        entries=(
            ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=2),
            ExperimentSetEntry(
                set_id="stim", set_revision=3, shuffle_trials=True, shuffle_seed=7
            ),
        ),
    )
    values.update(overrides)
    return ExperimentComposition(**values)


def test_a_composition_round_trips_through_its_record():
    original = make_composition(description="ABAB day")

    restored = ExperimentComposition.from_record(original.to_record())

    assert restored == original


def test_an_empty_composition_is_allowed_so_it_can_be_built_up():
    assert ExperimentComposition(
        experiment_id="draft", name="Draft", entries=()
    ).entries == ()


def test_an_entry_rejects_a_repeat_below_one():
    with pytest.raises(ValueError, match="repeat must be between"):
        ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=0)


def test_an_entry_rejects_a_repeat_above_the_ceiling():
    with pytest.raises(ValueError, match="repeat must be between"):
        ExperimentSetEntry(
            set_id="baseline", set_revision=1, repeat=MAX_SET_REPEAT + 1
        )


def test_an_entry_rejects_a_non_positive_set_revision():
    with pytest.raises(ValueError, match="revision must be positive"):
        ExperimentSetEntry(set_id="baseline", set_revision=0)


def test_experiment_id_is_normalised():
    assert make_composition(experiment_id="  Day3  ").experiment_id == "day3"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/experiment_composition_test.py -v'
```

Expected: FAIL with `ImportError: cannot import name 'ExperimentSetEntry'`.

- [ ] **Step 3: Write the implementation**

Append to `tools/acquisition/model/trial_protocol_set.py`, and add `Optional` to the `typing` import at the top:

```python
EXPERIMENT_SCHEMA_VERSION = 1

#: Upper bound on how many times one set may appear through a single entry.
#: Large enough for any real session, small enough that a typo cannot compile
#: a million-trial document.
MAX_SET_REPEAT = 100


@dataclass(frozen=True)
class ExperimentSetEntry:
    """One appearance of a set inside an experiment, pinned to a revision."""

    set_id: str
    set_revision: int
    repeat: int = 1
    shuffle_trials: bool = False
    shuffle_seed: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(
            self, "set_id", normalize_identifier(self.set_id, field="set_id")
        )
        if int(self.set_revision) < 1:
            raise ValueError("Set revision must be positive")
        if not 1 <= int(self.repeat) <= MAX_SET_REPEAT:
            raise ValueError(
                "Set repeat must be between 1 and {}".format(MAX_SET_REPEAT)
            )
        if self.shuffle_seed is not None:
            object.__setattr__(self, "shuffle_seed", int(self.shuffle_seed))

    def to_record(self) -> dict:
        return {
            "set_id": self.set_id,
            "set_revision": int(self.set_revision),
            "repeat": int(self.repeat),
            "shuffle_trials": bool(self.shuffle_trials),
            "shuffle_seed": self.shuffle_seed,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ExperimentSetEntry":
        return cls(
            set_id=record["set_id"],
            set_revision=int(record["set_revision"]),
            repeat=int(record.get("repeat", 1)),
            shuffle_trials=bool(record.get("shuffle_trials", False)),
            shuffle_seed=record.get("shuffle_seed"),
        )


@dataclass(frozen=True)
class ExperimentComposition:
    """An ordered list of set appearances that compiles to one protocol."""

    experiment_id: str
    name: str
    revision: int = 1
    entries: Tuple[ExperimentSetEntry, ...] = ()
    description: str = ""
    schema_version: int = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(
            self,
            "experiment_id",
            normalize_identifier(self.experiment_id, field="experiment_id"),
        )
        if not str(self.name).strip():
            raise ValueError("Experiment name cannot be empty")
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported experiment schema {}; expected {}".format(
                    self.schema_version, EXPERIMENT_SCHEMA_VERSION
                )
            )
        if int(self.revision) < 1:
            raise ValueError("Experiment revision must be positive")
        # An empty experiment is allowed so it can be built up in the editor.
        # compile_experiment refuses to compile one.
        object.__setattr__(self, "entries", tuple(self.entries))

    def to_record(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
            "entries": [item.to_record() for item in self.entries],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ExperimentComposition":
        stored = int(record.get("schema_version", -1))
        if stored != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported experiment schema {}; expected {}".format(
                    stored, EXPERIMENT_SCHEMA_VERSION
                )
            )
        return cls(
            schema_version=EXPERIMENT_SCHEMA_VERSION,
            experiment_id=record["experiment_id"],
            name=record["name"],
            description=record.get("description", ""),
            revision=int(record.get("revision", 1)),
            entries=tuple(
                ExperimentSetEntry.from_record(item)
                for item in record.get("entries", ())
            ),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/experiment_composition_test.py -v'
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/model/trial_protocol_set.py tests/experiment_composition_test.py
git commit -m "feat(protocol): describe an experiment as ordered set appearances"
```

---

### Task 3: The compiler

The core of this feature. Everything else is storage and buttons.

**Files:**
- Create: `tools/acquisition/model/experiment_compiler.py`
- Test: `tests/experiment_compiler_test.py`

**Interfaces:**
- Consumes: `TrialProtocolSet`, `ExperimentComposition`, `ExperimentSetEntry` from Tasks 1 and 2; `ProtocolScope`, `TrialOverride`, `TrialProtocolDocument` from `trial_protocol_schedule`.
- Produces:
  - `CompiledInstance(epoch_name: str, set_id: str, set_revision: int, repeat_index: int, shuffle_seed: Optional[int], trial_ids: Tuple[int, ...])` with `to_record()`
  - `CompiledExperiment(document: TrialProtocolDocument, instances: Tuple[CompiledInstance, ...], composition: ExperimentComposition)`
  - `compile_experiment(composition: ExperimentComposition, set_library: Mapping[str, TrialProtocolSet]) -> CompiledExperiment`

  `set_library` is any mapping from `set_id` to `TrialProtocolSet`; Task 4's repository supplies one.

  **Refinement of the spec:** the spec described `compile_experiment` returning a bare `TrialProtocolDocument` and stamping provenance into it. It returns `CompiledExperiment` instead, because the compiler must hand back seeds it drew so the caller can persist them, and because stamping new fields into `TrialProtocolDocument` would require bumping `PROTOCOL_SCHEMA_VERSION`, which the runner and session evidence both read. Machine-readable provenance lives on `CompiledExperiment.instances`; the document carries a human-readable `description`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiment_compiler_test.py`:

```python
import pytest

from tools.acquisition.model.experiment_compiler import compile_experiment
from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
)
from tools.acquisition.model.trial_protocol_set import (
    ExperimentComposition,
    ExperimentSetEntry,
    TrialProtocolSet,
)


def make_library():
    baseline = TrialProtocolSet(
        set_id="baseline",
        name="Baseline",
        revision=1,
        trial_count=2,
        defaults=ProtocolPatch.from_mapping({"enabled": True, "pre_reveal_ms": 10}),
    )
    stim = TrialProtocolSet(
        set_id="stim",
        name="Stim",
        revision=3,
        trial_count=3,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
        bulk_overrides=(
            ProtocolScope.create("tail", [2, 3], {"pre_reveal_ms": 40}),
        ),
        trial_overrides=(TrialOverride.create(1, {"pre_reveal_ms": 99}),),
    )
    return {"baseline": baseline, "stim": stim}


def compose(*entries, **overrides):
    values = dict(experiment_id="day3", name="Day 3", entries=entries)
    values.update(overrides)
    return ExperimentComposition(**values)


def test_entries_concatenate_in_order():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(set_id="stim", set_revision=3),
    )

    result = compile_experiment(composition, make_library())

    assert result.document.trial_count == 5
    assert [item.epoch_name for item in result.instances] == ["baseline", "stim"]
    assert result.instances[0].trial_ids == (1, 2)
    assert result.instances[1].trial_ids == (3, 4, 5)


def test_repeat_expands_into_separately_named_instances():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=3)
    )

    result = compile_experiment(composition, make_library())

    assert result.document.trial_count == 6
    assert [item.epoch_name for item in result.instances] == [
        "baseline-1",
        "baseline-2",
        "baseline-3",
    ]


def test_the_same_set_in_two_entries_gets_distinct_epoch_names():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(set_id="stim", set_revision=3),
        ExperimentSetEntry(set_id="baseline", set_revision=1),
    )

    result = compile_experiment(composition, make_library())

    names = [item.epoch_name for item in result.instances]
    assert names == ["baseline-1", "stim", "baseline-2"]
    assert len(set(names)) == len(names)


def test_within_set_precedence_survives_compilation():
    composition = compose(ExperimentSetEntry(set_id="stim", set_revision=3))

    result = compile_experiment(composition, make_library())
    rows = result.document.resolve()

    # Trial 1 takes the set's own trial override, which beats the bulk scope.
    assert rows[0].row.pre_reveal_ms == 99
    # Trials 2 and 3 take the bulk scope, which beats the set defaults.
    assert rows[1].row.pre_reveal_ms == 40
    assert rows[2].row.pre_reveal_ms == 40


def test_each_trial_reports_its_set_as_the_source_of_the_set_defaults():
    composition = compose(ExperimentSetEntry(set_id="baseline", set_revision=1))

    result = compile_experiment(composition, make_library())
    sources = dict(result.document.resolve()[0].sources)

    assert sources["enabled"] == "epoch:baseline"


def test_shuffling_is_reproducible_from_the_stored_seed():
    composition = compose(
        ExperimentSetEntry(
            set_id="stim", set_revision=3, shuffle_trials=True, shuffle_seed=11
        )
    )
    library = make_library()

    first = compile_experiment(composition, library)
    second = compile_experiment(composition, library)

    def overridden_trial(result):
        return [item.trial_id for item in result.document.trial_overrides]

    assert overridden_trial(first) == overridden_trial(second)


def test_shuffling_without_a_seed_draws_one_and_returns_it():
    composition = compose(
        ExperimentSetEntry(set_id="stim", set_revision=3, shuffle_trials=True)
    )

    result = compile_experiment(composition, make_library())

    seed = result.composition.entries[0].shuffle_seed
    assert seed is not None
    assert result.instances[0].shuffle_seed == seed


def test_shuffling_permutes_within_the_instance_range_only():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(
            set_id="stim", set_revision=3, shuffle_trials=True, shuffle_seed=5
        ),
    )

    result = compile_experiment(composition, make_library())

    assert result.instances[1].trial_ids == (3, 4, 5)
    for override in result.document.trial_overrides:
        assert 3 <= override.trial_id <= 5


def test_a_revision_mismatch_names_the_offending_entry():
    composition = compose(ExperimentSetEntry(set_id="stim", set_revision=2))

    with pytest.raises(ValueError, match="Entry 1 pins set 'stim' revision 2"):
        compile_experiment(composition, make_library())


def test_an_unknown_set_names_the_offending_entry():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(set_id="missing", set_revision=1),
    )

    with pytest.raises(ValueError, match="Entry 2 references unknown set 'missing'"):
        compile_experiment(composition, make_library())


def test_an_empty_experiment_refuses_to_compile():
    with pytest.raises(ValueError, match="no set entries"):
        compile_experiment(compose(), make_library())


def test_the_compiled_document_keeps_the_experiment_identity():
    composition = compose(ExperimentSetEntry(set_id="baseline", set_revision=1))

    result = compile_experiment(composition, make_library())

    assert result.document.protocol_id == "day3"
    assert result.document.name == "Day 3"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/experiment_compiler_test.py -v'
```

Expected: collection error, `ModuleNotFoundError: No module named 'tools.acquisition.model.experiment_compiler'`.

- [ ] **Step 3: Write the compiler**

Create `tools/acquisition/model/experiment_compiler.py`:

```python
"""Flatten an ordered set composition into one runnable protocol document.

The compiler emits one epoch per set appearance. TrialProtocolDocument.resolve
applies patches in the order defaults, epochs, blocks, bulk overrides, trial
overrides, so mapping a set's own defaults onto its epoch and its own scopes
onto the document's bulk and trial layers preserves within-set precedence
exactly, and every trial keeps a source label naming the set it came from.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Tuple

from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolScope,
    TrialOverride,
    TrialProtocolDocument,
)
from tools.acquisition.model.trial_protocol_set import (
    ExperimentComposition,
    TrialProtocolSet,
)


#: The identifier rule in trial_protocol_schedule caps names at 64 characters.
_MAX_NAME = 64


def _bounded_name(*parts: object) -> str:
    """Join identifier parts with '-', keeping a valid identifier <= 64 chars."""
    joined = "-".join(str(part) for part in parts if str(part) != "")
    if len(joined) <= _MAX_NAME:
        return joined
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    return "{}-{}".format(joined[: _MAX_NAME - 9], digest)


@dataclass(frozen=True)
class CompiledInstance:
    """Provenance for one appearance of a set in a compiled experiment."""

    epoch_name: str
    set_id: str
    set_revision: int
    repeat_index: int
    shuffle_seed: Optional[int]
    trial_ids: Tuple[int, ...]

    def to_record(self) -> dict:
        return {
            "epoch_name": self.epoch_name,
            "set_id": self.set_id,
            "set_revision": self.set_revision,
            "repeat_index": self.repeat_index,
            "shuffle_seed": self.shuffle_seed,
            "trial_ids": list(self.trial_ids),
        }


@dataclass(frozen=True)
class CompiledExperiment:
    """A runnable document plus the provenance of how it was produced.

    ``composition`` is the input with any drawn seed written back, so the
    caller can save it and reproduce this exact compile later.
    """

    document: TrialProtocolDocument
    instances: Tuple[CompiledInstance, ...]
    composition: ExperimentComposition


def _local_to_global(
    count: int,
    base: int,
    shuffle: bool,
    seed: Optional[int],
    entry_index: int,
    repeat_index: int,
) -> Dict[int, int]:
    """Map a set's local trial ids onto the contiguous global range at base."""
    order = list(range(1, int(count) + 1))
    if shuffle:
        # A string seed keeps this reproducible across processes; random.Random
        # does not accept a tuple. Including the entry and repeat indices makes
        # repeats of one entry differ while the whole compile stays derivable
        # from the single seed stored on the entry.
        generator = random.Random(
            "{}:{}:{}".format(seed, entry_index, repeat_index)
        )
        generator.shuffle(order)
    return {local: base + position for position, local in enumerate(order)}


def _describe(
    composition: ExperimentComposition,
    instances: Tuple[CompiledInstance, ...],
) -> str:
    parts = ", ".join(
        "{} rev {}".format(item.set_id, item.set_revision) for item in instances
    )
    return "Compiled from experiment {} revision {}: {}".format(
        composition.experiment_id, composition.revision, parts
    )


def compile_experiment(
    composition: ExperimentComposition,
    set_library: Mapping[str, TrialProtocolSet],
) -> CompiledExperiment:
    """Flatten an experiment into one document the existing runner accepts."""
    if not composition.entries:
        raise ValueError("Cannot compile an experiment with no set entries")

    # Pre-count appearances so a set used once keeps its plain name and a set
    # used more than once is numbered, including across separate entries.
    totals = Counter()
    for entry in composition.entries:
        totals[entry.set_id] += int(entry.repeat)
    seen = Counter()

    epochs: List[ProtocolScope] = []
    bulk_overrides: List[ProtocolScope] = []
    trial_overrides: List[TrialOverride] = []
    instances: List[CompiledInstance] = []
    resolved_entries = []
    next_trial_id = 1

    for entry_index, entry in enumerate(composition.entries):
        protocol_set = set_library.get(entry.set_id)
        if protocol_set is None:
            raise ValueError(
                "Entry {} references unknown set {!r}".format(
                    entry_index + 1, entry.set_id
                )
            )
        if int(protocol_set.revision) != int(entry.set_revision):
            raise ValueError(
                "Entry {} pins set {!r} revision {}, but the library holds "
                "revision {}".format(
                    entry_index + 1,
                    entry.set_id,
                    entry.set_revision,
                    protocol_set.revision,
                )
            )

        seed = entry.shuffle_seed
        if entry.shuffle_trials and seed is None:
            # Draw once and write it back, so a compile is never irreproducible.
            seed = random.randrange(1, 2 ** 31)
        resolved_entries.append(replace(entry, shuffle_seed=seed))

        for repeat_index in range(1, int(entry.repeat) + 1):
            base = next_trial_id
            count = int(protocol_set.trial_count)
            seen[entry.set_id] += 1
            epoch_name = (
                protocol_set.set_id
                if totals[entry.set_id] == 1
                else _bounded_name(protocol_set.set_id, seen[entry.set_id])
            )
            mapping = _local_to_global(
                count, base, entry.shuffle_trials, seed, entry_index, repeat_index
            )
            trial_ids = tuple(sorted(mapping.values()))

            epochs.append(
                ProtocolScope.create(
                    epoch_name, trial_ids, protocol_set.defaults.to_mapping()
                )
            )
            for scope in protocol_set.bulk_overrides:
                bulk_overrides.append(
                    ProtocolScope.create(
                        _bounded_name(epoch_name, scope.name),
                        [mapping[local] for local in scope.trial_ids],
                        scope.patch.to_mapping(),
                    )
                )
            for override in protocol_set.trial_overrides:
                trial_overrides.append(
                    TrialOverride.create(
                        mapping[override.trial_id], override.patch.to_mapping()
                    )
                )
            instances.append(
                CompiledInstance(
                    epoch_name=epoch_name,
                    set_id=protocol_set.set_id,
                    set_revision=int(protocol_set.revision),
                    repeat_index=repeat_index,
                    shuffle_seed=seed if entry.shuffle_trials else None,
                    trial_ids=trial_ids,
                )
            )
            next_trial_id += count

    frozen_instances = tuple(instances)
    document = TrialProtocolDocument(
        protocol_id=composition.experiment_id,
        name=composition.name,
        trial_count=next_trial_id - 1,
        epochs=tuple(epochs),
        bulk_overrides=tuple(bulk_overrides),
        trial_overrides=tuple(trial_overrides),
        description=_describe(composition, frozen_instances),
    )
    return CompiledExperiment(
        document=document,
        instances=frozen_instances,
        composition=replace(composition, entries=tuple(resolved_entries)),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/experiment_compiler_test.py -v'
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/model/experiment_compiler.py tests/experiment_compiler_test.py
git commit -m "feat(protocol): compile ordered set compositions into one document"
```

---

### Task 4: Storage and app model wiring

**Files:**
- Create: `tools/acquisition/model/trial_protocol_set_repository.py`
- Modify: `tools/acquisition/model/app_model.py` — the constructor block at lines 515-524, and the configuration-load block at lines 7040-7055
- Test: `tests/trial_protocol_set_repository_test.py`

**Interfaces:**
- Consumes: `TrialProtocolSet`, `ExperimentComposition` from Tasks 1-2; `CompiledExperiment` from Task 3.
- Produces:
  - `TrialProtocolSetRepository(root: Path)` with `documents`, `errors`, `get(set_id)`, `reload()`, `save(document)`, `library() -> Dict[str, TrialProtocolSet]`
  - `ExperimentCompositionRepository(root: Path)` with `documents`, `errors`, `get(experiment_id)`, `reload()`, `save(document)`
  - `AppModel.trial_protocol_sets -> Tuple[TrialProtocolSet, ...]`
  - `AppModel.experiment_compositions -> Tuple[ExperimentComposition, ...]`
  - `AppModel.compile_and_save_experiment(experiment_id: str) -> TrialProtocolDocument`

  Both repositories mirror `TrialProtocolRepository`'s shape, including its atomic write and its rule that a corrupt file keeps its cached version rather than disappearing. `save` auto-increments the revision the same way, so a recompiled experiment replaces its protocol in place.

- [ ] **Step 1: Write the failing tests**

Create `tests/trial_protocol_set_repository_test.py`:

```python
import json

import pytest

from tools.acquisition.model.trial_protocol_schedule import ProtocolPatch
from tools.acquisition.model.trial_protocol_set import (
    ExperimentComposition,
    ExperimentSetEntry,
    TrialProtocolSet,
)
from tools.acquisition.model.trial_protocol_set_repository import (
    ExperimentCompositionRepository,
    TrialProtocolSetRepository,
)


def make_set(set_id="baseline"):
    return TrialProtocolSet(
        set_id=set_id,
        name="Baseline",
        trial_count=2,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )


def test_a_saved_set_reloads_from_disk(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set())

    reloaded = TrialProtocolSetRepository(tmp_path)
    reloaded.reload()

    assert reloaded.get("baseline").name == "Baseline"


def test_saving_increments_the_revision(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()

    first = repository.save(make_set())
    second = repository.save(make_set())

    assert first.revision == 1
    assert second.revision == 2


def test_library_maps_ids_to_sets(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set("baseline"))
    repository.save(make_set("stim"))

    library = repository.library()

    assert sorted(library) == ["baseline", "stim"]


def test_a_corrupt_file_is_reported_and_does_not_lose_the_cached_set(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set())
    (tmp_path / "baseline.json").write_text("{ not json", encoding="utf-8")

    repository.reload()

    assert repository.errors
    assert repository.get("baseline") is not None


def test_a_filename_that_disagrees_with_its_id_is_refused(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set())
    record = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    (tmp_path / "wrong-name.json").write_text(json.dumps(record), encoding="utf-8")

    repository.reload()

    assert any("filename must be" in message for message in repository.errors.values())


def test_a_saved_experiment_reloads_from_disk(tmp_path):
    repository = ExperimentCompositionRepository(tmp_path)
    repository.reload()
    repository.save(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=1),),
        )
    )

    reloaded = ExperimentCompositionRepository(tmp_path)
    reloaded.reload()

    assert reloaded.get("day3").entries[0].set_id == "baseline"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/trial_protocol_set_repository_test.py -v'
```

Expected: collection error, `ModuleNotFoundError: No module named 'tools.acquisition.model.trial_protocol_set_repository'`.

- [ ] **Step 3: Write the repositories**

Create `tools/acquisition/model/trial_protocol_set_repository.py`:

```python
"""Atomic storage for reusable trial sets and the experiments built on them.

Both repositories mirror TrialProtocolRepository: one file per document, an
atomic write, and a reload that keeps the cached copy of a file that has become
unreadable rather than making the document vanish from the library.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from tools.acquisition.model.atomic_session_io import atomic_write_json
from tools.acquisition.model.trial_protocol_set import (
    ExperimentComposition,
    TrialProtocolSet,
)


class _JsonDocumentRepository:
    """Shared mechanics for a directory of individually isolated documents."""

    #: Subclasses set these three.
    document_type = None
    id_field = ""
    label = ""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._documents = {}
        self._path_to_id: Dict[Path, str] = {}
        self._errors: Dict[str, str] = {}

    @property
    def documents(self) -> Tuple:
        with self._lock:
            return tuple(self._documents[key] for key in sorted(self._documents))

    @property
    def errors(self) -> Mapping[str, str]:
        with self._lock:
            return dict(self._errors)

    def get(self, document_id: str):
        with self._lock:
            return self._documents.get(str(document_id).strip().lower())

    def reload(self) -> Tuple:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            previous = dict(self._documents)
            previous_paths = dict(self._path_to_id)
            loaded = {}
            path_to_id: Dict[Path, str] = {}
            errors: Dict[str, str] = {}
            for path in sorted(self.root.glob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as stream:
                        document = self.document_type.from_record(json.load(stream))
                    identifier = getattr(document, self.id_field)
                    expected_name = "{}.json".format(identifier)
                    if path.name != expected_name:
                        raise ValueError(
                            "filename must be {!r} for {} {!r}".format(
                                expected_name, self.label, identifier
                            )
                        )
                    if identifier in loaded:
                        raise ValueError(
                            "duplicate {} ID {!r}".format(self.label, identifier)
                        )
                    loaded[identifier] = document
                    path_to_id[path.resolve()] = identifier
                except Exception as error:
                    errors[path.as_posix()] = "{}: {}".format(
                        type(error).__name__, error
                    )
                    cached_id = previous_paths.get(path.resolve())
                    if cached_id is not None and cached_id in previous:
                        loaded.setdefault(cached_id, previous[cached_id])
                        path_to_id[path.resolve()] = cached_id
            self._documents = loaded
            self._path_to_id = path_to_id
            self._errors = errors
            return self.documents

    def save(self, document):
        """Save the next revision, mirroring TrialProtocolRepository.save."""
        with self._lock:
            identifier = getattr(document, self.id_field)
            current = self._documents.get(identifier)
            if current is None:
                next_revision = max(1, int(document.revision))
            else:
                next_revision = int(current.revision) + 1
            saved = replace(document, revision=next_revision)
            path = self.root / "{}.json".format(identifier)
            atomic_write_json(path, saved.to_record())
            self._documents[identifier] = saved
            self._path_to_id[path.resolve()] = identifier
            self._errors.pop(path.as_posix(), None)
            return saved


class TrialProtocolSetRepository(_JsonDocumentRepository):
    document_type = TrialProtocolSet
    id_field = "set_id"
    label = "set"

    def library(self) -> Dict[str, TrialProtocolSet]:
        """Mapping the compiler accepts as its set_library argument."""
        with self._lock:
            return dict(self._documents)


class ExperimentCompositionRepository(_JsonDocumentRepository):
    document_type = ExperimentComposition
    id_field = "experiment_id"
    label = "experiment"
```

- [ ] **Step 4: Run the repository tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/trial_protocol_set_repository_test.py -v'
```

Expected: 6 passed.

- [ ] **Step 5: Wire both repositories into AppModel**

In `tools/acquisition/model/app_model.py`, add to the imports beside the existing `TrialProtocolRepository` import:

```python
from tools.acquisition.model.trial_protocol_set_repository import (
    ExperimentCompositionRepository,
    TrialProtocolSetRepository,
)
from tools.acquisition.model.experiment_compiler import compile_experiment
```

In the constructor, immediately after `self._trial_protocol_repository.reload()`:

```python
        self._trial_protocol_set_repository = TrialProtocolSetRepository(
            Path(preferences.configuration_location).expanduser()
            / "trial_protocol_sets"
        )
        self._trial_protocol_set_repository.reload()
        self._experiment_repository = ExperimentCompositionRepository(
            Path(preferences.configuration_location).expanduser()
            / "trial_experiments"
        )
        self._experiment_repository.reload()
```

In the configuration-load block, inside the same `with self._trial_protocol_lock:` that rebinds `self._trial_protocol_repository`, add the matching rebind so a loaded configuration brings its own set and experiment libraries:

```python
            self._trial_protocol_set_repository = TrialProtocolSetRepository(
                self._loaded_config_dir_path / "trial_protocol_sets"
            )
            self._trial_protocol_set_repository.reload()
            self._experiment_repository = ExperimentCompositionRepository(
                self._loaded_config_dir_path / "trial_experiments"
            )
            self._experiment_repository.reload()
```

Add three members next to the existing protocol accessors:

```python
    @property
    def trial_protocol_sets(self):
        with self._trial_protocol_lock:
            return self._trial_protocol_set_repository.documents

    @property
    def experiment_compositions(self):
        with self._trial_protocol_lock:
            return self._experiment_repository.documents

    def compile_and_save_experiment(self, experiment_id: str):
        """Compile an experiment and publish it as a selectable protocol."""
        with self._trial_protocol_lock:
            composition = self._experiment_repository.get(experiment_id)
            if composition is None:
                raise ValueError(
                    "Unknown experiment {!r}".format(experiment_id)
                )
            compiled = compile_experiment(
                composition, self._trial_protocol_set_repository.library()
            )
            # Persist any seed the compiler drew, so this compile is
            # reproducible, before publishing the protocol it produced.
            self._experiment_repository.save(compiled.composition)
            saved = self._trial_protocol_repository.save(compiled.document)
            return saved
```

- [ ] **Step 6: Write and run an app model integration test**

Append to `tests/trial_protocol_set_repository_test.py`:

```python
def test_app_model_compiles_an_experiment_into_the_protocol_library(
    app_model, tmp_path
):
    app_model._trial_protocol_set_repository = TrialProtocolSetRepository(
        tmp_path / "sets"
    )
    app_model._trial_protocol_set_repository.reload()
    app_model._trial_protocol_set_repository.save(make_set())
    app_model._experiment_repository = ExperimentCompositionRepository(
        tmp_path / "experiments"
    )
    app_model._experiment_repository.reload()
    app_model._experiment_repository.save(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(
                ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=2),
            ),
        )
    )

    saved = app_model.compile_and_save_experiment("day3")

    assert saved.protocol_id == "day3"
    assert saved.trial_count == 4
    assert app_model._trial_protocol_repository.get("day3") is not None
```

This follows the pattern in `tests/test_app_model.py`, which replaces `app_model._trial_protocol_repository` with one rooted at `tmp_path`.

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/trial_protocol_set_repository_test.py -v'
```

Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add tools/acquisition/model/trial_protocol_set_repository.py tools/acquisition/model/app_model.py tests/trial_protocol_set_repository_test.py
git commit -m "feat(protocol): store trial sets and experiments beside protocols"
```

---

### Task 5: The set and experiment sidebar

**Files:**
- Create: `tools/acquisition/view/protocol_set_sidebar.py`
- Modify: `tools/acquisition/view/protocol_content.py` — the layout block in `__init__` around lines 200-255
- Test: `tests/protocol_set_sidebar_test.py`

**Interfaces:**
- Consumes: `AppModel.trial_protocol_sets`, `AppModel.experiment_compositions`, `AppModel.compile_and_save_experiment` from Task 4; `ExperimentSetEntry` from Task 2.
- Produces:
  - `ProtocolSetSidebar(app_model, parent=None)` — a `QWidget`
  - `sidebar.selection_changed` — `Signal(str, str)` carrying `(kind, identifier)` where `kind` is `"set"` or `"experiment"`
  - `sidebar.refresh() -> None`
  - `sidebar.current_selection() -> Tuple[str, str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/protocol_set_sidebar_test.py`:

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.model.trial_protocol_schedule import ProtocolPatch  # noqa: E402
from tools.acquisition.model.trial_protocol_set import (  # noqa: E402
    ExperimentComposition,
    ExperimentSetEntry,
    TrialProtocolSet,
)
from tools.acquisition.model.trial_protocol_set_repository import (  # noqa: E402
    ExperimentCompositionRepository,
    TrialProtocolSetRepository,
)
from tools.acquisition.view.protocol_set_sidebar import (  # noqa: E402
    ProtocolSetSidebar,
)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def stocked_app_model(app_model, tmp_path):
    sets = TrialProtocolSetRepository(tmp_path / "sets")
    sets.reload()
    sets.save(
        TrialProtocolSet(
            set_id="baseline",
            name="Baseline",
            trial_count=2,
            defaults=ProtocolPatch.from_mapping({"enabled": True}),
        )
    )
    experiments = ExperimentCompositionRepository(tmp_path / "experiments")
    experiments.reload()
    experiments.save(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=1),),
        )
    )
    app_model._trial_protocol_set_repository = sets
    app_model._experiment_repository = experiments
    return app_model


def test_the_sidebar_lists_saved_sets(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)

    labels = [
        sidebar.set_list.item(index).text()
        for index in range(sidebar.set_list.count())
    ]

    assert any("Baseline" in label for label in labels)


def test_the_sidebar_lists_saved_experiments(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)

    labels = [
        sidebar.experiment_list.item(index).text()
        for index in range(sidebar.experiment_list.count())
    ]

    assert any("Day 3" in label for label in labels)


def test_selecting_a_set_reports_its_identity(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    seen = []
    sidebar.selection_changed.connect(lambda kind, key: seen.append((kind, key)))

    sidebar.set_list.setCurrentRow(0)

    assert seen[-1] == ("set", "baseline")


def test_compiling_publishes_the_experiment_as_a_protocol(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    sidebar.experiment_list.setCurrentRow(0)

    sidebar.compile_selected_experiment()

    assert stocked_app_model._trial_protocol_repository.get("day3") is not None


def test_compiling_with_no_experiment_selected_reports_rather_than_raises(
    qapp, stocked_app_model
):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    sidebar.experiment_list.setCurrentRow(-1)

    sidebar.compile_selected_experiment()

    assert "Select an experiment" in sidebar.status_label.text()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/protocol_set_sidebar_test.py -v'
```

Expected: collection error, `ModuleNotFoundError: No module named 'tools.acquisition.view.protocol_set_sidebar'`.

- [ ] **Step 3: Write the sidebar**

Create `tools/acquisition/view/protocol_set_sidebar.py`:

```python
"""Set library and experiment builder beside the trial protocol table."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ProtocolSetSidebar(QWidget):
    """Choose a set to edit, or an experiment to arrange and compile."""

    selection_changed = Signal(str, str)

    def __init__(self, app_model, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._app_model = app_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Sets"))
        self.set_list = QListWidget()
        self.set_list.currentRowChanged.connect(self._set_selected)
        layout.addWidget(self.set_list, stretch=1)

        layout.addWidget(QLabel("Experiments"))
        self.experiment_list = QListWidget()
        self.experiment_list.currentRowChanged.connect(self._experiment_selected)
        layout.addWidget(self.experiment_list, stretch=1)

        buttons = QHBoxLayout()
        self.compile_button = QPushButton("Compile and save")
        self.compile_button.setToolTip(
            "Flatten the selected experiment into a protocol you can run"
        )
        self.compile_button.clicked.connect(self.compile_selected_experiment)
        buttons.addWidget(self.compile_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #5f6772;")
        layout.addWidget(self.status_label)

        self.refresh()

    def current_selection(self) -> Tuple[str, str]:
        item = self.experiment_list.currentItem()
        if item is not None:
            return "experiment", item.data(Qt.ItemDataRole.UserRole)
        item = self.set_list.currentItem()
        if item is not None:
            return "set", item.data(Qt.ItemDataRole.UserRole)
        return "", ""

    def refresh(self) -> None:
        self._fill(
            self.set_list,
            [
                (item.set_id, "{} ({} trials, rev {})".format(
                    item.name, item.trial_count, item.revision
                ))
                for item in self._app_model.trial_protocol_sets
            ],
        )
        self._fill(
            self.experiment_list,
            [
                (item.experiment_id, "{} ({} entries, rev {})".format(
                    item.name, len(item.entries), item.revision
                ))
                for item in self._app_model.experiment_compositions
            ],
        )

    def compile_selected_experiment(self) -> None:
        item = self.experiment_list.currentItem()
        if item is None:
            self.status_label.setText("Select an experiment to compile.")
            return
        experiment_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            saved = self._app_model.compile_and_save_experiment(experiment_id)
        except Exception as error:
            logger.exception("Could not compile experiment %s", experiment_id)
            self.status_label.setText(
                "{}: {}".format(type(error).__name__, error)
            )
            return
        self.status_label.setText(
            "Compiled {} into protocol {} revision {}, {} trials.".format(
                experiment_id, saved.protocol_id, saved.revision, saved.trial_count
            )
        )
        self.refresh()

    @staticmethod
    def _fill(widget: QListWidget, rows) -> None:
        previous = widget.currentItem()
        previous_key = (
            None if previous is None else previous.data(Qt.ItemDataRole.UserRole)
        )
        widget.blockSignals(True)
        widget.clear()
        for key, label in rows:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            widget.addItem(item)
        widget.blockSignals(False)
        if previous_key is None:
            return
        for index in range(widget.count()):
            if widget.item(index).data(Qt.ItemDataRole.UserRole) == previous_key:
                widget.setCurrentRow(index)
                return

    def _set_selected(self, row: int) -> None:
        if row < 0:
            return
        self.experiment_list.setCurrentRow(-1)
        self.selection_changed.emit(
            "set", self.set_list.item(row).data(Qt.ItemDataRole.UserRole)
        )

    def _experiment_selected(self, row: int) -> None:
        if row < 0:
            return
        self.set_list.setCurrentRow(-1)
        self.selection_changed.emit(
            "experiment",
            self.experiment_list.item(row).data(Qt.ItemDataRole.UserRole),
        )
```

- [ ] **Step 4: Place the sidebar beside the table**

In `tools/acquisition/view/protocol_content.py`, add the import:

```python
from tools.acquisition.view.protocol_set_sidebar import ProtocolSetSidebar
```

Replace the single `content_layout.addWidget(table, stretch=1)` line with a horizontal split that puts the sidebar to the left of the table:

```python
        body = QHBoxLayout()
        self._set_sidebar = ProtocolSetSidebar(self._app_model)
        self._set_sidebar.setMaximumWidth(280)
        self._set_sidebar.selection_changed.connect(self._sidebar_selection_changed)
        body.addWidget(self._set_sidebar)
        body.addWidget(table, stretch=1)
        content_layout.addLayout(body, stretch=1)
```

Add the selection handler next to `_protocol_selected`:

```python
    def _sidebar_selection_changed(self, kind: str, identifier: str) -> None:
        """Show a set's own trials, or a compiled experiment read-only."""
        if kind == "experiment":
            self._edit_status.setText(
                "Experiment {} selected. Compile it to edit or run its "
                "trials.".format(identifier)
            )
        elif kind == "set":
            self._edit_status.setText("Set {} selected.".format(identifier))
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/protocol_set_sidebar_test.py -v'
```

Expected: 5 passed.

- [ ] **Step 6: Run the full suite for regressions**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/ -q'
```

Expected: no new failures against the branch baseline. The known pre-existing failures are the two in `camera_discovery_test.py`, the two in `signal_stream_ui_test.py`, and a timing-flaky case in `video_capture_record_test.py`.

- [ ] **Step 7: Verify on the rig by eye**

Launch the application on the rig. Create two sets, build an experiment that uses one of them twice, compile it, then select the compiled protocol for a session and confirm the trial table shows the expected trial count and that the `Sources` column names the set each trial came from.

- [ ] **Step 8: Document and commit**

Add a section on sets and composition to `docs/acquisition/session-trials-protocols.md`, after `## Ordered trial protocol editor`, covering: what a set is, that experiments pin set revisions, that compiling publishes an ordinary protocol, and that shuffling is reproducible from the stored seed.

```bash
git add tools/acquisition/view/protocol_set_sidebar.py tools/acquisition/view/protocol_content.py tests/protocol_set_sidebar_test.py docs/acquisition/session-trials-protocols.md
git commit -m "feat(view): build experiments from saved trial sets"
```

- [ ] **Step 9: Commit the sidebar, then continue to Task 6**

Task 5 lists and compiles sets but cannot create one. The feature is not usable until Task 6 lands; do not cherry-pick until then.

---

### Task 6: Author a set from the protocol editor

Without this, nothing can create a set's trial content and the feature is inert. Rather than duplicate the 29-column editor's edit path, a set is authored **through** the existing protocol editor: convert a protocol you have built into a set, and open a set back into a scratch protocol to revise it.

**Files:**
- Modify: `tools/acquisition/model/trial_protocol_set.py` (append the two converters)
- Modify: `tools/acquisition/model/app_model.py` (two methods beside `compile_and_save_experiment`)
- Modify: `tools/acquisition/view/protocol_set_sidebar.py` (two buttons)
- Test: `tests/trial_protocol_set_conversion_test.py`

**Interfaces:**
- Consumes: `TrialProtocolSet` from Task 1; `TrialProtocolDocument` from `trial_protocol_schedule`; the repositories from Task 4.
- Produces:
  - `set_from_document(document: TrialProtocolDocument, *, set_id: str, name: str) -> TrialProtocolSet`
  - `document_from_set(protocol_set: TrialProtocolSet, *, protocol_id: str, name: str) -> TrialProtocolDocument`
  - `AppModel.save_protocol_as_set(protocol_id: str, set_id: str, name: str) -> TrialProtocolSet`
  - `AppModel.open_set_as_protocol(set_id: str, protocol_id: str, name: str) -> TrialProtocolDocument`
  - `ProtocolSetSidebar.save_selected_protocol_as_set(set_id, name)`, `ProtocolSetSidebar.open_selected_set_in_editor(protocol_id, name)`

- [ ] **Step 1: Write the failing tests**

Create `tests/trial_protocol_set_conversion_test.py`:

```python
import pytest

from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
    TrialProtocolDocument,
)
from tools.acquisition.model.trial_protocol_set import (
    TrialProtocolSet,
    document_from_set,
    set_from_document,
)


def make_document(**overrides):
    values = dict(
        protocol_id="draft",
        name="Draft",
        trial_count=4,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
        bulk_overrides=(
            ProtocolScope.create("late", [3, 4], {"pre_reveal_ms": 25}),
        ),
        trial_overrides=(TrialOverride.create(1, {"pre_reveal_ms": 5}),),
    )
    values.update(overrides)
    return TrialProtocolDocument(**values)


def test_a_document_converts_to_a_set_with_the_same_trial_content():
    result = set_from_document(make_document(), set_id="baseline", name="Baseline")

    assert result.set_id == "baseline"
    assert result.trial_count == 4
    assert result.defaults.to_mapping() == {"enabled": True}
    assert [item.name for item in result.bulk_overrides] == ["late"]
    assert [item.trial_id for item in result.trial_overrides] == [1]


def test_converting_a_document_that_has_epochs_is_refused():
    document = make_document(
        epochs=(ProtocolScope.create("phase", [1, 2], {"enabled": True}),)
    )

    with pytest.raises(ValueError, match="epochs or blocks"):
        set_from_document(document, set_id="baseline", name="Baseline")


def test_a_set_converts_back_into_an_editable_document():
    original = make_document()
    protocol_set = set_from_document(original, set_id="baseline", name="Baseline")

    document = document_from_set(
        protocol_set, protocol_id="baseline-draft", name="Baseline draft"
    )

    assert document.protocol_id == "baseline-draft"
    assert document.trial_count == original.trial_count
    assert document.epochs == ()


def test_a_round_trip_preserves_every_resolved_row():
    original = make_document()

    protocol_set = set_from_document(original, set_id="baseline", name="Baseline")
    restored = document_from_set(
        protocol_set, protocol_id="draft", name="Draft"
    )

    assert [item.row for item in restored.resolve()] == [
        item.row for item in original.resolve()
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/trial_protocol_set_conversion_test.py -v'
```

Expected: FAIL with `ImportError: cannot import name 'set_from_document'`.

- [ ] **Step 3: Write the converters**

Append to `tools/acquisition/model/trial_protocol_set.py`, adding `TrialProtocolDocument` to the imports from `trial_protocol_schedule`:

```python
def set_from_document(
    document: "TrialProtocolDocument",
    *,
    set_id: str,
    name: str,
) -> TrialProtocolSet:
    """Snapshot a protocol's trial content as a reusable set.

    Epochs and blocks are refused rather than flattened. Within a protocol they
    are the older, protocol-local grouping; a set is the cross-protocol one, and
    the compiler writes epochs itself. Flattening an epoch patch into the bulk
    layer would also change its precedence relative to blocks, so a silent
    conversion could alter what the trials actually do.
    """
    if document.epochs or document.blocks:
        raise ValueError(
            "Protocol {} uses epochs or blocks, which a set cannot hold. Express "
            "those groups as bulk overrides, or build the experiment from "
            "several sets instead.".format(document.protocol_id)
        )
    return TrialProtocolSet(
        set_id=set_id,
        name=name,
        trial_count=int(document.trial_count),
        defaults=document.defaults,
        bulk_overrides=tuple(document.bulk_overrides),
        trial_overrides=tuple(document.trial_overrides),
        description=document.description,
    )


def document_from_set(
    protocol_set: TrialProtocolSet,
    *,
    protocol_id: str,
    name: str,
) -> "TrialProtocolDocument":
    """Materialise a set as an ordinary protocol so the editor can revise it."""
    return TrialProtocolDocument(
        protocol_id=protocol_id,
        name=name,
        trial_count=int(protocol_set.trial_count),
        defaults=protocol_set.defaults,
        bulk_overrides=tuple(protocol_set.bulk_overrides),
        trial_overrides=tuple(protocol_set.trial_overrides),
        description=protocol_set.description,
    )
```

- [ ] **Step 4: Add the app model methods**

In `tools/acquisition/model/app_model.py`, add beside `compile_and_save_experiment`, and extend the `trial_protocol_set` import to include both converters:

```python
    def save_protocol_as_set(self, protocol_id: str, set_id: str, name: str):
        """Capture a protocol you have built as a reusable set."""
        with self._trial_protocol_lock:
            document = self._trial_protocol_repository.get(protocol_id)
            if document is None:
                raise ValueError("Unknown protocol {!r}".format(protocol_id))
            return self._trial_protocol_set_repository.save(
                set_from_document(document, set_id=set_id, name=name)
            )

    def open_set_as_protocol(self, set_id: str, protocol_id: str, name: str):
        """Publish a set as an editable protocol, to revise and capture again."""
        with self._trial_protocol_lock:
            protocol_set = self._trial_protocol_set_repository.get(set_id)
            if protocol_set is None:
                raise ValueError("Unknown set {!r}".format(set_id))
            return self._trial_protocol_repository.save(
                document_from_set(
                    protocol_set, protocol_id=protocol_id, name=name
                )
            )
```

- [ ] **Step 5: Add the two sidebar buttons**

In `tools/acquisition/view/protocol_set_sidebar.py`, add to the button row built in `__init__`:

```python
        self.capture_button = QPushButton("Set from protocol")
        self.capture_button.setToolTip(
            "Save the protocol currently selected for the session as a reusable set"
        )
        self.capture_button.clicked.connect(self._capture_selected_protocol)
        buttons.addWidget(self.capture_button)
        self.open_set_button = QPushButton("Open set")
        self.open_set_button.setToolTip(
            "Publish the selected set as an editable protocol"
        )
        self.open_set_button.clicked.connect(self._open_selected_set)
        buttons.addWidget(self.open_set_button)
```

and the two handlers, which prompt for an identity the same way `ProtocolContent._ask_identity` already does:

```python
    def save_selected_protocol_as_set(self, set_id: str, name: str) -> None:
        protocol = self._app_model.selected_trial_protocol_id
        if not protocol:
            self.status_label.setText("Select a session protocol to capture.")
            return
        try:
            saved = self._app_model.save_protocol_as_set(protocol, set_id, name)
        except Exception as error:
            logger.exception("Could not capture protocol %s as a set", protocol)
            self.status_label.setText("{}: {}".format(type(error).__name__, error))
            return
        self.status_label.setText(
            "Saved set {} revision {}.".format(saved.set_id, saved.revision)
        )
        self.refresh()

    def open_selected_set_in_editor(self, protocol_id: str, name: str) -> None:
        item = self.set_list.currentItem()
        if item is None:
            self.status_label.setText("Select a set to open.")
            return
        set_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            saved = self._app_model.open_set_as_protocol(set_id, protocol_id, name)
        except Exception as error:
            logger.exception("Could not open set %s", set_id)
            self.status_label.setText("{}: {}".format(type(error).__name__, error))
            return
        self.status_label.setText(
            "Opened set {} as protocol {}. Edit it, then capture it again.".format(
                set_id, saved.protocol_id
            )
        )
        self.refresh()

    def _capture_selected_protocol(self) -> None:
        identity = self._ask_identity("New set")
        if identity is not None:
            self.save_selected_protocol_as_set(*identity)

    def _open_selected_set(self) -> None:
        identity = self._ask_identity("Open set as protocol")
        if identity is not None:
            self.open_selected_set_in_editor(*identity)

    def _ask_identity(self, title: str):
        identifier, accepted = QInputDialog.getText(self, title, "Identifier:")
        if not accepted or not identifier.strip():
            return None
        name, accepted = QInputDialog.getText(self, title, "Display name:")
        if not accepted or not name.strip():
            return None
        return identifier.strip(), name.strip()
```

Add `QInputDialog` to the `PySide6.QtWidgets` imports.

Confirm the accessor name for the selected protocol before writing `selected_trial_protocol_id`:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && grep -n "selected_ordered_protocol\|def selected_trial_protocol" tools/acquisition/model/app_model.py | head'
```

Use whatever that shows; do not add a parallel accessor.

- [ ] **Step 6: Write the UI tests**

Append to `tests/protocol_set_sidebar_test.py`:

```python
def test_capturing_with_no_protocol_selected_reports_rather_than_raises(
    qapp, stocked_app_model
):
    sidebar = ProtocolSetSidebar(stocked_app_model)

    sidebar.save_selected_protocol_as_set("captured", "Captured")

    assert "Select a session protocol" in sidebar.status_label.text()


def test_opening_a_set_publishes_it_as_an_editable_protocol(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    sidebar.set_list.setCurrentRow(0)

    sidebar.open_selected_set_in_editor("baseline-draft", "Baseline draft")

    assert stocked_app_model._trial_protocol_repository.get("baseline-draft")
```

- [ ] **Step 7: Run every test in this plan**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/trial_protocol_set_test.py tests/experiment_composition_test.py tests/experiment_compiler_test.py tests/trial_protocol_set_repository_test.py tests/protocol_set_sidebar_test.py tests/trial_protocol_set_conversion_test.py -v'
```

Expected: 44 passed.

- [ ] **Step 8: Run the full suite for regressions**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/ -q'
```

Expected: no new failures against the branch baseline.

- [ ] **Step 9: Verify the whole loop on the rig**

Launch the application on the rig and complete one full cycle, which is the only proof the feature works: build a protocol in the editor, capture it as a set, build a second set, compose an experiment that uses the first set twice and the second once with shuffling on, compile it, select the compiled protocol for a session, and confirm the trial count, the trial order, and that the `Sources` column names the set each trial came from. Then recompile and confirm the shuffle reproduces from the stored seed.

- [ ] **Step 10: Document, commit and cherry-pick**

Add the sets and composition section to `docs/acquisition/session-trials-protocols.md` after `## Ordered trial protocol editor`, covering what a set is, that experiments pin set revisions, that compiling publishes an ordinary protocol, that shuffling is reproducible from the stored seed, and that sets are authored by capturing a protocol rather than in a separate editor. Record in `planning.md` why an experiment compiles to epochs rather than extending the protocol schema, and why sets reuse the protocol editor instead of duplicating it.

```bash
git add tools/acquisition/model/trial_protocol_set.py tools/acquisition/model/app_model.py tools/acquisition/view/protocol_set_sidebar.py tests/trial_protocol_set_conversion_test.py tests/protocol_set_sidebar_test.py docs/acquisition/session-trials-protocols.md planning.md
git commit -m "feat(protocol): author trial sets through the protocol editor"
```

Then cherry-pick this feature's commits onto `demo-mode-and-presentation` and run the six new test files there.
