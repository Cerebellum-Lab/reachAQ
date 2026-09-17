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

    Provenance is returned rather than stamped into the document: adding fields
    to TrialProtocolDocument would mean bumping PROTOCOL_SCHEMA_VERSION, which
    the runner and session evidence both read, for something the epoch names
    already carry.
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
    # used more than once is numbered, including across separate entries. Two
    # entries naming the same set would otherwise collide on a duplicate epoch
    # name, which TrialProtocolDocument rejects.
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
