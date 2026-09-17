"""Baseline, stimulation, and washout epoch blocks.

reachAQ already owns a hierarchical authoring model: a protocol document
carries named epoch and block scopes that patch a set of trial IDs, and every
resolved row records which scope supplied each value.  That mechanism is
strictly more expressive than a fixed runtime epoch cycle, so this module does
not add a second one.

Instead it generates the conventional block layout as ordinary
:class:`ProtocolScope` epochs, which the author then applies to a document like
any other scope.  The layout matches the reach-training reference: the first
block is baseline with stimulation disabled, and subsequent blocks alternate
stimulation and washout.

Because these are ordinary scopes, an author is free to edit, extend, or
discard them afterwards, and the runtime needs no epoch-specific behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from tools.acquisition.model.trial_protocol_schedule import ProtocolScope


EPOCH_LAYOUT_VERSION = 1
DEFAULT_EPOCH_BLOCK_SIZE = 15
MAX_EPOCH_BLOCK_SIZE = 10_000

BASELINE = "baseline"
STIMULATION = "stimulation"
WASHOUT = "washout"

# Stimulation is disabled during baseline and washout; the caller supplies the
# values that turn it on for a stimulation block.
DISABLED_STIMULUS_VALUES = {"stimulus_assignment": "disabled"}


def epoch_kind(trial_id: int, block_size: int) -> str:
    """Return the block kind a scored trial falls in."""

    trial_id = max(1, int(trial_id))
    block_size = _validated_block_size(block_size)
    block_index = (trial_id - 1) // block_size
    if block_index == 0:
        return BASELINE
    return STIMULATION if block_index % 2 == 1 else WASHOUT


def epoch_assignment(trial_id: int, block_size: int) -> Dict[str, Any]:
    """Describe the block a scored trial falls in, for labels and evidence."""

    trial_id = max(1, int(trial_id))
    block_size = _validated_block_size(block_size)
    block_index = (trial_id - 1) // block_size
    trial_in_block = ((trial_id - 1) % block_size) + 1
    kind = epoch_kind(trial_id, block_size)
    if kind == BASELINE:
        epoch_number = 0
        label = "Baseline epoch"
    else:
        epoch_number = (block_index + 1) // 2
        label = (
            f"Stimulation epoch #{epoch_number}"
            if kind == STIMULATION
            else f"Washout epoch #{epoch_number}"
        )
    return {
        "layout_version": EPOCH_LAYOUT_VERSION,
        "block_index": block_index,
        "block_size": block_size,
        "epoch_number": epoch_number,
        "epoch_kind": kind,
        "trial_index_in_block": trial_in_block,
        "progress_label": f"{trial_in_block}/{block_size} {label}",
        "stimulation_allowed": kind == STIMULATION,
    }


def build_epoch_scopes(
    trial_count: int,
    block_size: int = DEFAULT_EPOCH_BLOCK_SIZE,
    *,
    stimulation_values: Optional[Mapping[str, object]] = None,
    baseline_values: Optional[Mapping[str, object]] = None,
    washout_values: Optional[Mapping[str, object]] = None,
) -> Tuple[ProtocolScope, ...]:
    """Return baseline/stimulation/washout scopes covering ``trial_count``.

    ``stimulation_values`` carries whatever turns stimulation on for this
    protocol, for example an assignment plus a trigger profile.  Baseline and
    washout default to an explicitly disabled assignment rather than to
    whatever the document happens to inherit.
    """

    trial_count = int(trial_count)
    if trial_count < 1:
        raise ValueError("Epoch layout requires at least one trial")
    block_size = _validated_block_size(block_size)
    if not stimulation_values:
        raise ValueError("Stimulation epochs require stimulation values")

    values_by_kind = {
        BASELINE: dict(baseline_values or DISABLED_STIMULUS_VALUES),
        STIMULATION: dict(stimulation_values),
        WASHOUT: dict(washout_values or DISABLED_STIMULUS_VALUES),
    }

    grouped: Dict[int, list] = {}
    for trial_id in range(1, trial_count + 1):
        grouped.setdefault((trial_id - 1) // block_size, []).append(trial_id)

    scopes = []
    for block_index, trial_ids in sorted(grouped.items()):
        assignment = epoch_assignment(trial_ids[0], block_size)
        kind = assignment["epoch_kind"]
        epoch_number = assignment["epoch_number"]
        name = (
            BASELINE
            if kind == BASELINE
            else f"{kind}-{epoch_number}"
        )
        scopes.append(
            ProtocolScope.create(name, trial_ids, values_by_kind[kind])
        )
    return tuple(scopes)


def _validated_block_size(block_size: int) -> int:
    block_size = int(block_size)
    if not 1 <= block_size <= MAX_EPOCH_BLOCK_SIZE:
        raise ValueError(
            f"Epoch block size must be between 1 and {MAX_EPOCH_BLOCK_SIZE}"
        )
    return block_size
