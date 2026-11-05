"""What a laser command output does when the process holding it dies.

The kill test records, for every row of a matrix of process states and ways of
dying, what the command output reads at a few instants around the death. This
module is the part with no hardware in it: the rows, the readings taken off a
sample log, the verdict on each row, and the table and record it is written to.
Voltages are volts at the reader; the reader is independent of the process
under test, so a row's readings are the output's, not the victim's.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

TIMINGS = ("software", "hardware")
WAVEFORMS = ("dc", "pulse")
MODES = ("clean", "sigterm", "sigkill", "exception", "abort", "hang")

REFUSED_BY_DESIGN = "refused_by_design"
UNKNOWN = "unknown"
TRAIN_RAN_ON = "train_ran_on"
SAFE_ON_DEATH = "safe_on_death"
HELD_UNTIL_REOPEN = "held_until_reopen"
HELD_UNTIL_RESET = "held_until_reset"
HELD_THROUGH_RESET = "held_through_reset"

SOFTWARE_PULSE_REFUSAL = (
    "run_synchronized_pulse_train refuses a pulse train unless hardware_timed=True"
)

SCHEMA = 1


@dataclass(frozen=True)
class KillCase:
    mode: str
    timing: str
    waveform: str
    repeat: int = 1

    @property
    def case_id(self) -> str:
        text = f"{self.mode}-{self.timing}-{self.waveform}"
        if self.timing == "hardware" and self.waveform == "pulse":
            text += f"-r{self.repeat}"
        return text

    @property
    def refusal(self) -> Optional[str]:
        if self.timing == "software" and self.waveform == "pulse":
            return SOFTWARE_PULSE_REFUSAL
        return None


def kill_cases(*, pulse_repeats: int = 3) -> Tuple[KillCase, ...]:
    """The matrix, ordered timing, waveform, mode, repeat.

    A hardware pulse row repeats, because a kill that lands on a pulse's low
    phase reads safe at the instant after the kill. The software pulse row is
    refused by the controller before any mode could act, so one row carries
    that refusal.
    """
    cases = []
    for timing in TIMINGS:
        for waveform in WAVEFORMS:
            if (timing, waveform) == ("software", "pulse"):
                cases.append(KillCase("clean", timing, waveform))
                continue
            repeats = pulse_repeats if (timing, waveform) == ("hardware", "pulse") else 1
            for mode in MODES:
                for repeat in range(1, repeats + 1):
                    cases.append(KillCase(mode, timing, waveform, repeat))
    return tuple(cases)


@dataclass(frozen=True)
class LevelSample:
    """One reader chunk: when it ended and the mean, minimum and maximum in it."""

    time: float
    mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class LevelReading:
    mean: float
    minimum: float
    maximum: float
    #: Reader chunks the reading covers.
    sample_count: int
    instrument: str


def _reading_of(chunks: Sequence[LevelSample], instrument: str) -> Optional[LevelReading]:
    if not chunks:
        return None
    return LevelReading(
        mean=sum(chunk.mean for chunk in chunks) / len(chunks),
        minimum=min(chunk.minimum for chunk in chunks),
        maximum=max(chunk.maximum for chunk in chunks),
        sample_count=len(chunks),
        instrument=instrument,
    )


def level_at(
    samples: Sequence[LevelSample],
    time: float,
    *,
    window_s: float = 0.05,
    instrument: str,
) -> Optional[LevelReading]:
    """The reading over the window ending at ``time``, or None with no sample in it."""
    return _reading_of(
        [s for s in samples if time - window_s < s.time <= time], instrument
    )


def level_between(
    samples: Sequence[LevelSample], start: float, end: float, *, instrument: str
) -> Optional[LevelReading]:
    """The reading over ``start < sample.time <= end``, or None with no sample in it."""
    return _reading_of([s for s in samples if start < s.time <= end], instrument)


@dataclass(frozen=True)
class KillObservation:
    case: KillCase
    before: Optional[LevelReading]
    after_kill: Optional[LevelReading]
    hold: Optional[LevelReading]
    after_hold: Optional[LevelReading]
    after_reopen: Optional[LevelReading]
    after_reset: Optional[LevelReading]
    reopen_error: Optional[str]
    reset_error: Optional[str]
    victim_exit: str
    #: Hand-read DMM values keyed by instant (the observation's field names).
    dmm: Mapping[str, float] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()


def _within(reading: Optional[LevelReading], volts: float, tolerance: float) -> bool:
    """A missing reading is not evidence of anything, so it is never within."""
    return reading is not None and abs(reading.mean - volts) <= tolerance


def judge(
    observation: KillObservation,
    *,
    minimum_volts: float,
    level_volts: float,
    tolerance_volts: float = 0.05,
) -> str:
    """The verdict on one row.

    The hold window is read before the after-kill instant is trusted: a kill
    that lands on a pulse's low phase reads safe at +0.10 s while the train
    goes on generating.
    """
    if observation.case.refusal:
        return REFUSED_BY_DESIGN
    if observation.after_kill is None:
        return UNKNOWN
    hold = observation.hold
    if (
        hold is not None
        and hold.maximum >= level_volts - tolerance_volts
        and hold.minimum <= minimum_volts + tolerance_volts
    ):
        return TRAIN_RAN_ON
    if _within(observation.after_kill, minimum_volts, tolerance_volts):
        return SAFE_ON_DEATH
    if _within(observation.after_reopen, minimum_volts, tolerance_volts):
        return HELD_UNTIL_REOPEN
    if _within(observation.after_reset, minimum_volts, tolerance_volts):
        return HELD_UNTIL_RESET
    return HELD_THROUGH_RESET


_NOT_AVAILABLE = "n/a"
_COLUMNS = (
    "case", "timing", "waveform", "mode", "rep", "before", "after kill",
    "hold min..max", "after 5 s", "after reopen", "after reset", "victim exit",
    "instrument", "verdict",
)
#: The dmm key (an instant) and the column its value is printed in.
_DMM_COLUMNS = {
    "before": "before",
    "after_kill": "after kill",
    "hold": "hold min..max",
    "after_hold": "after 5 s",
    "after_reopen": "after reopen",
    "after_reset": "after reset",
}


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _volts(reading: Optional[LevelReading]) -> str:
    return _NOT_AVAILABLE if reading is None else f"{reading.mean:.3f}"


def _instrument_of(observation: KillObservation) -> str:
    for reading in (
        observation.before, observation.after_kill, observation.hold,
        observation.after_hold, observation.after_reopen, observation.after_reset,
    ):
        if reading is not None:
            return reading.instrument
    return _NOT_AVAILABLE


def _row(observation: KillObservation, verdict: str) -> str:
    case = observation.case
    hold = observation.hold
    after_kill = _volts(observation.after_kill)
    if case.refusal:
        after_kill = case.refusal
    after_reopen = _volts(observation.after_reopen)
    if observation.after_reopen is None and observation.reopen_error:
        after_reopen += f" ({observation.reopen_error})"
    after_reset = _volts(observation.after_reset)
    if observation.after_reset is None and observation.reset_error:
        after_reset += f" ({observation.reset_error})"
    cells = {
        "case": case.case_id,
        "timing": case.timing,
        "waveform": case.waveform,
        "mode": case.mode,
        "rep": str(case.repeat),
        "before": _volts(observation.before),
        "after kill": after_kill,
        "hold min..max": (
            _NOT_AVAILABLE if hold is None else f"{hold.minimum:.3f}..{hold.maximum:.3f}"
        ),
        "after 5 s": _volts(observation.after_hold),
        "after reopen": after_reopen,
        "after reset": after_reset,
        "victim exit": observation.victim_exit,
        "instrument": _instrument_of(observation),
        "verdict": verdict,
    }
    cells = {column: _escape(text) for column, text in cells.items()}
    for instant, value in observation.dmm.items():
        column = _DMM_COLUMNS.get(instant)
        if column is None:
            # An instant with no column of its own is still shown, not dropped.
            cells["instrument"] += f"<br>dmm.{_escape(instant)}={value:g}"
        else:
            cells[column] += f"<br>dmm={value:g}"
    return "| " + " | ".join(cells[column] for column in _COLUMNS) + " |"


def format_table(
    observations: Sequence[KillObservation],
    *,
    minimum_volts: float,
    level_volts: float,
    tolerance_volts: float = 0.05,
) -> str:
    """A Markdown table, one row per observation, one line per row."""
    lines = [
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "|".join(" --- " for _ in _COLUMNS) + "|",
    ]
    for observation in observations:
        verdict = judge(
            observation,
            minimum_volts=minimum_volts,
            level_volts=level_volts,
            tolerance_volts=tolerance_volts,
        )
        lines.append(_row(observation, verdict))
    return "\n".join(lines)


def to_record(observations: Sequence[KillObservation], **header) -> dict:
    """The observations as JSON-ready data, after the header the caller names."""
    reserved = {"schema", "cases"} & set(header)
    if reserved:
        raise ValueError(f"header keys {sorted(reserved)} are written by to_record")
    return {
        "schema": SCHEMA,
        **header,
        "cases": [dataclasses.asdict(observation) for observation in observations],
    }


def _reading_from(data: Optional[Mapping]) -> Optional[LevelReading]:
    return None if data is None else LevelReading(**data)


def from_record(record: Mapping) -> Tuple[KillObservation, ...]:
    """The observations in a record, as ``to_record`` wrote them."""
    if record.get("schema") != SCHEMA:
        raise ValueError(f"unsupported record schema {record.get('schema')!r}")
    observations = []
    for data in record["cases"]:
        fields = dict(data)
        fields["case"] = KillCase(**data["case"])
        for name in ("before", "after_kill", "hold", "after_hold",
                     "after_reopen", "after_reset"):
            fields[name] = _reading_from(data[name])
        fields["dmm"] = dict(data.get("dmm", {}))
        fields["notes"] = tuple(data.get("notes", ()))
        observations.append(KillObservation(**fields))
    return tuple(observations)
