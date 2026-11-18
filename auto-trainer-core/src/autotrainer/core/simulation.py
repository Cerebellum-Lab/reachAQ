"""Hardware-free simulation switch, shared time base and cross-process signal bus."""

import dataclasses
import json
import os
from pathlib import Path

ENV_SIMULATION = "REACHAQ_SIMULATION"
ENV_SIM_BUS = "REACHAQ_SIM_BUS"
ENV_SIM_EPOCH = "REACHAQ_SIM_EPOCH"


def is_simulation() -> bool:
    return os.environ.get(ENV_SIMULATION) == "1"


def sim_epoch() -> float:
    """Perf-counter time of simulated exposure 0; every process reads the same value."""
    return float(os.environ.get(ENV_SIM_EPOCH, "0.0"))


@dataclasses.dataclass(frozen=True)
class SimSignalEvent:
    line: str
    start: float
    end: float
    value: float = 1.0


def publish_signal(line: str, start: float, end: float, value: float = 1.0) -> None:
    """Append one event to the bus file; no-op when REACHAQ_SIM_BUS is unset."""
    path = os.environ.get(ENV_SIM_BUS)
    if not path:
        return
    record = json.dumps(
        {"line": line, "start": start, "end": end, "value": value}
    )
    # One short write plus flush keeps a line whole across appending processes.
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(record + "\n")
        f.flush()


class SimSignalBusReader:
    """Tails the bus file, returning only complete lines not yet seen."""

    def __init__(self, path):
        self._path = Path(path)
        self._offset = 0

    def poll(self) -> tuple[SimSignalEvent, ...]:
        try:
            with open(self._path, "rb") as f:
                f.seek(self._offset)
                data = f.read()
        except FileNotFoundError:
            return ()
        end = data.rfind(b"\n")
        if end < 0:
            return ()  # nothing complete yet; keep the partial tail for next time
        self._offset += end + 1
        events = []
        for raw in data[: end + 1].splitlines():
            if not raw.strip():
                continue
            d = json.loads(raw)
            events.append(
                SimSignalEvent(
                    line=d["line"],
                    start=float(d["start"]),
                    end=float(d["end"]),
                    value=float(d.get("value", 1.0)),
                )
            )
        return tuple(events)
