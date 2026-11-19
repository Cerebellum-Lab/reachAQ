"""Hardware-free simulation switch, shared time base and cross-process signal bus."""

import dataclasses
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_SIMULATION = "REACHAQ_SIMULATION"
ENV_SIM_BUS = "REACHAQ_SIM_BUS"
ENV_SIM_EPOCH = "REACHAQ_SIM_EPOCH"
ENV_SIM_CAMERA_FPS = "REACHAQ_SIM_CAMERA_FPS"
ENV_SIM_CAMERA_EXPOSURE_S = "REACHAQ_SIM_CAMERA_EXPOSURE_S"
DEFAULT_SIM_CAMERA_FPS = 150.0
DEFAULT_SIM_CAMERA_EXPOSURE_S = 0.002


def is_simulation() -> bool:
    return os.environ.get(ENV_SIMULATION) == "1"


def sim_epoch() -> float:
    """Perf-counter time of simulated exposure 0; every process reads the same value."""
    return float(os.environ.get(ENV_SIM_EPOCH, "0.0"))


def sim_camera_fps() -> float:
    """The simulated exposure grid's rate; the NI stream and the cameras must agree on it."""
    return float(os.environ.get(ENV_SIM_CAMERA_FPS, DEFAULT_SIM_CAMERA_FPS))


def sim_camera_exposure_s() -> float:
    """The simulated exposure, in seconds."""
    return float(os.environ.get(ENV_SIM_CAMERA_EXPOSURE_S, DEFAULT_SIM_CAMERA_EXPOSURE_S))


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
        self._warned_bad_line = False

    def poll(self) -> tuple[SimSignalEvent, ...]:
        try:
            with open(self._path, "rb") as f:
                if os.fstat(f.fileno()).st_size < self._offset:
                    # Shrink-only: one fresh bus per run, so a file that is
                    # replaced by a longer one is not detected.
                    self._offset = 0  # truncated or recreated: read it from the start
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
            try:
                d = json.loads(raw)
                event = SimSignalEvent(
                    line=d["line"],
                    start=float(d["start"]),
                    end=float(d["end"]),
                    value=float(d.get("value", 1.0)),
                )
            except (ValueError, KeyError, TypeError) as err:
                # One bad line must not cost the batch its good ones.
                if not self._warned_bad_line:
                    self._warned_bad_line = True
                    logger.warning("skipping malformed simulation bus line in %s: %r (%s)",
                                   self._path, raw[:200], err)
                continue
            events.append(event)
        return tuple(events)
