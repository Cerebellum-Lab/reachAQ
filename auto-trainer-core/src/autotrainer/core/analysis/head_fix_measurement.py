import dataclasses
from typing import List


@dataclasses.dataclass
class HeadFixMeasurement:
    when: float = 0   # realtime timestamp, i.e: "UNIX" timestamp, in float seconds as normally.
    timestamp: int = 0  # perf counter timestamp, in integer nanoseconds
    switch: bool = False
    pressure: float = 0
    temperature: float = 0
    humidity: float = 0
    spectrum: List[float] = dataclasses.field(default_factory=list)
    head_contact: bool = False
