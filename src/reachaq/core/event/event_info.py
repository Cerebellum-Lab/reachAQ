from dataclasses import dataclass

from reachaq.core.api import ApiEvent


@dataclass(frozen=True)
class EventInfo(ApiEvent):
    pass
