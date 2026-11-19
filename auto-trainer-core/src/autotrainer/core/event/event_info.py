from dataclasses import dataclass

from autotrainer.core.api import ApiEvent


@dataclass(frozen=True)
class EventInfo(ApiEvent):
    pass
