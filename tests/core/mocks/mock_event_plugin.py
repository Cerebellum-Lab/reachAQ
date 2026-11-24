from typing import Optional

from reachaq.core import EventManagerPlugin, EventInfo, ProjectInfo


class MockEventPlugin(EventManagerPlugin):
    def __init__(self):
        self.project = None
        self.enabled = True
        self.last_event: Optional[EventInfo] = None
        self.event_count: int = 0
        self.flushed = False
        self.closed = False

    def set_project(self, project: Optional[ProjectInfo]) -> None:
        self.project = project

    def set_enable(self, enable: bool) -> None:
        self.enabled = enable

    def process_event(self, info: EventInfo, repeat_count: int) -> None:
        self.last_event = info
        self.event_count += 1

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True
