from typing import Protocol, Optional

from ..project import ProjectInfo
from .event_info import EventInfo


class EventManagerPlugin(Protocol):
    """
    Interface for event manager plugins.
    """

    def set_project(self, project: Optional[ProjectInfo]) -> None:
        """
        Called when the project changes.

        Args:
             project: The new project info (or none).
        """

    def set_enable(self, enable: bool) -> None:
        """
        Called when the event manager is enabled or disabled.

        Args:
            enable: True if the event manager is enabled, false otherwise.
        """

    def process_event(self, info: EventInfo, repeat_count: int) -> None:
        """
        Called when an event occurs.

        Args:
            info: The event info.
            repeat_count: Number of times this event has been repeated until a different event was received.
        """

    def flush(self):
        """
        For event sinks that may buffer output, this should forcibly flush any pending output.
        """

    def close(self) -> None:
        """
        Called when the event manager is closed.  Any pending output should be flushed, resources released, etc.
        """
