import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, List, TypeVar

from typing_extensions import Self


@dataclass(frozen=True)
class Notification:
    """
    A class representing a notification with a type and optional source and context objects.
    """
    event_type: Any
    """
    The type of the event. This is used to identify the event and determine which observers should be notified.
    """
    source: Optional[Any] = None
    """
    The source of the event. This is typically the object that generated the event.
    """
    context: Optional[Any] = None


TCallable = TypeVar("TCallable", bound=Callable[[Notification], None])


class NotificationCenter:
    """
    A publish and subscribe implementation for general message passing where publishers can remain unaware of if there
    are listeners and who they are.

    Each process has a default notification center.  Additional instances may be created to organize notifications in
    specific contexts.
    """

    @classmethod
    def default_center(cls) -> Self:
        """
        Returns the default instance of NotificationCenter.
        """
        if not hasattr(cls, "_default_center"):
            cls._default_center = cls()

        return cls._default_center

    def __init__(self):
        self._subscribers: Dict[str, List[TCallable]] = {}
        self._lock = threading.Lock()

    def add_observer(self, event_type: Any, callback: TCallable):
        """
        Subscribe to a specific event type with a callback function.
        """
        with self._lock:
            self._add_observer(event_type, callback)

    def _add_observer(self, event_type, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def remove_observer(self, event_type: Any, callback: TCallable):
        """
        Unsubscribe from a specific event type.
        """
        with self._lock:
            self._remove_observer(event_type, callback)

    def _remove_observer(self, event_type: Any, callback: TCallable):
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def post_notification(self, **kwargs):
        """
        Notify all subscribers of a specific event type.
        """
        if "notification" in kwargs:
            notification = kwargs["notification"]
            assert isinstance(notification, Notification), "notification must be of type Notification"
        else:
            event_type = kwargs["event_type"]
            assert event_type is not None, "event_type must not be None"
            notification = Notification(event_type, kwargs.get("source"), kwargs.get("context"))

        with self._lock:
            if notification.event_type in self._subscribers:
                for callback in self._subscribers[notification.event_type]:
                    callback(notification)
