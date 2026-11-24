from enum import Enum
from typing import Tuple, Callable, Any, get_type_hints

from events import Events

from reachaq.core import EventManager
from reachaq.core.logging import get_verbose_logger

AnyOldValue = AnyNewValue = Any


logger = get_verbose_logger(__name__)


class StateMachineEvents(Events):

    state_changed: Callable[[AnyOldValue, AnyNewValue], None]
    property_changed: Callable[[str, AnyNewValue, AnyOldValue], None]


class StateMachine:
    """Generic state machine/object mixin, with events handling"""

    _events_class = StateMachineEvents

    class Properties(str, Enum):
        STATE_PROPERTY = "state"

    def __init__(self, *, initial_state, event_names: Tuple[str, ...] = ()):
        super().__init__()
        self._state = initial_state
        event_names = sorted(event_names + tuple(get_type_hints(self._events_class)))
        logger.debug("%s: creating events instance for events: %s",
                     self.__class__.__name__, event_names)
        self._events = self._events_class(event_names)
        self._event_manager = EventManager.default()
        self.post_event_content = self._event_manager.post_event_content

    @property
    def events(self):
        return self._events

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_value):
        old_value, self._state = self._state, new_value
        if new_value == old_value:
            return
        logger.verbose("%s state changed: %s -> %s", self.__class__.__name__, old_value, new_value)
        self._events.state_changed(old_value, new_value)
        self._events.property_changed(StateMachine.Properties.STATE_PROPERTY, new_value, old_value)
