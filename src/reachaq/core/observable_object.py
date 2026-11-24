import os
from typing import Protocol, Callable, Any, TypeVar
from typing_extensions import Self

import verboselogs
from events import Events

from reachaq.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)


_debug_properties_change = {
    v.strip() for v in os.getenv("AUTOTRAINER_DEBUG_PROPERTIES", "").split(",")
}


HandlerT = TypeVar("HandlerT", bound=Callable[..., None])

AnyNewValue = Any
AnyOldValue = Any


class EventHandler(Protocol[HandlerT]):
    """Single event handler"""

    __call__: HandlerT

    def __iadd__(self, handler: HandlerT) -> Self:
        """Register event handler"""

    def __isub__(self, handler: HandlerT) -> Self:
        """Unregister event handler"""


class ObservableObjectProtocol(Protocol):
    """Allow classes that do not inherit from ObservableObject to indicate they support the property_changed event."""

    property_changed: EventHandler[Callable[[str, AnyNewValue, AnyOldValue], None]]


class ObservableObject(Events):
    """
    Defines a class with observable property change events.  This is a common pattern for UI frameworks in
    particular where some form of a view layer and some type of model layer need to communicate changes in a decoupled
    way.  However, it is applicable in many other situations as well.

    The property_changed event is used by one or more listeners to respond to property changes.

    _on_property_changed is a convenience method that will only generate an event if the new value does not pass
    the == test with the old value.  This prevents needless events and more importantly endless recursion when two
    objects are listening to each other to stay in sync (e.g., UI and model where a change may originate in either).
    """

    # type hint for dynamic event function (added by Events):
    property_changed: EventHandler[Callable[[str, AnyNewValue, AnyOldValue], None]]

    def __init__(self, event_names=()):
        super().__init__(event_names + ("property_changed",))

    # def __bool__(self):
    #     # NB: was used to make the `if model:` in Machine.__init__ method works,
    #     # when tried mixing an ObservableObject with transitions.Machine
    #     # but there were other issues too, so keeping commented, as ref. at worst.
    #     # NB2: this is because the __bool__ method is otherwise overridden by the Events.__len__ method
    #     #  which returns the nbr of events handled by the given instance.
    #     return True

    def _on_property_changed(self, name: str, new_value: AnyNewValue, old_value: AnyOldValue):
        """
        Generate a property-changed event if the new value does not pass the == test with the old value.

        This is a convenience method for only generating an event when the value has actually changed (as defined by
        `==`).  If `==` is not an appropriate test for a given property, a subclass should override this method for
        handling that property, or the caller should use the property_changed event directly.

        A common pattern for subclasses would be:

        ```
        def set_age(self, value: int):
            prev, self._age = self._age, value  # assign the value before call the _on_property_changed event
            self._on_property_changed("age", value, prev)
        ```

        This will only generate an event if the new value is different from the old value and update the member variable
        with the new value if it changed.

        An important note for event handlers is that the property on the object will be the old value at the time the
        event is handled.  Handlers must use the `new_value` argument to get the new value.

        Args:
            name (str): Name of the property that changed.
            new_value: New value of the property.
            old_value: Old/current value of the property.
        """
        if old_value == new_value:
            return old_value

        if __debug__:
            if name in _debug_properties_change:
                logger.debug("%s: property %r: from %s to %s", self, name, old_value, new_value,
                             stack_info=logger.level == verboselogs.SPAM)

        self.property_changed(name, new_value, old_value)
        return new_value
