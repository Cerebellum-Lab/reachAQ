from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class ApiEvent:
    """One published event.

    ``kind`` is normally an ``ApiEventKind``; plain ints are accepted. ``when``
    and ``index`` are the wall time and ``perf_counter_ns`` stamp set by the
    event manager, ``context`` the event payload.
    """
    kind: int
    when: Optional[datetime] = None
    index: Optional[int] = None
    context: Optional[Any] = None

    def is_same(self, other: "ApiEvent") -> bool:
        """True when ``other`` repeats this event: same kind and equal payload.

        The event manager coalesces an uninterrupted run of repeats into one
        record plus a repeat count. Timestamps are not compared.
        """
        return self.kind == other.kind and self.context == other.context
