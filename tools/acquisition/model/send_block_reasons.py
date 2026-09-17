"""Let several subsystems hold the pellet send without overwriting each other.

``BehaviorAlgorithm.pellet_send_block_reason`` is a single string, and
``can_send_pellet()`` treats any non-empty value as a block. That works while
one subsystem owns it. It stops working the moment two do: intertrial analysis
already sets it while waiting for a retried trial's result, and inter-trial
timing needs to hold the send as well. With one slot, whichever writes last
wins, and whichever clears first unblocks a send the other still wants held -
which would be a pellet delivered during analysis, or one delivered before the
inter-trial target elapsed.

So the reasons are held by name here and rendered into that one string. Clearing
one leaves the others in place. The rendered text is operator-facing, so it
stays a readable sentence rather than a set of keys.
"""

import threading
import typing


class SendBlockReasons:
    """Named, independently clearable reasons to hold the pellet send."""

    def __init__(self, publish: typing.Optional[typing.Callable[[str], None]] = None):
        self._lock = threading.RLock()
        self._reasons: typing.Dict[str, str] = {}
        self._publish = publish

    def set(self, name: str, detail: str) -> str:
        """Hold the send under ``name``. An empty detail clears that name."""
        if not detail:
            return self.clear(name)
        with self._lock:
            self._reasons[str(name)] = str(detail)
            return self._render_and_publish()

    def clear(self, name: str) -> str:
        with self._lock:
            self._reasons.pop(str(name), None)
            return self._render_and_publish()

    def clear_all(self) -> str:
        with self._lock:
            self._reasons.clear()
            return self._render_and_publish()

    def holds(self, name: str) -> bool:
        with self._lock:
            return str(name) in self._reasons

    @property
    def is_blocked(self) -> bool:
        with self._lock:
            return bool(self._reasons)

    @property
    def names(self) -> typing.Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._reasons))

    def render(self) -> str:
        with self._lock:
            return self._render()

    def _render(self) -> str:
        # Sorted so the same set of holds always reads the same way, rather
        # than changing with insertion order between sessions.
        return "; ".join(detail for _name, detail in sorted(self._reasons.items()))

    def _render_and_publish(self) -> str:
        rendered = self._render()
        if self._publish is not None:
            self._publish(rendered)
        return rendered
