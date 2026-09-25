"""Keep the shared NI-DAQ input stream running whenever nothing else needs it.

The stream used to start only from a button, or when System Mode started, so
reachAQ opened with every input graph dark and the operator had to know to
press Start before a laser pulse or a tone could be seen. Ben asked for it to
start by itself, for every NI-DAQ channel. The rule lives here, in one place,
rather than in each view that happens to show a trace.

What it decides is small:

  * when asked (the configuration loaded, a hardware refresh, a new DAQ
    ports plan, acquisition stopping), start the stream if it is enabled and
    configured, the application says it may, and nobody holds it;
  * a holder - the DAQ Monitor, a configuration load - stops it and keeps it
    stopped until every holder lets go, then it starts again;
  * a failed start is left failed. The monitor keeps the error, where the
    Hardware panel and the stream labels show it, and nothing here retries
    it: a retry loop would relaunch a broken worker forever, each attempt
    clearing the error the operator needs to read. The next request - a
    hardware refresh, for one - tries again.

Starting runs device discovery and the exact-task preflight, which take
seconds and can take the full timeouts on a sick driver, so requests start
the stream on a thread of their own and return at once. Most callers are on
the Qt thread.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional, Tuple

from autotrainer.core.logging import get_verbose_logger


logger = get_verbose_logger(__name__)


class NidaqStreamAutoStart:
    """Start the stream when wanted; stop it while anything holds it."""

    def __init__(
        self,
        monitor,
        *,
        may_start: Callable[[], bool],
        background: bool = True,
    ):
        self._monitor = monitor
        self._may_start = may_start
        self._background = background
        # Two locks, so a request from the Qt thread never waits on a start
        # in progress. The state lock is only ever held for a few lines.
        self._state_lock = threading.Lock()
        # Held across the decision to start and the start itself, and by a
        # pause across its stop, so a pause cannot land between the two and
        # be undone by a start that had already decided to run.
        self._start_lock = threading.RLock()
        self._holders: Dict[object, str] = {}
        self._thread: Optional[threading.Thread] = None
        self._pending = False
        self._closed = False

    @property
    def pause_reasons(self) -> Tuple[str, ...]:
        with self._state_lock:
            return tuple(self._holders.values())

    def request(self) -> None:
        """Start the stream if it should be running and is not."""
        with self._state_lock:
            if self._closed:
                return
            if self._background:
                if self._thread is not None:
                    # One start at a time; this one looks again when it ends.
                    self._pending = True
                    return
                # Started under the lock, so wait() never finds a thread
                # that exists but has not been started.
                self._thread = threading.Thread(
                    target=self._run, name="nidaq-stream-autostart", daemon=True)
                self._thread.start()
                return
        self._start_if_wanted()

    def pause(self, holder, reason: str) -> None:
        """Stop the stream and keep it stopped until `holder` resumes it."""
        with self._state_lock:
            self._holders[holder] = str(reason)
        with self._start_lock:
            self._monitor.stop()
            self._monitor.show_paused(str(reason))

    def resume(self, holder) -> None:
        """Let go; the stream starts again once nothing else holds it."""
        with self._state_lock:
            if self._holders.pop(holder, None) is None:
                return
            remaining = tuple(self._holders.values())
        if remaining:
            self._monitor.show_paused(remaining[-1])
            return
        self._monitor.show_paused("")
        self.request()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for a start in progress; True when none is left running."""
        with self._state_lock:
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def close(self, timeout: float = 2.0) -> None:
        with self._state_lock:
            self._closed = True
        self.wait(timeout)

    def _run(self) -> None:
        while True:
            try:
                self._start_if_wanted()
            except Exception:
                logger.exception("Automatic NI-DAQ input stream start failed")
            with self._state_lock:
                if self._pending and not self._closed:
                    self._pending = False
                    continue
                self._thread = None
                return

    def _start_if_wanted(self) -> bool:
        with self._start_lock:
            with self._state_lock:
                if self._closed or self._holders:
                    return False
            monitor = self._monitor
            if monitor.is_running or monitor.is_starting:
                return False
            if not (monitor.hardware_enabled and monitor.configuration.is_enabled):
                return False
            if not self._may_start():
                return False
            return bool(monitor.start())
