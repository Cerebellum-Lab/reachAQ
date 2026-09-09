"""Fire Tone 2 at its deadline, accurately enough to be worth measuring.

A `threading.Timer` is not good enough here. The cue interval is a drawn
experimental quantity, so a Tone 2 delivered late is error in the independent
variable, and a plain sleep-for-a-duration timer inherits the scheduler's
wake-up latency: measured on this rig at 900 Hz against CPU load, a periodic
host wake-up was 3.662 ms late at p99 and 6.050 ms at worst.

So this timer sleeps against an *absolute* deadline and spins out the last
couple of milliseconds, where sleep granularity stops being trustworthy.

Real-time priority is deliberately NOT the default, which is the opposite of
what this module first did. Measured, 300 deadlines at 20 ms with 16 competing
processes:

    normal priority        p50 0.0005 ms   p99 0.0007 ms   max 0.0007 ms
    SCHED_FIFO (70)        p50 0.0005 ms   p99 0.0007 ms   max 7.1800 ms

The spin already holds the CPU, so priority buys nothing at the deadline - and
a busy-spinning real-time thread is exactly what Linux RT throttling exists to
suspend (`sched_rt_runtime_us` reserves 5% of each period for normal tasks),
which is the most likely source of that 7.18 ms outlier. Raising the priority
made the tail 10,000x worse. It stays available through the environment
variable for a rig that measures otherwise, but off by default.

It also records when it actually fired. Timing that is asserted but never
measured tends to be wrong, so the achieved lateness travels with the trial
evidence instead of being assumed to be zero.

Two things this does NOT solve, and they must not be conflated with it:

  * Whatever the callback then does. A tone routed over CAN pays the transport's
    own latency and jitter after this timer fires, which can be larger than
    everything measured above. That path needs bench measurement against
    recorded NI-DAQ edges before any timing claim is made about Tone 2 itself.
  * Real-time priority is best effort. Without an rtprio allowance the thread
    runs normally and the recorded lateness will show it.
"""

import os
import threading
import time
import typing

from autotrainer.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)

CUE_TIMER_PRIORITY_ENV_VAR = "REACHAQ_CUE_TIMER_RT_PRIORITY"

# Off by default; see the module note. When a rig does opt in, this is the
# suggested value: below the stim capture loop's 80, because that loop owns a
# 5 ms closed-loop deadline and a cue is not worth preempting it.
DEFAULT_CUE_TIMER_PRIORITY = None
SUGGESTED_CUE_TIMER_PRIORITY = 70

# Sleep is only trusted to this resolution; the remainder is spun out. Kept
# small because a spinning thread at real-time priority is expensive.
DEFAULT_SPIN_MARGIN_SECONDS = 0.002

# Cancellation is checked at least this often while waiting.
_MAX_SLEEP_CHUNK_SECONDS = 0.050


def _requested_priority(environ=None) -> typing.Optional[int]:
    """
    Return the opt-in real-time priority, or None to stay at normal priority.

    Every unusable value resolves to None rather than to a fallback priority:
    measurement says raising the priority costs a 7 ms tail and buys nothing, so
    a typo must not be the thing that enables it.
    """
    source = os.environ if environ is None else environ
    raw = (source.get(CUE_TIMER_PRIORITY_ENV_VAR) or "").strip().lower()
    if not raw or raw in {"0", "off", "false", "no", "none", "disabled"}:
        return None
    try:
        priority = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; staying at normal priority",
                       CUE_TIMER_PRIORITY_ENV_VAR, raw)
        return None
    if not 1 <= priority <= 99:
        logger.warning("%s=%r is outside 1..99; staying at normal priority",
                       CUE_TIMER_PRIORITY_ENV_VAR, raw)
        return None
    return priority


class CueTimer:
    """
    Call back once at an absolute deadline, and report how late it was.

    The clock, sleep and thread factory are injected so tests can drive the wait
    deterministically instead of waiting in real time.
    """

    def __init__(
        self,
        *,
        priority: typing.Optional[int] = None,
        spin_margin_seconds: float = DEFAULT_SPIN_MARGIN_SECONDS,
        clock: typing.Callable[[], float] = time.perf_counter,
        sleep: typing.Callable[[float], None] = time.sleep,
        thread_factory: typing.Optional[typing.Callable[..., threading.Thread]] = None,
        apply_priority: typing.Optional[typing.Callable[..., bool]] = None,
        environ: typing.Optional[typing.Mapping[str, str]] = None,
    ):
        self._priority = _requested_priority(environ) if priority is None else priority
        self._spin_margin = max(0.0, float(spin_margin_seconds))
        self._clock = clock
        self._sleep = sleep
        self._thread_factory = thread_factory or (
            lambda **kwargs: threading.Thread(daemon=True, **kwargs)
        )
        self._apply_priority = apply_priority
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: typing.Optional[threading.Thread] = None
        self._deadline: typing.Optional[float] = None

    @property
    def is_pending(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def deadline_perf_time(self) -> typing.Optional[float]:
        with self._lock:
            return self._deadline

    def _raise_priority(self) -> bool:
        if self._priority is None:
            return False
        apply_priority = self._apply_priority
        if apply_priority is None:
            try:
                from autotrainer.video.realtime_priority import apply_realtime_priority
            except Exception as err:  # pragma: no cover - import guard
                logger.info("real-time priority unavailable for the cue timer: %s", err)
                return False
            apply_priority = apply_realtime_priority
        try:
            return bool(apply_priority(priority=self._priority))
        except Exception as err:
            logger.warning("could not raise the cue timer thread: %s", err)
            return False

    def schedule(self, deadline_perf_time: float, callback) -> None:
        """
        Run ``callback(fired_at, lateness_seconds)`` at ``deadline_perf_time``.

        A deadline already in the past fires immediately rather than being
        skipped, so a late arm still delivers the cue and records how late.
        """
        with self._lock:
            if self.is_pending:
                raise RuntimeError("A cue timer is already pending")
            self._cancel.clear()
            self._deadline = float(deadline_perf_time)
            thread = self._thread = self._thread_factory(
                target=self._wait_and_fire,
                args=(float(deadline_perf_time), callback),
                name="CueTimer",
            )
        thread.start()

    def cancel(self) -> bool:
        """Cancel a pending fire. Returns True if it was still pending."""
        with self._lock:
            pending = self.is_pending
            self._cancel.set()
            return pending

    def join(self, timeout: typing.Optional[float] = None) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _wait_and_fire(self, deadline: float, callback) -> None:
        raised = self._raise_priority()
        if not raised and self._priority is not None:
            logger.info("cue timer is running at normal priority; Tone 2 lateness "
                        "will reflect scheduler delay")

        while not self._cancel.is_set():
            remaining = deadline - self._clock()
            if remaining <= self._spin_margin:
                break
            self._sleep(min(remaining - self._spin_margin, _MAX_SLEEP_CHUNK_SECONDS))

        # Spin out the remainder. Sleep granularity is not trustworthy at the
        # millisecond scale, and this window is deliberately short.
        while not self._cancel.is_set() and self._clock() < deadline:
            pass

        if self._cancel.is_set():
            logger.debug("cue timer cancelled before firing")
            return

        fired_at = self._clock()
        lateness = fired_at - deadline
        try:
            callback(fired_at, lateness)
        except Exception as err:
            logger.exception("cue timer callback failed: %s", err)
