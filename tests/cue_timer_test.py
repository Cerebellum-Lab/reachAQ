"""Tests for the Tone 2 deadline timer.

The cue interval is a drawn experimental quantity, so lateness in delivering
Tone 2 is error in the independent variable. Two properties therefore matter
more than the mechanics:

  * it waits against an absolute deadline, not a duration, so a late or slow
    arm does not shift the cue further out; and
  * it reports the lateness it actually achieved, so the timing can be checked
    against recorded NI edges rather than assumed.

The clock and sleep are injected throughout so these run instantly and
deterministically instead of waiting in real time.
"""

import threading

import pytest

from tools.acquisition.model.cue_timer import (
    CUE_TIMER_PRIORITY_ENV_VAR,
    DEFAULT_CUE_TIMER_PRIORITY,
    SUGGESTED_CUE_TIMER_PRIORITY,
    CueTimer,
)


class _FakeClock:
    """A clock that only advances when the timer sleeps or spins."""

    def __init__(self, start=0.0, spin_step=0.0005):
        self.now = start
        self.spin_step = spin_step
        self.sleeps = []

    def __call__(self):
        # Every read advances slightly, so a spin loop always terminates.
        self.now += self.spin_step
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _run_inline(**kwargs):
    """Build a timer whose 'thread' runs inline, so tests stay deterministic."""
    started = []

    class _InlineThread:
        def __init__(self, target, args=(), name=None):
            self._target = target
            self._args = args
            self.name = name
            self._done = False

        def start(self):
            started.append(self)
            self._target(*self._args)
            self._done = True

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    timer = CueTimer(
        thread_factory=lambda **kw: _InlineThread(**kw),
        apply_priority=lambda **kw: True,
        **kwargs,
    )
    return timer, started


def test_it_fires_at_the_deadline_and_reports_lateness():
    clock = _FakeClock()
    timer, _ = _run_inline(clock=clock, sleep=clock.sleep)
    fired = []

    timer.schedule(0.100, lambda at, late: fired.append((at, late)))

    assert len(fired) == 1
    fired_at, lateness = fired[0]
    assert fired_at >= 0.100
    assert lateness >= 0.0
    # The spin margin is 2 ms, so it must not overshoot by anything like that.
    assert lateness < 0.002


def test_it_waits_against_an_absolute_deadline_not_a_duration():
    """A slow arm must not push the cue further out."""
    clock = _FakeClock(start=0.050)
    timer, _ = _run_inline(clock=clock, sleep=clock.sleep)
    fired = []

    timer.schedule(0.100, lambda at, late: fired.append(at))

    assert fired and fired[0] >= 0.100
    # Total slept is bounded by the remaining 50 ms, not the full 100 ms.
    assert sum(clock.sleeps) <= 0.050


def test_a_deadline_in_the_past_fires_immediately():
    """A late arm still delivers the cue, and records how late it was."""
    clock = _FakeClock(start=1.000)
    timer, _ = _run_inline(clock=clock, sleep=clock.sleep)
    fired = []

    timer.schedule(0.500, lambda at, late: fired.append(late))

    assert len(fired) == 1
    assert fired[0] > 0.4, "lateness must be reported, not hidden"
    assert clock.sleeps == [], "must not sleep for a deadline already passed"


def test_it_sleeps_in_bounded_chunks_so_cancel_is_responsive():
    clock = _FakeClock()
    timer, _ = _run_inline(clock=clock, sleep=clock.sleep)
    timer.schedule(5.000, lambda at, late: None)
    assert clock.sleeps, "expected the timer to sleep"
    assert max(clock.sleeps) <= 0.050


def test_cancelling_before_the_deadline_suppresses_the_callback():
    fired = []
    timer = CueTimer(apply_priority=lambda **kw: True)
    ready = threading.Event()

    def callback(at, late):
        fired.append(at)

    # Far enough out that the cancel lands first.
    timer.schedule(timer._clock() + 30.0, callback)
    ready.set()
    assert timer.cancel() is True
    timer.join(5)
    assert fired == []


def test_cancel_reports_false_when_nothing_is_pending():
    timer = CueTimer(apply_priority=lambda **kw: True)
    assert timer.cancel() is False


def test_scheduling_twice_is_refused():
    timer = CueTimer(apply_priority=lambda **kw: True)
    timer.schedule(timer._clock() + 30.0, lambda at, late: None)
    try:
        with pytest.raises(RuntimeError, match="already pending"):
            timer.schedule(timer._clock() + 30.0, lambda at, late: None)
    finally:
        timer.cancel()
        timer.join(5)


def test_a_failing_callback_does_not_kill_the_thread():
    clock = _FakeClock()
    timer, _ = _run_inline(clock=clock, sleep=clock.sleep)

    def explode(at, late):
        raise RuntimeError("tone send failed")

    timer.schedule(0.010, explode)  # must not propagate


def test_real_time_priority_is_off_by_default():
    """Measured: raising the priority costs a 7 ms tail and buys nothing.

    The spin already holds the CPU at the deadline, so priority cannot help,
    and a busy-spinning real-time thread is what Linux RT throttling suspends.
    300 deadlines at 20 ms against 16 competing processes gave max 0.0007 ms at
    normal priority against 7.18 ms at SCHED_FIFO 70.
    """
    assert DEFAULT_CUE_TIMER_PRIORITY is None
    attempts = []
    timer = CueTimer(apply_priority=lambda **kw: attempts.append(kw) or True)
    assert timer._priority is None

    timer.schedule(timer._clock() - 1.0, lambda at, late: None)
    timer.join(5)
    assert attempts == [], "must not request a real-time policy by default"


def test_the_suggested_priority_yields_to_the_stim_loop():
    """If a rig does opt in, the 5 ms closed loop at 80 must still win."""
    assert SUGGESTED_CUE_TIMER_PRIORITY < 80


def test_opting_in_requests_the_priority_but_tolerates_refusal():
    """No rtprio allowance must not stop the cue from firing."""
    clock = _FakeClock()
    attempts = []

    class _InlineThread:
        def __init__(self, target, args=(), name=None):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    def refuse(**kwargs):
        attempts.append(kwargs.get("priority"))
        return False

    timer = CueTimer(
        clock=clock, sleep=clock.sleep,
        thread_factory=lambda **kw: _InlineThread(**kw),
        apply_priority=refuse,
        environ={CUE_TIMER_PRIORITY_ENV_VAR: "70"},
    )
    fired = []
    timer.schedule(0.010, lambda at, late: fired.append(at))

    assert attempts == [70]
    assert len(fired) == 1, "the cue must still fire at normal priority"


@pytest.mark.parametrize("value,expected", [
    ("", None),
    ("55", 55),
    ("abc", None),
    ("0", None),
    ("off", None),
    ("100", None),
    ("-5", None),
])
def test_the_priority_is_opt_in_and_bounded(value, expected):
    """A typo must never be the thing that enables real-time priority."""
    timer = CueTimer(environ={CUE_TIMER_PRIORITY_ENV_VAR: value},
                     apply_priority=lambda **kw: True)
    assert timer._priority == expected


def test_the_env_var_name_is_stable():
    assert CUE_TIMER_PRIORITY_ENV_VAR == "REACHAQ_CUE_TIMER_RT_PRIORITY"
