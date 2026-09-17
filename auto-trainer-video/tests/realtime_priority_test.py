"""Tests for putting the stim capture thread on SCHED_FIFO.

The behaviour that matters is what happens when it does NOT work. A rig without
an rtprio allowance, a developer machine on Windows, and a container without
CAP_SYS_NICE all have to keep acquiring at normal priority. A raise here would
stop the camera process from ever reaching its capture loop.

The second thing that matters is that success is read back rather than assumed.
A silent no-op would present as a working real-time loop while behaving exactly
like the untuned one - the same failure mode as TensorFlow silently falling back
to CPU, which produced an entirely invalid benchmark earlier in this work.
"""

import os

import pytest

from autotrainer.video import realtime_priority
from autotrainer.video.realtime_priority import (
    DEFAULT_STIM_RT_PRIORITY,
    STIM_RT_PRIORITY_ENV_VAR,
    apply_realtime_priority,
    requested_priority,
)


def test_unset_requests_the_default_priority():
    assert requested_priority({}) == DEFAULT_STIM_RT_PRIORITY


def test_default_priority_is_below_the_kernel_realtime_threads():
    """migration/* run at 99; staying under it keeps the machine recoverable."""
    assert 1 <= DEFAULT_STIM_RT_PRIORITY < 99


@pytest.mark.parametrize("value", ["0", "off", "false", "no", "none", "disabled", "OFF"])
def test_the_variable_can_disable_it(value):
    assert requested_priority({STIM_RT_PRIORITY_ENV_VAR: value}) is None


def test_an_explicit_priority_is_used():
    assert requested_priority({STIM_RT_PRIORITY_ENV_VAR: "42"}) == 42


@pytest.mark.parametrize("value", ["abc", "", "   ", "-5", "0.5", "100", "999"])
def test_an_unusable_value_falls_back_rather_than_raising(value):
    """A typo must not silently drop the loop to normal priority."""
    assert requested_priority({STIM_RT_PRIORITY_ENV_VAR: value}) == DEFAULT_STIM_RT_PRIORITY


def test_the_env_var_name_is_stable():
    """Operators put this in the rig's service environment."""
    assert STIM_RT_PRIORITY_ENV_VAR == "REACHAQ_STIM_RT_PRIORITY"


def test_disabled_never_touches_the_scheduler(monkeypatch):
    called = {"set": False}

    def should_not_run(*args, **kwargs):
        called["set"] = True

    monkeypatch.setattr(realtime_priority.os, "sched_setscheduler", should_not_run,
                        raising=False)
    assert apply_realtime_priority(environ={STIM_RT_PRIORITY_ENV_VAR: "0"}) is False
    assert called["set"] is False


def test_an_unsupported_platform_is_not_an_error(monkeypatch):
    """Windows has no scheduler API; the capture loop must still start."""
    monkeypatch.setattr(realtime_priority, "realtime_priority_supported", lambda: False)
    assert apply_realtime_priority(priority=80) is False


def test_permission_denied_is_reported_and_tolerated(monkeypatch):
    """The untuned-rig case: ulimit -r is 0, so the request is refused."""
    monkeypatch.setattr(realtime_priority, "realtime_priority_supported", lambda: True)

    def deny(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(realtime_priority.os, "sched_setscheduler", deny, raising=False)
    assert apply_realtime_priority(priority=80) is False


def test_other_os_errors_are_tolerated(monkeypatch):
    monkeypatch.setattr(realtime_priority, "realtime_priority_supported", lambda: True)

    def fail(*args, **kwargs):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(realtime_priority.os, "sched_setscheduler", fail, raising=False)
    assert apply_realtime_priority(priority=80) is False


def test_success_is_read_back_from_the_scheduler(monkeypatch):
    monkeypatch.setattr(realtime_priority, "realtime_priority_supported", lambda: True)
    recorded = {}

    def record(pid, policy, param):
        recorded["pid"] = pid
        recorded["policy"] = policy
        recorded["priority"] = param.sched_priority

    monkeypatch.setattr(realtime_priority.os, "sched_setscheduler", record, raising=False)
    monkeypatch.setattr(realtime_priority.os, "SCHED_FIFO", 1, raising=False)
    monkeypatch.setattr(realtime_priority.os, "sched_getscheduler", lambda pid: 1,
                        raising=False)
    monkeypatch.setattr(realtime_priority.os, "sched_param",
                        lambda value: type("P", (), {"sched_priority": value})(),
                        raising=False)

    assert apply_realtime_priority(priority=80) is True
    assert recorded["pid"] == 0, "must apply to the calling thread"
    assert recorded["priority"] == 80


def test_a_silent_no_op_is_reported_as_failure(monkeypatch):
    """setscheduler returning without applying must not read as success."""
    monkeypatch.setattr(realtime_priority, "realtime_priority_supported", lambda: True)
    monkeypatch.setattr(realtime_priority.os, "sched_setscheduler",
                        lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(realtime_priority.os, "SCHED_FIFO", 1, raising=False)
    # Still reporting the normal policy afterwards.
    monkeypatch.setattr(realtime_priority.os, "sched_getscheduler", lambda pid: 0,
                        raising=False)
    monkeypatch.setattr(realtime_priority.os, "sched_param",
                        lambda value: type("P", (), {"sched_priority": value})(),
                        raising=False)

    assert apply_realtime_priority(priority=80) is False


@pytest.mark.skipif(not hasattr(os, "sched_setscheduler"),
                    reason="no scheduler API on this platform")
def test_the_real_call_never_raises():
    """Against the real OS: permitted or not, this returns a bool."""
    assert apply_realtime_priority(priority=1) in (True, False)


def test_the_capture_loop_only_raises_priority_for_the_stim_camera():
    """Every camera process going real-time would take CPU from the others."""
    import inspect

    from autotrainer.video import video_capture

    source = inspect.getsource(video_capture.VideoCapture._run_capture_loop)
    assert "apply_realtime_priority()" in source
    guard = source.split("apply_realtime_priority()")[0]
    assert "if self._stim_detector is not None:" in guard
