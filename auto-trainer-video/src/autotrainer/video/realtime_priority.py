"""Run the stim-camera capture thread at real-time priority.

The 900 Hz closed loop misses its 5 ms budget under CPU contention, and the
cause is not the detector's own work. Measured on the reachAQ rig, 60 s at
900 Hz pinned to physical P-cores against 16 competing processes:

    policy         decision p99   total max   deadline misses   cycles in 60 s
    SCHED_OTHER        0.090 ms    6.050 ms   13 (0.026%)       49,480
    SCHED_FIFO 80      0.056 ms    0.155 ms    0                54,000

The detector never exceeds 0.09 ms even under load; the entire tail was
SCHED_OTHER wake-up latency, which peaked at 6.05 ms. SCHED_FIFO bounds it to
0.022 ms, removes every missed deadline, and restores the full 900 Hz cadence.
A PREEMPT_RT kernel is not needed for this: stock Linux already preempts
SCHED_OTHER for SCHED_FIFO immediately.

Priority 80 sits below the kernel's own real-time threads (migration/* run at
99), so a runaway loop cannot lock the machine out. Linux RT throttling is a
further backstop: sched_rt_runtime_us reserves 5% of each period for normal
tasks by default.

This needs an rtprio allowance, which a stock install does not grant:

    sudo bash rig-latency-setup.sh rtprio-enable    # writes limits.d, adds group

Without it the request fails and the loop keeps running exactly as before, at
normal priority. That is deliberate: a rig that has not been tuned must still
acquire.
"""

import os
import typing

from autotrainer.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)

STIM_RT_PRIORITY_ENV_VAR = "REACHAQ_STIM_RT_PRIORITY"

# Below the kernel's own RT threads, above everything in userspace.
DEFAULT_STIM_RT_PRIORITY = 80

_DISABLED_VALUES = frozenset({"0", "off", "false", "no", "none", "disabled"})

# SCHED_FIFO priorities are 1..99 on Linux.
_MIN_PRIORITY = 1
_MAX_PRIORITY = 99


def realtime_priority_supported() -> bool:
    """Whether this interpreter can request a real-time policy at all.

    False on Windows, where the scheduler API does not exist. Checked rather
    than assumed so the capture process behaves the same on a developer machine
    as on the rig.
    """
    return hasattr(os, "sched_setscheduler") and hasattr(os, "SCHED_FIFO")


def requested_priority(
    environ: typing.Optional[typing.Mapping[str, str]] = None,
) -> typing.Optional[int]:
    """
    Return the configured SCHED_FIFO priority, or None when disabled.

    An environment variable rather than a SystemConfiguration field, for the
    same reason as REACHAQ_INFERENCE_ROI and REACHAQ_POSE_BACKEND:
    SystemConfiguration.version is pinned at 57 and rejects any other value, so
    a new field would stop deployed rigs loading their configuration.
    """
    source = os.environ if environ is None else environ
    raw = source.get(STIM_RT_PRIORITY_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_STIM_RT_PRIORITY

    text = raw.strip().lower()
    if text in _DISABLED_VALUES:
        logger.info("%s=%r: stim capture stays at normal priority",
                    STIM_RT_PRIORITY_ENV_VAR, raw)
        return None

    try:
        priority = int(text)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d",
                       STIM_RT_PRIORITY_ENV_VAR, raw, DEFAULT_STIM_RT_PRIORITY)
        return DEFAULT_STIM_RT_PRIORITY

    if not _MIN_PRIORITY <= priority <= _MAX_PRIORITY:
        logger.warning("%s=%r is outside %d..%d; using %d",
                       STIM_RT_PRIORITY_ENV_VAR, raw, _MIN_PRIORITY, _MAX_PRIORITY,
                       DEFAULT_STIM_RT_PRIORITY)
        return DEFAULT_STIM_RT_PRIORITY

    return priority


def apply_realtime_priority(
    priority: typing.Optional[int] = None,
    environ: typing.Optional[typing.Mapping[str, str]] = None,
) -> bool:
    """
    Put the calling thread on SCHED_FIFO. Return True only if it took effect.

    Never raises. Acquisition must start on an untuned rig, on Windows, and in
    a container without CAP_SYS_NICE, so every failure here is reported and
    then tolerated. The return value is read back from the scheduler rather
    than inferred from the call not raising.
    """
    if priority is None:
        priority = requested_priority(environ)
    if priority is None:
        return False

    if not realtime_priority_supported():
        logger.info("real-time scheduling is not available on this platform; "
                    "stim capture stays at normal priority")
        return False

    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError:
        logger.warning(
            "not permitted to run the stim capture thread at SCHED_FIFO %d. "
            "The 900 Hz loop can miss its 5 ms budget under CPU load without it "
            "(measured: 6.05 ms worst case, versus 0.155 ms with it). Grant an "
            "rtprio allowance, or set %s=0 to silence this.",
            priority, STIM_RT_PRIORITY_ENV_VAR)
        return False
    except OSError as err:
        logger.warning("could not set SCHED_FIFO %d for stim capture: %s", priority, err)
        return False

    # Confirm rather than assume: a silent no-op here would look like a working
    # real-time loop while behaving exactly like the untuned one.
    try:
        applied = os.sched_getscheduler(0)
    except OSError as err:
        logger.warning("could not read back the scheduling policy: %s", err)
        return False
    if applied != os.SCHED_FIFO:
        logger.warning("scheduling policy is %s after requesting SCHED_FIFO", applied)
        return False

    logger.notice("stim capture thread running SCHED_FIFO at priority %d", priority)
    return True
