"""Hold the next automatic pellet send until the inter-trial target elapses.

`InterTrialTimingPolicy` decides; this owns the consequences of that decision:
holding the send, re-checking when the target is due, releasing the hold, and
recording how the timing actually came out.

The hold goes through `SendBlockReasons` rather than straight to
`BehaviorAlgorithm.pellet_send_block_reason`, because intertrial *analysis*
already owns that string. Two subsystems writing one slot means whichever
clears first releases a pellet the other still wants held.

A late trial is never delayed further. The policy reports `READY_LATE` and this
releases immediately, because padding an already-late trial only compounds the
drift rather than restoring the intended spacing.

Nothing here is active unless the operator enables inter-trial timing. A
disabled policy always answers send-now, so this can be wired unconditionally.
"""

import time
import typing

from autotrainer.core.logging import get_verbose_logger

from .cue_timer import CueTimer
from .intertrial_timing import (
    InterTrialDecision,
    InterTrialTimingEvaluation,
    InterTrialTimingPolicy,
)

logger = get_verbose_logger(__name__)

# The name this subsystem holds the send under.
BLOCK_NAME = "intertrial_timing"


class InterTrialSendGate:
    """Own the send hold for one inter-trial interval."""

    def __init__(
        self,
        policy: typing.Optional[InterTrialTimingPolicy] = None,
        block_reasons=None,
        *,
        timer_factory: typing.Optional[typing.Callable[[], CueTimer]] = None,
        clock: typing.Callable[[], float] = time.perf_counter,
        observe: typing.Optional[typing.Callable[[InterTrialTimingEvaluation], None]] = None,
    ):
        self._policy = policy or InterTrialTimingPolicy()
        self._block_reasons = block_reasons
        self._timer_factory = timer_factory or CueTimer
        self._clock = clock
        self._observe = observe
        self._timer: typing.Optional[CueTimer] = None

    @property
    def policy(self) -> InterTrialTimingPolicy:
        return self._policy

    @property
    def is_holding(self) -> bool:
        return bool(self._block_reasons is not None and self._block_reasons.holds(BLOCK_NAME))

    def arm(self, dispatch_perf_time: float) -> InterTrialTimingEvaluation:
        """
        Start the interval at a dispatched retrieval and hold if it is early.

        Returns the evaluation so the caller can record it as trial evidence.
        """
        self._cancel_timer()
        if not self._policy.is_enabled:
            self._release_hold()
            return self._policy.evaluate(self._clock())

        self._policy.arm(dispatch_perf_time)
        return self._refresh()

    def evaluate(self, *, is_retry: bool = False) -> InterTrialTimingEvaluation:
        """Decide whether a send that is ready now may go out."""
        return self._policy.evaluate(self._clock(), is_retry=is_retry)

    def release(self, *, disarm: bool = True) -> None:
        """Release the hold, whether or not the target was reached."""
        self._cancel_timer()
        self._release_hold()
        if disarm:
            self._policy.disarm()

    # ------------------------------------------------------------------ internals

    def _refresh(self) -> InterTrialTimingEvaluation:
        now = self._clock()
        evaluation = self._policy.evaluate(now)

        if evaluation.decision is InterTrialDecision.WAIT:
            remaining = evaluation.remaining_seconds
            self._set_hold(
                f"waiting {remaining:.2f} s for the inter-trial target"
            )
            self._schedule(now + max(remaining, 0.0))
            return evaluation

        self._release_hold()
        if evaluation.was_late and evaluation.lateness_seconds is not None:
            # Recorded, not corrected. The interval was already longer than the
            # target and holding the send would only make it longer still.
            logger.info(
                "inter-trial target missed by %.3f s; sending without further delay",
                evaluation.lateness_seconds,
            )
        self._report(evaluation)
        return evaluation

    def _on_target_due(self, fired_at: float, lateness_seconds: float) -> None:
        try:
            self._refresh()
        except Exception as err:
            # A failure here would leave the send held forever, which is worse
            # than an inaccurate interval.
            logger.exception("inter-trial re-check failed, releasing the hold: %s", err)
            self._release_hold()

    def _schedule(self, deadline_perf_time: float) -> None:
        timer = self._timer = self._timer_factory()
        timer.schedule(deadline_perf_time, self._on_target_due)

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _set_hold(self, detail: str) -> None:
        if self._block_reasons is not None:
            self._block_reasons.set(BLOCK_NAME, detail)

    def _release_hold(self) -> None:
        if self._block_reasons is not None:
            self._block_reasons.clear(BLOCK_NAME)

    def _report(self, evaluation: InterTrialTimingEvaluation) -> None:
        if self._observe is None:
            return
        try:
            self._observe(evaluation)
        except Exception as err:
            logger.warning("inter-trial timing observer failed: %s", err)
