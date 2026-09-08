"""Pre-cue pellet authorization.

Before Tone 1 fires, the pellet the animal is about to reach for should be
verified as actually present.  Without this a trial can run to completion,
consume a cue, and be scored against a pellet that was never delivered.

reach-training performs this check against camera ROIs.  reachAQ has no
operator ROI surface: presence evidence comes from the pellet board, so the
methods here are board-evidence methods rather than ROI methods.

Two methods are offered:

* ``PELLET_PRESENCE`` authorizes from recent presence evidence already being
  reported by the board.
* ``BEFORE_SEND`` asks the board to confirm immediately before the send, which
  costs a round trip but closes the window between the last report and the cue.

Both require firmware that advertises the ``pellet_precheck`` capability.  When
the capability is absent, the method is unavailable and the caller is told so
explicitly rather than silently falling back to no check.  Authorization is
bounded: a board that does not answer within the timeout fails the trial rather
than blocking it.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Dict, Optional


PELLET_PRECHECK_CAPABILITY = "pellet_precheck"

DEFAULT_PRESENCE_FRESHNESS_MS = 500
DEFAULT_AUTHORIZATION_TIMEOUT_MS = 750
MAX_AUTHORIZATION_TIMEOUT_MS = 10_000
MAX_PRESENCE_FRESHNESS_MS = 10_000


class PelletPrecheckMethod(str, enum.Enum):
    DISABLED = "disabled"
    PELLET_PRESENCE = "pellet_presence"
    BEFORE_SEND = "before_send"

    @property
    def label(self) -> str:
        return {
            PelletPrecheckMethod.DISABLED: "Disabled",
            PelletPrecheckMethod.PELLET_PRESENCE: "Pellet presence evidence",
            PelletPrecheckMethod.BEFORE_SEND: "Confirm before send",
        }[self]

    @property
    def requires_firmware(self) -> bool:
        return self is not PelletPrecheckMethod.DISABLED


class PelletPrecheckOutcome(str, enum.Enum):
    AUTHORIZED = "authorized"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    EVIDENCE_STALE = "evidence_stale"
    UNSUPPORTED = "unsupported"

    @property
    def authorized(self) -> bool:
        return self in {
            PelletPrecheckOutcome.AUTHORIZED,
            PelletPrecheckOutcome.NOT_REQUIRED,
        }

    @property
    def is_pending(self) -> bool:
        return self is PelletPrecheckOutcome.PENDING

    @property
    def trial_outcome(self) -> Optional[str]:
        """The TrialOutcome value a failure maps to, or None when not failed."""

        return {
            PelletPrecheckOutcome.FAILED: "pellet_precheck_failed",
            PelletPrecheckOutcome.TIMED_OUT: (
                "pellet_precheck_authorization_timeout"
            ),
            PelletPrecheckOutcome.EVIDENCE_STALE: "pellet_precheck_failed",
            PelletPrecheckOutcome.UNSUPPORTED: "pellet_precheck_failed",
        }.get(self)


@dataclasses.dataclass(frozen=True)
class PelletPrecheckConfiguration:
    method: PelletPrecheckMethod = PelletPrecheckMethod.DISABLED
    presence_freshness_ms: int = DEFAULT_PRESENCE_FRESHNESS_MS
    authorization_timeout_ms: int = DEFAULT_AUTHORIZATION_TIMEOUT_MS

    def __post_init__(self):
        object.__setattr__(self, "method", PelletPrecheckMethod(self.method))
        freshness = int(self.presence_freshness_ms)
        if not 1 <= freshness <= MAX_PRESENCE_FRESHNESS_MS:
            raise ValueError(
                "Presence freshness must be between 1 and "
                f"{MAX_PRESENCE_FRESHNESS_MS} ms"
            )
        timeout = int(self.authorization_timeout_ms)
        if not 1 <= timeout <= MAX_AUTHORIZATION_TIMEOUT_MS:
            raise ValueError(
                "Authorization timeout must be between 1 and "
                f"{MAX_AUTHORIZATION_TIMEOUT_MS} ms"
            )
        object.__setattr__(self, "presence_freshness_ms", freshness)
        object.__setattr__(self, "authorization_timeout_ms", timeout)

    def to_record(self) -> Dict[str, Any]:
        record = dataclasses.asdict(self)
        record["method"] = self.method.value
        return record


@dataclasses.dataclass(frozen=True)
class PelletPrecheckEvidence:
    """What the board reported, and when."""

    observed_at: Optional[float] = None
    pellet_present: Optional[bool] = None
    authorized: Optional[bool] = None
    responded_at: Optional[float] = None
    firmware_response: str = ""

    def __post_init__(self):
        for field in ("observed_at", "responded_at"):
            value = getattr(self, field)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field} must be finite")


@dataclasses.dataclass(frozen=True)
class PelletPrecheckResult:
    outcome: PelletPrecheckOutcome
    method: PelletPrecheckMethod
    detail: str = ""
    evidence_age_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    firmware_response: str = ""

    @property
    def authorized(self) -> bool:
        return self.outcome.authorized

    @property
    def is_pending(self) -> bool:
        return self.outcome.is_pending

    @property
    def trial_outcome(self) -> Optional[str]:
        return self.outcome.trial_outcome

    def to_record(self) -> Dict[str, Any]:
        record = dataclasses.asdict(self)
        record["outcome"] = self.outcome.value
        record["method"] = self.method.value
        record["trial_outcome"] = self.trial_outcome
        return record


def capability_available(reported_capabilities) -> bool:
    return PELLET_PRECHECK_CAPABILITY in set(reported_capabilities or ())


def evaluate_precheck(
    configuration: PelletPrecheckConfiguration,
    *,
    now_perf_time: float,
    evidence: Optional[PelletPrecheckEvidence] = None,
    reported_capabilities=(),
) -> PelletPrecheckResult:
    """Decide whether the cue is authorized for this trial."""

    method = configuration.method
    if method is PelletPrecheckMethod.DISABLED:
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.NOT_REQUIRED,
            method=method,
            detail="Pellet precheck is disabled",
        )

    if not capability_available(reported_capabilities):
        # Never silently degrade to no check: the operator selected a method.
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.UNSUPPORTED,
            method=method,
            detail=(
                "Pellet board firmware does not advertise the "
                f"{PELLET_PRECHECK_CAPABILITY} capability"
            ),
        )

    now_perf_time = float(now_perf_time)
    if not math.isfinite(now_perf_time):
        raise ValueError("Evaluation time must be finite")
    evidence = evidence or PelletPrecheckEvidence()

    if method is PelletPrecheckMethod.PELLET_PRESENCE:
        return _evaluate_presence(configuration, now_perf_time, evidence)
    return _evaluate_before_send(configuration, now_perf_time, evidence)


def _evaluate_presence(
    configuration: PelletPrecheckConfiguration,
    now_perf_time: float,
    evidence: PelletPrecheckEvidence,
) -> PelletPrecheckResult:
    method = PelletPrecheckMethod.PELLET_PRESENCE
    if evidence.observed_at is None or evidence.pellet_present is None:
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.EVIDENCE_STALE,
            method=method,
            detail="No pellet presence evidence was reported",
        )
    age_ms = (now_perf_time - float(evidence.observed_at)) * 1000.0
    if age_ms > configuration.presence_freshness_ms:
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.EVIDENCE_STALE,
            method=method,
            detail=(
                f"Presence evidence is {age_ms:.0f} ms old, older than the "
                f"{configuration.presence_freshness_ms} ms window"
            ),
            evidence_age_ms=age_ms,
        )
    if not evidence.pellet_present:
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.FAILED,
            method=method,
            detail="The board reports no pellet present",
            evidence_age_ms=age_ms,
        )
    return PelletPrecheckResult(
        outcome=PelletPrecheckOutcome.AUTHORIZED,
        method=method,
        detail="Recent presence evidence authorizes the cue",
        evidence_age_ms=age_ms,
    )


def _evaluate_before_send(
    configuration: PelletPrecheckConfiguration,
    now_perf_time: float,
    evidence: PelletPrecheckEvidence,
) -> PelletPrecheckResult:
    method = PelletPrecheckMethod.BEFORE_SEND
    requested_at = evidence.observed_at
    if requested_at is None:
        raise ValueError("Before-send precheck requires the request time")
    if evidence.authorized is None or evidence.responded_at is None:
        elapsed_ms = (now_perf_time - float(requested_at)) * 1000.0
        if elapsed_ms >= configuration.authorization_timeout_ms:
            return PelletPrecheckResult(
                outcome=PelletPrecheckOutcome.TIMED_OUT,
                method=method,
                detail=(
                    "The board did not authorize within "
                    f"{configuration.authorization_timeout_ms} ms"
                ),
                latency_ms=elapsed_ms,
                firmware_response=evidence.firmware_response,
            )
        # Still inside the window; the caller keeps waiting. This is distinct
        # from a timeout so a pending answer never scores the trial.
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.PENDING,
            method=method,
            detail="Awaiting board authorization",
            latency_ms=elapsed_ms,
            firmware_response=evidence.firmware_response,
        )

    latency_ms = (float(evidence.responded_at) - float(requested_at)) * 1000.0
    if latency_ms > configuration.authorization_timeout_ms:
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.TIMED_OUT,
            method=method,
            detail=(
                f"The board answered after {latency_ms:.0f} ms, beyond the "
                f"{configuration.authorization_timeout_ms} ms timeout"
            ),
            latency_ms=latency_ms,
            firmware_response=evidence.firmware_response,
        )
    if not evidence.authorized:
        return PelletPrecheckResult(
            outcome=PelletPrecheckOutcome.FAILED,
            method=method,
            detail="The board refused to authorize the send",
            latency_ms=latency_ms,
            firmware_response=evidence.firmware_response,
        )
    return PelletPrecheckResult(
        outcome=PelletPrecheckOutcome.AUTHORIZED,
        method=method,
        detail="The board authorized the send",
        latency_ms=latency_ms,
        firmware_response=evidence.firmware_response,
    )
