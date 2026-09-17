"""Select the source of reach state for the cue-timing gate.

``CueGateState.reach_active`` decides whether Tone 2 is allowed to fire, and two
different subsystems can answer it:

``live_tracking``
    The live pose stream. Available whenever live inference runs, and its
    samples carry the pose response's own capture time.

``stim_camera``
    The stim-camera first-reach detector. Sub-millisecond and already the
    closed-loop decision path, but it only exists when a stim camera is
    configured and armed.

Neither is right for every rig, so the source is selected rather than assumed,
and ``none`` keeps the current behaviour where reach never blocks the cue.

The limits here exist because a stale or missing reach observation is more
dangerous than no observation at all: it would let the gate believe the animal
is still, using evidence from seconds ago. Every observation is therefore
rejected unless it is fresh, and an unavailable or failing provider reports
nothing rather than a default.
"""

import dataclasses
import enum
import math
import os
import typing

from autotrainer.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)

REACH_STATE_SOURCE_ENV_VAR = "REACHAQ_REACH_STATE_SOURCE"
REACH_STATE_MAX_AGE_ENV_VAR = "REACHAQ_REACH_STATE_MAX_AGE_MS"

DEFAULT_MAX_AGE_MS = 250
MIN_MAX_AGE_MS = 1
MAX_MAX_AGE_MS = 5000


class ReachStateSource(str, enum.Enum):
    """Which subsystem answers "is the animal reaching right now?"."""

    NONE = "none"
    LIVE_TRACKING = "live_tracking"
    STIM_CAMERA = "stim_camera"

    @property
    def display_name(self) -> str:
        return {
            ReachStateSource.NONE: "Not used",
            ReachStateSource.LIVE_TRACKING: "Live tracking",
            ReachStateSource.STIM_CAMERA: "Stim camera",
        }[self]


_SOURCE_ALIASES = {
    "none": ReachStateSource.NONE,
    "off": ReachStateSource.NONE,
    "disabled": ReachStateSource.NONE,
    "live_tracking": ReachStateSource.LIVE_TRACKING,
    "live-tracking": ReachStateSource.LIVE_TRACKING,
    "live": ReachStateSource.LIVE_TRACKING,
    "pose": ReachStateSource.LIVE_TRACKING,
    "stim_camera": ReachStateSource.STIM_CAMERA,
    "stim-camera": ReachStateSource.STIM_CAMERA,
    "stimcam": ReachStateSource.STIM_CAMERA,
    "stim": ReachStateSource.STIM_CAMERA,
}


@dataclasses.dataclass(frozen=True)
class ReachStateConfiguration:
    """Bounded configuration for reach observation."""

    source: ReachStateSource = ReachStateSource.NONE
    max_age_ms: int = DEFAULT_MAX_AGE_MS

    def __post_init__(self):
        source = ReachStateSource(self.source)
        max_age_ms = int(self.max_age_ms)
        if not MIN_MAX_AGE_MS <= max_age_ms <= MAX_MAX_AGE_MS:
            raise ValueError(
                f"Reach observation age must be between {MIN_MAX_AGE_MS} and "
                f"{MAX_MAX_AGE_MS} ms, found {max_age_ms}"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "max_age_ms", max_age_ms)

    @property
    def max_age_seconds(self) -> float:
        return self.max_age_ms / 1000.0

    def to_record(self) -> typing.Dict[str, typing.Any]:
        return {"source": self.source.value, "max_age_ms": self.max_age_ms}

    @classmethod
    def from_environment(cls, environ=None) -> "ReachStateConfiguration":
        """
        Read the configuration from the process environment.

        Environment rather than SystemConfiguration for the same reason as the
        other runtime switches on this branch: SystemConfiguration.version is
        pinned at 57 and rejects any other value. This should move into the
        protocol schema at the next version bump, alongside the rest of
        CueTimingConfiguration.

        Never raises: an unusable value falls back to the default with a
        warning, because refusing to start a session over a typo is worse than
        running with reach gating off.
        """
        source_map = os.environ if environ is None else environ

        raw_source = (source_map.get(REACH_STATE_SOURCE_ENV_VAR) or "").strip().lower()
        if not raw_source:
            source = ReachStateSource.NONE
        else:
            source = _SOURCE_ALIASES.get(raw_source)
            if source is None:
                logger.warning("%s=%r is not one of %s; reach gating stays off",
                               REACH_STATE_SOURCE_ENV_VAR, raw_source,
                               [item.value for item in ReachStateSource])
                source = ReachStateSource.NONE

        raw_age = (source_map.get(REACH_STATE_MAX_AGE_ENV_VAR) or "").strip()
        max_age_ms = DEFAULT_MAX_AGE_MS
        if raw_age:
            try:
                candidate = int(raw_age)
            except ValueError:
                logger.warning("%s=%r is not an integer; using %d ms",
                               REACH_STATE_MAX_AGE_ENV_VAR, raw_age, DEFAULT_MAX_AGE_MS)
            else:
                if MIN_MAX_AGE_MS <= candidate <= MAX_MAX_AGE_MS:
                    max_age_ms = candidate
                else:
                    logger.warning("%s=%r is outside %d..%d ms; using %d ms",
                                   REACH_STATE_MAX_AGE_ENV_VAR, raw_age,
                                   MIN_MAX_AGE_MS, MAX_MAX_AGE_MS, DEFAULT_MAX_AGE_MS)

        return cls(source=source, max_age_ms=max_age_ms)


class LiveTrackingReachProvider:
    """Answer the cue gate's reach question from the most recent pose sample.

    The criterion is that a hand has a 3D location in that sample. A hand only
    gets one when both cameras saw it above the confidence gate and
    triangulation succeeded, so "located" already carries "seen confidently by
    the stereo pair" - it is not a bare presence flag.

    A distance-to-pellet threshold would be a stronger criterion and is the
    obvious next step, but what distance counts as a reach is a scientific
    decision rather than one to invent here. When that is settled it belongs on
    this class, where the gate already reads from.

    None means unknown, never "not reaching": an empty buffer, a sample without
    locations, or a buffer that raises are all absence of evidence, and the
    resolver's staleness bound then applies to whatever this does return.
    """

    DEFAULT_HANDS = ("R_Hand", "L_Hand")

    def __init__(self, live_tracking, *, hands=DEFAULT_HANDS):
        self._live_tracking = live_tracking
        self._hands = tuple(hands)

    def __call__(self):
        try:
            sample = self._live_tracking.latest()
        except Exception as err:
            logger.warning("live tracking buffer failed: %s", err)
            return None
        if sample is None:
            return None
        try:
            reaching = any(
                sample.location(name) is not None for name in self._hands)
            # The capture time, not the time the pose was computed: staleness is
            # about how old the evidence is, not how long it took to produce.
            observed_at = float(sample.source_perf)
        except Exception as err:
            logger.warning("live tracking sample is unusable: %s", err)
            return None
        return reaching, observed_at


class ReachStateResolver:
    """
    Read reach state from the configured source, or report that it is unknown.

    Providers are injected rather than looked up, so this stays testable and so
    a rig without live inference or without a stim camera simply has no provider
    for that source. A provider returns ``(reach_active, observed_at)`` where
    ``observed_at`` is the monotonic time the evidence was *captured*.
    """

    def __init__(self, configuration=None, live_tracking_provider=None,
                 stim_camera_provider=None):
        self._configuration = configuration or ReachStateConfiguration()
        self._providers = {
            ReachStateSource.LIVE_TRACKING: live_tracking_provider,
            ReachStateSource.STIM_CAMERA: stim_camera_provider,
        }

    @property
    def configuration(self) -> ReachStateConfiguration:
        return self._configuration

    @property
    def is_enabled(self) -> bool:
        return self._configuration.source is not ReachStateSource.NONE

    def is_available(self) -> bool:
        """Whether the configured source has a provider at all."""
        if not self.is_enabled:
            return False
        return self._providers.get(self._configuration.source) is not None

    def observe(self, now_perf_time: float):
        """
        Return ``(reach_active, observed_at)``, or None when reach is unknown.

        None is returned for every failure mode - source off, no provider, the
        provider raised, a malformed reply, a non-finite timestamp, an
        observation from the future, or one older than the configured limit -
        so a caller can never mistake missing evidence for "not reaching".
        """
        if not self.is_enabled:
            return None

        provider = self._providers.get(self._configuration.source)
        if provider is None:
            logger.debug("reach source %s has no provider", self._configuration.source.value)
            return None

        try:
            observation = provider()
        except Exception as err:
            logger.warning("reach source %s failed: %s", self._configuration.source.value, err)
            return None

        if observation is None:
            return None
        try:
            reach_active, observed_at = observation
        except (TypeError, ValueError):
            logger.warning("reach source %s returned %r, expected (bool, time)",
                           self._configuration.source.value, observation)
            return None

        observed_at = float(observed_at)
        if not math.isfinite(observed_at):
            logger.warning("reach source %s reported a non-finite observation time",
                           self._configuration.source.value)
            return None

        age = now_perf_time - observed_at
        if age < 0:
            # A clock that runs backwards is a bug, not a fresh observation.
            logger.warning("reach source %s reported an observation %.3fs in the future",
                           self._configuration.source.value, -age)
            return None
        if age > self._configuration.max_age_seconds:
            logger.debug("reach observation is %.3fs old, limit %.3fs",
                         age, self._configuration.max_age_seconds)
            return None

        return bool(reach_active), observed_at
