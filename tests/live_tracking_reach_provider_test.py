"""The cue gate's reach state, answered from live pose.

`ReachStateResolver` has always accepted a `live_tracking_provider` and nothing
ever supplied one, so selecting `live_tracking` as the source produced "unknown"
forever and Tone 2's reach gate had no pose implementation at all.

The criterion is deliberately the simplest defensible one: the animal is
reaching when a hand has a 3D location in the most recent pose sample. A hand
only gets one when both cameras saw it above the confidence gate and
triangulation succeeded, so "located" already means "seen, confidently, by the
stereo pair". A distance-to-pellet threshold would be a stronger criterion and
is the obvious next step, but the distance that counts as a reach is a
scientific decision rather than one to invent here.

Every failure mode reports unknown rather than "not reaching", matching the
rest of this path: an empty buffer, a malformed sample, or a provider that
raises must never be read as evidence the animal is still.
"""

import pytest

from tools.acquisition.model.live_tracking_buffer import (
    LiveTrackingBuffer,
    LiveTrackingSample,
    TrackingLocation,
)
from tools.acquisition.model.reach_state_source import (
    LiveTrackingReachProvider,
    ReachStateConfiguration,
    ReachStateResolver,
    ReachStateSource,
)


def _sample(locations=(), *, sequence=1, start=100.0, end=100.02):
    return LiveTrackingSample(
        sequence=sequence,
        primary_frame_ids=(sequence,),
        primary_frame_perf_times=(start,),
        source_start_perf=start,
        source_end_perf=end,
        processing_perf=end + 0.001,
        pellet_seen=True,
        locations_3d=tuple(
            TrackingLocation(name, 1.0, 2.0, 3.0) for name in locations),
        offsets_3d=(),
    )


# --- the buffer accessor -----------------------------------------------------


def test_an_empty_buffer_has_no_latest_sample():
    assert LiveTrackingBuffer().latest() is None


def test_latest_returns_the_most_recent_sample():
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(sequence=1))
    buffer.append(_sample(sequence=2))
    assert buffer.latest().sequence == 2


def test_latest_does_not_go_through_snapshot():
    """snapshot() copies the whole deque - up to 18000 samples - and the cue
    gate asks for the latest sample on every poll."""
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(sequence=7))

    def _fail():
        raise AssertionError("latest() copied the buffer")

    buffer.snapshot = _fail
    assert buffer.latest().sequence == 7


# --- the provider ------------------------------------------------------------


def test_no_samples_yet_is_unknown_not_still():
    """Before the first pose arrives there is no evidence either way."""
    assert LiveTrackingReachProvider(LiveTrackingBuffer())() is None


def test_a_located_hand_is_a_reach():
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(["R_Hand"]))
    active, observed_at = LiveTrackingReachProvider(buffer)()
    assert active is True
    assert observed_at == pytest.approx(100.01)


def test_either_hand_counts():
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(["L_Hand"]))
    assert LiveTrackingReachProvider(buffer)()[0] is True


def test_no_located_hand_is_not_reaching():
    """The pellet being visible is not a reach."""
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(["Pellet", "Nose"]))
    active, _observed_at = LiveTrackingReachProvider(buffer)()
    assert active is False


def test_the_observation_time_is_when_the_frame_was_captured():
    """Not when the pose was computed: staleness is judged from capture."""
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(["R_Hand"], start=50.0, end=50.04))
    assert LiveTrackingReachProvider(buffer)()[1] == pytest.approx(50.02)


def test_only_the_configured_hands_count():
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(["L_Hand"]))
    provider = LiveTrackingReachProvider(buffer, hands=("R_Hand",))
    assert provider()[0] is False


def test_a_broken_buffer_is_unknown_rather_than_an_exception():
    """A raising provider would be caught by the resolver, but not by callers
    that use the provider directly."""

    class _Broken:
        def latest(self):
            raise RuntimeError("buffer is gone")

    assert LiveTrackingReachProvider(_Broken())() is None


# --- through the resolver ----------------------------------------------------


def _resolver(buffer, max_age_ms=250):
    return ReachStateResolver(
        ReachStateConfiguration(source=ReachStateSource.LIVE_TRACKING,
                                max_age_ms=max_age_ms),
        live_tracking_provider=LiveTrackingReachProvider(buffer),
    )


def test_the_resolver_reports_a_reach_from_live_tracking():
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(["R_Hand"], start=100.0, end=100.02))
    assert _resolver(buffer).observe(100.1) == (True, pytest.approx(100.01))


def test_a_stale_sample_is_unknown():
    """A pose from a previous trial must not decide this one."""
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(["R_Hand"], start=100.0, end=100.02))
    assert _resolver(buffer).observe(102.0) is None


def test_live_tracking_is_available_once_a_provider_is_supplied():
    """The regression this fixes: the seam existed and nothing filled it."""
    assert _resolver(LiveTrackingBuffer()).is_available() is True
