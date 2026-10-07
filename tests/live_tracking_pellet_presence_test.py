"""The cue gate's pellet presence, answered from live pose.

TrialActionExecutor has always accepted a `pellet_presence_provider` and app_model
never supplied one, so presence was unknown on every evaluation, the policy read
that as stale, and no cued Tone 2 could fire.

The criterion is the pose response's own `pellet_seen` in the most recent
sample, observed at that sample's capture time. Freshness is the cue policy's
gate staleness limit, which already judges the oldest evidence in the gate.
Every failure mode reports unknown rather than a default.
"""

import pytest

from tools.acquisition.model.live_tracking_buffer import (
    LiveTrackingBuffer,
    LiveTrackingSample,
)
from tools.acquisition.model.reach_state_source import LiveTrackingPelletPresenceProvider


def _sample(*, pellet_seen, start=100.0, end=100.02):
    return LiveTrackingSample(
        sequence=1,
        primary_frame_ids=(1,),
        primary_frame_perf_times=(start,),
        source_start_perf=start,
        source_end_perf=end,
        processing_perf=end + 0.001,
        pellet_seen=pellet_seen,
        locations_3d=(),
        offsets_3d=(),
    )


def test_no_samples_yet_is_unknown():
    assert LiveTrackingPelletPresenceProvider(LiveTrackingBuffer())() is None


def test_a_seen_pellet_is_present_at_the_capture_time():
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(pellet_seen=True, start=100.0, end=100.02))
    present, observed_at = LiveTrackingPelletPresenceProvider(buffer)()
    assert present is True
    assert observed_at == pytest.approx(100.01)


def test_an_unseen_pellet_is_absent():
    buffer = LiveTrackingBuffer()
    buffer.append(_sample(pellet_seen=False))
    assert LiveTrackingPelletPresenceProvider(buffer)()[0] is False


def test_a_broken_buffer_is_unknown_rather_than_an_exception():
    class _Broken:
        def latest(self):
            raise RuntimeError("buffer is gone")

    assert LiveTrackingPelletPresenceProvider(_Broken())() is None
