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

from reachaq.app.model.live_tracking_buffer import (
    LiveTrackingBuffer,
    LiveTrackingSample,
)
from reachaq.app.model.reach_state_source import LiveTrackingPelletPresenceProvider


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



# ------------------------------------------------ the source, and the ROI


class _RoiState:
    def __init__(self, latest):
        self._latest = latest

    def latest(self):
        return self._latest


def test_the_presence_source_defaults_to_live_tracking():
    from reachaq.app.model.reach_state_source import (
        PelletPresenceSource,
        pellet_presence_source_from_environment,
    )

    assert pellet_presence_source_from_environment({}) is PelletPresenceSource.LIVE_TRACKING
    assert pellet_presence_source_from_environment(
        {"REACHAQ_PELLET_PRESENCE_SOURCE": "ROI"}) is PelletPresenceSource.ROI
    assert pellet_presence_source_from_environment(
        {"REACHAQ_PELLET_PRESENCE_SOURCE": "none"}) is PelletPresenceSource.NONE
    assert pellet_presence_source_from_environment(
        {"REACHAQ_PELLET_PRESENCE_SOURCE": "typo"}) is PelletPresenceSource.LIVE_TRACKING


def test_the_roi_provider_answers_from_the_first_cameras_roi():
    from reachaq.app.model.reach_state_source import RoiPelletPresenceProvider

    provider = RoiPelletPresenceProvider(
        lambda: (_RoiState((False, 7.5, 12.0)), _RoiState((True, 8.0, 200.0))))

    assert provider() == (False, 7.5)


def test_the_roi_provider_is_unknown_without_an_roi_or_a_judged_frame():
    from reachaq.app.model.reach_state_source import RoiPelletPresenceProvider

    assert RoiPelletPresenceProvider(lambda: ())() is None
    assert RoiPelletPresenceProvider(lambda: (_RoiState(None),))() is None

    def broken():
        raise RuntimeError("camera gone")

    assert RoiPelletPresenceProvider(broken)() is None


def test_each_source_makes_its_provider():
    from reachaq.app.model.reach_state_source import (
        PelletPresenceSource,
        RoiPelletPresenceProvider,
        make_pellet_presence_provider,
    )

    kwargs = dict(live_tracking=LiveTrackingBuffer(), roi_states=lambda: ())
    assert isinstance(
        make_pellet_presence_provider(PelletPresenceSource.LIVE_TRACKING, **kwargs),
        LiveTrackingPelletPresenceProvider)
    assert isinstance(
        make_pellet_presence_provider(PelletPresenceSource.ROI, **kwargs),
        RoiPelletPresenceProvider)
    assert make_pellet_presence_provider(PelletPresenceSource.NONE, **kwargs) is None


def test_a_cameras_pellet_roi_params_give_the_app_its_roi_state(app_model):
    from reachaq.video import PelletRoiConfiguration

    assert app_model.load_configuration() is True
    camera = app_model.reach_cameras[0]
    camera.is_enabled = True
    conf = camera.save_configuration(runtime_mode=True)
    conf.params.update({
        "pellet_roi_x": 10, "pellet_roi_y": 20, "pellet_roi_width": 8,
        "pellet_roi_height": 6, "pellet_roi_threshold": 90,
    })
    camera.load_configuration(conf, runtime_mode=True)

    assert camera._pellet_roi_configuration == PelletRoiConfiguration(10, 20, 8, 6, 90.0)
    assert app_model._pellet_roi_states() == (camera.pellet_roi_state,)
    camera.pellet_roi_state.publish(True, 4.0, 120.0)
    assert app_model._pellet_roi_states()[0].latest() == (True, 4.0, 120.0)
