"""Tests for selecting and bounding the cue gate's reach-state source.

The dangerous case is not a missing observation, it is a stale one. If a two
second old pose sample were accepted, the gate would decide the animal is still
using evidence from a different trial, and Tone 2 would fire when it should have
been withheld. Every failure mode therefore has to report "unknown" rather than
"not reaching", so `observe()` returning None is the assertion that matters
throughout.
"""

import pytest

from tools.acquisition.model.reach_state_source import (
    DEFAULT_MAX_AGE_MS,
    MAX_MAX_AGE_MS,
    MIN_MAX_AGE_MS,
    REACH_STATE_MAX_AGE_ENV_VAR,
    REACH_STATE_SOURCE_ENV_VAR,
    ReachStateConfiguration,
    ReachStateResolver,
    ReachStateSource,
)


def _resolver(source, provider, max_age_ms=DEFAULT_MAX_AGE_MS):
    configuration = ReachStateConfiguration(source=source, max_age_ms=max_age_ms)
    if source is ReachStateSource.STIM_CAMERA:
        return ReachStateResolver(configuration, stim_camera_provider=provider)
    return ReachStateResolver(configuration, live_tracking_provider=provider)


def test_the_default_leaves_reach_gating_off():
    """Existing rigs must not start blocking cues because of a new module."""
    configuration = ReachStateConfiguration()
    assert configuration.source is ReachStateSource.NONE
    assert ReachStateResolver(configuration).observe(100.0) is None


@pytest.mark.parametrize("value,expected", [
    ("live_tracking", ReachStateSource.LIVE_TRACKING),
    ("live-tracking", ReachStateSource.LIVE_TRACKING),
    ("LIVE", ReachStateSource.LIVE_TRACKING),
    ("pose", ReachStateSource.LIVE_TRACKING),
    ("stim_camera", ReachStateSource.STIM_CAMERA),
    ("stimcam", ReachStateSource.STIM_CAMERA),
    ("  STIM  ", ReachStateSource.STIM_CAMERA),
    ("none", ReachStateSource.NONE),
    ("off", ReachStateSource.NONE),
])
def test_the_source_is_selectable_by_environment(value, expected):
    configuration = ReachStateConfiguration.from_environment(
        {REACH_STATE_SOURCE_ENV_VAR: value}
    )
    assert configuration.source is expected


@pytest.mark.parametrize("value", ["", "   ", "camera", "true", "1"])
def test_an_unusable_source_falls_back_to_off(value):
    """A typo must not silently select a source, nor stop the session."""
    configuration = ReachStateConfiguration.from_environment(
        {REACH_STATE_SOURCE_ENV_VAR: value}
    )
    assert configuration.source is ReachStateSource.NONE


def test_the_age_limit_is_selectable_and_bounded():
    assert ReachStateConfiguration.from_environment(
        {REACH_STATE_MAX_AGE_ENV_VAR: "40"}
    ).max_age_ms == 40
    for value in ("0", "-1", str(MAX_MAX_AGE_MS + 1), "abc", "1.5"):
        configuration = ReachStateConfiguration.from_environment(
            {REACH_STATE_MAX_AGE_ENV_VAR: value}
        )
        assert configuration.max_age_ms == DEFAULT_MAX_AGE_MS, value


@pytest.mark.parametrize("value", [0, -1, MAX_MAX_AGE_MS + 1])
def test_constructing_with_an_out_of_range_limit_is_an_error(value):
    """An explicit argument is a programming error, unlike a typo in the env."""
    with pytest.raises(ValueError, match="between"):
        ReachStateConfiguration(source=ReachStateSource.LIVE_TRACKING, max_age_ms=value)


@pytest.mark.parametrize("value", [MIN_MAX_AGE_MS, MAX_MAX_AGE_MS])
def test_the_bounds_themselves_are_accepted(value):
    assert ReachStateConfiguration(max_age_ms=value).max_age_ms == value


def test_a_fresh_observation_is_returned_with_its_capture_time():
    """observed_at must be when the evidence was captured, not when it was read."""
    resolver = _resolver(ReachStateSource.LIVE_TRACKING, lambda: (True, 99.95))
    assert resolver.observe(100.0) == (True, 99.95)


def test_a_stale_observation_is_unknown_not_inactive():
    resolver = _resolver(ReachStateSource.LIVE_TRACKING, lambda: (False, 90.0),
                         max_age_ms=100)
    assert resolver.observe(100.0) is None


def test_an_observation_exactly_at_the_limit_is_accepted():
    resolver = _resolver(ReachStateSource.LIVE_TRACKING, lambda: (True, 99.9),
                         max_age_ms=100)
    assert resolver.observe(100.0) == (True, 99.9)


def test_an_observation_from_the_future_is_rejected():
    """A backwards clock is a bug, not a fresh sample."""
    resolver = _resolver(ReachStateSource.LIVE_TRACKING, lambda: (True, 101.0))
    assert resolver.observe(100.0) is None


def test_a_missing_provider_is_unknown():
    configuration = ReachStateConfiguration(source=ReachStateSource.STIM_CAMERA)
    resolver = ReachStateResolver(configuration)
    assert resolver.is_enabled is True
    assert resolver.is_available() is False
    assert resolver.observe(100.0) is None


def test_the_wrong_provider_does_not_satisfy_the_selection():
    """Selecting the stim camera must not silently read live tracking."""
    configuration = ReachStateConfiguration(source=ReachStateSource.STIM_CAMERA)
    resolver = ReachStateResolver(configuration, live_tracking_provider=lambda: (True, 100.0))
    assert resolver.observe(100.0) is None


def test_a_failing_provider_is_unknown():
    def explode():
        raise RuntimeError("inference process died")

    assert _resolver(ReachStateSource.LIVE_TRACKING, explode).observe(100.0) is None


@pytest.mark.parametrize("reply", [None, (True,), "reaching", 5, (True, float("nan")),
                                   (True, float("inf"))])
def test_a_malformed_reply_is_unknown(reply):
    assert _resolver(ReachStateSource.LIVE_TRACKING, lambda: reply).observe(100.0) is None


def test_reach_active_is_coerced_to_bool():
    resolver = _resolver(ReachStateSource.LIVE_TRACKING, lambda: (1, 100.0))
    active, _ = resolver.observe(100.0)
    assert active is True


def test_both_sources_are_selectable_against_their_own_provider():
    live = ReachStateResolver(
        ReachStateConfiguration(source=ReachStateSource.LIVE_TRACKING),
        live_tracking_provider=lambda: (True, 100.0),
        stim_camera_provider=lambda: (False, 100.0),
    )
    stim = ReachStateResolver(
        ReachStateConfiguration(source=ReachStateSource.STIM_CAMERA),
        live_tracking_provider=lambda: (True, 100.0),
        stim_camera_provider=lambda: (False, 100.0),
    )
    assert live.observe(100.0)[0] is True
    assert stim.observe(100.0)[0] is False


def test_the_env_var_names_are_stable():
    """Operators put these in the rig's service environment."""
    assert REACH_STATE_SOURCE_ENV_VAR == "REACHAQ_REACH_STATE_SOURCE"
    assert REACH_STATE_MAX_AGE_ENV_VAR == "REACHAQ_REACH_STATE_MAX_AGE_MS"


def test_configuration_is_recordable_for_session_evidence():
    configuration = ReachStateConfiguration(
        source=ReachStateSource.STIM_CAMERA, max_age_ms=40
    )
    assert configuration.to_record() == {"source": "stim_camera", "max_age_ms": 40}
