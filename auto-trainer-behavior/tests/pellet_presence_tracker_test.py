import math

from autotrainer.behavior import PelletPresenceTracker
from autotrainer.core.pose_elements import SceneElement
from autotrainer.inference import PoseResponse


def test_tracks_any_and_all_camera_presence_independently():
    tracker = PelletPresenceTracker()
    response = PoseResponse(
        perf_c=12.0,
        parts_flags=(
            {SceneElement.Pellet: True},
            {SceneElement.Pellet: False},
            {SceneElement.Pellet: False},
        ),
    )

    assert tracker.update_pose(response, in_session=False) is False
    assert tracker.is_recently_seen(
        SceneElement.Pellet, 1.0, use_any_camera=True, perf_now=12.0
    ) is True
    assert tracker.is_recently_seen(
        SceneElement.Pellet, 1.0, use_any_camera=False, perf_now=12.0
    ) is False


def test_session_mouse_seen_is_edge_triggered_and_resettable():
    tracker = PelletPresenceTracker()

    assert tracker.update_mouse(True, in_session=False, perf_now=1.0) is False
    assert tracker.session_mouse_seen is False
    assert tracker.update_mouse(True, in_session=True, perf_now=2.0) is True
    assert tracker.update_mouse(True, in_session=True, perf_now=3.0) is False
    assert tracker.session_mouse_seen is True
    tracker.reset_session()
    assert tracker.session_mouse_seen is False


def test_unknown_parts_have_infinite_age_and_last_seen():
    tracker = PelletPresenceTracker()

    assert tracker.presence_age(SceneElement.Pellet) == -math.inf
    assert tracker.last_seen(SceneElement.Triangle) == -math.inf
