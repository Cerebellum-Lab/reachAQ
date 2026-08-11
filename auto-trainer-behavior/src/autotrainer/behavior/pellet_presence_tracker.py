import math
from typing import Optional

from autotrainer.core import get_perf_now
from autotrainer.core.pose_elements import SceneElement, ScenePartsPresenceContext
from autotrainer.inference import PoseResponse
from autotrainer.inference.pose_algorithm import update_scene_elements_context_from_pose


class PelletPresenceTracker:
    """Own live scene-part presence state used by pellet automation.

    The tracker deliberately has no event or state-machine dependencies.  Its
    caller decides when a recording session starts and how a newly observed
    animal should be published.
    """

    def __init__(self) -> None:
        self._any_camera = ScenePartsPresenceContext()
        self._all_cameras = ScenePartsPresenceContext()
        self._session_mouse_seen = False
        self._mouse_seen_last_perf_c = -math.inf

    @property
    def any_camera(self) -> ScenePartsPresenceContext:
        return self._any_camera

    @property
    def all_cameras(self) -> ScenePartsPresenceContext:
        return self._all_cameras

    @property
    def session_mouse_seen(self) -> bool:
        return self._session_mouse_seen

    def reset_session(self) -> None:
        self._session_mouse_seen = False

    def update_pose(self, pose_response: PoseResponse, *, in_session: bool) -> bool:
        update_scene_elements_context_from_pose(
            self._any_camera,
            self._all_cameras,
            pose_response,
        )
        return self.update_mouse(
            pose_response.mouse_seen,
            in_session=in_session,
            perf_now=pose_response.perf_c,
        )

    def update_part(
        self,
        part: str,
        seen: bool,
        *,
        perf_now: Optional[float] = None,
    ) -> None:
        self._any_camera.update_part_seen(part, seen, perf_now=perf_now)
        self._all_cameras.update_part_seen(part, seen, perf_now=perf_now)

    def update_mouse(
        self,
        seen: bool,
        *,
        in_session: bool,
        perf_now: Optional[float] = None,
    ) -> bool:
        """Update mouse presence and return whether session presence changed."""
        if perf_now is None:
            perf_now = get_perf_now()
        self.update_part(SceneElement.Nose, seen, perf_now=perf_now)
        if seen:
            self._mouse_seen_last_perf_c = perf_now
        newly_seen = in_session and seen and not self._session_mouse_seen
        if in_session and seen:
            self._session_mouse_seen = True
        return newly_seen

    def is_recently_seen(
        self,
        part: str,
        missing_time: float,
        *,
        use_any_camera: bool = False,
        perf_now: Optional[float] = None,
    ) -> bool:
        context = self._any_camera if use_any_camera else self._all_cameras
        if perf_now is None:
            perf_now = get_perf_now()
        return context.get_recently_seen(part, missing_time, perf_now=perf_now)

    def presence_age(self, part: str) -> float:
        return self._any_camera.get_presence_age(part)

    def last_seen(self, part: str) -> float:
        return self._any_camera.present_last_perf_c.get(part, -math.inf)

    @property
    def mouse_last_seen_age(self) -> float:
        return get_perf_now() - self._mouse_seen_last_perf_c
