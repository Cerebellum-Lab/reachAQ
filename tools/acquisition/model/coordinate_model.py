"""Live coordinate validation separate from acquisition orchestration."""

from __future__ import annotations

import math
import time

from autotrainer.core import Offset3DTuple
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.pose_elements import SceneElement


logger = get_verbose_logger(__name__)


class CoordinateModel:
    """Own live diamond-coordinate validity state.

    Calibration, diamond/triangle transforms, and saved coordinate data remain
    in their established core/behavior models. This object owns only runtime
    validation and its warning/error edge state.
    """

    def __init__(self) -> None:
        self.enabled = True
        self.report_error = False
        self.previous_diamond = Offset3DTuple.get_nan()
        self.previous_valid_perf = -math.inf
        self.warned_bad = False
        self.triggered_bad = False

    def validate_pose(
        self,
        response,
        configuration,
        *,
        inference_started_perf: float,
        paused: bool,
        perf_now: float | None = None,
        minimum_invalid_duration: float = 5.0,
        inference_warmup_duration: float = 3.0,
        maximum_distance: float = 5.0,
    ) -> str | None:
        if not self.enabled or paused or configuration is None:
            return None
        location = response.locations_3d.get(SceneElement.Diamond)
        raw_location = response.raw_loc_3d.get(SceneElement.Diamond)
        if location is None or raw_location is None:
            return None

        self.previous_diamond = location
        difference = location - configuration.diamond_coord
        raw_difference = raw_location - configuration.raw_diamond_coord
        now = time.perf_counter() if perf_now is None else float(perf_now)
        if (
            difference.distance > maximum_distance
            or raw_difference.distance > maximum_distance
        ):
            if not self.warned_bad:
                logger.warning(
                    "Diamond coordinate invalid: %s ; dist=%.2f raw=%.2f ; pose=%s",
                    location.humanize(n_digits=2),
                    difference.distance,
                    raw_difference.distance,
                    response,
                )
                self.warned_bad = True
        else:
            self.previous_valid_perf = now
            self.warned_bad = False
            self.triggered_bad = False

        if (
            now - inference_started_perf > inference_warmup_duration
            and now - self.previous_valid_perf > minimum_invalid_duration
            and not self.triggered_bad
        ):
            self.triggered_bad = True
            message = (
                "Could not ensure valid diamond position for too long.\n\n"
                "Please re-execute a diamond-triangle calibration via menu "
                "Tools -> Calibrate Coordinate System\n\n"
                "Automatic pause is disabled in reachAQ, so acquisition was "
                "not paused automatically."
            )
            if self.report_error:
                return message
            logger.error(
                "Bad diamond coord check: distance=%.2f ; %s vs %s",
                difference.distance,
                location.humanize(),
                configuration.diamond_coord.humanize(),
            )
        return None
