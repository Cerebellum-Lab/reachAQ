import math

from autotrainer.api import ApiAlarmKind
from autotrainer.core import get_perf_now
from autotrainer.core.analysis.alarm_detector import AlarmDetector
from autotrainer.core.configuration.animal_presence_configuration import GlobalAnimalPresenceConfig
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.video_detection import PresenceDetectionAttrs

logger = get_verbose_logger(__name__)


class GlobalAnimalPresenceAlarm(AlarmDetector[GlobalAnimalPresenceConfig]):

    config_cls = GlobalAnimalPresenceConfig
    alarm_api_kind = ApiAlarmKind.animalImmobile

    use_daemon = True

    def __init__(
        self, *,
        topcam_presence: PresenceDetectionAttrs,
    ):
        super().__init__()
        self._topcam_presence = topcam_presence

    def _check_state(self):
        perf_now = get_perf_now()
        cfg = self._config
        topcam_presence = self._topcam_presence
        if topcam_presence is None:
            logger.verbose("topcam presence is not configured")
            return None
        top_cam_pres_age = perf_now - topcam_presence.last_presence_start_perf_c
        delay_seconds = cfg.presence_missing_delay_hours * 3600
        timer_delay = 1
        diff_started = perf_now - self._p_started
        if diff_started < delay_seconds:
            new_engaged = False
            timer_delay = delay_seconds - diff_started
            top_cam_miss = math.nan
        else:
            top_cam_miss = delay_seconds - top_cam_pres_age
            if top_cam_miss <= 0:
                new_engaged = True
            else:
                new_engaged = False
                timer_delay = top_cam_miss
        logger.debug("engaged=%s top_cam_miss=%.1f ; new_delay=%.1f",
                     new_engaged, top_cam_miss, timer_delay)
        self.is_engaged = new_engaged
        return timer_delay
