import math
from typing import Optional, Tuple, List

from reachaq.core.api import ApiDetectorKind
from reachaq.core.configuration.pellet_misplaced_config import PelletMisplacedDetectorConfiguration

from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from reachaq.core import get_perf_now, Offset3DTuple, calculate_std_dev_manual
from reachaq.core.logging import get_verbose_logger
from reachaq.core.analysis.detector import BaseDetector


logger = get_verbose_logger(__name__)


_offset_nans = Offset3DTuple(math.nan, math.nan, math.nan)


class PelletMisplacedDetector(BaseDetector[PelletMisplacedDetectorConfiguration]):

    config_cls = PelletMisplacedDetectorConfiguration

    use_daemon = True
    default_timer_delay = 0.25  # default delay between each check of the possible condition(s)
    # NB: should be at least 2 times smaller than aggregate_duration, otherwise given we expire too old data
    # before checking the condition, we could always have the data set empty.

    detector_api_kind = ApiDetectorKind.pelletMisplaced

    def __init__(self):
        super().__init__()
        self._prev_data: List[Tuple[float, Offset3DTuple]] = []
        self._dcs_config: Optional[DiamondTriangleOffsetConfig] = None

    @property
    def dcs_config(self) -> Optional[DiamondTriangleOffsetConfig]:
        return self._dcs_config

    @dcs_config.setter
    def dcs_config(self, value: Optional[DiamondTriangleOffsetConfig]):
        logger.verbose("got new DCS config: %s", value)
        self._dcs_config = value

    def _check_state(self) -> Optional[float]:
        cfg = self._config
        prev_data = self._prev_data
        engaged = False
        p_now = get_perf_now()
        if p_now - self._p_started < cfg.aggregate_duration:
            # wait at least 1 window before consider
            return None
        idx = 0
        while idx < len(prev_data):
            perf_c = prev_data[idx][0]
            if p_now - perf_c <= cfg.aggregate_duration:
                break
            idx += 1
        del prev_data[:idx]
        if cfg.use_dcs_y_low_limit:
            # logger.debug("Checking %s", prev_positions)
            if len(prev_data) > 0 and all(pos.y < cfg.dcs_y_low_limit for _, pos in prev_data):
                engaged = True
        if not self._is_engaged and engaged:
            if len(prev_data) < 2:
                if len(prev_data) == 0:
                    avg_pos, stdev_pos = _offset_nans, _offset_nans
                else:
                    avg_pos, stdev_pos = prev_data[-1][1], Offset3DTuple(0, 0, 0)
            else:
                avg_pos, stdev_pos = calculate_std_dev_manual([d[1] for d in prev_data])
            logger.notice("New engaged, dcs_pos=%s stdev=%s ; all=%s",
                          avg_pos.humanize(), stdev_pos.humanize(), prev_data)
        self.is_engaged = engaged
        return None

    def update(self, pellet_inference_3d: Optional[Offset3DTuple]):
        dcs_cfg = self._dcs_config
        if dcs_cfg is None or not dcs_cfg.fully_valid:  # refuse to guess with not fully valid
            return
        if pellet_inference_3d is None:
            # no need put None 3d loc, the check is done by timer
            return
        rel_to_diamond = dcs_cfg.diamond_coord - pellet_inference_3d
        dcs_pellet_3d = dcs_cfg.inference_to_diamond(rel_to_diamond)
        if __debug__:
            # should not be needed with fully_valid check above.
            if any(map(math.isnan, dcs_pellet_3d)):
                logger.debug("filtering bad pellet3d : %s", pellet_inference_3d)
                return
        p_now = get_perf_now()
        with self._lock:
            self._prev_data.append((p_now, dcs_pellet_3d))
        # self.check_state()
        # no: we don't need resolution/check frequency as high as the input source
