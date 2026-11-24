import math
import time
from typing import Optional, Callable, Dict, List, Set

from reachaq.core import get_perf_now, get_verbose_logger
from reachaq.core.analysis.detector import BaseDetector
from reachaq.core.configuration.watchdog_config import WatchdogConfig


logger = get_verbose_logger(__name__)


WatchdogEvaluatePerfCounterT = Callable[[], float]


class WatchdogMonitor(BaseDetector[WatchdogConfig]):

    config_cls = WatchdogConfig

    use_daemon = True
    default_timer_delay = 1  # "refresh" rate

    def __init__(self):
        super().__init__()
        self._watchdogs: Dict[str, WatchdogEvaluatePerfCounterT] = {}
        self._engaged_watchdogs: Set[str] = set()

    @property
    def registered_watchdogs(self) -> List[str]:
        with self._lock:
            return list(self._watchdogs)

    def register_watchdog(self, key: str, getter: WatchdogEvaluatePerfCounterT):
        logger.verbose("registering %s as watchdog with %s", key, getter)
        with self._lock:
            self._watchdogs[key] = getter

    def unregister_watchdog(self, key: str):
        with self._lock:
            popped = self._watchdogs.pop(key, None)
        if popped is not None:
            logger.verbose("unregistered %s as watchdog", key)

    @property
    def engaged_watchdogs(self) -> List[str]:
        with self._lock:
            return sorted(self._engaged_watchdogs)

    def _check_state(self) -> Optional[float]:
        # NB: using get_perf_now leads to issue in test,
        # given get_perf_now is patched in tests, and sometimes we jump of big value with it,
        # making possible watchdog timeout, given possible watchdog producers are not refreshing "continuously",
        # but only ~few times per second max.
        p_now = time.perf_counter()  # get_perf_now()
        cfg = self._config
        engaged = set()
        prev_engaged = self._engaged_watchdogs
        logger.spam("checking %s watchdogs", len(self._watchdogs))
        for key, watchdog in self._watchdogs.items():
            watch_perf_c = watchdog()
            if watch_perf_c is None:
                logger.warning("watchdog %s None", key)
            if watch_perf_c is None or math.isnan(watch_perf_c):
                continue
            delay = p_now - watch_perf_c
            if delay > cfg.timeout_trigger_delay:
                engaged.add(key)
                if key not in prev_engaged:
                    logger.error("detected watchdog %s timed out: %.1f ; p_now=%.1f", key, delay, p_now)
            elif key in prev_engaged:
                logger.notice("watchdog %s resumed live, delay=%.1f ; p_now=%.1f", key, delay, p_now)
        self._engaged_watchdogs = engaged
        self.is_engaged = len(engaged) > 0
