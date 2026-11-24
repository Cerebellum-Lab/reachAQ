import math
import queue
import threading
import time
from typing import Dict, Tuple, Optional, Union, ClassVar, TypeVar, Type, Generic

from reachaq.core.api import ApiEventKind, ApiDetectorKind
from reachaq.core import ObservableObject, get_perf_now
from reachaq.core.configuration.detector import DetectorConfig
from reachaq.core.event import post_api_detector_event_content
from reachaq.core.event.event_manager import EventManager
from reachaq.core.logging import get_verbose_logger
from reachaq.core.multiproc import no_op_timer, make_daemon_timer


logger = get_verbose_logger(__name__)


_request_check_state = object()


DetectorConfigT = TypeVar("DetectorConfigT", bound=DetectorConfig)


class BaseDetector(ObservableObject, Generic[DetectorConfigT]):

    IS_ENGAGED = "is_engaged"
    IS_ENGAGED_PROPERTY = IS_ENGAGED  # synonym

    CONFIG = "config"

    config_cls: Type[DetectorConfigT] = DetectorConfig

    use_daemon: bool = False
    default_timer_delay: Optional[float] = None

    detector_api_kind: ClassVar[Optional[ApiDetectorKind]] = None
    default_detector_enabled: ClassVar[bool] = True  # raw base detectors are always "enabled" by default

    def __init__(self, *, name: Optional[str] = None, config: Optional[DetectorConfigT] = None):
        super().__init__()
        if name is None:
            name = self.__class__.__qualname__
        self._name = name
        self._config = self.config_cls() if config is None else config
        self._running = False
        self._p_started = -math.inf
        self._is_engaged = False
        self._force_engaged = False  # only used for dev/testing
        self._engaged_perf_c = -math.inf
        self._disengaged_perf_c = -math.inf
        self._cur_timer = no_op_timer
        self._thread_queue: Optional[Tuple[threading.Thread, queue.Queue]] = None
        self._lock = threading.RLock()
        self._checking_state = False
        self._logger = get_verbose_logger(self.__class__.__module__)
        self._event_manager = EventManager.default()

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> DetectorConfigT:
        return self._config

    @config.setter
    def config(self, config: DetectorConfigT):
        self._set_config(config)

    def _set_config(self, value: DetectorConfigT):
        self._logger.debug("got new config: %s", value)
        prev, self._config = self._config, value
        self.property_changed(self.CONFIG, value, prev)
        self.check_state()  # force check_state even if same config

    def post_detector_event(self, detector_id: int, active: bool, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = self._running
        post_api_detector_event_content(self._event_manager, detector_id, active, enabled)

    @property
    def running(self):
        return self._running

    @property
    def is_engaged(self):
        return self._is_engaged or self._force_engaged

    @is_engaged.setter
    def is_engaged(self, value):
        self.set_is_engaged(value)

    def _custom_set_is_engaged(self):
        """this is for subclass to customize their logic on is_engaged changed"""

    def set_is_engaged(self, engaged: bool):
        engaged |= self._force_engaged
        prev, self._is_engaged = self._is_engaged, engaged
        if prev == engaged:
            return
        perf_now = get_perf_now()
        if engaged:
            self._engaged_perf_c = perf_now
        else:
            self._disengaged_perf_c = perf_now
        self._logger.notice("is_engaged -> %s (age previous = %.1f)",
                            engaged, perf_now - (self._disengaged_perf_c if engaged else self._engaged_perf_c))
        kind = self.detector_api_kind
        if kind is not None:
            self.post_detector_event(kind, engaged, self.default_detector_enabled)
        self._custom_set_is_engaged()  # before the property changed event
        self.property_changed(self.IS_ENGAGED, engaged, prev)

    @property
    def engaged_age(self):
        with self._lock:
            return (get_perf_now() if self._is_engaged else self._disengaged_perf_c) - self._engaged_perf_c

    @property
    def disengaged_age(self):
        with self._lock:
            return (get_perf_now() if not self._is_engaged else self._engaged_perf_c) - self._disengaged_perf_c

    def _make_new_timer(self, delay: float):
        self._cur_timer.cancel()  # safer
        timer = self._cur_timer = make_daemon_timer(delay, self.check_state)
        timer.start()
        self._logger.verbose("created timer to check_state within %.1fs", delay)

    def _check_state(self) -> Optional[float]:
        raise NotImplementedError

    def check_state_if_not_detector_thread(self):
        th_q = self._thread_queue
        if th_q is None or threading.current_thread() != th_q[0]:
            self.check_state()

    def check_state(self, *, force: bool=False):
        with self._lock:
            if not self._running and not force:
                return None
            th_q = self._thread_queue
            if th_q is not None:
                # if it's daemon detector, and it's running, put request_check_state to it:
                th, th_q = th_q
                if th is not threading.current_thread() and th.is_alive():
                    th_q.put(_request_check_state)
                    return None
            if self._checking_state:
                logger.warning("checking_state already in progress")
                return None
            self._checking_state = True
            try:
                next_delay = self._check_state()
            except Exception as err:
                self._logger.exception("_check_state() failed: %s", err)
                next_delay = 1
                # if not daemon detector, this will create a timer for another check in 1s
            finally:
                self._checking_state = False
            if next_delay is None:
                next_delay = self.default_timer_delay
            if next_delay is not None and not self.use_daemon:
                # "recurrent/timed" detector
                self._make_new_timer(next_delay)
            else:
                # daemon detector or detector that will resume by explicit refresh,
                # or that is manually checked.
                self._cur_timer.cancel()
                self._cur_timer = no_op_timer
        return next_delay

    def _start(self):
        """Allow any subclass to customize its start procedure. super() should be called."""

    def start(self):
        with self._lock:
            if self._running:
                self._logger.debug("requested start but already running")
                return
            self._logger.verbose("%s: starting monitor", self.__class__.__name__)
            self._running = True
            self.is_engaged = False  # force reset "engaged" to False
            self._p_started = get_perf_now()
            self._start()
            if self.use_daemon:
                cmd_queue = queue.Queue(maxsize=32)
                thread = threading.Thread(name=self.__class__.__name__, target=self._daemon_run, daemon=True,
                                          args=(cmd_queue,))
                self._thread_queue = thread, cmd_queue  # set before start
                thread.start()
            else:
                self.check_state()

    def _daemon_run(self, cmd_queue):
        self._logger.info("%s running", self.__class__.__name__)
        while self._running:
            delay = self.check_state()  # always check immediately
            if delay is None:
                delay = self.default_timer_delay
                if delay is None:
                    delay = 1
            # Now limit max delay to 60 seconds,
            # so that any change in config will be handled at most every 60 seconds.
            # Some detector, like global animal presence, could use very long delay between check,
            # so this ensures that if its setting relating to the delay is changed, then the new one will be handled,
            # ~relatively quickly.
            # Otherwise, we would need to restart any of them that has its config updated.
            delay = min(60., delay)
            try:
                r = cmd_queue.get(timeout=delay)
            except queue.Empty:
                continue
            cmd_queue.task_done()
            if r is _request_check_state:
                continue
            if r is None:
                break
            logger.warning("unhandled command object: %r", r)
        # end while True
        self._logger.verbose("%s: exiting main loop", self.__class__.__name__)

    def _stop(self):
        pass

    def stop(self):
        log = self._logger
        with self._lock:
            self._cur_timer.cancel()
            if not self._running:
                return
            log.verbose("%s: stopping monitor", self.__class__.__name__)
            self._running = False
            thread_queue = self._thread_queue
            self._thread_queue = None
            self._stop()
        # don't try join check thread with the lock acquired, given that can deadlock otherwise.
        if thread_queue is not None:
            thread, q = thread_queue
            # assert isinstance(q, queue.Queue)
            if thread != threading.current_thread():
                if thread.is_alive():
                    q.put(None)
                log.debug("Joining check thread %s", thread)
                thread.join(3)
                if thread.is_alive():
                    self._logger.warning("check thread still alive, but continuing anyway")
                else:
                    log.verbose("joined check thread %s", thread)

    def restart(self):
        self._logger.notice("Restarting %s", self.__class__.__name__)
        self.stop()
        self.start()

    def force_engaged(self, engaged: bool) -> None:
        """
        Primarily used for testing.  This will force the detector to be engaged if called with True.
        """
        self._force_engaged = engaged
        self.is_engaged = engaged
