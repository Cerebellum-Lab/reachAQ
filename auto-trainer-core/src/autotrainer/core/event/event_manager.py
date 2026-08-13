import atexit
import collections
import dataclasses
import threading
import time
from threading import Thread
from datetime import datetime
from queue import Queue, Empty, Full
from typing import Optional, List, Any

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.project import ProjectInfo

from .event_info import EventInfo
from .event_manager_plugin import EventManagerPlugin
from .file_event_plugin import FileEventPlugin
from .logger_event_plugin import LoggerEventPlugin

logger = get_verbose_logger(__name__)


class EventQueueFullError(RuntimeError):
    """An event could not enter the bounded dispatcher queue in time."""


@dataclasses.dataclass(frozen=True)
class _QueuedEvent:
    info: EventInfo
    enqueued_perf_time: float


class EventManager:
    """
    Events, in the context of the autotrainer local application, are a set of messages that track specific changes and
    actions of interest.  They are a subset of information that is likely to be logged overall, and are made to conform
    to a specific structure for reading in conjunction with data files during analysis.

    By default, events are also sent to the default logger.  It is generally not necessary to post an event _and_ send
    the same information explicitly to the logger.

    Repeat behavior:  The objective with repeat events (same event without interruption of another event type) is to
    * Immediately output the first instance of the event (as would normally happen)
    * When a new event type is received, output the number of repeats (not counting that first instance already output)
      * Per the first bullet, this new event type will also be immediately output
    """

    _instance: "EventManager"
    _class_lock = threading.Lock()

    @staticmethod
    def _remove_cls_instance():
        try:
            del EventManager._instance
        except AttributeError:
            pass

    @staticmethod
    def default() -> "EventManager":
        """
        Creates (if needed) and returns the default instance.
        """
        cls = EventManager
        with cls._class_lock:
            instance: Optional[EventManager] = getattr(cls, "_instance", None)
            if instance is None:
                instance = cls._instance = cls("EventManagerInstance")
                instance.register_plugin(LoggerEventPlugin())
                instance.register_plugin(FileEventPlugin())
        return instance

    @staticmethod
    def try_close_default():
        """
        Close the default instance if it exists.  Do not spin one up just to check (e.g.,
        EventManager.default().close() if one was never created).
        """
        cls = EventManager
        with cls._class_lock:
            cls_inst: Optional[EventManager] = getattr(cls, "_instance", None)
            if cls_inst is not None:
                report = cls_inst.close()
                if report and report.get("workerStopped"):
                    cls._remove_cls_instance()

    def __init__(
        self,
        key="",
        *,
        queue_capacity: int = 4096,
        producer_wait_seconds: float = 0.05,
        shutdown_timeout_seconds: float = 5.0,
    ):
        if key != "EventManagerInstance":
            raise Exception("Use EventManager.default() to access and instance.")

        self._plugins: List[EventManagerPlugin] = []
        self._lock = threading.RLock()
        self._closing = False
        self._producer_wait_seconds = max(0.001, float(producer_wait_seconds))
        self._shutdown_timeout_seconds = max(0.1, float(shutdown_timeout_seconds))
        self._enqueue_times = collections.deque()
        self._active_event_enqueued_perf: Optional[float] = None
        self._accepted_count = 0
        self._delivered_count = 0
        self._failed_count = 0
        self._last_shutdown_report = None

        self._project_info = None

        # Callers should expect requests to post an event return as quickly as possible.  Events are pushed to a queue
        # so that processing can be done in a separate thread as resources allow.
        self._write_queue = Queue(maxsize=max(1, int(queue_capacity)))
        self._write_thread = Thread(
            target=self._process_queue,
            name=f"{self.__class__.__name__}",
            daemon=True,  # allow the main thread/process to exit even if this thread is still alive
        )
        self._write_thread.start()

    @property
    def is_valid(self):
        return self._write_thread is not None and self._write_queue is not None

    @property
    def project(self) -> ProjectInfo:
        with self._lock:
            return self._project_info

    @project.setter
    def project(self, value: ProjectInfo) -> None:
        """
        ProjectInfo is an optional property.  If set, it is used to generate the location and name of the event file in
        the expected format.

        Args
            value: ProjectInfo object.  This is used to determine the location of the event file.
        """
        with self._lock:
            self._project_info = value
            plugins = tuple(self._plugins)
        for plugin in plugins:
            plugin.set_project(value)

    @property
    def plugins(self) -> List[EventManagerPlugin]:
        # Do not let callers modify ths list. register/unregister are available for this.  This is already dangerous
        # enough - giving them access to the plugins themselves.
        with self._lock:
            return self._plugins.copy()

    def register_plugin(self, plugin: EventManagerPlugin) -> None:
        """
        Registers a plugin with the event manager.

        Args:
            plugin: The plugin to register.
        """
        with self._lock:
            if self._closing:
                raise RuntimeError("Cannot register an event plugin during shutdown")
            if plugin in self._plugins:
                return
            self._plugins.append(plugin)
            project = self._project_info
        plugin.set_project(project)

    def unregister_plugin(self, plugin: EventManagerPlugin) -> None:
        """
        Unregisters a plugin with the event manager.

        Args:
            plugin: The plugin to unregister.
        """
        with self._lock:
            present = plugin in self._plugins
            if present:
                self._plugins.remove(plugin)
        if present:
            # Even though the caller must have a reference to the plugin, take responsibility to close, if needed.
            # In the future, there may be a way to unregister by some kind of key/type/identifier that doesn't require
            # explicit access to the plugin instance by the caller.
            plugin.close()

    def flush(self):
        with self._lock:
            plugins = tuple(self._plugins)
        for plugin in plugins:
            plugin.flush()

    @property
    def diagnostics(self) -> dict:
        with self._lock:
            now = time.perf_counter()
            pending_times = tuple(self._enqueue_times)
            if self._active_event_enqueued_perf is not None:
                pending_times = (*pending_times, self._active_event_enqueued_perf)
            return {
                "capacity": (
                    0 if self._write_queue is None else self._write_queue.maxsize
                ),
                "pending": len(pending_times),
                "oldestAgeSeconds": (
                    0.0
                    if not pending_times
                    else max(0.0, now - min(pending_times))
                ),
                "accepted": self._accepted_count,
                "delivered": self._delivered_count,
                "failed": self._failed_count,
                "closing": self._closing,
                "lastShutdown": self._last_shutdown_report,
            }

    def close(self, timeout: Optional[float] = None):
        """
        Closes the event manager.  This is required to stop any internal threads and allow a clean exit.  This should
        only be called when the application or script is closing or otherwise finished with the event manager as an
        instance, including the `default` cannot be restarted.
        """
        cls_inst = getattr(EventManager, "_instance", None)
        with self._lock:
            if self._closing and self._write_thread is None:
                return self._last_shutdown_report
            self._closing = True
            wt = self._write_thread
            wq = self._write_queue
        timeout = self._shutdown_timeout_seconds if timeout is None else max(0.0, float(timeout))
        deadline = time.perf_counter() + timeout
        if wt is not None:
            if wq is not None:
                remaining = max(0.0, deadline - time.perf_counter())
                try:
                    wq.put(None, timeout=remaining)
                except Full:
                    logger.error("Event queue remained full during bounded shutdown")
            wt.join(max(0.0, deadline - time.perf_counter()))
            if not wt.is_alive():
                self._write_thread = None
        worker_stopped = wt is None or not wt.is_alive()
        if worker_stopped:
            with self._lock:
                plugins = tuple(self._plugins)
            for plugin in plugins:
                plugin.set_enable(False)
        # Once stopped, account for anything unexpectedly left behind. Never
        # call Queue.join() here: an unresponsive plugin must not make shutdown
        # unbounded.
        if wq is not None and worker_stopped:
            while True:
                try:
                    item = wq.get_nowait()
                    logger.warning("dropped unhandled %s: %s", type(item), item)
                    with self._lock:
                        if isinstance(item, _QueuedEvent):
                            try:
                                self._enqueue_times.remove(item.enqueued_perf_time)
                            except ValueError:
                                pass
                            self._failed_count += 1
                    wq.task_done()
                except Empty:
                    break
        with self._lock:
            if worker_stopped:
                self._write_queue = None
            report = {
                "delivered": self._delivered_count,
                "pending": self.diagnostics["pending"],
                "failed": self._failed_count,
                "workerStopped": worker_stopped,
            }
            self._last_shutdown_report = report
        logger.info("Event manager shutdown report: %s", report)
        if cls_inst is self and worker_stopped:
            self._remove_cls_instance()
        return report

    def post_event_content(self, kind: int, data: Optional[Any] = None, when: Optional[datetime] = None,
                           index: int = None):
        """
        Add an event info instance to the event manager output queue.  This is a convenience method that creates an
        EventInfo object and optionally populates timestamp related fields as the time this method is called.  If timing
        information is critical, those arguments should be explicitly set, or `post_event()` should be used with a
        preconstructed `EventInfo` instance with the desired timestamp fields.

        Args:
            kind: See EventInfo.kind for a detailed description.
            data: See EventInfo.kind for a detailed description.
            when: See EventInfo.kind for a detailed description.
            index: See EventInfo.kind for a detailed description.

        """
        info = EventInfo(kind,
                         when=datetime.now() if when is None else when,
                         index=time.perf_counter_ns() if index is None else index,
                         context=data)

        self.post_event(info)

    def post_event(self, info: EventInfo):
        """
        Posts an event info instance to the event manager.

        Args:
            info:

        Returns:

        """
        if info is None:
            # "~paranoid" check but that will prevent the non-desired stop of the work thread.
            raise RuntimeError("post_event(None) refused")
        with self._lock:
            wq = self._write_queue
            if wq is None or self._closing:
                raise RuntimeError("Event manager is closed")
        queued = _QueuedEvent(info, time.perf_counter())
        with self._lock:
            if self._write_queue is not wq or self._closing:
                raise RuntimeError("Event manager is closed")
            self._enqueue_times.append(queued.enqueued_perf_time)
        try:
            wq.put(queued, timeout=self._producer_wait_seconds)
        except Full as error:
            with self._lock:
                try:
                    self._enqueue_times.remove(queued.enqueued_perf_time)
                except ValueError:
                    pass
                self._failed_count += 1
                diagnostics = self.diagnostics
            raise EventQueueFullError(
                "Event dispatcher queue is full: "
                f"capacity={diagnostics['capacity']} pending={diagnostics['pending']} "
                f"oldest_age={diagnostics['oldestAgeSeconds']:.3f}s kind={info.kind}"
            ) from error
        with self._lock:
            self._accepted_count += 1

    def has_pending(self) -> bool:
        """
        This is primarily for testing and diagnostics.  It should not be relied upon absolutely as this class will not
        guarantee that whatever implementation is used to queue events for processing will accurately report the state
        of empty at all times.

        Returns: True if there are pending events to process by the handlers.  Might be accurate, might not.
        """
        wq = self._write_queue
        if wq is None:
            return False
        return not wq.empty()

    def _process_queue(self):
        # Because we a) use plugins and b) allow for EventInfo->is_same to be overridden, we may be given a plugin that
        # errors on every process_event, or an EventInfo that errors on every is_same.  This would bury the log if we
        # reported it every time.  It may also be a one time error for the plugin, so we don't want to just skip/remove
        # if there is an error.
        # This will log the first occurrence of each of the above, but not spam the log if it is a recurring issue.

        is_same_error_reported = False
        process_event_error_reported = False
        last_event_info: Optional[EventInfo] = None
        repeat_event_count = 0
        input_q = self._write_queue
        got_data = False
        do_process = self._process_event

        last_p_flush = time.perf_counter()

        while True:
            if got_data:
                input_q.task_done()
            p_now = time.perf_counter()
            if p_now - last_p_flush > 5:
                self.flush()
                last_p_flush = p_now
            try:
                # Workaround or current Jetson behavior w/ queue.get(timeout=).
                queued = input_q.get(timeout=0.5)
                got_data = True
                if queued is None:
                    logger.verbose("got exit sentinel, exiting main loop")
                    break
                if isinstance(queued, _QueuedEvent):
                    info = queued.info
                    with self._lock:
                        try:
                            self._enqueue_times.remove(queued.enqueued_perf_time)
                        except ValueError:
                            pass
                        self._active_event_enqueued_perf = (
                            queued.enqueued_perf_time
                        )
                else:
                    info = queued
            except Empty:
                got_data = False
                continue

            if not isinstance(info, EventInfo):
                logger.warning("unexpected event info: type=%s value=%s", type(info), info)
                with self._lock:
                    self._active_event_enqueued_perf = None
                continue

            try:
                is_same = False if last_event_info is None else info.is_same(last_event_info)
            except Exception as err:
                if not is_same_error_reported:
                    is_same_error_reported = True
                    logger.exception("info.is_same() failed: %s", err)
            else:
                if is_same:
                    repeat_event_count += 1
                    last_event_info = info
                    with self._lock:
                        self._active_event_enqueued_perf = None
                    continue
            try:
                if last_event_info is not None and repeat_event_count > 0:
                    do_process(last_event_info, repeat_event_count)
                    repeat_event_count = 0

                last_event_info = info
                do_process(info)
            except Exception as err:  # Coming from an arbitrary plugin process_event() - cannot predict type of error.
                # TODO (maybe): track exceptions per plugin.  After some number N exceptions, disable the plugin.
                if not process_event_error_reported:
                    logger.exception("process queue info (%s) failed: %s", info, err)
                    process_event_error_reported = True
            finally:
                with self._lock:
                    self._active_event_enqueued_perf = None
        # end while True

        if last_event_info is not None and repeat_event_count > 0:
            do_process(last_event_info, repeat_event_count)

        # if got_data: always True.
        input_q.task_done()

        with self._lock:
            plugins = tuple(self._plugins)
        for plugin in plugins:
            plugin.close()

    def _process_event(self, info: EventInfo, repeat_count: int = 0):
        with self._lock:
            plugins = tuple(self._plugins)
        failed = False
        for plugin in plugins:
            logger.spam("plugin %s: processing event %s", plugin, info)
            try:
                plugin.process_event(info, repeat_count)
            except Exception:
                failed = True
                logger.exception("Event plugin %s failed for %s", plugin, info)
        with self._lock:
            event_count = 1 + int(repeat_count)
            if failed:
                self._failed_count += event_count
            else:
                self._delivered_count += event_count


atexit.register(EventManager.try_close_default)
