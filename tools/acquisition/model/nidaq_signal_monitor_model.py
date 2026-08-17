from __future__ import annotations

import dataclasses
import logging
import math
import queue
import signal
import threading
import time
from typing import Iterable, Optional

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    NidaqDeviceIdentity,
    NidaqTimingConfiguration,
    NidaqTimingPlan,
    ObservableObject,
    ProjectInfo,
)
from autotrainer.core.logging import (
    ALWAYS_CONSOLE_LOG_ATTRIBUTE,
    get_verbose_logger,
    install_log_exception_hook,
    log_hardware_initialization,
)
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.core.project import ProjectDependentProtocol
from autotrainer.device import NidaqSignalStreamController
from tools.acquisition.model.nidaq_sample_ring import SharedNidaqSampleRing
from tools.acquisition.model.nidaq_channel_plan import with_display_channels
from tools.acquisition.model.nidaq_discovery import discover_nidaq_devices
from tools.acquisition.model.nidaq_timing import build_nidaq_timing_plan
from tools.acquisition.model.nidaq_timing import (
    resolve_nidaq_multidevice_probe,
    resolve_nidaq_device_aliases,
    resolve_nidaq_device_names,
    resolve_nidaq_stream_configuration,
)
from tools.acquisition.model.nidaq_preflight import run_isolated_nidaq_preflight


logger = get_verbose_logger(__name__)


def _resolve_physical_channel_device(channel: str, aliases) -> str:
    parts = str(channel).strip("/").split("/", 1)
    if len(parts) != 2:
        return str(channel)
    return f"{aliases.get(parts[0], parts[0])}/{parts[1]}"


_WORKER_READY = "ready"
_WORKER_ERROR = "error"
_WORKER_STOPPED = "stopped"
_WORKER_LOG = "log"
_DEFAULT_STARTUP_TIMEOUT_SECONDS = 10.0


def _put_worker_message(message_queue, message) -> None:
    """Best-effort child-to-parent delivery without letting a full queue hang NI cleanup."""
    try:
        message_queue.put(message, timeout=0.5)
    except (BrokenPipeError, EOFError, OSError, queue.Full):
        pass


class _NidaqWorkerLogHandler(logging.Handler):
    """Relay child logs over its disposable stream queue, not the global log queue."""

    def __init__(self, message_queue):
        super().__init__(logging.NOTSET)
        self._message_queue = message_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = (
                record.levelno,
                record.name,
                self.format(record),
                bool(getattr(record, ALWAYS_CONSOLE_LOG_ATTRIBUTE, False)),
            )
            _put_worker_message(self._message_queue, (_WORKER_LOG, payload))
        except Exception:
            self.handleError(record)


def _nidaq_signal_stream_worker(
    configuration,
    timing_plan,
    message_queue,
    sample_ring,
    stop_event,
    log_dict_config,
) -> None:
    """Own every NI-DAQmx call in a disposable process.

    Loading the NI Linux runtime can hang or terminate the interpreter.  Keeping
    that native boundary out of the Qt process ensures either failure is reported
    without taking the application UI with it.
    """
    del log_dict_config  # retained in the worker signature for injected test workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    root_logger = logging.getLogger()
    root_logger.handlers = [_NidaqWorkerLogHandler(message_queue)]
    root_logger.setLevel(logging.NOTSET)
    install_log_exception_hook()

    controller = None
    try:
        controller = NidaqSignalStreamController(
            configuration,
            timing_plan=timing_plan,
        )
        controller.start()
        _put_worker_message(message_queue, (_WORKER_READY, None))
        while not stop_event.is_set():
            block = controller.read_chunk()
            if block.sample_count > 0:
                sample_ring.write_block(block)
    except BaseException as exc:
        message = str(exc) or exc.__class__.__name__
        logger.exception("NI-DAQ signal stream worker failed")
        _put_worker_message(message_queue, (_WORKER_ERROR, message))
    finally:
        if controller is not None:
            try:
                controller.close()
            except BaseException as exc:
                message = str(exc) or exc.__class__.__name__
                logger.exception("Failed to close NI-DAQ signal stream worker")
                _put_worker_message(message_queue, (_WORKER_ERROR, message))
        _put_worker_message(message_queue, (_WORKER_STOPPED, None))


class NidaqSignalMonitorModel(ObservableObject, ProjectDependentProtocol):
    CONFIGURATION = "configuration"
    HARDWARE_ENABLED = "hardware_enabled"
    IS_STARTING = "is_starting"
    IS_RUNNING = "is_running"
    STATUS_MESSAGE = "status_message"
    ERROR_MESSAGE = "error_message"
    TIMING_PLAN = "timing_plan"

    def __init__(
        self,
        *,
        mp_ctx=None,
        worker_target=_nidaq_signal_stream_worker,
        device_discovery=discover_nidaq_devices,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
        exact_preflight=None,
    ):
        super().__init__()
        self._configuration = NidaqSignalStreamConfiguration()
        self._hardware_enabled = False
        self._project: Optional[ProjectInfo] = None
        self._mp_ctx = get_mp_ctx() if mp_ctx is None else mp_ctx
        self._worker_target = worker_target
        self._device_discovery = device_discovery
        self._startup_timeout_seconds = startup_timeout_seconds
        self._exact_preflight = (
            run_isolated_nidaq_preflight
            if exact_preflight is None and worker_target is _nidaq_signal_stream_worker
            else exact_preflight
        )
        self._display_refresh_rate_hz = 60.0
        self._process = None
        self._message_queue = None
        self._process_stop_event = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._is_starting = False
        self._is_running = False
        self._status_message = "NI-DAQ signal stream disabled"
        self._error_message = ""
        self._timing_configuration = NidaqTimingConfiguration()
        self._hardware_timed_output_devices = tuple()
        self._hardware_timed_output_channels = tuple()
        self._device_identities: tuple[NidaqDeviceIdentity, ...] = tuple()
        self._runtime_device_aliases = {}
        self._timing_plan: Optional[NidaqTimingPlan] = None
        self._sample_ring = SharedNidaqSampleRing(self._configuration, mp_ctx=self._mp_ctx)

    @property
    def configuration(self) -> NidaqSignalStreamConfiguration:
        return self._configuration

    @property
    def hardware_enabled(self) -> bool:
        return self._hardware_enabled

    @property
    def project(self) -> Optional[ProjectInfo]:
        return self._project

    @project.setter
    def project(self, value: Optional[ProjectInfo]):
        with self._lock:
            self._project = value

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_starting(self) -> bool:
        return self._is_starting

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def timing_plan(self) -> Optional[NidaqTimingPlan]:
        return self._timing_plan

    @property
    def runtime_device_aliases(self):
        return dict(self._runtime_device_aliases)

    @property
    def display_refresh_rate_hz(self) -> float:
        return self._display_refresh_rate_hz

    @property
    def sample_ring(self) -> SharedNidaqSampleRing:
        with self._lock:
            expected_names = tuple(channel.name for channel in self._configuration.channels)
            required_capacity = max(
                self._configuration.read_chunk_size * 4,
                int(math.ceil(
                    self._configuration.sample_rate_hz
                    * self._configuration.rolling_window_seconds
                )),
            )
            ring = self._sample_ring
            if (
                ring.channel_names != expected_names
                or not math.isclose(ring.sample_rate_hz, self._configuration.sample_rate_hz)
                or ring.capacity < required_capacity
            ):
                if self._process is not None:
                    raise RuntimeError("cannot replace NI-DAQ sample ring while stream is active")
                ring = SharedNidaqSampleRing(self._configuration, mp_ctx=self._mp_ctx)
                self._sample_ring = ring
            return ring

    @property
    def effective_read_chunk_size(self) -> int:
        return max(
            1,
            int(round(self._configuration.sample_rate_hz / self._display_refresh_rate_hz)),
        )

    def set_display_refresh_rate(self, refresh_rate_hz: float) -> None:
        refresh_rate_hz = float(refresh_rate_hz)
        if not math.isfinite(refresh_rate_hz) or refresh_rate_hz < 1.0:
            refresh_rate_hz = 60.0
        previous_chunk_size = self.effective_read_chunk_size
        if math.isclose(refresh_rate_hz, self._display_refresh_rate_hz, rel_tol=0.001):
            return
        was_active = self._is_running or self._is_starting
        self._display_refresh_rate_hz = refresh_rate_hz
        if was_active and self.effective_read_chunk_size != previous_chunk_size:
            self.stop()
            self.start()

    def load_configuration(self, configuration: NidaqSignalStreamConfiguration) -> None:
        self.stop()
        prev = self._configuration
        self._configuration = configuration
        self._on_property_changed(self.CONFIGURATION, configuration, prev)
        if configuration.is_enabled and self._hardware_enabled:
            self._set_status("NI-DAQ signal stream stopped")
            self._set_error("")
            log_hardware_initialization(
                logger,
                "SKIP | NI-DAQ signal stream | awaiting manual or acquisition start",
            )
        elif configuration.is_enabled:
            self._set_status("NI-DAQ hardware disabled; signal stream stopped")
            self._set_error("")
            log_hardware_initialization(logger, "SKIP | NI-DAQ signal stream | NI-DAQ hardware disabled")
        else:
            self._set_status("NI-DAQ signal stream disabled")
            self._set_error("")
            log_hardware_initialization(logger, "SKIP | NI-DAQ signal stream | disabled")

    def set_hardware_enabled(self, enabled: bool, *, auto_start: bool = False) -> None:
        enabled = bool(enabled)
        previous = self._hardware_enabled
        if enabled == previous:
            return
        self._hardware_enabled = enabled
        self._on_property_changed(self.HARDWARE_ENABLED, enabled, previous)
        if not enabled:
            self.stop()
            self._set_status("NI-DAQ hardware disabled; signal stream stopped")
            self._set_error("")
        elif auto_start and self._configuration.is_enabled:
            self.start()

    def save_configuration(self) -> NidaqSignalStreamConfiguration:
        return self._configuration

    def configure_timing(
        self,
        configuration: NidaqTimingConfiguration,
        *,
        hardware_timed_output_devices: Iterable[str] = tuple(),
        hardware_timed_output_channels: Iterable[str] = tuple(),
        device_identities: Iterable[NidaqDeviceIdentity] = tuple(),
    ) -> None:
        if self._is_running or self._is_starting:
            raise RuntimeError("cannot change NI-DAQ timing while acquisition is active")
        self._timing_configuration = configuration
        self._hardware_timed_output_devices = tuple(dict.fromkeys(
            str(device) for device in hardware_timed_output_devices if device
        ))
        self._hardware_timed_output_channels = tuple(dict.fromkeys(
            str(channel) for channel in hardware_timed_output_channels if channel
        ))
        self._device_identities = tuple(device_identities)
        self._runtime_device_aliases = {}
        self._set_timing_plan(None)

    def set_stream_channels(
        self,
        channels: Iterable[NidaqSignalChannelConfiguration],
    ) -> None:
        channels = tuple(channels)
        if channels == self._configuration.channels:
            return
        was_running = self._is_running or self._is_starting
        self.stop()
        previous = self._configuration
        self._configuration = dataclasses.replace(
            previous,
            channels=channels,
            is_enabled=bool(channels),
        )
        self._on_property_changed(self.CONFIGURATION, self._configuration, previous)
        self._set_error("")
        if was_running and self._hardware_enabled and channels:
            self.start()
        elif not channels:
            self._set_status("NI-DAQ signal stream has no selected signals")
        else:
            self._set_status("NI-DAQ signal stream stopped")

    def set_display_channels(self, channel_names: Iterable[str]) -> None:
        previous = self._configuration
        configuration = with_display_channels(previous, channel_names)
        if configuration == previous:
            return
        self._configuration = configuration
        self._on_property_changed(self.CONFIGURATION, configuration, previous)

    def start(self) -> bool:
        with self._lock:
            if self._is_running or self._is_starting:
                return True
            configuration = dataclasses.replace(
                self._configuration,
                read_chunk_size=self.effective_read_chunk_size,
            )
            if not self._hardware_enabled:
                self._set_status("NI-DAQ hardware disabled; signal stream stopped")
                self._set_error("")
                log_hardware_initialization(logger, "SKIP | NI-DAQ signal stream | NI-DAQ hardware disabled")
                return False
            if not configuration.is_enabled:
                self._set_status("NI-DAQ signal stream disabled")
                self._set_error("")
                log_hardware_initialization(logger, "SKIP | NI-DAQ signal stream | disabled")
                return False
            started = time.perf_counter()
            log_hardware_initialization(
                logger,
                "START | NI-DAQ signal stream | sample_rate=%s channels=%s",
                configuration.sample_rate_hz,
                tuple(channel.physical_channel for channel in configuration.channels),
            )
            try:
                devices, discovery_error = self._device_discovery()
                if discovery_error:
                    raise RuntimeError(discovery_error)
                aliases = resolve_nidaq_device_aliases(
                    self._device_identities,
                    devices,
                )
                configuration = resolve_nidaq_stream_configuration(
                    configuration,
                    aliases,
                )
                hardware_timed_output_devices = resolve_nidaq_device_names(
                    self._hardware_timed_output_devices,
                    aliases,
                )
                hardware_timed_output_channels = tuple(
                    _resolve_physical_channel_device(channel, aliases)
                    for channel in self._hardware_timed_output_channels
                )
                timing_plan = build_nidaq_timing_plan(
                    configuration,
                    self._timing_configuration,
                    devices,
                    hardware_timed_output_devices=hardware_timed_output_devices,
                    hardware_timed_output_channels=hardware_timed_output_channels,
                )
                self._set_timing_plan(timing_plan)
                if not timing_plan.is_valid:
                    raise RuntimeError(timing_plan.reason)
                if self._exact_preflight is not None:
                    preflight = self._exact_preflight(configuration, timing_plan)
                    if not preflight.is_valid:
                        raise RuntimeError(
                            "NI-DAQ exact task preflight failed at "
                            f"{preflight.stage}: {preflight.error}. "
                            f"{preflight.corrective_action}"
                        )
                    timing_plan = resolve_nidaq_multidevice_probe(
                        timing_plan,
                        getattr(
                            preflight,
                            "multidevice_probe_status",
                            timing_plan.multidevice_probe_status,
                        ),
                    )
                    self._set_timing_plan(timing_plan)
                self._runtime_device_aliases = dict(aliases)
                message_queue = self._mp_ctx.Queue(maxsize=16)
                stop_event = self._mp_ctx.Event()
                sample_ring = self.sample_ring
                sample_ring.reset()
                process = self._mp_ctx.Process(
                    target=self._worker_target,
                    name="nidaq-signal-stream",
                    args=(
                        configuration,
                        timing_plan,
                        message_queue,
                        sample_ring,
                        stop_event,
                        None,
                    ),
                    daemon=True,
                )
                process.start()
            except Exception as exc:
                logger.exception("Failed to launch NI-DAQ signal stream worker")
                log_hardware_initialization(
                    logger,
                    "FAILED | NI-DAQ signal stream | elapsed=%.3fs error=%s",
                    time.perf_counter() - started,
                    str(exc) or exc.__class__.__name__,
                    level=logging.ERROR,
                )
                self._process = None
                self._message_queue = None
                self._process_stop_event = None
                self._set_starting(False)
                self._set_running(False)
                self._set_status("NI-DAQ signal stream stopped")
                self._set_error(str(exc) or exc.__class__.__name__)
                return False

            self._process = process
            self._message_queue = message_queue
            self._process_stop_event = stop_event
            self._set_error("")
            self._set_status("Starting NI-DAQ signal stream...")
            self._set_starting(True)
            thread = threading.Thread(
                target=self._run,
                args=(process, message_queue, stop_event, started),
                name="nidaq_signal_monitor",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            process = self._process
            stop_event = self._process_stop_event
            was_starting = self._is_starting
            if process is None:
                self._set_starting(False)
                self._set_running(False)
                return
            self._set_status("Stopping NI-DAQ signal stream...")
            if stop_event is not None:
                stop_event.set()
            # A worker still loading NI-DAQmx cannot observe the event.  Stop it
            # immediately instead of making the Qt caller wait on the driver.
            if was_starting and process.is_alive():
                process.terminate()

        if thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if process.is_alive():
            logger.warning("NI-DAQ signal stream worker did not stop; terminating it")
            process.terminate()
            process.join(timeout=0.5)
        self._finalize_worker(process)

    def close(self) -> None:
        self.stop()

    def _run(self, process, message_queue, stop_event, started: float) -> None:
        worker_error = ""
        ready = False
        unexpected_exit = False
        try:
            while True:
                if (
                    process.is_alive()
                    and not ready
                    and time.perf_counter() - started >= self._startup_timeout_seconds
                ):
                    worker_error = (
                        f"NI-DAQmx runtime did not become ready within "
                        f"{self._startup_timeout_seconds:g} seconds; its isolated worker was stopped"
                    )
                    logger.error(worker_error)
                    stop_event.set()
                    process.terminate()
                    process.join(timeout=0.5)
                    break
                try:
                    kind, payload = message_queue.get(timeout=0.01)
                except queue.Empty:
                    if not process.is_alive():
                        unexpected_exit = not stop_event.is_set()
                        process.join(timeout=0)
                        break
                    continue
                if kind == _WORKER_READY:
                    ready = True
                    with self._lock:
                        if process is not self._process:
                            break
                        self._set_error("")
                        self._set_starting(False)
                        self._set_running(True)
                        self._set_status("NI-DAQ signal stream running")
                    log_hardware_initialization(
                        logger,
                        "READY | NI-DAQ signal stream | elapsed=%.3fs worker_pid=%s",
                        time.perf_counter() - started,
                        process.pid,
                    )
                elif kind == _WORKER_ERROR:
                    message = str(payload)
                    if not worker_error:
                        worker_error = message
                    logger.error("NI-DAQ signal stream worker error: %s", message)
                elif kind == _WORKER_LOG:
                    level, logger_name, message, always_console = payload
                    extra = {ALWAYS_CONSOLE_LOG_ATTRIBUTE: True} if always_console else None
                    logging.getLogger(logger_name).log(
                        level,
                        "NI-DAQ worker pid=%s | %s",
                        process.pid,
                        message,
                        extra=extra,
                    )
                elif kind == _WORKER_STOPPED:
                    process.join(timeout=0.5)
                    break
        except Exception as exc:
            worker_error = str(exc) or exc.__class__.__name__
            logger.exception("Failed while monitoring NI-DAQ signal stream worker")
        finally:
            if process.is_alive():
                stop_event.set()
                process.terminate()
                process.join(timeout=0.5)
            exitcode = process.exitcode
            if not worker_error and exitcode not in (0, None) and unexpected_exit:
                if exitcode < 0:
                    try:
                        signal_name = signal.Signals(-exitcode).name
                    except ValueError:
                        signal_name = f"signal {-exitcode}"
                    worker_error = (
                        f"NI-DAQ signal stream worker crashed with {signal_name} "
                        f"(exit code {exitcode}); check the NI-DAQmx driver and selected device channels"
                    )
                else:
                    worker_error = f"NI-DAQ signal stream worker exited unexpectedly (exit code {exitcode})"
            if worker_error:
                log_hardware_initialization(
                    logger,
                    "FAILED | NI-DAQ signal stream | elapsed=%.3fs worker_pid=%s error=%s",
                    time.perf_counter() - started,
                    process.pid,
                    worker_error,
                    level=logging.ERROR,
                )
                self._set_error(worker_error)
            self._finalize_worker(process)

    def _finalize_worker(self, process) -> None:
        with self._lock:
            if process is not self._process:
                return
            self._process = None
            self._message_queue = None
            self._process_stop_event = None
            self._thread = None
            self._set_starting(False)
            self._set_running(False)
            if self._configuration.is_enabled:
                self._set_status("NI-DAQ signal stream stopped")

    def _set_running(self, value: bool) -> None:
        prev, self._is_running = self._is_running, value
        self._on_property_changed(self.IS_RUNNING, value, prev)

    def _set_starting(self, value: bool) -> None:
        prev, self._is_starting = self._is_starting, value
        self._on_property_changed(self.IS_STARTING, value, prev)

    def _set_status(self, value: str) -> None:
        prev, self._status_message = self._status_message, value
        self._on_property_changed(self.STATUS_MESSAGE, value, prev)

    def _set_error(self, value: str) -> None:
        prev, self._error_message = self._error_message, value
        self._on_property_changed(self.ERROR_MESSAGE, value, prev)

    def _set_timing_plan(self, value: Optional[NidaqTimingPlan]) -> None:
        previous, self._timing_plan = self._timing_plan, value
        self._on_property_changed(self.TIMING_PLAN, value, previous)
