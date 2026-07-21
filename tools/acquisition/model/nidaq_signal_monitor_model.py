from __future__ import annotations

import csv
import dataclasses
import logging
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, TextIO

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
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
from autotrainer.device import NidaqSignalSampleBlock, NidaqSignalStreamController


logger = get_verbose_logger(__name__)


_WORKER_READY = "ready"
_WORKER_SAMPLE = "sample"
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


def _nidaq_signal_stream_worker(configuration, message_queue, stop_event, log_dict_config) -> None:
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
        controller = NidaqSignalStreamController(configuration)
        controller.start()
        _put_worker_message(message_queue, (_WORKER_READY, None))
        while not stop_event.is_set():
            block = controller.read_chunk()
            if block.sample_count > 0:
                _put_worker_message(message_queue, (_WORKER_SAMPLE, block))
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


class NidaqSignalMonitorEvents:
    sample_block_received = Callable[[NidaqSignalSampleBlock], None]


class NidaqSignalMonitorModel(ObservableObject, ProjectDependentProtocol):
    CONFIGURATION = "configuration"
    HARDWARE_ENABLED = "hardware_enabled"
    IS_STARTING = "is_starting"
    IS_RUNNING = "is_running"
    STATUS_MESSAGE = "status_message"
    ERROR_MESSAGE = "error_message"
    RECORDING_PATH = "recording_path"

    sample_block_received: NidaqSignalMonitorEvents.sample_block_received

    def __init__(
        self,
        *,
        mp_ctx=None,
        worker_target=_nidaq_signal_stream_worker,
        startup_timeout_seconds: float = _DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ):
        super().__init__(("sample_block_received",))
        self._configuration = NidaqSignalStreamConfiguration()
        self._hardware_enabled = False
        self._project: Optional[ProjectInfo] = None
        self._mp_ctx = get_mp_ctx() if mp_ctx is None else mp_ctx
        self._worker_target = worker_target
        self._startup_timeout_seconds = startup_timeout_seconds
        self._process = None
        self._message_queue = None
        self._process_stop_event = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._recording_file: Optional[TextIO] = None
        self._recording_writer: Optional[csv.writer] = None
        self._recording_path: Optional[Path] = None
        self._recording_blocked = False
        self._is_starting = False
        self._is_running = False
        self._status_message = "NI-DAQ signal stream disabled"
        self._error_message = ""

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
            self._recording_blocked = False
            self._close_recording_file()
            if self._is_running:
                self._open_recording_file()

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
    def recording_path(self) -> Optional[Path]:
        return self._recording_path

    def load_configuration(self, configuration: NidaqSignalStreamConfiguration) -> None:
        self.stop()
        prev = self._configuration
        self._configuration = configuration
        self._recording_blocked = False
        self._on_property_changed(self.CONFIGURATION, configuration, prev)
        if configuration.is_enabled and self._hardware_enabled:
            self.start()
        elif configuration.is_enabled:
            self._set_status("NI-DAQ hardware disabled; signal stream stopped")
            self._set_error("")
            log_hardware_initialization(logger, "SKIP | NI-DAQ signal stream | NI-DAQ hardware disabled")
        else:
            self._set_status("NI-DAQ signal stream disabled")
            self._set_error("")
            log_hardware_initialization(logger, "SKIP | NI-DAQ signal stream | disabled")

    def set_hardware_enabled(self, enabled: bool, *, auto_start: bool = True) -> None:
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
        self._recording_blocked = False
        self._on_property_changed(self.CONFIGURATION, self._configuration, previous)
        self._set_error("")
        if was_running and self._hardware_enabled and channels:
            self.start()
        elif not channels:
            self._set_status("NI-DAQ signal stream has no selected signals")
        else:
            self._set_status("NI-DAQ signal stream stopped")

    def start(self) -> bool:
        with self._lock:
            if self._is_running or self._is_starting:
                return True
            configuration = self._configuration
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
                message_queue = self._mp_ctx.Queue(maxsize=16)
                stop_event = self._mp_ctx.Event()
                process = self._mp_ctx.Process(
                    target=self._worker_target,
                    name="nidaq-signal-stream",
                    args=(configuration, message_queue, stop_event, None),
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
            self._recording_blocked = False
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
                self._close_recording_file()
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
                    kind, payload = message_queue.get(timeout=0.1)
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
                        self._recording_blocked = False
                        self._set_error("")
                        self._open_recording_file()
                        self._set_starting(False)
                        self._set_running(True)
                        self._set_status("NI-DAQ signal stream running")
                    log_hardware_initialization(
                        logger,
                        "READY | NI-DAQ signal stream | elapsed=%.3fs worker_pid=%s",
                        time.perf_counter() - started,
                        process.pid,
                    )
                elif kind == _WORKER_SAMPLE:
                    self._write_block(payload)
                    self.sample_block_received(payload)
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
            self._close_recording_file()
            self._set_starting(False)
            self._set_running(False)
            if self._configuration.is_enabled:
                self._set_status("NI-DAQ signal stream stopped")

    def _open_recording_file(self) -> None:
        if not self._configuration.record_to_acquisition:
            self._set_recording_path(None)
            return
        if self._recording_blocked:
            return
        project = self._project
        if project is None:
            self._set_recording_path(None)
            return
        try:
            source = project.get_source_path(self._configuration.output_name)
            path = Path(f"{source.full_path}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            file = path.open("a", newline="")
            writer = csv.writer(file)
            if path.stat().st_size == 0:
                writer.writerow(
                    ["wall_time", "perf_time", "sample_index"]
                    + [channel.name for channel in self._configuration.channels]
                )
                file.flush()
            self._recording_file = file
            self._recording_writer = writer
            self._set_recording_path(path)
        except Exception as exc:
            logger.exception("Failed to open NI-DAQ signal recording file")
            self._recording_blocked = True
            self._close_recording_file()
            self._set_error(f"NI-DAQ signal recording disabled: {str(exc) or exc.__class__.__name__}")

    def _write_block(self, block: NidaqSignalSampleBlock) -> None:
        with self._lock:
            writer = self._recording_writer
            file = self._recording_file
            if writer is None or file is None:
                return
            try:
                for sample_offset in range(block.sample_count):
                    writer.writerow(
                        [
                            block.wall_time + sample_offset / block.sample_rate_hz,
                            block.perf_time + sample_offset / block.sample_rate_hz,
                            block.sample_index + sample_offset,
                        ]
                        + [
                            block.values.get(channel.name, ())[sample_offset]
                            if sample_offset < len(block.values.get(channel.name, ()))
                            else ""
                            for channel in block.channels
                        ]
                    )
                file.flush()
            except Exception as exc:
                logger.exception("Failed to write NI-DAQ signal recording samples")
                self._recording_blocked = True
                self._close_recording_file()
                self._set_error(f"NI-DAQ signal recording stopped: {str(exc) or exc.__class__.__name__}")

    def _close_recording_file(self) -> None:
        file = self._recording_file
        self._recording_file = None
        self._recording_writer = None
        if file is not None:
            try:
                file.close()
            except Exception as exc:
                logger.exception("Failed to close NI-DAQ signal recording file")
                self._set_error(str(exc) or exc.__class__.__name__)

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

    def _set_recording_path(self, value: Optional[Path]) -> None:
        prev, self._recording_path = self._recording_path, value
        self._on_property_changed(self.RECORDING_PATH, value, prev)
