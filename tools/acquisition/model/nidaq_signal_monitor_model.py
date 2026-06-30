from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Callable, Optional, TextIO

from autotrainer.core import NidaqSignalStreamConfiguration, ObservableObject, ProjectInfo
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.project import ProjectDependentProtocol
from autotrainer.device import NidaqSignalSampleBlock, NidaqSignalStreamController


logger = get_verbose_logger(__name__)


class NidaqSignalMonitorEvents:
    sample_block_received = Callable[[NidaqSignalSampleBlock], None]


class NidaqSignalMonitorModel(ObservableObject, ProjectDependentProtocol):
    CONFIGURATION = "configuration"
    IS_RUNNING = "is_running"
    STATUS_MESSAGE = "status_message"
    ERROR_MESSAGE = "error_message"
    RECORDING_PATH = "recording_path"

    sample_block_received: NidaqSignalMonitorEvents.sample_block_received

    def __init__(self):
        super().__init__(("sample_block_received",))
        self._configuration = NidaqSignalStreamConfiguration()
        self._project: Optional[ProjectInfo] = None
        self._controller: Optional[NidaqSignalStreamController] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._recording_file: Optional[TextIO] = None
        self._recording_writer: Optional[csv.writer] = None
        self._recording_path: Optional[Path] = None
        self._recording_blocked = False
        self._is_running = False
        self._status_message = "NI-DAQ signal stream disabled"
        self._error_message = ""

    @property
    def configuration(self) -> NidaqSignalStreamConfiguration:
        return self._configuration

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
        if configuration.is_enabled:
            self.start()
        else:
            self._set_status("NI-DAQ signal stream disabled")
            self._set_error("")

    def save_configuration(self) -> NidaqSignalStreamConfiguration:
        return self._configuration

    def start(self) -> bool:
        with self._lock:
            if self._is_running:
                return True
            configuration = self._configuration
            if not configuration.is_enabled:
                self._set_status("NI-DAQ signal stream disabled")
                self._set_error("")
                return False
            try:
                controller = NidaqSignalStreamController(configuration)
                controller.start()
            except Exception as exc:
                logger.exception("Failed to start NI-DAQ signal stream")
                self._controller = None
                self._set_running(False)
                self._set_status("NI-DAQ signal stream stopped")
                self._set_error(str(exc) or exc.__class__.__name__)
                return False

            self._controller = controller
            self._stop_event.clear()
            self._recording_blocked = False
            self._set_error("")
            self._open_recording_file()
            thread = threading.Thread(target=self._run, name="nidaq_signal_monitor", daemon=True)
            self._thread = thread
            self._set_status("NI-DAQ signal stream running")
            self._set_running(True)
            thread.start()
            return True

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            self._close_recording_file()
            return
        self._stop_event.set()
        controller = self._controller
        if controller is not None:
            try:
                controller.close()
            except Exception as exc:
                logger.exception("Failed while stopping NI-DAQ signal stream")
                self._set_error(str(exc) or exc.__class__.__name__)
        if thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                self._set_error("Timed out waiting for NI-DAQ signal stream thread to stop")
                return
        self._thread = None
        self._controller = None
        self._close_recording_file()
        self._set_running(False)
        if self._configuration.is_enabled:
            self._set_status("NI-DAQ signal stream stopped")

    def close(self) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                controller = self._controller
                if controller is None:
                    break
                block = controller.read_chunk()
                if block.sample_count <= 0:
                    continue
                self._write_block(block)
                self.sample_block_received(block)
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("NI-DAQ signal stream failed")
                self._set_status("NI-DAQ signal stream stopped")
                self._set_error(str(exc) or exc.__class__.__name__)
        finally:
            self._stop_event.set()
            try:
                controller = self._controller
                if controller is not None:
                    controller.close()
            except Exception as exc:
                logger.exception("Failed to close NI-DAQ signal stream after worker exit")
                self._set_error(str(exc) or exc.__class__.__name__)
            self._close_recording_file()
            self._set_running(False)

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

    def _set_status(self, value: str) -> None:
        prev, self._status_message = self._status_message, value
        self._on_property_changed(self.STATUS_MESSAGE, value, prev)

    def _set_error(self, value: str) -> None:
        prev, self._error_message = self._error_message, value
        self._on_property_changed(self.ERROR_MESSAGE, value, prev)

    def _set_recording_path(self, value: Optional[Path]) -> None:
        prev, self._recording_path = self._recording_path, value
        self._on_property_changed(self.RECORDING_PATH, value, prev)
