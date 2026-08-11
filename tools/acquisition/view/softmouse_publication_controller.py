from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from tools.acquisition.model.app_model_status import SessionRecordingStatus


class SoftMousePublicationController(QObject):
    """Run the shared SoftMouse publisher without blocking the Qt event loop."""

    running_changed = Signal(bool)
    status_changed = Signal(str)

    def __init__(
        self,
        app_model,
        parent=None,
        *,
        program: str | None = None,
        arguments: tuple[str, ...] | None = None,
        working_directory: Path | None = None,
    ):
        super().__init__(parent)
        self._app_model = app_model
        self._program = program or sys.executable
        self._arguments = arguments or ("-m", "tools.softmouse_sync.cli")
        self._working_directory = Path(
            working_directory or Path(__file__).resolve().parents[3]
        )
        self._status = "Ready to sync the shared SoftMouse publication"
        self._process = QProcess(self)
        self._process.setProcessChannelMode(
            QProcess.ProcessChannelMode.SeparateChannels
        )
        self._process.started.connect(self._on_started)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_process_error)

    @classmethod
    def for_model(cls, app_model, parent=None):
        controller = getattr(app_model, "_softmouse_publication_controller", None)
        if controller is None:
            controller = cls(app_model, parent)
            app_model._softmouse_publication_controller = controller
        return controller

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> bool:
        if self.is_running:
            self._set_status("A SoftMouse sync is already running")
            return False
        if (
            self._app_model.session_recording_status
            is not SessionRecordingStatus.READY
        ):
            self._set_status("SoftMouse sync is unavailable during recording")
            return False
        self._process.setProgram(self._program)
        self._process.setArguments(list(self._arguments))
        self._process.setWorkingDirectory(self._working_directory.as_posix())
        self._set_status("Starting SoftMouse sync…")
        self._process.start()
        self.running_changed.emit(True)
        return True

    def _on_started(self) -> None:
        self._set_status("Downloading current Christie animals from SoftMouse…")

    def _on_process_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._set_status(
                f"SoftMouse sync could not start: {self._process.errorString()}"
            )
            self.running_changed.emit(False)

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        stdout = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        stderr = bytes(self._process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if exit_code != 0:
            detail = self._last_output_line(stderr) or self._last_output_line(stdout)
            self._set_status(detail or f"SoftMouse sync failed (exit {exit_code})")
            self.running_changed.emit(False)
            return

        publication = self._last_output_line(stdout) or "Shared publication updated"
        self._set_status(f"{publication}; refreshing this computer's cache…")
        try:
            self._app_model.refresh_animal_metadata()
        except Exception as exc:
            self._set_status(
                f"Shared publication updated, but local cache refresh failed: {exc}"
            )
        else:
            self._set_status(f"{publication}; local cache refreshed")
        self.running_changed.emit(False)

    def _set_status(self, value: str) -> None:
        self._status = value
        self.status_changed.emit(value)

    @staticmethod
    def _last_output_line(value: str) -> str:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        return lines[-1] if lines else ""
