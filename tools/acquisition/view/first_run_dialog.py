"""First-run configuration step, shown before the main window.

It opens when there is no configuration file yet, or when the configured data
folder cannot be written (a drive not mounted, say). It names the
configuration, data and log locations, checks the data folder, shows what
hardware is present, and writes the configuration or the new data folder.
"""

from __future__ import annotations

import html
import os
import threading
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core.logging import get_verbose_logger
from tools.acquisition.model import first_run
from tools.acquisition.model.first_run import FirstRunAssessment, HardwareCheck

logger = get_verbose_logger(__name__)

SKIP_ENV = "REACHAQ_SKIP_FIRST_RUN"

OK_STYLE = "color: #166534;"
PROBLEM_STYLE = "color: #b00020; font-weight: 600;"
MISSING_STYLE = "color: #b45309;"


class _Relay(QObject):
    hardware_checked = Signal(object)


class FirstRunDialog(QDialog):
    def __init__(
        self,
        assessment: FirstRunAssessment,
        *,
        log_location: str = "",
        hardware_checks=first_run.HARDWARE_CHECKS,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.assessment = assessment
        self._hardware_checks = hardware_checks
        self._relay = _Relay()
        self._relay.hardware_checked.connect(self._show_hardware, Qt.ConnectionType.QueuedConnection)
        self._check_thread: Optional[threading.Thread] = None
        self.written_path: Optional[Path] = None

        new_install = not assessment.configuration_exists
        self.setWindowTitle("Set up reachAQ" if new_install else "Data folder unavailable")
        self.setMinimumWidth(640)

        intro = QLabel(
            "No configuration was found on this computer. Choose where recordings are saved; "
            "the rest of the configuration can be changed later in the app."
            if new_install else
            "The data folder in this computer's configuration cannot be used. If it is on a "
            "drive that is not mounted, mount it and choose Check again. Otherwise choose "
            "another folder, or start without changing it: recording stays unavailable "
            "until the folder can be written."
        )
        intro.setWordWrap(True)

        paths = QGroupBox("Locations")
        form = QFormLayout(paths)
        config_label = QLabel(str(assessment.configuration_path)
                              + ("  (will be created)" if new_install else ""))
        config_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Configuration:", config_label)
        self.data_edit = QLineEdit(str(assessment.data_folder))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        data_row = QHBoxLayout()
        data_row.addWidget(self.data_edit, 1)
        data_row.addWidget(browse)
        form.addRow("Data folder:", data_row)
        self.data_status = QLabel()
        self.data_status.setWordWrap(True)
        form.addRow("", self.data_status)
        log_label = QLabel(log_location or "In the data folder, one log per day")
        form.addRow("Logs:", log_label)
        # Checked a moment after typing stops: each check writes a probe file,
        # and a network path can take a while to answer.
        self._data_timer = QTimer(self)
        self._data_timer.setSingleShot(True)
        self._data_timer.setInterval(400)
        self._data_timer.timeout.connect(self._check_data_folder)
        self.data_edit.textChanged.connect(lambda _text: self._data_timer.start())

        hardware = QGroupBox("Hardware found")
        hardware_layout = QVBoxLayout(hardware)
        # One wrapped label per device: a grid of wrapped labels gets no
        # height-for-width and its rows overlap.
        self.hardware_list = QVBoxLayout()
        self.hardware_list.setSpacing(6)
        hardware_layout.addLayout(self.hardware_list)
        self.hardware_note = QLabel(
            "Missing hardware does not stop setup: enable each device in File ▸ Hardware once it is connected.")
        self.hardware_note.setWordWrap(True)
        hardware_layout.addWidget(self.hardware_note)
        self.recheck_button = QPushButton("Check again")
        self.recheck_button.clicked.connect(self.start_hardware_check)
        self.recheck_button.clicked.connect(self._check_data_folder)
        recheck_row = QHBoxLayout()
        recheck_row.addStretch(1)
        recheck_row.addWidget(self.recheck_button)
        hardware_layout.addLayout(recheck_row)

        self.buttons = QDialogButtonBox()
        self.accept_button = self.buttons.addButton(
            "Create configuration and start" if new_install else "Use this folder and start",
            QDialogButtonBox.ButtonRole.AcceptRole)
        self.start_unchanged_button = None
        if not new_install:
            self.start_unchanged_button = self.buttons.addButton(
                "Start without changing", QDialogButtonBox.ButtonRole.ActionRole)
            self.start_unchanged_button.clicked.connect(self._start_unchanged)
        self.buttons.addButton("Quit", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(paths)
        layout.addWidget(hardware)
        layout.addWidget(self.buttons)

        self._check_data_folder()
        self.start_hardware_check()

    # -- data folder -----------------------------------------------------

    def data_folder(self) -> Path:
        return Path(self.data_edit.text().strip()).expanduser()

    def _browse(self) -> None:
        start = self.data_folder()
        while not start.exists() and start != start.parent:
            start = start.parent
        chosen = QFileDialog.getExistingDirectory(self, "Data folder", str(start))
        if chosen:
            self.data_edit.setText(chosen)

    def _is_configured_folder(self, path: Path) -> bool:
        return (self.assessment.configuration_exists
                and path.as_posix() == Path(self.assessment.data_folder).expanduser().as_posix())

    def _check_data_folder(self) -> first_run.FolderCheck:
        self._data_timer.stop()
        text = self.data_edit.text().strip()
        if not text:
            self.data_status.setText("Choose a folder.")
            self.data_status.setStyleSheet(PROBLEM_STYLE)
            self.accept_button.setEnabled(False)
            return first_run.FolderCheck(Path(), False, "Choose a folder.")
        check = first_run.check_folder(self.data_folder())
        if check.ok:
            self.data_status.setText(check.summary)
            self.data_status.setStyleSheet(
                MISSING_STYLE if (check.free_bytes or 0) < first_run.LOW_SPACE_BYTES else OK_STYLE)
        elif not check.path.exists() and not self._is_configured_folder(check.path):
            # A new folder is created on accept, if its parent can be written.
            self.data_status.setText("The folder will be created.")
            self.data_status.setStyleSheet(MISSING_STYLE)
        else:
            # The configured folder is never created here: when its drive is
            # not mounted, a new folder at the mount point would take the
            # recordings onto the system disk and hide them once it mounts.
            self.data_status.setText(check.summary)
            self.data_status.setStyleSheet(PROBLEM_STYLE)
            self.accept_button.setEnabled(False)
            return check
        self.accept_button.setEnabled(True)
        return check

    # -- hardware --------------------------------------------------------

    def start_hardware_check(self) -> None:
        if self._check_thread is not None and self._check_thread.is_alive():
            return
        self.recheck_button.setEnabled(False)
        self._clear_hardware()
        self.hardware_list.addWidget(QLabel("Checking…"))

        def run(checks=self._hardware_checks):
            self._relay.hardware_checked.emit(first_run.run_hardware_checks(checks))

        self._check_thread = threading.Thread(target=run, name="first-run-hardware", daemon=True)
        self._check_thread.start()

    def _clear_hardware(self) -> None:
        while self.hardware_list.count():
            item = self.hardware_list.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _show_hardware(self, results: List[HardwareCheck]) -> None:
        self._clear_hardware()
        for result in results:
            color = "#166534" if result.found else "#b45309"
            mark = "✓" if result.found else "–"
            text = (f"<span style='color:{color}; font-weight:600'>{mark} {html.escape(result.name)}</span>"
                    f" · {html.escape(result.detail.lstrip('✓!→ '))}")
            if result.remedy:
                text += f"<br>{html.escape(result.remedy)}"
            label = QLabel(text)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            self.hardware_list.addWidget(label)
        self.recheck_button.setEnabled(True)
        self.hardware_results = list(results)

    # -- accept ----------------------------------------------------------

    def _start_unchanged(self) -> None:
        logger.warning("First run: started with the data folder unusable: %s (%s)",
                       self.assessment.data_folder, self.assessment.data_check.problem)
        self.accept()

    def _accept(self) -> None:
        folder = self.data_folder()
        check = first_run.check_folder(folder, create=not self._is_configured_folder(folder))
        if not check.ok:
            self._check_data_folder()
            QMessageBox.warning(self, "Data folder", f"{check.path}\n\n{check.problem}")
            return
        try:
            if self.assessment.configuration_exists:
                backup = first_run.set_data_folder(self.assessment.configuration_path, check.path)
                if backup is not None:
                    logger.notice("First run: data folder set to %s; previous configuration kept at %s",
                                  check.path, backup)
                self.written_path = self.assessment.configuration_path
            else:
                self.written_path = first_run.create_configuration(
                    self.assessment.configuration_path, check.path)
                logger.notice("First run: configuration created at %s with data folder %s",
                              self.written_path, check.path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Configuration not written", str(error))
            return
        self.accept()


def first_run_needed(configuration_path: Path) -> Optional[FirstRunAssessment]:
    assessment = first_run.assess(configuration_path)
    if assessment.configuration_problem:
        logger.warning("First run: %s", assessment.configuration_problem)
    return assessment if assessment.needed else None


def run_first_run_if_needed(configuration_path: Path, *, log_location: str = "") -> bool:
    """Show the first-run step when it is needed. False: the operator quit."""
    if os.environ.get(SKIP_ENV):
        return True
    assessment = first_run_needed(configuration_path)
    if assessment is None:
        return True
    if QApplication.platformName() in ("offscreen", "minimal"):
        # Nobody can answer a dialog here; the app's own checks still apply.
        logger.warning(
            "First-run step skipped on the %s platform: configuration_exists=%s data folder: %s",
            QApplication.platformName(), assessment.configuration_exists, assessment.data_check.summary)
        return True
    dialog = FirstRunDialog(assessment, log_location=log_location)
    return dialog.exec() == QDialog.DialogCode.Accepted
