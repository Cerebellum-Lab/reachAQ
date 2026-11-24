"""The operator alert banner, the alert log, and the diagnostic bundle export.

``install_operator_alerts`` is the one entry point the main window calls. It
adds a banner row under the toolbar (hidden while there is nothing to show), a
Help menu with "Alerts…" and "Export Diagnostic Bundle…", and a dialog for the
alerts that mean a recording ended early or data were not written.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core.logging import get_verbose_logger
from tools.acquisition.model.diagnostic_bundle import (
    BundleSources,
    build_diagnostic_bundle,
    default_bundle_name,
    default_credential_files,
)
from tools.acquisition.model.operator_alerts import (
    AlertSeverity,
    OperatorAlert,
    OperatorAlertBridge,
    OperatorAlertCenter,
)

logger = get_verbose_logger(__name__)

#: background, border/stripe, text
SEVERITY_COLORS = {
    AlertSeverity.WARNING: ("#fff7e0", "#b45309", "#4a2c00"),
    AlertSeverity.ERROR: ("#fdecec", "#b00020", "#4a0010"),
    AlertSeverity.CRITICAL: ("#b00020", "#7a0016", "#ffffff"),
}
SEVERITY_LABELS = {
    AlertSeverity.WARNING: "Warning",
    AlertSeverity.ERROR: "Error",
    AlertSeverity.CRITICAL: "Critical",
}


def _time_text(wall_time: float) -> str:
    return _dt.datetime.fromtimestamp(wall_time).strftime("%H:%M:%S")


def _alert_record(alert: OperatorAlert) -> dict:
    record = dataclasses.asdict(alert)
    record["severity"] = SEVERITY_LABELS[alert.severity]
    record["time"] = _dt.datetime.fromtimestamp(alert.raised_wall_time).isoformat(timespec="seconds")
    return record


class _Relay(QObject):
    """Carries center callbacks, from any thread, onto the GUI thread."""

    changed = Signal()
    dialog_requested = Signal(object)
    export_finished = Signal(object, str)


class _ElidedLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = max(self.width(), 40)
        self.setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width))


class OperatorAlertBanner(QFrame):
    """The most severe alert the operator has not dismissed."""

    def __init__(self, center: OperatorAlertCenter, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._center = center
        self._shown: Optional[OperatorAlert] = None
        self.setObjectName("OperatorAlertBanner")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.title_label = QLabel()
        self.title_label.setObjectName("OperatorAlertTitle")
        # Single elided lines, the full text in the tooltip and the alert log:
        # a wrapped label inside a toolbar keeps the height of its first layout.
        self.message_label = _ElidedLabel()
        self.remedy_label = _ElidedLabel()
        self.more_label = QLabel()
        self.details_button = QPushButton("Details…")
        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.clicked.connect(self._dismiss)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        text.addWidget(self.title_label)
        text.addWidget(self.message_label)
        text.addWidget(self.remedy_label)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 6, 4)
        row.addLayout(text, 1)
        for widget in (self.more_label, self.details_button, self.dismiss_button):
            row.addWidget(widget, 0, Qt.AlignmentFlag.AlignTop)
        self.refresh()

    @property
    def shown_alert(self) -> Optional[OperatorAlert]:
        return self._shown

    def refresh(self) -> None:
        visible = self._center.visible_alerts()
        self._shown = visible[0] if visible else None
        # The window shows or hides the row this banner sits in.
        if self._shown is None:
            return
        alert = self._shown
        background, border, color = SEVERITY_COLORS[alert.severity]
        self.setStyleSheet(
            f"#OperatorAlertBanner {{background: {background}; border-left: 6px solid {border};"
            f" border-bottom: 1px solid {border};}}"
            f"#OperatorAlertBanner QLabel {{color: {color};}}"
            "#OperatorAlertTitle {font-weight: 700;}"
        )
        repeat = f" (×{alert.count})" if alert.count > 1 else ""
        self.title_label.setText(f"{SEVERITY_LABELS[alert.severity]}: {alert.title}{repeat}")
        message = " ".join(alert.message.split())
        self.message_label.set_full_text(message)
        self.remedy_label.set_full_text(f"What to do: {alert.remedy}" if alert.remedy else "")
        self.remedy_label.setVisible(bool(alert.remedy))
        others = len(visible) - 1
        self.more_label.setText(f"+{others} more" if others > 0 else "")
        self.more_label.setVisible(others > 0)

    def _dismiss(self) -> None:
        if self._shown is not None:
            self._center.dismiss(self._shown.key)


class OperatorAlertLogDialog(QDialog):
    COLUMNS = ("Time", "Severity", "Alert", "Detail", "What to do", "State")

    def __init__(self, center: OperatorAlertCenter, export, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Alerts")
        self.resize(980, 420)
        self._center = center
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.export_button = buttons.addButton(
            "Export Diagnostic Bundle…", QDialogButtonBox.ButtonRole.ActionRole)
        self.export_button.clicked.connect(export)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        self.empty_label = QLabel("No alerts since reachAQ started.")
        layout.addWidget(self.empty_label)
        layout.addWidget(self.table, 1)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self) -> None:
        active_keys = {alert.key: alert for alert in self._center.active_alerts()}
        history = tuple(reversed(self._center.history()))
        self.empty_label.setVisible(not history)
        self.table.setRowCount(len(history))
        for row, alert in enumerate(history):
            current = active_keys.get(alert.key)
            if not alert.active:
                state = "dismissed" if alert.dismissed else "cleared"
            elif current is not None and current.sequence == alert.sequence:
                state = "active, dismissed" if current.dismissed else "active"
            else:
                state = "superseded"
            values = (
                _time_text(alert.raised_wall_time),
                SEVERITY_LABELS[alert.severity],
                alert.title,
                alert.message,
                alert.remedy,
                state,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row, column, item)


class OperatorAlertsUi:
    """Everything ``install_operator_alerts`` made, kept alive by the window."""

    def __init__(self, window: QMainWindow, app_model, preferences):
        self.window = window
        self._app_model = app_model
        self._preferences = preferences
        self.center = OperatorAlertCenter()
        self.bridge = OperatorAlertBridge(app_model, self.center)
        self._relay = _Relay()
        self._relay.changed.connect(self._refresh, Qt.ConnectionType.QueuedConnection)
        self._relay.dialog_requested.connect(self._show_dialog, Qt.ConnectionType.QueuedConnection)
        self._relay.export_finished.connect(self._on_export_finished, Qt.ConnectionType.QueuedConnection)
        self.center.changed.append(self._relay.changed.emit)
        self.center.dialog_requested.append(self._relay.dialog_requested.emit)
        self._log_dialog: Optional[OperatorAlertLogDialog] = None
        self._alert_box: Optional[QMessageBox] = None
        self._export_thread: Optional[threading.Thread] = None

        self.banner = OperatorAlertBanner(self.center)
        self.banner.details_button.clicked.connect(self.show_alert_log)
        toolbar = self.toolbar = QToolBar("Alerts", window)
        toolbar.setObjectName("OperatorAlertToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.toggleViewAction().setVisible(False)
        toolbar.setStyleSheet("QToolBar#OperatorAlertToolbar {border: 0; padding: 0; spacing: 0;}")
        self._banner_action = toolbar.addWidget(self.banner)
        window.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        window.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self.alerts_action = QAction("Alerts…", window)
        self.alerts_action.triggered.connect(self.show_alert_log)
        self.export_action = QAction("Export Diagnostic Bundle…", window)
        self.export_action.triggered.connect(self.export_bundle)
        help_menu = self.help_menu = window.menuBar().addMenu("Help")
        help_menu.addAction(self.alerts_action)
        help_menu.addAction(self.export_action)

        self.bridge.connect()
        self._refresh()

    # -- banner and log --------------------------------------------------

    def _refresh(self) -> None:
        self.banner.refresh()
        self.toolbar.setVisible(self.banner.shown_alert is not None)
        if self._log_dialog is not None and self._log_dialog.isVisible():
            self._log_dialog.refresh()

    def show_alert_log(self) -> None:
        if self._log_dialog is None:
            self._log_dialog = OperatorAlertLogDialog(self.center, self.export_bundle, self.window)
        self._log_dialog.refresh()
        self._log_dialog.show()
        self._log_dialog.raise_()
        self._log_dialog.activateWindow()

    def _show_dialog(self, alert: OperatorAlert) -> None:
        """Window-modal and opened with open(), so no thread waits on it. One
        at a time: a second while one is open stays on the banner and log."""
        if self._alert_box is not None and self._alert_box.isVisible():
            return
        box = self._alert_box = QMessageBox(self.window)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(alert.title)
        box.setText(alert.title)
        informative = alert.message
        if alert.remedy:
            informative = f"{informative}\n\nWhat to do: {alert.remedy}"
        box.setInformativeText(informative)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        export = box.addButton("Export Diagnostic Bundle…", QMessageBox.ButtonRole.ActionRole)
        export.clicked.connect(self.export_bundle)
        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.open()

    # -- diagnostic bundle -----------------------------------------------

    def bundle_sources(self) -> BundleSources:
        model = self._app_model
        configuration_path = None
        try:
            configuration_path = Path(model.get_config_location())
        except Exception:
            logger.exception("Diagnostic bundle: configuration location unavailable")
        output_roots = []
        try:
            if model.output_location:
                output_roots.append(Path(model.output_location))
        except Exception:
            logger.exception("Diagnostic bundle: output location unavailable")
        log_dirs = []
        log_location = getattr(self._preferences, "log_location", "")
        if log_location:
            log_dirs.append(Path(log_location))
        try:
            subsystems = {
                key: {**dataclasses.asdict(status), "state": status.state.value}
                for key, status in model.subsystem_statuses.items()
            }
        except Exception as error:
            subsystems = {"error": str(error)}
        app_state = {}
        for name in ("status", "session_recording_status", "acquisition_started",
                     "recording_blockers", "recording_ending_reason"):
            try:
                value = getattr(model, name)
            except Exception as error:
                value = f"unavailable: {error}"
            app_state[name] = getattr(value, "value", value)
        return BundleSources(
            configuration_path=configuration_path,
            preferences_path=Path(self._preferences.settings_file_name)
            if getattr(self._preferences, "settings_file_name", "") else None,
            output_roots=tuple(output_roots),
            log_dirs=tuple(log_dirs),
            subsystem_snapshot=subsystems,
            alerts=[_alert_record(alert) for alert in self.center.history()],
            app_state=app_state,
            credential_files=default_credential_files(),
        )

    def export_bundle(self) -> None:
        if self._export_thread is not None and self._export_thread.is_alive():
            self.window.statusBar().showMessage("A diagnostic bundle is already being written", 5000)
            return
        suggested = str(Path.home() / default_bundle_name())
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Export Diagnostic Bundle", suggested, "Zip archive (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        sources = self.bundle_sources()
        self.export_action.setEnabled(False)
        self.window.statusBar().showMessage("Writing diagnostic bundle…", 0)

        def run(path=Path(path), sources=sources):
            try:
                result = build_diagnostic_bundle(path, sources)
            except Exception as error:
                logger.warning("Diagnostic bundle failed: %s", error, exc_info=True)
                self._relay.export_finished.emit(None, str(error) or type(error).__name__)
            else:
                logger.info("Diagnostic bundle written to %s (%s bytes, %s redactions)",
                            result.path, result.size_bytes, result.redactions)
                self._relay.export_finished.emit(result, "")

        self._export_thread = threading.Thread(target=run, name="diagnostic-bundle", daemon=True)
        self._export_thread.start()

    def _on_export_finished(self, result, error: str) -> None:
        self.export_action.setEnabled(True)
        self.window.statusBar().clearMessage()
        if result is None:
            QMessageBox.warning(self.window, "Diagnostic bundle not written", error)
            return
        skipped = f"\n\n{len(result.skipped)} item(s) were skipped; see manifest.json." if result.skipped else ""
        QMessageBox.information(
            self.window,
            "Diagnostic bundle written",
            f"{result.path}\n\n{result.size_bytes / 1e6:.1f} MB, {len(result.members)} files, "
            f"{result.redactions} value(s) redacted.{skipped}\n\n"
            "Passwords, tokens and keyring entries are not included.",
        )

    def close(self) -> None:
        self.bridge.disconnect()


def install_operator_alerts(window: QMainWindow, app_model, preferences) -> OperatorAlertsUi:
    return OperatorAlertsUi(window, app_model, preferences)
