"""The 3D calibration's operator pages: which cameras to calibrate, the result, and the
confirmation Record asks for when the cameras differ from the active calibration's.

A camera a calibration left 2D only is named, with its reason, wherever the operator looks:
first and in red on the result page, and in the confirmation's details.
"""

import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QRadioButton, QVBoxLayout, QWidget)

from autotrainer.inference.multires_calibration.extract import MAX_CAMERAS, MIN_CAMERAS
from autotrainer.inference.multires_calibration.report import NOT_CALIBRATED, QualityReport
from tools.acquisition.model.calibration_geometry import camera_serial

SCOUT = "scout"
HANDWRITTEN = "handwritten"
NOT_AVAILABLE_YET = "not available yet"
SCOUT_TOOLTIP = ("the stage first visits a coarse grid and every camera looks for the board; the waypoints are"
                 " chosen from what they saw, or the hand-written list when fewer than 12 suit")
HANDWRITTEN_TOOLTIP = "the fixed list of 37 waypoints"
SCOUT_ADVICE_TITLE = "Some cameras saw too little of the board"
SCOUT_ADVICE_CONTINUE = "Continue: the cameras named may end up 2D only"
#: The red of every "2D only" mark.
UNCALIBRATED_COLOR = "#c62828"

SETUP_DIFFERS_TITLE = "Calibration looks needed"
SETUP_DIFFERS_TEXT = "Camera setup differs from the active calibration; a calibration looks needed. Record anyway?"
_SWAPPED_PREFIX = "swapped: "


@dataclass(frozen=True)
class CalibrationRequest:
    """What the operator chose: the cameras in camera-list order, primary first; the move list,
    "scout" or "handwritten"; and whether to capture reference images."""

    cameras: List[str]
    move_list: str
    capture_references: bool


def camera_unavailable_reason(camera) -> Optional[str]:
    """Why `camera` (a VideoCaptureModel) cannot be calibrated, or None when it can."""
    if not camera.is_enabled:
        return "disabled"
    config = camera.active_config
    scheme = str(getattr(config, "scheme", "") or "").lower()
    if scheme != "spinnaker":
        return f"no hardware (scheme {scheme or 'none'})"
    return None


def _camera_text(camera) -> str:
    serial = camera_serial(camera.active_config)
    return f"{camera.name} ({serial or 'no serial'})"


class CalibrationDialog(QDialog):
    """One checkbox per configured camera, all ticked but those that cannot be calibrated.

    Start is enabled with 2 to 6 ticked. request() gives the choice.
    """

    def __init__(self, cameras: Sequence, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Make 3D calibration")
        # Primary first, then the rest in camera-list order: the trigger order make_3d_calib keeps.
        self._cameras = [c for c in cameras if c.is_primary] + [c for c in cameras if not c.is_primary]
        self._boxes: Dict[str, QCheckBox] = {}
        self._reasons: Dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Cameras to calibrate ({MIN_CAMERAS} to {MAX_CAMERAS}):"))
        grid = QGridLayout()
        for row, camera in enumerate(self._cameras):
            reason = camera_unavailable_reason(camera)
            box = QCheckBox(_camera_text(camera))
            box.setChecked(reason is None)
            box.setEnabled(reason is None)
            label = QLabel(reason or "")
            if reason is not None:
                box.setToolTip(reason)
                label.setToolTip(reason)
                label.setStyleSheet("color: gray;")
            box.toggled.connect(self._update_start)
            grid.addWidget(box, row, 0)
            grid.addWidget(label, row, 1)
            self._boxes[camera.name] = box
            self._reasons[camera.name] = label
        layout.addLayout(grid)

        moves = QGroupBox("Move list")
        moves_layout = QHBoxLayout(moves)
        self.scout_button = QRadioButton("automatic (scout)")
        self.scout_button.setToolTip(SCOUT_TOOLTIP)
        self.scout_button.setChecked(True)
        self.handwritten_button = QRadioButton("hand-written")
        self.handwritten_button.setToolTip(HANDWRITTEN_TOOLTIP)
        group = self._move_group = QButtonGroup(self)
        group.addButton(self.scout_button)
        group.addButton(self.handwritten_button)
        moves_layout.addWidget(self.scout_button)
        moves_layout.addWidget(self.handwritten_button)
        layout.addWidget(moves)

        self.references_box = QCheckBox("Capture reference images")
        self.references_box.setEnabled(False)
        self.references_box.setToolTip(NOT_AVAILABLE_YET)
        layout.addWidget(self.references_box)

        self._count_label = QLabel("")
        layout.addWidget(self._count_label)

        buttons = QDialogButtonBox()
        self.start_button = buttons.addButton("Start", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_start()

    def checkbox(self, name: str) -> QCheckBox:
        return self._boxes[name]

    def reason_label(self, name: str) -> QLabel:
        return self._reasons[name]

    def _ticked(self) -> List[str]:
        return [camera.name for camera in self._cameras if self._boxes[camera.name].isChecked()]

    def _update_start(self, *_):
        count = len(self._ticked())
        fits = MIN_CAMERAS <= count <= MAX_CAMERAS
        self.start_button.setEnabled(fits)
        self._count_label.setText(
            f"{count} cameras ticked" + ("" if fits else f"; a calibration takes {MIN_CAMERAS} to {MAX_CAMERAS}"))

    def request(self) -> CalibrationRequest:
        return CalibrationRequest(
            cameras=self._ticked(),
            move_list=SCOUT if self.scout_button.isChecked() else HANDWRITTEN,
            capture_references=self.references_box.isChecked(),
        )


def scout_advice_text(diagnosis: Dict[str, str]) -> str:
    """The scout's advice, one "<camera>: <advice>" line per camera that saw too little, then what Continue does."""
    lines = [f"{camera}: {advice}" for camera, advice in diagnosis.items()]
    return "\n".join([*lines, "", SCOUT_ADVICE_CONTINUE + "."])


def confirm_scout_advice(parent, diagnosis: Dict[str, str]) -> bool:
    """Show the scout's advice before the capture; True for Continue, False for Cancel."""
    box = QMessageBox(QMessageBox.Icon.Warning, SCOUT_ADVICE_TITLE, scout_advice_text(diagnosis), parent=parent)
    go_on = box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(go_on)
    box.exec()
    return box.clickedButton() is go_on


def uncalibrated_lines(report: QualityReport) -> List[str]:
    """"<camera> NOT CALIBRATED (2D only): <reason>", one per camera the report left 2D only."""
    return [f"{v.name} {NOT_CALIBRATED}: {v.reason}" for v in report.cameras.values() if not v.calibrated]


def result_page(report: QualityReport) -> str:
    """The report as rich text: each 2D-only camera as a red banner line at the top, then the
    verdict, then the report. The text report's own 2D-only lines are left out, as the
    banner says them."""
    banner = "".join(
        f'<p style="color: {UNCALIBRATED_COLOR}; font-weight: bold;">{html.escape(line)}</p>'
        for line in uncalibrated_lines(report)
    )
    body = "\n".join(line for line in report.to_text().splitlines() if not line.startswith(NOT_CALIBRATED))
    return f"{banner}<pre>{html.escape(body)}</pre>"


def setup_difference_details(difference, reasons: Dict[str, str]) -> List[str]:
    """The details lines of the confirmation: which cameras were added, removed or swapped."""
    lines = []
    if difference.added:
        lines.append(f"Added, not in the calibration: {', '.join(difference.added)}")
    if difference.removed:
        lines.append(f"Removed, calibrated but no longer configured: {', '.join(difference.removed)}")
    for name in difference.swapped:
        reason = reasons.get(name, "")
        reason = reason[len(_SWAPPED_PREFIX):] if reason.startswith(_SWAPPED_PREFIX) else reason
        lines.append(f"Swapped for another camera body: {name}" + (f", {reason}" if reason else ""))
    return lines


def uncalibrated_off_note(difference) -> Optional[str]:
    if not difference.uncalibrated_off:
        return None
    return f"Calibrated cameras not recording: {', '.join(difference.uncalibrated_off)}"


def confirm_camera_setup(parent, app_model) -> Tuple[bool, Optional[str]]:
    """Ask before Record when the recording's cameras differ from the active calibration's.

    Returns (whether to record, the passive note for the status bar). Added, removed or
    swapped cameras ask; Yes acknowledges the difference (acknowledge_setup_difference) for
    this start. Calibrated cameras that are only not recording are the note, without asking.
    Only the GUI's Record asks: headless and API starts record the mismatch unacknowledged.
    """
    difference = app_model.camera_setup_difference(app_model.recording_camera_names)
    if difference.differs:
        details = setup_difference_details(difference, app_model.uncalibrated_reasons)
        rsp = QMessageBox.question(
            parent,
            SETUP_DIFFERS_TITLE,
            "\n\n".join([SETUP_DIFFERS_TEXT, "\n".join(details)]),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if rsp != QMessageBox.StandardButton.Yes:
            return False, None
        app_model.acknowledge_setup_difference()
    return True, uncalibrated_off_note(difference)
