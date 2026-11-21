"""Preferences -> Behavior: the two moved-camera checks, each with its switch and its limits.

The reference-image check (checks.reference_image) runs at Record; the pose-consistency
check (checks.pose_consistency) runs in offline 3D. Every field shows the value in
calibrations/settings.yaml, or the documented default where it is unset or invalid, and an
edit is saved alone (AppModel.update_moved_camera_check), keeping every other key. An entry
that is not a number in the field's range is refused beside the field and not saved.
Per-camera overrides stay in the YAML.
"""

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit

from autotrainer.inference.analysis.offline_3d import ConsistencyLimits, pose_consistency_settings
from autotrainer.pyside import QSwitch
from tools.acquisition.model import reference_image_check as ric

REFERENCE_CHECK = ric.REFERENCE_KEY
POSE_CHECK = "pose_consistency"
INVALID_STYLE = "border: 1px solid #c62828;"
ERROR_COLOR = "#c62828"


def per_camera_hint(check: str) -> str:
    return f"Per-camera overrides: edit checks.{check}.per_camera in calibrations/settings.yaml"


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: type  # float or int
    low: float
    high: float
    default: Optional[float]
    optional: bool = False  # empty saves null: the limit is off

    def text(self, value) -> str:
        if value is None:
            return ""
        return str(int(value)) if self.kind is int else f"{float(value):g}"

    def range_text(self) -> str:
        return f"{self.text(self.low)} to {self.text(self.high)}"

    def tooltip(self) -> str:
        default = "empty (off)" if self.default is None else self.text(self.default)
        tip = f"{self.label}: {self.range_text()}; default {default}"
        return tip + ("; empty means off" if self.optional else "")

    def parse(self, text: str) -> Tuple[bool, Optional[float]]:
        """(valid, value) for `text`: a number of the field's kind within its range, or empty
        for an optional field (None). Never raises: "inf" and "nan" are invalid."""
        text = text.strip()
        if not text:
            return self.optional, None
        try:
            value = float(text)
        except ValueError:
            return False, None
        if not math.isfinite(value):
            return False, None
        if self.kind is int:
            if value != int(value):
                return False, None
            value = int(value)
        return self.low <= value <= self.high, value


_REFERENCE_DEFAULTS = ric.Limits()
REFERENCE_FIELDS = (
    Field("max_shift_px", "max shift (px)", float, 0.5, 50, _REFERENCE_DEFAULTS.max_shift_px),
    Field("max_rotation_deg", "max rotation (\N{DEGREE SIGN})", float, 0.1, 20, _REFERENCE_DEFAULTS.max_rotation_deg),
    Field("max_scale_pct", "max scale change (%)", float, 0.1, 20, _REFERENCE_DEFAULTS.max_scale_pct),
    Field("min_quality", "min match quality", float, 0.0, 1.0, _REFERENCE_DEFAULTS.min_quality),
)
_POSE_DEFAULTS = ConsistencyLimits()
POSE_FIELDS = (
    Field("median_px", "median limit (px)", float, 0.5, 100, _POSE_DEFAULTS.median_px),
    Field("p95_px", "p95 limit (px)", float, 1, 200, _POSE_DEFAULTS.p95_px, optional=True),
    Field("min_frames", "minimum frames", int, 1, 1_000_000, _POSE_DEFAULTS.min_frames),
    Field("min_observations", "minimum observations", int, 1, 10_000_000, _POSE_DEFAULTS.min_observations),
    Field("min_confidence", "confidence cutoff", float, 0.0, 1.0, _POSE_DEFAULTS.min_confidence),
)


class MovedCameraChecks(QGroupBox):
    """The switches and limits of both checks, read from settings.yaml when built: Preferences
    builds a new one each time it opens."""

    def __init__(self, app_model, parent=None):
        super().__init__("Moved-camera checks", parent)
        self._app_model = app_model
        self.switches: Dict[str, QSwitch] = {}
        self.fields: Dict[Tuple[str, str], QLineEdit] = {}
        self.errors: Dict[str, QLabel] = {}
        # The text each field shows as stored: leaving a field unchanged saves nothing.
        self._stored: Dict[Tuple[str, str], str] = {}

        settings = app_model.library_settings()
        reference = ric.reference_check_settings(settings)
        checks = settings.get(ric.CHECKS_KEY)
        pose = pose_consistency_settings(checks.get(POSE_CHECK) if isinstance(checks, Mapping) else None)

        layout = self._layout = QGridLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._add_check(
            0, REFERENCE_CHECK, "Reference image:", reference.enabled, reference.limits, REFERENCE_FIELDS,
            "At Record, compares each camera's view with its calibration's reference image and asks before"
            " recording when a camera looks shifted, rotated or rescaled. Off by default. Needs a calibration"
            " with reference images.")
        self._add_check(
            3, POSE_CHECK, "Pose consistency:", pose.enabled, pose.limits, POSE_FIELDS,
            "In offline 3D, flags a camera whose 2D pose disagrees with the others' triangulated points"
            " (its reprojection residual). Off by default.")

    def _add_check(self, row, check, title, enabled, limits, fields, tooltip):
        layout = self._layout
        hint = per_camera_hint(check)
        label = QLabel(title)
        label.setToolTip(f"{tooltip}\n{hint}")
        switch = self.switches[check] = QSwitch()
        switch.setToolTip(f"{tooltip}\n{hint}")
        switch.setChecked(enabled)
        switch.stateChanged.connect(lambda state, check=check: self._switch_changed(check, state != 0))
        layout.addWidget(label, row, 0)
        layout.addWidget(switch, row, 1, Qt.AlignmentFlag.AlignLeft)

        values = QHBoxLayout()
        for field in fields:
            text = self._stored[(check, field.key)] = field.text(getattr(limits, field.key))
            edit = self.fields[(check, field.key)] = QLineEdit(text)
            edit.setPlaceholderText("off" if field.default is None else field.text(field.default))
            edit.setToolTip(f"{field.tooltip()}\n{hint}")
            edit.setMaximumWidth(90)
            edit.editingFinished.connect(lambda check=check, field=field: self._field_edited(check, field))
            values.addWidget(QLabel(f"{field.label}:"))
            values.addWidget(edit)
        values.addStretch(1)
        layout.addLayout(values, row + 1, 0, 1, 3)

        error = self.errors[check] = QLabel("")
        error.setStyleSheet(f"color: {ERROR_COLOR};")
        error.setVisible(False)
        layout.addWidget(error, row + 2, 0, 1, 3)

    def _switch_changed(self, check: str, enabled: bool) -> None:
        if check == REFERENCE_CHECK:
            self._app_model.reference_image_check_enabled = enabled
        else:
            self._app_model.update_moved_camera_check(check, enabled=enabled)

    def _field_edited(self, check: str, field: Field) -> None:
        edit = self.fields[(check, field.key)]
        error = self.errors[check]
        if edit.text().strip() == self._stored[(check, field.key)]:
            edit.setStyleSheet("")
            self._clear_error(error, field)
            return
        valid, value = field.parse(edit.text())
        if not valid:
            edit.setStyleSheet(INVALID_STYLE)
            needs = f"a whole number from {field.range_text()}" if field.kind is int else (
                f"a number from {field.range_text()}")
            error.setText(f"{field.label}: not saved, it needs {needs}" + (", or empty for off" if field.optional
                                                                           else ""))
            error.setVisible(True)
            return
        edit.setStyleSheet("")
        self._clear_error(error, field)
        self._app_model.update_moved_camera_check(check, **{field.key: value})
        self._stored[(check, field.key)] = edit.text().strip()

    @staticmethod
    def _clear_error(error: QLabel, field: Field) -> None:
        if error.text().startswith(f"{field.label}:"):
            error.setText("")
            error.setVisible(False)
