"""Build laser pulse trains and keep them as profiles.

A profile is the pulse train and nothing else. Which laser fires it and how
it is started are chosen where it is used: on a laser tab, in Test stim, or in
a protocol row. The unsaved state of these controls is the builder draft,
which a laser tab can fire without saving a revision for every change.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autotrainer.pyside import PGWidget
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.view.compact_panel import CollapsibleSection, compact_plot_axes

#: The identifier the builder draft is fired under.
DRAFT_PROFILE_ID = "builder-draft"
#: Past these, a field only gets emptier; the extra width goes to the preview.
_FIELD_MAXIMUM_WIDTH = 170
_WIDE_FIELD_MAXIMUM_WIDTH = 400
#: Field columns grow first; an empty last column takes what is left once
#: the fields reach their maximum width.
_FIELD_COLUMN_STRETCH = 10


def pulse_shape_refusal(profile: LaserPulseProfile) -> str:
    """Why this train cannot be generated, or empty."""
    if profile.pulse_count > 1 and profile.frequency_hz:
        period_ms = 1000.0 / profile.frequency_hz
        if profile.pulse_duration_ms > period_ms:
            return (
                f"Pulse width {profile.pulse_duration_ms:g} ms exceeds the "
                f"{period_ms:g} ms period at {profile.frequency_hz:g} Hz"
            )
    return ""


def preview_points(profile: LaserPulseProfile, rest_volts: float = 0.0) -> Tuple[list, list]:
    """The command waveform as step points, for the preview plot."""
    duration_s = profile.pulse_duration_ms / 1000.0
    baseline_s = profile.baseline_ms / 1000.0
    post_stim_s = profile.post_stim_ms / 1000.0
    period_s = (1.0 / profile.frequency_hz) if profile.frequency_hz else duration_s
    x_values = [0.0]
    y_values = [rest_volts]
    current_t = 0.0

    def horizontal(to_t: float) -> None:
        nonlocal current_t
        if to_t <= current_t:
            return
        x_values.append(to_t)
        y_values.append(y_values[-1])
        current_t = to_t

    def transition(value: float) -> None:
        x_values.append(current_t)
        y_values.append(y_values[-1])
        x_values.append(current_t)
        y_values.append(value)

    horizontal(baseline_s)
    for pulse_index in range(profile.pulse_count):
        pulse_start = baseline_s + pulse_index * period_s
        horizontal(pulse_start)
        transition(profile.amplitude_volts)
        horizontal(pulse_start + duration_s)
        transition(rest_volts)
    horizontal(current_t + post_stim_s)
    return x_values, y_values


def _ms_spinbox(value: float) -> QDoubleSpinBox:
    spinbox = QDoubleSpinBox()
    spinbox.setDecimals(3)
    spinbox.setSingleStep(1.0)
    spinbox.setSuffix(" ms")
    spinbox.setRange(0.0, 600000.0)
    spinbox.setValue(value)
    return spinbox


class PulseBuilderTab(QWidget):
    """Every control that shapes a pulse train, and the saved profiles."""

    profiles_changed = Signal()
    draft_changed = Signal()

    def __init__(self, app_model, set_status: Callable[[str, bool], None], parent=None):
        super().__init__(parent)
        self._app_model = app_model
        self._set_status = set_status
        self._rest_volts = 0.0
        self._can_edit = True

        self.setObjectName("PulseBuilderTab")
        self.setStyleSheet(
            "#PulseBuilderTab QPushButton {min-height: 18px; padding: 1px 8px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(3)

        library = QHBoxLayout()
        library.setSpacing(4)
        self._profile_selector = QComboBox()
        self._profile_selector.setToolTip("Load a saved profile into the builder")
        # Sized to a few characters rather than its longest "name — summary"
        # entry, which made the builder wider than the docked panel.
        self._profile_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._profile_selector.setMinimumContentsLength(6)
        self._profile_selector.setMaximumWidth(_WIDE_FIELD_MAXIMUM_WIDTH)
        self._delete_button = QPushButton("Delete")
        self._save_button = QPushButton("Save profile…")
        library.addWidget(QLabel("Profile:"))
        library.addWidget(self._profile_selector, stretch=1)
        library.addWidget(self._delete_button)
        library.addWidget(self._save_button)
        library.addStretch(0)
        layout.addLayout(library)

        self.sections = {
            "pulse_train": CollapsibleSection("Pulse train", "pulse_train"),
            # Folded away by default: they matter only with the PMT shutter on.
            "pmt_margins": CollapsibleSection(
                "PMT shutter margins", "pmt_margins", expanded=False),
        }
        self._amplitude = QDoubleSpinBox()
        self._amplitude.setDecimals(3)
        self._amplitude.setSingleStep(0.050)
        self._amplitude.setSuffix(" V")
        self._duration_ms = _ms_spinbox(10.0)
        self._baseline_ms = _ms_spinbox(0.0)
        self._post_stim_ms = _ms_spinbox(0.0)
        self._pulse_count = QSpinBox()
        self._pulse_count.setRange(1, 100000)
        self._frequency_hz = QDoubleSpinBox()
        self._frequency_hz.setRange(0.1, 100000.0)
        self._frequency_hz.setDecimals(3)
        self._frequency_hz.setValue(10.0)
        self._frequency_hz.setSuffix(" Hz")
        self._pmt_open_lead_ms = _ms_spinbox(0.0)
        self._pmt_close_lag_ms = _ms_spinbox(0.0)
        self._controls = (
            self._amplitude, self._duration_ms, self._baseline_ms,
            self._post_stim_ms, self._pulse_count, self._frequency_hz,
            self._pmt_open_lead_ms, self._pmt_close_lag_ms,
        )
        # Two fields to a row: in one column the train and the preview did
        # not both fit the docked panel.
        for section_name, fields in (
            ("pulse_train", (
                ("Amplitude:", self._amplitude), ("Pulse width:", self._duration_ms),
                ("Baseline:", self._baseline_ms), ("Post-stim:", self._post_stim_ms),
                ("Count:", self._pulse_count), ("Frequency:", self._frequency_hz),
            )),
            ("pmt_margins", (
                ("PMT open lead:", self._pmt_open_lead_ms),
                ("PMT close lag:", self._pmt_close_lag_ms),
            )),
        ):
            section = self.sections[section_name]
            grid = QGridLayout(section.content)
            grid.setContentsMargins(4, 0, 2, 2)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(2)
            for index, (text, widget) in enumerate(fields):
                row, column = divmod(index, 2)
                label = QLabel(text)
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                widget.setMaximumWidth(_FIELD_MAXIMUM_WIDTH)
                grid.addWidget(label, row, 2 * column)
                grid.addWidget(widget, row, 2 * column + 1)
            grid.setColumnStretch(1, _FIELD_COLUMN_STRETCH)
            grid.setColumnStretch(3, _FIELD_COLUMN_STRETCH)
            grid.setColumnStretch(4, 1)
            layout.addWidget(section)

        self._preview_plot = PGWidget()
        self._preview_plot.setBackground("w")
        self._preview_plot.getAxis("bottom").setLabel("Time", units="s")
        self._preview_plot.getAxis("left").setLabel("Command", units="V")
        compact_plot_axes(self._preview_plot)
        self._preview_plot.setMouseEnabled(x=False, y=False)
        self._preview_curve = self._preview_plot.plot(
            [], [], pen=pg.mkPen(color=(30, 90, 180), width=2))
        self._preview_status = QLabel("")
        self._preview_status.setWordWrap(True)
        layout.addWidget(self._preview_plot, stretch=1)
        layout.addWidget(self._preview_status)

        for control in (self._duration_ms, self._baseline_ms, self._post_stim_ms,
                        self._frequency_hz, self._amplitude,
                        self._pmt_open_lead_ms, self._pmt_close_lag_ms):
            control.valueChanged.connect(self._on_draft_changed)
        self._pulse_count.valueChanged.connect(self._on_draft_changed)
        self._profile_selector.currentIndexChanged.connect(self._on_profile_selected)
        self._save_button.clicked.connect(self._save)
        self._delete_button.clicked.connect(self._delete)
        self.refresh_profiles()
        self._refresh_preview()

    def set_amplitude_range(self, low: float, high: float) -> None:
        """The widest range any configured laser accepts; each laser checks its own."""
        self._rest_volts = float(low)
        self._amplitude.setRange(float(low), float(high))
        self._refresh_preview()

    def set_controls_enabled(self, can_edit: bool) -> None:
        self._can_edit = bool(can_edit)
        for control in (*self._controls, self._profile_selector, self._save_button):
            control.setEnabled(can_edit)
        self._refresh_delete_enabled()

    def draft_profile(self) -> Optional[LaserPulseProfile]:
        """The train on the controls, or None when it cannot be generated."""
        try:
            profile = self._profile_from_controls(DRAFT_PROFILE_ID)
        except ValueError:
            return None
        return None if pulse_shape_refusal(profile) else profile

    def refresh_profiles(self) -> None:
        previous = self._profile_selector.currentData()
        self._profile_selector.blockSignals(True)
        self._profile_selector.clear()
        self._profile_selector.addItem("(new profile)", None)
        state = getattr(self._app_model, "trial_protocol_state", {}) or {}
        for item in state.get("laser_profiles", ()):
            self._profile_selector.addItem(
                "{} — {}".format(item["profile_id"], item.get("summary", "")),
                item["profile_id"])
        index = self._profile_selector.findData(previous)
        self._profile_selector.setCurrentIndex(max(0, index))
        self._profile_selector.blockSignals(False)
        self._refresh_delete_enabled()

    def _refresh_delete_enabled(self) -> None:
        # Both conditions, wherever either changes: refreshing the list after
        # a save used to re-enable Delete while the tab was locked.
        self._delete_button.setEnabled(
            self._can_edit and bool(self._profile_selector.currentData()))

    def _profile_from_controls(self, profile_id: str, revision: int = 1) -> LaserPulseProfile:
        count = self._pulse_count.value()
        return LaserPulseProfile(
            profile_id=profile_id,
            revision=revision,
            amplitude_volts=self._amplitude.value(),
            pulse_duration_ms=self._duration_ms.value(),
            pulse_count=count,
            frequency_hz=self._frequency_hz.value() if count > 1 else None,
            baseline_ms=self._baseline_ms.value(),
            post_stim_ms=self._post_stim_ms.value(),
            pmt_open_lead_ms=self._pmt_open_lead_ms.value(),
            pmt_close_lag_ms=self._pmt_close_lag_ms.value(),
        )

    def _on_profile_selected(self, *_args) -> None:
        profile_id = self._profile_selector.currentData()
        self._refresh_delete_enabled()
        if not profile_id:
            return
        profile = self._app_model.laser_profile(profile_id)
        if profile is None:
            self._set_status(f"Laser profile {profile_id!r} is no longer saved", True)
            return
        for control in self._controls:
            control.blockSignals(True)
        try:
            self._amplitude.setValue(float(profile.amplitude_volts))
            self._duration_ms.setValue(float(profile.pulse_duration_ms))
            self._baseline_ms.setValue(float(profile.baseline_ms))
            self._post_stim_ms.setValue(float(profile.post_stim_ms))
            self._pulse_count.setValue(int(profile.pulse_count))
            if profile.frequency_hz:
                self._frequency_hz.setValue(float(profile.frequency_hz))
            self._pmt_open_lead_ms.setValue(float(profile.pmt_open_lead_ms))
            self._pmt_close_lag_ms.setValue(float(profile.pmt_close_lag_ms))
        finally:
            for control in self._controls:
                control.blockSignals(False)
        # The spinbox clamps to the builder's range without a word, and a
        # save would then store the clamped value under this profile's name.
        # Rounding to the displayed decimals is not worth reporting.
        shown = self._amplitude.value()
        if abs(shown - profile.amplitude_volts) > 10 ** -self._amplitude.decimals():
            self._set_status(
                f"Profile {profile_id!r} is {profile.amplitude_volts:g} V; the "
                f"builder allows {self._amplitude.minimum():g}.."
                f"{self._amplitude.maximum():g} V, so it now shows {shown:g} V",
                False,
            )
        self._on_draft_changed()

    def _on_draft_changed(self, *_args) -> None:
        self._refresh_preview()
        self.draft_changed.emit()

    def _refresh_preview(self) -> None:
        draft = self.draft_profile()
        if draft is None:
            self._preview_curve.setData([], [])
            try:
                refusal = pulse_shape_refusal(self._profile_from_controls(DRAFT_PROFILE_ID))
            except ValueError as error:
                refusal = str(error)
            self._preview_status.setText(refusal)
            return
        x_values, y_values = preview_points(draft, self._rest_volts)
        self._preview_curve.setData(x_values, y_values)
        self._preview_plot.setXRange(0.0, max(x_values[-1], 0.001), padding=0.02)
        self._preview_status.setText(draft.summary())

    def _ask_profile_name(self, suggested: str) -> Optional[str]:
        name, accepted = QInputDialog.getText(
            self, "Save laser profile", "Profile name:", text=suggested)
        return name.strip() if accepted and name.strip() else None

    def _confirm_delete(self, profile_id: str) -> bool:
        answer = QMessageBox.question(
            self, "Delete laser profile",
            f"Delete laser profile {profile_id!r}? Protocols that use it refuse the delete.")
        return answer == QMessageBox.StandardButton.Yes

    def _save(self) -> None:
        draft = self.draft_profile()
        if draft is None:
            self._set_status(self._preview_status.text() or "The pulse train is not valid", True)
            return
        name = self._ask_profile_name(self._profile_selector.currentData() or "pulse")
        if name is None:
            return
        # A laser tab's picker fires the draft for this identifier, so picking
        # a profile saved under it would fire the draft instead. Protocol rows
        # lowercase profile identifiers, hence the case-insensitive check.
        if name.lower() == DRAFT_PROFILE_ID:
            self._set_status(
                f"{DRAFT_PROFILE_ID!r} is reserved for the unsaved builder draft; "
                "choose another name",
                True,
            )
            return
        values = draft.to_record()
        values.pop("revision")
        values["profile_id"] = name
        try:
            saved = self._app_model.save_laser_profile(**values)
        except Exception as error:
            self._set_status(str(error) or type(error).__name__, True)
            return
        self.refresh_profiles()
        self._profile_selector.blockSignals(True)
        self._profile_selector.setCurrentIndex(
            max(0, self._profile_selector.findData(saved.profile_id)))
        self._profile_selector.blockSignals(False)
        self._refresh_delete_enabled()
        self._set_status(
            f"Saved laser profile {saved.profile_id!r} (revision {saved.revision})", False)
        self.profiles_changed.emit()

    def _delete(self) -> None:
        profile_id = self._profile_selector.currentData()
        if not profile_id or not self._confirm_delete(profile_id):
            return
        try:
            self._app_model.delete_stimulus_profile("laser", profile_id)
        except Exception as error:
            self._set_status(str(error) or type(error).__name__, True)
            return
        self.refresh_profiles()
        self._set_status(f"Deleted laser profile {profile_id!r}", False)
        self.profiles_changed.emit()
