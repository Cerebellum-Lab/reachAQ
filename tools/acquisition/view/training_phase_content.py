from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget, QStackedWidget, QSizePolicy, QScrollArea, \
    QHBoxLayout

from autotrainer.core.logging import get_verbose_logger

from autotrainer.pyside import CardWidget
from autotrainer.pyside.StackedContent import StackedWidget

from autotrainer.training import TrainingPlan, TrainingPhase

from autotrainer.pyside.content_widget import invoke_method

logger = get_verbose_logger(__name__)


some_light_gray = "#C7C5C5"


class TrainingPhaseCard(CardWidget):

    # TODO: the constructor takes the current phase: convert to constructor without phase,
    #  and add a set_training_phase method to allow update in-place,
    #  instead of having to fully reconstruct always the card.

    def __init__(self, phase: TrainingPhase):
        super().__init__(title="Current Phase")

        label = self._header_right_label = QLabel(phase.name)
        self.header.setRightContent(label)

        widget = QWidget()
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        widget.setContentsMargins(0, 0, 0, 0)
        self.setContentWidget(widget)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 4, 2, 0)
        layout.setSpacing(4)

        label = self._description_label = QLabel(phase.description)
        label.setContentsMargins(0, 0, 0, 0)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setStyleSheet("color: gray")
        label.setWordWrap(True)
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignTop) # | Qt.AlignmentFlag.AlignLeft)

        border_with_padding_style = """QLabel {
            border: 1px solid gray;
            margin: 0px;
            padding: 2px;
        }"""

        sub = QHBoxLayout()
        sub.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(sub)

        left = QVBoxLayout()
        left.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)
        sub.addLayout(left)

        dev = self._make_device(phase)
        dev.setStyleSheet(border_with_padding_style)
        left.addWidget(dev)

        predicate = self._make_predicates(phase)
        predicate.setStyleSheet(border_with_padding_style)
        left.addWidget(predicate)

        left.addStretch(1)  # allows to consume any remaining space

        behavior = self._make_behavior(phase)
        behavior.setStyleSheet(border_with_padding_style)
        sub.addWidget(behavior, stretch=0)
        sub.addStretch(1)

    def _apply_background(self, grid):
        for row in range(grid.rowCount()):
            item = grid.itemAtPosition(row, 0)
            if item and item.widget():
                w = item.widget()
                w.setStyleSheet(f"{w.styleSheet()}; background-color: lightgray;")

    def _make_unit_label(self, txt):
        label = QLabel(txt)
        label.setStyleSheet("background-color: lightblue")
        return label

    def _make_sub_panel_head(self, txt: str):
        label = QLabel(f"<b>{txt}</b>")
        font = label.font()
        font.setPointSize(10)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # NB: this expanding of the panel head allows the panel to expand as necessarily to fill the layout containing
        # it (and its parent layout).
        # i.e: the panels on same column will have all the same width being the maximum one of them.
        return label

    def _make_device(self, phase: TrainingPhase):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        label = self._make_sub_panel_head("Device")
        layout.addWidget(label)

        grid = QGridLayout()
        layout.addLayout(grid)

        r = c = 0
        grid.addWidget(QLabel("Pellet Delivery"), r, c)
        label = self._device_pellet_delivery_enabled_label = QLabel("On" if phase.is_pellet_delivery_enabled else "Off")
        grid.addWidget(label, r, c + 1)
        r += 1

        if phase.is_pellet_delivery_enabled:
            grid.addWidget(QLabel("Pellet Cover"), r, c)
            label = self._device_pellet_cover_enabled_label = QLabel("On" if phase.is_pellet_cover_enabled else "Off")
            grid.addWidget(label, r, c + 1)
            r += 1

        for coord, value in zip("XYZ", (phase.send_x, phase.send_y, phase.send_z)):
            if value is not None:
                grid.addWidget(QLabel(f"Send-{coord}"), r, c)
                hbox = QHBoxLayout()
                hbox.addWidget(QLabel(f"{value}"))
                hbox.addWidget(self._make_unit_label("mm"))
                hbox.setStretch(0, 1)
                grid.addLayout(hbox, r, c + 1, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
                r += 1

        self._apply_background(grid)

        return widget

    def _make_predicates(self, phase: TrainingPhase):
        widget = QWidget()
        widget.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.setSpacing(0)
        label = self._make_sub_panel_head("Predicates & Actions")
        layout.addWidget(label, stretch=1)

        grid = QGridLayout()
        layout.addLayout(grid)

        r = c = 0
        grid.addWidget(QLabel("Fallback Conditions"), r, c)
        grid.addWidget(QLabel("Yes" if phase.fallback_predicate is not None else "No"), r, c + 1)
        r += 1
        grid.addWidget(QLabel("Advance Conditions"), r, c)
        grid.addWidget(QLabel("Yes" if phase.advance_predicate is not None else "No"), r, c + 1)
        r += 1
        grid.addWidget(QLabel("Trial Actions"), r, c)
        grid.addWidget(QLabel(str(len(phase.session_actions))), r, c + 1)

        self._apply_background(grid)

        return widget

    def _make_behavior(self, phase: TrainingPhase):
        widget = QWidget()
        widget.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.setSpacing(0)

        label = self._make_sub_panel_head("Behavior")
        layout.addWidget(label)

        grid = QGridLayout()
        layout.addLayout(grid)
        r = c = 0
        #
        grid.addWidget(QLabel("Pellet Shift"), r, c)
        grid.addWidget(QLabel("On" if phase.is_pellet_shift_enabled else "Off"), r, c + 1)
        r += 1

        self._apply_background(grid)

        return widget


class TrainingPhaseContent(StackedWidget):

    def __init__(self):
        super().__init__()
        self._phase_card_by_phase_id: Dict[str, QWidget] = {}
        self._phase_by_phase_id: Dict[str, TrainingPlan] = {}
        self._training_phase: Optional[TrainingPhase] = None
        card = self._empty_card = CardWidget(title="Current Phase")
        self.addWidget(card)

    @invoke_method
    def set_training_phase(self, phase: Optional[TrainingPhase], *, force_refresh: bool=False):
        self._training_phase = phase
        if phase is None:
            self.setCurrentWidget(self._empty_card)
            return

        card = self._phase_card_by_phase_id.get(phase.phase_id)
        if card is not None and not force_refresh:
            logger.debug("Using cache for phase to %s", phase)
            self.setCurrentWidget(card)
            return

        if card is not None:
            self.removeWidget(card)
            card.setParent(None)

        logger.debug("Adding new phase %s with %s actions", phase.phase_id, len(phase.session_actions))
        card = TrainingPhaseCard(phase)
        self.addWidget(card)
        self._phase_card_by_phase_id[phase.phase_id] = card
        self.setCurrentWidget(card)
