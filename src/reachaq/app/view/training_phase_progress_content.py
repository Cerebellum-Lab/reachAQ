
# NB: training plan and training phase progress are totally similar but phase progress has 1 extra item..
# todo: consider factorizing their common parts


from typing import Optional, List, Dict, Any, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget, QStackedWidget, QSizePolicy, QScrollArea, \
    QHBoxLayout, QFormLayout

from reachaq.core.logging import get_verbose_logger

from reachaq.pyside import CardWidget
from reachaq.behavior.training import TrainingPlan, TrainingPhase

from reachaq.pyside.content_widget import invoke_method

logger = get_verbose_logger(__name__)


def make_unit_label(txt, *, background_color="lightblue"):
    label = QLabel(txt)
    label.setStyleSheet(f"background-color: {background_color}")
    return label


class TrainingPhaseProgressContent(CardWidget):

    phase_desc_truncate_length = 92

    def __init__(self):
        super().__init__(title="Phase Progress")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._plan_card_by_plan_id: Dict[str, Tuple[CardWidget, QScrollArea]] = {}
        self._plans_by_plan_id: Dict[str, TrainingPlan] = {}
        self._phases_by_plan_id: Dict[str, List[QWidget]] = {}

        main_content = QWidget()
        # main_content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setContentWidget(main_content)

        main_layout = QHBoxLayout(main_content)
        main_layout.setSpacing(6)

        r = c = 0
        left = QGridLayout()
        left.addWidget(QLabel("Started"), r, c)
        label = self._started_label = QLabel("")
        left.addWidget(label, r, c + 1)
        r += 1
        left.addWidget(QLabel("Time in Training"), r, c)
        label = self._time_in_training_label = QLabel("")
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(make_unit_label("sec"))
        left.addLayout(hbox, r, c + 1)
        r += 1
        left.addWidget(QLabel("Trials"), r, c)
        label = self._sessions_label = QLabel("")
        left.addWidget(label, r, c + 1)
        r += 1
        left.addWidget(QLabel("Attempts"), r, c)
        label = self._attempts_label = QLabel("")
        left.addWidget(label, r, c + 1)

        r = c = 0
        right = QGridLayout()
        right.addWidget(QLabel("Pellets Presented"), r, c)
        label = self._pellets_presented_label = QLabel("")
        right.addWidget(label, r, c + 1)
        r += 1
        right.addWidget(QLabel("Pellets Consumed"), r, c)
        label = self._pellets_consumed_label = QLabel("")
        right.addWidget(label, r, c + 1)
        r += 1
        right.addWidget(QLabel("Successful Reaches"), r, c)
        label = self._successful_reaches_label = QLabel("")
        right.addWidget(label, r, c + 1)
        r += 1
        right.addWidget(QLabel("Total Reaches"), r, c)
        label = self._total_reaches_label = QLabel("")
        right.addWidget(label, r, c + 1)

        border_with_padding_style = """QLabel {
            border: 1px solid gray;
            padding: 4px;
        }"""

        main_content.setStyleSheet(border_with_padding_style)

        for side in (left, right):
            assert isinstance(side, QGridLayout)
            side.setContentsMargins(0, 0, 0, 0)
            side.setVerticalSpacing(0)
            side.setHorizontalSpacing(0)
            side.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            for row in range(side.rowCount()):
                item = side.itemAtPosition(row, 0)
                label = item.widget()
                label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                label.setContentsMargins(0, 0, 0, 0)
                label.setStyleSheet(f"{label.styleSheet()}; background-color: lightgray;")
                font = label.font()
                font.setBold(True)
                label.setFont(font)  # do not forget !
            side_widget = QWidget()
            side_widget.setLayout(side)
            main_layout.addWidget(side_widget)

    @invoke_method
    def set_training_phase_progress(self, phase: Optional[TrainingPhase]):
        logger.verbose("Setting training phase to %s", phase)
        if phase is None:
            for label in (
                self._started_label,
                self._time_in_training_label,
                self._sessions_label,
                self._attempts_label,
                self._pellets_presented_label,
                self._pellets_consumed_label,
                self._successful_reaches_label,
                self._total_reaches_label,
            ):
                label.setText("")
            return
        prog = phase.progress
        value = prog.timestamp
        self._started_label.setText(
            "NA" if value is None
            else f"{value.strftime('%Y/%m/%d %I:%M %p')}"
        )
        self._time_in_training_label.setText(f"{prog.time_in_training:.1f}")
        for label, value in (
            (self._sessions_label, prog.session_count),
            (self._attempts_label,  prog.phase_attempts),
            (self._pellets_presented_label, prog.pellets_presented),
            (self._pellets_consumed_label, prog.pellets_consumed),
            (self._successful_reaches_label, prog.successful_reaches),
            (self._total_reaches_label, prog.total_reaches)
        ):
            label.setText(f"{value}")

