from typing import Optional, List, Dict, Any, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget, QStackedWidget, QSizePolicy, QScrollArea

from autotrainer.core.logging import get_verbose_logger

from autotrainer.pyside import CardWidget
from autotrainer.pyside.StackedContent import StackedWidget

from autotrainer.behavior.training import TrainingPlan

from autotrainer.pyside.content_widget import invoke_method

logger = get_verbose_logger(__name__)


class TrainingPlanContent(StackedWidget):

    phase_desc_truncate_length = 92

    def __init__(self):
        super().__init__()
        # self.setMaximumHeight(250)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._plan_card_by_plan_id: Dict[str, Tuple[CardWidget, QScrollArea]] = {}
        self._plans_by_plan_id: Dict[str, TrainingPlan] = {}
        self._phases_by_plan_id: Dict[str, List[QWidget]] = {}
        card = self._empty_card = self._card = CardWidget(title="Protocol")
        self.addWidget(card)

    @staticmethod
    def _get_phase_widget_name(plan, phase):
        return f"plan_{plan.plan_id}_{phase.phase_id}_widget".replace(" ", "").replace("-", "_")

    def _get_phase_label_name(self, plan, phase):
        return f"{self._get_phase_widget_name(plan, phase)}_label"

    def _show_training_plan(self, card: CardWidget, scroll_area: QScrollArea, plan: TrainingPlan):
        self.setCurrentWidget(card)
        phase = plan.current_phase
        cur_phase_id = None if phase is None else plan.current_phase.phase_id
        logger.debug("showing %s with cur_phase_id = %s ; %s",
                     phase, cur_phase_id, "None" if phase is None else phase.name)
        for phase in plan.phases:
            phase_widget_name = self._get_phase_widget_name(plan, phase)
            widget = card.findChild(QWidget, phase_widget_name)
            if widget is None:
                logger.warning("%s not found", phase_widget_name)
            else:
                assert isinstance(widget, QWidget)
                label_phase_widget_name = self._get_phase_label_name(plan, phase)
                if cur_phase_id is not None and phase.phase_id == cur_phase_id:
                    scroll_area.ensureWidgetVisible(widget)
                    widget.setStyleSheet(f"""\
                    QWidget {{ background-color: lightgray; }}
                    #{label_phase_widget_name} {{ color: green; font-weight: bold; }}
                    #{phase_widget_name} {{
                    border: 1px solid gray; /* Sets the contour/border */
                    border-radius: 10px; /* Optional: rounds the corners */
                    /* padding: 2px; */ /* Optional: adds space inside the border */
                    }}
                    """)
                else:
                    widget.setStyleSheet("")
        self.update()

    @invoke_method
    def set_training_plan(self, plan: Optional[TrainingPlan]):
        logger.verbose("Setting training plan to %s", plan)
        if plan is None:
            self.setCurrentWidget(self._empty_card)
            return

        card_scroll_area = self._plan_card_by_plan_id.get(plan.plan_id)
        if card_scroll_area is not None:
            card, scroll_area = card_scroll_area
            logger.debug("Using cache for plan to %s", plan)
            self._show_training_plan(card, scroll_area, plan)
            return

        logger.debug("Adding new plan %s with %s phases", plan.plan_id, len(plan.phases))

        card = CardWidget(title="Protocol")
        card.header.setRightContent(QLabel(plan.name))

        content_widget = QWidget()
        content_widget.setContentsMargins(0, 0, 0, 0)

        vbox_layout = QVBoxLayout(content_widget)
        vbox_layout.setContentsMargins(0, 0, 0, 0)
        vbox_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        label = QLabel(plan.description)
        label.setContentsMargins(4, 4, 0, 0)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setStyleSheet("color: gray")
        vbox_layout.addWidget(label)

        scroll_area = QScrollArea()
        vbox_layout.addWidget(scroll_area)
        # scroll_area.setContentsMargins(0, 0, 0, 0)
        scroll_area.setStyleSheet("")
        scroll_area.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # scroll_area.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setWidgetResizable(True)

        content_widget.setStyleSheet("background-color: white")
        card.setContentWidget(content_widget)

        grid_widget = QWidget()
        # grid_widget.setContentsMargins(0, 0, 0, 0)
        grid_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        grid_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        scroll_area.setWidget(grid_widget)

        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        cur_col = cur_row = 0
        right_max_cols = (
            3 if len(plan.phases) > 6
            else 2 if len(plan.phases) > 3
            else 1
        )
        phase_widgets = self._phases_by_plan_id[plan.plan_id] = []
        for phase_nr, phase in enumerate(plan.phases, start=1):
            phase_widget_name = self._get_phase_widget_name(plan, phase)
            widget = QWidget()
            # widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
            # widget.setContentsMargins(0, 0, 0, 0)
            widget.setObjectName(phase_widget_name)
            phase_widgets.append(widget)
            layout = QVBoxLayout(widget)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(widget, cur_row, cur_col)
            label_phase_widget_name = self._get_phase_label_name(plan, phase)
            label = QLabel(f"{phase_nr}. {phase.name}")
            label.setObjectName(label_phase_widget_name)
            layout.addWidget(label)
            desc = phase.description
            if len(desc) > self.phase_desc_truncate_length:
                desc = phase.description[:self.phase_desc_truncate_length - 3]
                desc += "..."
            label = QLabel(desc)
            font = label.font()
            font.setPointSize(8)
            label.setFont(font)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(label)
            cur_col += 1
            if cur_col >= right_max_cols:
                cur_row += 1
                cur_col = 0

        self._plan_card_by_plan_id[plan.plan_id] = (card, scroll_area)
        self.addWidget(card)
        self._show_training_plan(card, scroll_area, plan)
