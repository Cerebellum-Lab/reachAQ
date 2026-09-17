"""Move one panel between its splitter slot, an overlay, and its own window."""

from __future__ import annotations

import enum
from typing import Optional

from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget


#: Fraction of the reference width an expanded panel covers. The panel is
#: right-aligned inside the reference, so it grows leftward over the panels it
#: covers instead of pushing them aside.
EXPANDED_WIDTH_FRACTION = 0.85


class PanelState(enum.Enum):
    DOCKED = "docked"
    EXPANDED = "expanded"
    DETACHED = "detached"


class DetachablePanelHost(QObject):
    """Own one panel widget's placement across three states.

    Both undocked states are real top-level windows rather than raised child
    widgets. The left side of the main window contains QtGLImageView, a
    QOpenGLWidget, and stacking a sibling above a QOpenGLWidget depends on
    platform compositing. A top-level window is composited by the window
    manager, so the overlay does not depend on widget stacking order at all,
    and detaching reuses the same code path.
    """

    state_changed = Signal(str)

    def __init__(
        self,
        panel: QWidget,
        splitter: QSplitter,
        index: int,
        *,
        title: str = "Panel",
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._panel = panel
        self._splitter = splitter
        self._index = int(index)
        self._title = title
        self._state = PanelState.DOCKED
        self._placeholder: Optional[QWidget] = None
        self._window: Optional[QWidget] = None
        self._reference = QRect()

    @property
    def state(self) -> PanelState:
        return self._state

    @property
    def window(self) -> Optional[QWidget]:
        return self._window

    @staticmethod
    def expanded_geometry(reference: QRect) -> QRect:
        """Right-aligned rectangle covering most of the reference."""
        width = max(1, int(reference.width() * EXPANDED_WIDTH_FRACTION))
        return QRect(
            reference.x() + reference.width() - width,
            reference.y(),
            width,
            reference.height(),
        )

    def expand(self) -> None:
        self._move_to_window(
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint,
            PanelState.EXPANDED,
        )
        self.track(self._reference)

    def detach(self, geometry: Optional[QRect] = None) -> None:
        self._move_to_window(Qt.WindowType.Window, PanelState.DETACHED)
        if geometry is not None and self._is_on_a_screen(geometry):
            self._window.setGeometry(QRect(geometry))

    def collapse(self) -> None:
        """Return the panel to its splitter slot from either undocked state."""
        if self._state is PanelState.DOCKED:
            return
        sizes = self._splitter.sizes()
        self._splitter.replaceWidget(self._index, self._panel)
        if self._placeholder is not None:
            self._placeholder.setParent(None)
            self._placeholder.deleteLater()
            self._placeholder = None
        self._splitter.setSizes(sizes)
        self._discard_window()
        self._set_state(PanelState.DOCKED)

    def reattach(self) -> None:
        """Alias for collapse, for callers that think in terms of detaching."""
        self.collapse()

    def track(self, reference: QRect) -> None:
        """Keep an expanded panel pinned to the right of the reference."""
        if reference is not None and reference.isValid():
            self._reference = QRect(reference)
        if self._state is not PanelState.EXPANDED or self._window is None:
            return
        if not self._reference.isValid():
            return
        self._window.setGeometry(self.expanded_geometry(self._reference))

    def _move_to_window(self, flags, state: PanelState) -> None:
        if self._state is PanelState.DOCKED:
            sizes = self._splitter.sizes()
            placeholder = QWidget()
            placeholder.setSizePolicy(self._panel.sizePolicy())
            self._splitter.replaceWidget(self._index, placeholder)
            self._placeholder = placeholder
            self._splitter.setSizes(sizes)
        window = QWidget(None, flags)
        window.setWindowTitle(self._title)
        layout = QVBoxLayout(window)
        layout.setContentsMargins(0, 0, 0, 0)
        # Reparent the panel into the new window before discarding the old one,
        # so the panel is never briefly owned by a window being destroyed.
        layout.addWidget(self._panel)
        previous = self._window
        self._window = window
        if previous is not None:
            previous.close()
            previous.deleteLater()
        window.show()
        self._set_state(state)

    def _discard_window(self) -> None:
        if self._window is None:
            return
        self._window.close()
        self._window.deleteLater()
        self._window = None

    def _set_state(self, state: PanelState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state.value)

    @staticmethod
    def _is_on_a_screen(geometry: QRect) -> bool:
        if not geometry.isValid() or geometry.width() < 1 or geometry.height() < 1:
            return False
        for screen in QGuiApplication.screens():
            if screen.availableGeometry().intersects(geometry):
                return True
        return False
