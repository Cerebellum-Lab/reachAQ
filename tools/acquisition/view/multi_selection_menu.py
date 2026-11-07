"""Menus whose independent checkable choices can be changed in one visit."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu


class MultiSelectionMenu(QMenu):
    """Keep the menu open when an enabled checkable action is selected."""

    @staticmethod
    def _is_persistent_action(action) -> bool:
        return bool(action is not None and action.isEnabled() and action.isCheckable())

    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.position().toPoint())
        if self._is_persistent_action(action):
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in {Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Space}:
            action = self.activeAction()
            if self._is_persistent_action(action):
                action.trigger()
                event.accept()
                return
        super().keyPressEvent(event)
