from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QWidget


class PersistentSplitter(QSplitter):
    """A visible, non-collapsing splitter whose position survives restarts."""

    def __init__(
        self,
        orientation: Qt.Orientation,
        preferences,
        settings_key: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(orientation, parent)
        self._preferences = preferences
        self._settings_key = settings_key
        self.setChildrenCollapsible(False)
        self.setHandleWidth(6)
        # Avoid repeatedly repainting high-rate plots while the handle moves.
        self.setOpaqueResize(False)
        self.setStyleSheet(
            "QSplitter::handle {background: #c9cdd3;}"
            "QSplitter::handle:hover {background: #7b8794;}"
            "QSplitter::handle:pressed {background: #52606d;}"
        )
        self.splitterMoved.connect(self._save_state)

    @property
    def settings_key(self) -> str:
        return self._settings_key

    def apply_saved_or_default_sizes(self, default_sizes: Iterable[int]) -> bool:
        state = self._preferences.splitter_state(self._settings_key)
        restored = bool(state) and self.restoreState(state)
        if not restored:
            self.setSizes([max(0, int(size)) for size in default_sizes])
        return restored

    def save_state(self) -> None:
        self._preferences.set_splitter_state(self._settings_key, self.saveState())

    def _save_state(self, _position: int, _index: int) -> None:
        self.save_state()
