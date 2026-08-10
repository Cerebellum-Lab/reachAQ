import logging
from typing import Optional

from PySide6.QtWidgets import QLabel

import qtawesome as qta


logger = logging.getLogger(__name__)


class StatusIcon(QLabel):
    def __init__(
        self,
        on_icon: Optional[str] = None,
        off_icon: Optional[str] = None,
        on_disabled_icon: Optional[str] = None,
        off_disabled_icon: Optional[str] = None,
        on_color: str ="green",
        off_color: str = "black",
        on_disabled_color: str = "orange",
        off_disabled_color: str = "gray",
        size: int = 18, parent=None,
        name: str = "NA",
    ):
        super(StatusIcon, self).__init__(parent)

        self._in_use = True
        self._cur_status = False
        self._name = name

        self._on = qta.icon(on_icon or 'fa5s.check-circle', color=on_color).pixmap(size, size)
        self._on_disabled = qta.icon(on_disabled_icon or 'fa5.check-circle', color=on_disabled_color).pixmap(size, size)
        self._off = qta.icon(off_icon or 'fa5s.times-circle', color=off_color).pixmap(size, size)
        self._off_disabled = qta.icon(off_disabled_icon or 'fa5s.times-circle', color=off_disabled_color).pixmap(size, size)

        self.setStatus(False)

    def setInUse(self, in_use: bool):
        self._in_use = in_use
        self.setStatus(self._cur_status)

    def setStatus(self, status: bool):
        self._cur_status = status
        in_use = self._in_use
        # logger.debug("setStatus[%s]: status=%s in_use=%s", self._name, status, in_use)
        self.setPixmap(
            (self._on if in_use else self._on_disabled) if status
            else (self._off if in_use else self._off_disabled)
        )
