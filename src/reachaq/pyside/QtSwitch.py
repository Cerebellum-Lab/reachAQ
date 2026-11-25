from PySide6.QtCore import Qt, QSize, QPoint, QPointF, QRectF, Slot, Property

from PySide6.QtWidgets import QCheckBox
from PySide6.QtGui import QColor, QBrush, QPaintEvent, QPen, QPainter


class QSwitch(QCheckBox):
    _transparent_pen = QPen(Qt.transparent)
    _light_grey_pen = QPen(Qt.lightGray)

    def __init__(self, parent=None, bar_color=Qt.gray, checked_color="#cfb87c", handle_color=Qt.white):
        super().__init__(parent)
        self._bar_brush = QBrush(bar_color)
        self._bar_checked_brush = QBrush(QColor(checked_color).lighter())
        self._bar_disabled_brush = QBrush(QColor(bar_color).lighter())

        self._handle_brush = QBrush(handle_color)
        self._handle_checked_brush = QBrush(QColor(checked_color))

        self.setContentsMargins(8, 0, 8, 0)
        self._handle_position = 0

        self.stateChanged.connect(self.handle_state_change)

    def sizeHint(self):
        return QSize(68, 25)

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def paintEvent(self, e: QPaintEvent):
        cont_rect = self.contentsRect()
        handle_radius = round(0.38 * cont_rect.height())

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.setPen(self._transparent_pen)
        bar_rect = QRectF(0, 0, cont_rect.width() - handle_radius, 0.80 * cont_rect.height())
        bar_rect.moveCenter(cont_rect.center())
        rounding = bar_rect.height() / 2

        # the handle will move along this line
        trail_length = cont_rect.width() - 2 * handle_radius
        x_pos = cont_rect.x() + handle_radius + trail_length * self._handle_position

        if self.isChecked():
            p.setBrush(self._bar_checked_brush if self.isEnabled() else self._bar_disabled_brush)
            p.drawRoundedRect(bar_rect, rounding, rounding)
            if not self.isEnabled():
                p.setPen(self._light_grey_pen)
            p.setBrush(self._handle_checked_brush if self.isEnabled() else self._bar_disabled_brush)
        else:
            p.setBrush(self._bar_brush if self.isEnabled() else self._bar_disabled_brush)
            p.drawRoundedRect(bar_rect, rounding, rounding)
            p.setPen(self._light_grey_pen)
            p.setBrush(self._handle_brush if self.isEnabled() else self._bar_disabled_brush)

        p.drawEllipse(QPointF(x_pos, bar_rect.center().y()), handle_radius, handle_radius)

        p.end()

    @Slot(int)
    def handle_state_change(self, value):
        self._handle_position = 1 if value else 0

    @Property(float)
    def handle_position(self):
        return self._handle_position

    @handle_position.setter
    def handle_position(self, pos):
        self._handle_position = pos
        self.update()

# Based on https://github.com/pythonguis/python-qtwidgets
#
# MIT License
#
# Copyright (c) 2019 Martin Fitzpatrick
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
