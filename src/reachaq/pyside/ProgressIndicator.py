from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import Qt, QColor, QPen, QPainterPath, QPainter
from PySide6.QtWidgets import QWidget


class ProgressIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._offset = 0
        self.setMinimumSize(50, 50)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_image)
        self._timer.start(int(1000/20))

    def _update_image(self):
        self._offset += 0.05
        if self._offset > 1:
            self._offset -= 1
        self.update()

    def paintEvent(self, e):
        if self.height() > self.width():
            self.setFixedWidth(self.height())
        if self.width() > self.height():
            self.setFixedHeight(self.width())
        pd = self._offset * 360
        p = QPainter(self)
        p.translate(4, 4)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        circle_width = self.width() - self.width() / 10
        width_half = circle_width / 2
        path.moveTo(width_half, 0)
        circle_rect = QRectF(self.rect().left() / 2, self.rect().top() / 2, circle_width,
                             self.height() - self.height() / 10)
        path.arcMoveTo(circle_rect, -pd)
        path.arcTo(circle_rect, -pd, 45)
        pen = QPen()
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        pen.setColor(QColor("#30b7e0"))
        pen_width = self.width() / 25
        pen.setWidth(pen_width)
        p.strokePath(path, pen)
