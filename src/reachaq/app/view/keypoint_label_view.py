"""Placing keypoints on one frame.

Scene coordinates are frame pixels with pixel centres on integers, the
convention the dataset and keypoint.train use: the pixmap is drawn at
(-0.5, -0.5), so clicking the middle of pixel (10, 20) stores (10, 20).

Left click places the selected part and selects the next one, so a frame is
labelled by clicking the parts in order. Drag moves a point. Right click on a
point, or Delete, clears it (the part is not visible). Wheel zooms around the
cursor. Pre-labelled points the operator has not confirmed are drawn hollow.
"""

import typing

import numpy
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

POINT_RADIUS = 2.5
# A right click or a press this close to a point (in frame pixels) hits it.
HIT_RADIUS = 4.0


def part_colours(count: int) -> typing.List[QColor]:
    """Distinct hues, evenly spaced, bright enough on a grey frame."""
    return [QColor.fromHsvF((i / max(1, count)) % 1.0, 0.85, 1.0) for i in range(count)]


class KeypointLabelView(QGraphicsView):
    points_changed = Signal(object)        # numpy (K, 2) float32, NaN = not visible
    selected_part_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pixmap = QGraphicsPixmapItem()
        self._pixmap.setOffset(-0.5, -0.5)
        self._pixmap.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.scene().addItem(self._pixmap)
        self._points = numpy.zeros((0, 2), "float32")
        self._parts: typing.List[str] = []
        self._colours: typing.List[QColor] = []
        self._items: typing.List[typing.Optional[QGraphicsEllipseItem]] = []
        self._labels: typing.List[typing.Optional[QGraphicsSimpleTextItem]] = []
        self._pending = False
        self._selected = 0
        self._dragging: typing.Optional[int] = None
        # A placing click selects the next part on release; a drag does not.
        self._advance = False
        self._size = 0

    # -- state ------------------------------------------------------------

    @property
    def selected_part(self) -> int:
        return self._selected

    @selected_part.setter
    def selected_part(self, index: int) -> None:
        if not self._parts:
            return
        index = int(index) % len(self._parts)
        if index != self._selected:
            self._selected = index
            self.selected_part_changed.emit(index)
        self._redraw()

    def points(self) -> numpy.ndarray:
        return self._points.copy()

    def set_frame(self, image: numpy.ndarray, points: numpy.ndarray,
                  parts: typing.Sequence[str], pending: bool) -> None:
        image = numpy.ascontiguousarray(image)
        rows, cols = image.shape[:2]
        qimage = QImage(image.data, cols, rows, cols, QImage.Format.Format_Grayscale8).copy()
        self._pixmap.setPixmap(QPixmap.fromImage(qimage))
        if rows != self._size:
            self._size = rows
            self.scene().setSceneRect(QRectF(-0.5, -0.5, cols, rows))
            self.fit()
        if list(parts) != self._parts:
            self._parts = list(parts)
            self._colours = part_colours(len(parts))
            self._selected = 0
        self._points = numpy.asarray(points, "float32").copy()
        self._pending = pending
        self._redraw()

    def set_pending(self, pending: bool) -> None:
        self._pending = pending
        self._redraw()

    def place(self, part: int, x: float, y: float) -> None:
        self._points[part] = (x, y)
        self._redraw()
        self.points_changed.emit(self.points())

    def clear(self, part: int) -> None:
        if not numpy.isnan(self._points[part]).all():
            self._points[part] = numpy.nan
            self._redraw()
            self.points_changed.emit(self.points())

    def fit(self) -> None:
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # -- drawing ----------------------------------------------------------

    def _redraw(self) -> None:
        for item in self._items + self._labels:
            if item is not None:
                self.scene().removeItem(item)
        self._items, self._labels = [], []
        for index, (x, y) in enumerate(self._points):
            if numpy.isnan(x) or numpy.isnan(y):
                self._items.append(None)
                self._labels.append(None)
                continue
            colour = self._colours[index]
            radius = POINT_RADIUS * (1.6 if index == self._selected else 1.0)
            item = QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
            item.setPos(QPointF(float(x), float(y)))
            pen = QPen(colour)
            pen.setCosmetic(True)
            pen.setWidthF(2.0)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush if self._pending else QBrush(colour))
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, False)
            self.scene().addItem(item)
            label = QGraphicsSimpleTextItem(self._parts[index])
            label.setBrush(QBrush(colour))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            label.setPos(QPointF(float(x) + radius, float(y) - radius))
            self.scene().addItem(label)
            self._items.append(item)
            self._labels.append(label)

    # -- input ------------------------------------------------------------

    def _nearest(self, scene: QPointF) -> typing.Optional[int]:
        if not len(self._points):
            return None
        distance = numpy.hypot(self._points[:, 0] - scene.x(), self._points[:, 1] - scene.y())
        distance = numpy.where(numpy.isnan(distance), numpy.inf, distance)
        index = int(numpy.argmin(distance))
        return index if distance[index] <= HIT_RADIUS else None

    def _inside(self, scene: QPointF) -> bool:
        return -0.5 <= scene.x() < self._size - 0.5 and -0.5 <= scene.y() < self._size - 0.5

    def mousePressEvent(self, event):  # noqa: N802 - Qt's name
        if not self._parts:
            return super().mousePressEvent(event)
        scene = self.mapToScene(event.position().toPoint())
        hit = self._nearest(scene)
        if event.button() == Qt.MouseButton.RightButton:
            if hit is not None:
                self.clear(hit)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if hit is not None and hit != self._selected:
                # Pressing an existing point grabs it for a drag.
                self._dragging = hit
                self._advance = False
                self.selected_part = hit
                return
            if self._inside(scene):
                self._dragging = self._selected
                self._advance = True
                self.place(self._selected, scene.x(), scene.y())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._dragging is not None:
            scene = self.mapToScene(event.position().toPoint())
            if self._inside(scene):
                self._points[self._dragging] = (scene.x(), scene.y())
                self._redraw()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._dragging is not None:
            part, self._dragging = self._dragging, None
            self.points_changed.emit(self.points())
            if self._advance and part == self._selected:
                self.selected_part = self._selected + 1
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):  # noqa: N802
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._parts:
            self.clear(self._selected)
            return
        if event.key() == Qt.Key.Key_Up and self._parts:
            self.selected_part = self._selected - 1
            return
        if event.key() == Qt.Key.Key_Down and self._parts:
            self.selected_part = self._selected + 1
            return
        if event.key() == Qt.Key.Key_0:
            self.resetTransform()
            self.fit()
            return
        super().keyPressEvent(event)
