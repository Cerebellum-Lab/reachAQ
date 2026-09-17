import os
from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QSurfaceFormat, QBrush, QPixmap, QPen, QPainter, QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget, QGraphicsView, QGraphicsScene, QHBoxLayout, QGraphicsPixmapItem, \
    QGraphicsEllipseItem, QGraphicsItem

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.pose_elements import SceneElement
from autotrainer.inference import PoseLocation


logger = get_verbose_logger(__name__)

# Colours for the parts this rig has always drawn, kept so the overlay looks
# the same as before. Any other part the model emits gets a colour from
# _FALLBACK_COLOURS, picked by name so it stays the same across restarts.
_PART_COLOURS = {
    SceneElement.Pellet: Qt.GlobalColor.red,
    SceneElement.Star: Qt.GlobalColor.magenta,
    SceneElement.Diamond: Qt.GlobalColor.blue,
    SceneElement.Triangle: Qt.GlobalColor.green,
    SceneElement.Nose: Qt.GlobalColor.cyan,
    SceneElement.Mouth: Qt.GlobalColor.darkYellow,
    SceneElement.Tongue_mid: Qt.GlobalColor.darkRed,
    SceneElement.Tongue_tip: Qt.GlobalColor.darkMagenta,
    SceneElement.RH_flat: Qt.GlobalColor.yellow,
    SceneElement.RH_spread: Qt.GlobalColor.darkGreen,
    SceneElement.RH_grab: Qt.GlobalColor.darkCyan,
    SceneElement.LH_flat: Qt.GlobalColor.white,
    SceneElement.LH_spread: Qt.GlobalColor.lightGray,
    SceneElement.LH_grab: Qt.GlobalColor.gray,
}

_FALLBACK_COLOURS = (
    Qt.GlobalColor.red, Qt.GlobalColor.green, Qt.GlobalColor.blue,
    Qt.GlobalColor.cyan, Qt.GlobalColor.magenta, Qt.GlobalColor.yellow,
    Qt.GlobalColor.white, Qt.GlobalColor.darkRed, Qt.GlobalColor.darkGreen,
    Qt.GlobalColor.darkBlue,
)

# The markers are landmarks rather than anatomy and want to stay legible
# under a cluster of animal parts, so they keep the original larger dot.
_MARKER_PARTS = frozenset((
    SceneElement.Star, SceneElement.Diamond, SceneElement.Triangle,
    SceneElement.Pellet,
))


class QGLImageView(QWidget):
    def __init__(self, width: int = 450, height: int = 300):
        """Image view for a camera, (width, height) is the dimension of the output model"""
        super().__init__()

        self.setContentsMargins(0, 0, 0, 0)

        self._width = float(width)
        self._height = float(height)

        self._width_factor = 1
        self._height_factor = 1
        self._raw_img_scale_w = self._raw_img_scale_h = 1
        self._raw_img_w = self._width
        self._raw_img_h = self._height
        self._data_width = None
        self._data_height = None

        self._reach_overlay_scene_item: Optional[QGraphicsPixmapItem] = None

        self._scene = QGraphicsScene(0, 0, width, height)

        brush = QBrush(Qt.GlobalColor.black)
        self._scene.setBackgroundBrush(brush)

        self._widget = QOpenGLWidget()
        view = self._view = QGraphicsView(self._scene)
        view.setStyleSheet("border: 0x")

        sformat = QSurfaceFormat()
        sformat.setSamples(4)
        self._widget.setFormat(sformat)
        view.setViewport(self._widget)
        view.setFixedSize(width, height)
        view.setContentsMargins(0, 0, 0, 0)

        # Created on first sight of a part rather than from a fixed list. That
        # list named six elements, two of which - L_Hand and R_Hand - are
        # composites the pose algorithm never emits, so those two slots waited
        # for keys that never arrived while Mouth, both tongue points and all
        # six real hand parts had no slot at all. Ten of the model's fourteen
        # keypoints could not be drawn at any confidence and nothing said so.
        # Driving this from what actually arrives means a retrained model with
        # new keypoints draws without editing this file.
        self._points: Dict[str, QGraphicsEllipseItem] = {}
        self._overlay_parts: Optional[frozenset] = None

        self._pixmap = None
        self._cur_image = None  # image must remain active to prevent segfault when pixmap continue use it.

        layout = QHBoxLayout()
        layout.addWidget(view)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)

        self._count = 0

    @property
    def size_factor(self) -> Tuple[float, float]:
        width_f = self._raw_img_scale_w / self._width_factor
        height_f = self._raw_img_scale_h / self._height_factor
        return width_f, height_f

    def set_data_size(self, width: int, height: int):
        # data size is the output model resolution
        self._pixmap = None
        self._data_width = width
        self._data_height = height
        self._width_factor = self._data_width / self._width
        self._height_factor = self._data_height / self._height

    def set_scale_aspect_ratio(self, scale_w: float, scale_h: float):
        # used when source image is not same aspect ratio than the one used by self
        self._raw_img_scale_w = scale_w
        self._raw_img_scale_h = scale_h

    def set_reach_overlay(self, px: Optional[QPixmap]):
        prev = self._reach_overlay_scene_item
        if prev is not None:
            self._scene.removeItem(prev)
            self._reach_overlay_scene_item = None
        if px is not None:
            px_item = self._reach_overlay_scene_item = self._scene.addPixmap(px)
            px_item.setZValue(150)
            # logger.info("configured scene item %s ; %s ; %s", px_item, px.size(), self._view.size())

    def set_data(
        self,
        image: QImage,
        *,
        text_overlay: Optional[str] = None,
        text_color: Qt.GlobalColor = Qt.GlobalColor.yellow,
    ):
        # retain a ref the used image to keep it alive after calling function also return
        self._cur_image = image
        pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_RGBA8888))
        if text_overlay:
            painter = QPainter(pixmap)
            font = QFont("Sans-serif", 12)
            painter.setFont(font)
            painter.setPen(text_color)
            with painter:
                painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, text_overlay)

        if self._pixmap is None:
            self._pixmap = QGraphicsPixmapItem(pixmap)
            self._pixmap.setZValue(0)
            self._pixmap.setPos(0, 0)
            self._scene.addItem(self._pixmap)
        else:
            self._pixmap.setPixmap(pixmap)

    def set_overlay_parts(self, parts) -> None:
        """Restrict the overlay to these part names; empty or None means all.

        Configured rather than hard-coded, so narrowing a cluttered view never
        again means a part becomes undrawable with nothing to say so.
        """
        self._overlay_parts = frozenset(parts) if parts else None
        for name, widget_point in self._points.items():
            if not self._is_part_shown(name):
                widget_point.setVisible(False)

    def _is_part_shown(self, name: str) -> bool:
        return self._overlay_parts is None or name in self._overlay_parts

    def _point_for(self, name: str) -> QGraphicsEllipseItem:
        """The dot for one part, created the first time that part is seen."""
        widget_point = self._points.get(name)
        if widget_point is not None:
            return widget_point
        colour = _PART_COLOURS.get(name)
        if colour is None:
            # Stable across restarts: the same part keeps the same colour.
            colour = _FALLBACK_COLOURS[hash(str(name)) % len(_FALLBACK_COLOURS)]
        size = 5.0 if name in _MARKER_PARTS else 3.0
        widget_point = QGraphicsEllipseItem(0, 0, size, size)
        pen = QPen(colour)
        pen.setWidth(1)
        widget_point.setPen(pen)
        widget_point.setBrush(QBrush(colour))
        widget_point.setZValue(100)
        widget_point.setPos(-10, -10)
        widget_point.setVisible(False)
        self._scene.addItem(widget_point)
        self._points[name] = widget_point
        return widget_point

    def set_points(self, points: Dict[str, PoseLocation]):
        width_f, height_f = self.size_factor
        # Iterate what arrived, not a fixed registry, so a part is drawn
        # whenever the model reports it. Parts absent this frame are hidden
        # below rather than left showing a stale position.
        for name, values in points.items():
            if values is None or not self._is_part_shown(name):
                continue
            # values are in coordinates (self._data_width, self._data_height)
            x = values.x * width_f
            y = values.y * height_f
            widget_point = self._point_for(name)
            if x < 0 or y < 0 or x > self._width or y > self._height:
                widget_point.setVisible(False)
            else:
                widget_point.setPos(x, y)
                widget_point.setVisible(True)
        for name, widget_point in self._points.items():
            if name not in points:
                widget_point.setVisible(False)
