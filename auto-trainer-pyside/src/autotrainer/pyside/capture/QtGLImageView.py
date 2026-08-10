import os
from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QSurfaceFormat, QPixmap, QPen, QPainter, QFont
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget, QGraphicsView, QGraphicsScene, QHBoxLayout, QGraphicsPixmapItem, \
    QGraphicsEllipseItem, QGraphicsItem

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.pose_elements import SceneElement
from autotrainer.inference import PoseLocation


logger = get_verbose_logger(__name__)


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

        self._points: Dict[SceneElement, QGraphicsEllipseItem] = {}

        def add_managed_point(color, elem: SceneElement, *, size_w: float=5, size_h: float=5):
            point = self._points[elem] = QGraphicsEllipseItem(0, 0, size_w, size_h)
            pen = QPen(color)
            pen.setWidth(1)
            point.setPen(pen)
            point.setBrush(QBrush(color))
            point.setZValue(100)
            point.setPos(-10, -10)
            self._scene.addItem(point)

        add_managed_point(Qt.GlobalColor.red, SceneElement.Pellet)
        add_managed_point(Qt.GlobalColor.magenta, SceneElement.Star)
        add_managed_point(Qt.GlobalColor.blue, SceneElement.Diamond)
        add_managed_point(Qt.GlobalColor.green, SceneElement.Triangle)
        add_managed_point(Qt.GlobalColor.white, SceneElement.L_Hand)
        add_managed_point(Qt.GlobalColor.yellow, SceneElement.R_Hand)
        if os.getenv("AUTOTRAINER_SHOW_NOSE"):
            add_managed_point(Qt.GlobalColor.cyan, SceneElement.Nose, size_w=1.75, size_h=1.75)
        # was previously used until we had L/R_Hand :
        # add_managed_point(Qt.GlobalColor.white, SceneElement.LH_grab)
        # add_managed_point(Qt.GlobalColor.yellow, SceneElement.RH_grab)

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

    def set_points(self, points: Dict[str, PoseLocation]):
        width_f, height_f = self.size_factor
        # values are in coordinates (self._data_width, self._data_height)
        for elem, widget_point in self._points.items():
            values = points.get(elem, None)
            if values is None:
                widget_point.setVisible(False)
                continue
            x = values.x * width_f
            y = values.y * height_f
            if x < 0 or y < 0 or x > self._width or y > self._height:
                widget_point.setVisible(False)
            else:
                widget_point.setPos(x, y)
                widget_point.setVisible(True)
