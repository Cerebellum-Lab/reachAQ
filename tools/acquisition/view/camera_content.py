from typing import Dict, Optional

from numpy import ndarray

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import QGridLayout, QSizePolicy

from autotrainer.core import NotificationCenter, TriggerNotification, Notification
from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference import PoseLocation
from autotrainer.video import VideoRecordMode

from autotrainer.pyside.capture.QtCaptureView import ImageData
from autotrainer.pyside import QCaptureView
from autotrainer.pyside.content_widget import ContentWidget, invoke_method

from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.video_capture_model import VideoCaptureModel

logger = get_verbose_logger(__name__)


class CameraContent(ContentWidget):
    def __init__(self, app_model: AppModel, capture_model: VideoCaptureModel):
        super().__init__()

        self._app_model = app_model
        self._model = capture_model

        self._text_overlay: Optional[str] = None
        self._text_overlay_color = Qt.GlobalColor.yellow

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        capture_view = self._capture_view = QCaptureView()
        capture_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        settings = self._settings = capture_view.settings
        capture_model.record_mode = VideoRecordMode.TRIGGER
        settings.setIsVideoCaptureEnabled(capture_model.is_enabled)
        settings.setIsVideoRecordEnabled(capture_model.is_recording_enabled)
        settings.setRecordMode(VideoRecordMode.TRIGGER)
        settings.setRecordModeVisible(False)
        settings.setStillImageCaptureEnabled(capture_model.is_still_capture_enabled)
        settings.setStillImageCaptureInterval(capture_model.still_image_capture_interval)

        capture_view.setCameras(capture_model.camera_list)
        capture_view.setCamera(capture_model.camera_source)
        capture_view.camera_changed.connect(self._camera_source_changed)

        settings.capture_enabled_changed.connect(self._camera_enabled_changed)
        settings.record_enabled_changed.connect(self._recording_enabled_changed)
        settings.record_mode_changed.connect(self._recording_enabled_changed)
        settings.image_capture_enabled_changed.connect(self._is_still_image_capture_enabled_changed)
        settings.image_capture_interval_changed.connect(self._still_image_capture_interval_changed)

        layout.addWidget(self._capture_view)

        self.setLayout(layout)

        NotificationCenter.default_center().add_observer(TriggerNotification.CAPTURE_ID, self._trigger_received)

        capture_model.property_changed += self._on_model_property_changed

        # Swap because model shape is row x col == height x width
        capture_view.setShape(capture_model.shape[1], capture_model.shape[0])

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._model.name!r})"

    @property
    def camera_view(self) -> QCaptureView:
        return self._capture_view

    def close(self):
        NotificationCenter.default_center().remove_observer(TriggerNotification.CAPTURE_ID, self._trigger_received)
        self._model.property_changed -= self._on_model_property_changed
        super().close()

    @Slot(ndarray, float)
    def refresh_image(self, data: ndarray, fps: float):
        row, col = data.shape
        self._capture_view.refresh_image(ImageData(data, col, row), fps)

    @invoke_method
    def set_is_editable(self, is_editable: bool):
        self._capture_view.set_is_editable(is_editable)

    @invoke_method
    def set_is_capture_active(self, is_active: bool):
        self._capture_view.set_is_capture_active(is_active)

    @invoke_method
    def update_image(self):
        self._capture_view.update_image()
        self._capture_view.update_pose()

    @invoke_method
    def refresh_pose(self, points: Dict[str, PoseLocation]):
        self._capture_view.refresh_pose(points)

    def _camera_source_changed(self, camera):
        self._model.camera_source = camera

    def _camera_enabled_changed(self, is_enabled):
        self._model.is_enabled = is_enabled

    def _recording_enabled_changed(self, is_enabled):
        self._model.is_recording_enabled = is_enabled
        self._model.record_mode = VideoRecordMode.TRIGGER

    def _is_still_image_capture_enabled_changed(self, is_enabled):
        self._model.is_still_capture_enabled = is_enabled

    def _still_image_capture_interval_changed(self, interval: float):
        self._model.still_image_capture_interval = interval

    @invoke_method
    def _trigger_received(self, notification: Notification):
        if self._model.is_enabled and self._model.record_mode == VideoRecordMode.TRIGGER:
            self._capture_view.recording_indicator_changed.emit(notification.context)

    @invoke_method
    def _on_model_property_changed(self, name, value, _):
        if name == VideoCaptureModel.CAMERA_PROP:
            self._capture_view.setCamera(value)
        elif name == VideoCaptureModel.IS_ENABLED_PROP:
            self._settings.setIsVideoCaptureEnabled(value)
        elif name == VideoCaptureModel.IS_RECORDING_ENABLED_PROP:
            self._settings.setIsVideoRecordEnabled(value)
        elif name == VideoCaptureModel.RECORD_MODE_PROP:
            self._settings.setRecordMode(value)
        elif name == VideoCaptureModel.IS_STILL_CAPTURE_ENABLED_PROP:
            self._settings.setStillImageCaptureEnabled(value)
        elif name == VideoCaptureModel.STILL_IMAGE_CAPTURE_INTERVAL_PROP:
            self._settings.setStillImageCaptureInterval(value)
        elif name == VideoCaptureModel.SHAPE_PROP:
            if value is not None and value[0] != 0 and value[1] != 0:
                # Swap because model shape is row x col == height x width
                self._capture_view.setShape(value[1], value[0])
        elif name == VideoCaptureModel.CAMERA_LIST_PROP:
            self._capture_view.setCameras(self._model.camera_list)
        elif name == VideoCaptureModel.TEXT_OVERLAY_PROP:
            self._text_overlay = value
            self._capture_view.set_text_overlay(
                value,
                color=self._text_overlay_color,
            )
        elif name == VideoCaptureModel.TEXT_OVERLAY_COLOR_PROP:
            color = getattr(Qt.GlobalColor, value, None)
            if color is None:
                logger.warning("unknown text overlay color %s", value)
                color = Qt.GlobalColor.yellow
            self._text_overlay_color = color
            self._capture_view.set_text_overlay(self._text_overlay, color=self._text_overlay_color)
        elif name == VideoCaptureModel.DISPLAY_DOTS_DETECTION_PROP:
            self._capture_view.set_display_dots_detection(value)
