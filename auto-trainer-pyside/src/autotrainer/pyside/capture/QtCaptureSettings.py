from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QComboBox, QHBoxLayout, QLineEdit


from autotrainer.core.logging import get_verbose_logger
from autotrainer.pyside.QtLabeledSwitch import QLabeledSwitch


logger = get_verbose_logger(__name__)


class QCaptureSettings(QWidget):
    capture_enabled_changed = Signal(bool)
    record_enabled_changed = Signal(bool)
    record_mode_changed = Signal(bool)
    image_capture_enabled_changed = Signal(bool)
    image_capture_interval_changed = Signal(float)

    def __init__(self, parent=None):
        super(QCaptureSettings, self).__init__(parent)

        layout = QGridLayout()

        layout.addWidget(QLabel("Video Capture:"), 0, 0)
        self._isCaptureEnabled = QLabeledSwitch()
        self._isCaptureEnabled.stateChanged.connect(self._is_capture_enabled_changed)
        layout.addWidget(self._isCaptureEnabled, 0, 1)

        self._record_mode_label = QLabel("Record Mode:")
        layout.addWidget(self._record_mode_label, 1, 0)
        self._record_mode = QComboBox()
        self._record_mode.setMaximumWidth(140)
        self._record_mode.addItem("Continuous")
        self._record_mode.addItem("Trigger")
        self._record_mode.currentIndexChanged.connect(lambda x:
                                                      self.record_mode_changed.emit(self.isRecordEnabled))
        layout.addWidget(self._record_mode, 1, 1, alignment=Qt.AlignRight)

        layout.addWidget(QLabel("Video Recording:"), 2, 0)
        self._isRecordEnabled = QLabeledSwitch()
        self._isRecordEnabled.stateChanged.connect(self._is_record_enabled_changed)
        layout.addWidget(self._isRecordEnabled, 2, 1)

        layout.addWidget(QLabel("Image Capture:"), 3, 0)

        self._isStillImageCaptureEnabled = QLabeledSwitch()
        self._isStillImageCaptureEnabled.stateChanged.connect(self._is_still_image_capture_enabled_changed)
        layout.addWidget(self._isStillImageCaptureEnabled, 3, 1)

        layout.addWidget(QLabel("Image Capture Interval:"), 4, 0)

        sell_interval_layout = QHBoxLayout()
        sell_interval_layout.setContentsMargins(0, 0, 0, 0)
        sell_interval_layout.setSpacing(2)
        self._stillImageCaptureInterval = QLineEdit()
        self._stillImageCaptureInterval.setMaximumWidth(50)
        self._stillImageCaptureInterval.textChanged.connect(self._still_image_capture_interval_changed)
        sell_interval_layout.addStretch(1)
        sell_interval_layout.addWidget(self._stillImageCaptureInterval)
        sell_interval_layout.addWidget(QLabel("seconds"))
        layout.addLayout(sell_interval_layout, 4, 1)

        layout.setRowStretch(5, 1)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("QCaptureSettings")
        self.setStyleSheet("#QCaptureSettings {background-color: #c9c9c9;}")
        self.setLayout(layout)

        self._is_capture_enabled_changed(False)

    @property
    def isTriggerRecordMode(self) -> bool:
        return self._record_mode.currentIndex() == 1

    @property
    def isCaptureEnabled(self) -> bool:
        return self._isCaptureEnabled.isChecked()

    @property
    def isRecordEnabled(self) -> bool:
        return self._isRecordEnabled.isChecked()

    @property
    def isImageCaptureEnabled(self) -> bool:
        return self._isStillImageCaptureEnabled.isChecked()

    @property
    def imageCaptureInterval(self) -> float:
        try:
            return float(self._stillImageCaptureInterval.text())
        except (ValueError, TypeError):
            pass

        return 0.0

    def setIsVideoCaptureEnabled(self, b: bool) -> None:
        self._isCaptureEnabled.setChecked(b)

    def setIsVideoRecordEnabled(self, b: bool) -> None:
        self._isRecordEnabled.setChecked(b)

    def setRecordMode(self, mode: int):
        if mode is None:
            self._record_mode.setCurrentIndex(-1)
        else:
            self._record_mode.setCurrentIndex(mode)

    def setRecordModeVisible(self, visible: bool) -> None:
        self._record_mode_label.setVisible(visible)
        self._record_mode.setVisible(visible)

    def setStillImageCaptureEnabled(self, b: bool) -> None:
        self._isStillImageCaptureEnabled.setChecked(b)

    def setStillImageCaptureInterval(self, seconds: float) -> None:
        self._stillImageCaptureInterval.setText(str(seconds))

    def _is_capture_enabled_changed(self, x: int) -> None:
        is_enabled = x != 0
        self.capture_enabled_changed.emit(is_enabled)

        self._isRecordEnabled.setEnabled(is_enabled)
        self._record_mode.setEnabled(is_enabled)

        self._isStillImageCaptureEnabled.setEnabled(is_enabled)
        self._stillImageCaptureInterval.setEnabled(is_enabled and self._isStillImageCaptureEnabled.isChecked())

    def _is_record_enabled_changed(self, x: int) -> None:
        is_enabled = x != 0
        self.record_enabled_changed.emit(is_enabled)

    def _is_still_image_capture_enabled_changed(self, x: int) -> None:
        is_enabled = x != 0
        self.image_capture_enabled_changed.emit(is_enabled)
        self._stillImageCaptureInterval.setEnabled(is_enabled)

    def _still_image_capture_interval_changed(self, value: str) -> None:
        try:
            interval = float(value)
        except (ValueError, TypeError):
            pass
        else:
            self.image_capture_interval_changed.emit(interval)
