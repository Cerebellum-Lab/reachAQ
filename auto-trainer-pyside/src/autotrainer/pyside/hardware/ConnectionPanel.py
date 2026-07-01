from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QPushButton

from ..CardWidget import CardWidget


class ConnectionPanel(QWidget):
    connecting = Signal()
    disconnected = Signal()

    def __init__(self, app_view_model):
        super().__init__()

        self._app_view_model = app_view_model
        self._app_view_model.property_changed += self._model_property_changed

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addStretch()

        self._connect_button = QPushButton("Connect")
        self._connect_button.clicked.connect(self._connect)
        layout.addWidget(self._connect_button, 0, Qt.AlignRight)

        self._connection_status = QLabel("Not Connected")
        self._connection_status.setStyleSheet("color: #20242a")
        self._connection_status.setContentsMargins(0, 0, 4, 0)

        panel = CardWidget(title="Connection", content_layout=layout, header_right_layout=self._connection_status)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(panel)
        self.setLayout(layout)

    def _connect(self):
        if self._app_view_model.is_connected:
            self._app_view_model.disconnect_from_device()
            self._connect_button.setText("Connect")
            self._connection_status.setText("Not Connected")
            self.disconnected.emit()
        else:
            self.connecting.emit()
            self._connection_status.setText("...")
            self._app_view_model.connect_to_device()
            self._connect_button.setText("Disconnect")

    def _model_property_changed(self, name: str, value: object, _: object):
        if name == "command_pending":
            self._connect_button.setEnabled((not value) and self._app_view_model.is_connected)
        elif name == "firmware_version":
            assert isinstance(value, str)
            self._connection_status.setText(f"Firmware: {value}")
