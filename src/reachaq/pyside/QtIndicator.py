from PySide6.QtWidgets import QLabel

ACTIVE_STYLE = "border: 1px solid gray; border-radius: 4; background-color: green;"
INACTIVE_STYLE = "border: 1px solid gray; border-radius: 4; background-color: red;"


class QtIndicator(QLabel):
    def __init__(self, parent=None, text: str = ""):
        super(QtIndicator, self).__init__(parent)

        self.setText(text)
        self.setStyleSheet(INACTIVE_STYLE)

    def setState(self, state: bool) -> None:
        self.setStyleSheet(ACTIVE_STYLE if state else INACTIVE_STYLE)
