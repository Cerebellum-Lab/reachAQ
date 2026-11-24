from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QFrame


class Separator(QFrame):
    def __init__(self, color: str = "#b9b9b9"):
        super().__init__()

        self.setFrameShape(QFrame.Shape.HLine)
        self.set_color(color)

    def set_color(self, color: str):
        palette = self.palette()
        palette.setColor(QPalette.WindowText, color)
        self.setPalette(palette)
