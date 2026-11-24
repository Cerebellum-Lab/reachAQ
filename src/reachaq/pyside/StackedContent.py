from PySide6.QtWidgets import QStackedWidget, QStackedLayout


class StackedWidget(QStackedWidget):
    """QStackedWidget which set its size hint and minimum size to the currently selected child widget"""

    def minimumSize(self):
        current = self.currentWidget()
        if current:
            # Check for specific minimum size, otherwise use sizeHint
            s = current.minimumSize()
            if s.isEmpty():
                s = current.minimumSizeHint()
            return s
        return super().minimumSize()

    def minimumSizeHint(self):
        current = self.currentWidget()
        if current:
            return current.minimumSizeHint()
        return super().minimumSizeHint()

    def sizeHint(self):
        current = self.currentWidget()
        if current:
            return current.sizeHint()
        return super().sizeHint()



class StackedLayout(QStackedLayout):

    def minimumSize(self):
        current = self.currentWidget()
        if current:
            # Check for specific minimum size, otherwise use sizeHint
            s = current.minimumSize()
            if s.isEmpty():
                s = current.minimumSizeHint()
            return s
        return super().minimumSize()

    def sizeHint(self):
        current = self.currentWidget()
        if current:
            return current.sizeHint()
        return super().sizeHint()
