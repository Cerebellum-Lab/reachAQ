import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from autotrainer.core import ObservableObject, Offset3DTuple
from tools.pellet_delivery.view.pellet_control import PelletControl


class _PelletModelStub(ObservableObject):
    def __init__(self):
        super().__init__()
        self.travel_limits = None
        self.xyz = Offset3DTuple(0.0, 0.0, 0.0)

    def send_home(self):
        pass

    def load_pellet(self):
        pass

    def send_pellet(self):
        pass

    def move_retract(self):
        pass

    def release_pellet(self):
        pass

    def cover_pellet(self):
        pass

    def move_x(self, _value):
        pass

    def move_y(self, _value):
        pass

    def move_z(self, _value):
        pass

    def set_x(self, _value):
        pass

    def set_y(self, _value):
        pass

    def set_z(self, _value):
        pass

    @staticmethod
    def to_motor_coordinates(value):
        return value

    @staticmethod
    def to_diamond_coordinates(value):
        return value


def test_retained_pellet_control_constructs_without_legacy_flag():
    app = QApplication.instance() or QApplication([])
    widget = PelletControl(_PelletModelStub())
    app.processEvents()
    try:
        assert widget._x_pos is not None
        assert widget._y_pos is not None
        assert widget._z_pos is not None
    finally:
        widget.close()
