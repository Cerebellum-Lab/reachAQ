from tools.acquisition.run_acquisition import _set_compact_initial_window_size


class _WindowStub:
    def __init__(self):
        self._width = 640
        self._height = 480
        self.adjusted = False

    def adjustSize(self):
        self.adjusted = True
        self._width = 1800
        self._height = 1000

    def width(self):
        return self._width

    def height(self):
        return self._height

    def resize(self, width, height):
        self._width = width
        self._height = height


def test_initial_window_is_reduced_by_300_pixels_in_each_dimension():
    window = _WindowStub()

    _set_compact_initial_window_size(window)

    assert window.adjusted
    assert (window.width(), window.height()) == (1500, 700)
