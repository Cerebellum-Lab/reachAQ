from tools.acquisition.run_acquisition import _show_default_window


class _WindowStub:
    def __init__(self):
        self.maximized = False
        self.geometry_restored = False

    def restore_normal_window_geometry(self):
        self.geometry_restored = True

    def showMaximized(self):
        self.maximized = True


def test_initial_window_is_maximized():
    window = _WindowStub()

    _show_default_window(window)

    assert window.geometry_restored
    assert window.maximized
