

import inspect
import re

import tools


def test_we_can_import_acquisition_main_window():
    from tools.acquisition.view import main_window


def test_main_window_subscribes_only_existing_callbacks():
    from tools.acquisition.view.main_window import MainWindow

    source = inspect.getsource(MainWindow.__init__)
    callback_names = set(re.findall(r"\+= self\.(\w+)", source))

    assert callback_names
    assert not [name for name in callback_names if not hasattr(MainWindow, name)]


def test_we_can_import_headless():
    from tools.acquisition import headless


def test_we_can_import_pellet_delivery_window():
    from tools.pellet_delivery.view import main_window
