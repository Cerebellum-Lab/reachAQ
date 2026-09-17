import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from autotrainer.pyside.content_widget import ContentWidget
from tools.acquisition.view import main_content as main_content_module
from tools.acquisition.view.detachable_panel import PanelState
from tools.acquisition.view.main_content import MainContent


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_content(qapp, app_model, monkeypatch):
    class _BehaviorPanelStub(ContentWidget):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr(main_content_module, "BehaviorContent", _BehaviorPanelStub)
    assert app_model.load_configuration() is True
    content = MainContent(app_model)
    content.resize(1500, 900)
    content.show()
    qapp.processEvents()
    yield content
    content.right_panel_host.collapse()
    content.close()


def test_the_right_side_tabs_start_docked(main_content):
    assert main_content.right_panel_host.state is PanelState.DOCKED
    assert main_content._main_splitter.widget(1) is main_content._right_side_tabs


def test_expanding_takes_the_tabs_out_of_the_splitter(main_content):
    host = main_content.right_panel_host
    panel = main_content._right_side_tabs

    host.expand()

    assert host.state is PanelState.EXPANDED
    assert main_content._main_splitter.widget(1) is not panel
    assert panel.window() is host.window


def test_collapsing_puts_the_tabs_back_in_their_slot(main_content):
    host = main_content.right_panel_host
    panel = main_content._right_side_tabs
    host.expand()

    host.collapse()

    assert host.state is PanelState.DOCKED
    assert main_content._main_splitter.widget(1) is panel


def test_the_expanded_panel_is_wider_than_its_docked_slot(main_content, qapp):
    host = main_content.right_panel_host
    docked_width = main_content._main_splitter.sizes()[1]

    main_content.update_right_panel_tracking()
    host.expand()
    qapp.processEvents()

    assert host.window.width() > docked_width


def test_detaching_and_reattaching_restores_the_slot(main_content):
    host = main_content.right_panel_host
    panel = main_content._right_side_tabs

    host.detach()
    assert host.state is PanelState.DETACHED

    host.reattach()

    assert host.state is PanelState.DOCKED
    assert main_content._main_splitter.widget(1) is panel


def test_the_placement_is_written_to_preferences(main_content, app_model):
    host = main_content.right_panel_host

    host.expand()

    assert app_model.preferences.right_panel_state == "expanded"


def test_the_toggle_buttons_expand_and_restore(main_content):
    host = main_content.right_panel_host

    main_content._expand_panel_button.click()
    assert host.state is PanelState.EXPANDED

    main_content._expand_panel_button.click()
    assert host.state is PanelState.DOCKED
