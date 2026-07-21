import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from autotrainer.pyside import PGWidget
from autotrainer.pyside.content_widget import ContentWidget
from tools.acquisition.view import main_content as main_content_module
from tools.acquisition.view.main_content import MainContent
from tools.acquisition.view.main_window import _allow_bidirectional_window_resizing
from tools.acquisition.view.persistent_splitter import PersistentSplitter


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_persistent_splitter_restores_user_sizes(qapp, user_pref):
    first = PersistentSplitter(Qt.Orientation.Horizontal, user_pref, "test_panels")
    first.addWidget(QWidget())
    first.addWidget(QWidget())
    first.resize(600, 200)
    first.show()
    first.setSizes([450, 150])
    qapp.processEvents()
    first.save_state()

    restored = PersistentSplitter(Qt.Orientation.Horizontal, user_pref, "test_panels")
    restored.addWidget(QWidget())
    restored.addWidget(QWidget())
    restored.resize(600, 200)
    restored.show()
    assert restored.apply_saved_or_default_sizes([300, 300])
    qapp.processEvents()

    assert restored.sizes()[0] > restored.sizes()[1]
    assert not restored.opaqueResize()
    assert restored.handleWidth() == 6
    first.close()
    restored.close()


def test_main_window_resizes_freely_in_width_and_height(qapp):
    window = QMainWindow()
    central_widget = QWidget()
    central_widget.setMinimumSize(1200, 800)
    window.setCentralWidget(central_widget)
    _allow_bidirectional_window_resizing(window)
    window.show()
    assert (window.minimumWidth(), window.minimumHeight()) == (320, 240)

    window.resize(480, 320)
    qapp.processEvents()
    assert (window.width(), window.height()) == (480, 320)

    window.resize(960, 720)
    qapp.processEvents()
    assert (window.width(), window.height()) == (960, 720)
    window.close()


def test_graph_widget_tracks_container_width_and_height(qapp):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    graph = PGWidget(container)
    layout.addWidget(graph)
    container.show()

    container.resize(800, 600)
    qapp.processEvents()
    large_size = graph.size()

    container.resize(400, 250)
    qapp.processEvents()
    small_size = graph.size()

    assert small_size.width() < large_size.width()
    assert small_size.height() < large_size.height()
    assert graph.minimumWidth() == graph.minimumHeight() == 0
    container.close()


def test_all_visible_main_panel_boundaries_are_splitters(qapp, app_model, monkeypatch):
    class _BehaviorPanelStub(ContentWidget):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr(main_content_module, "BehaviorContent", _BehaviorPanelStub)
    assert app_model.load_configuration() is True
    content = MainContent(app_model)
    try:
        content.resize(1500, 900)
        content.show()
        qapp.processEvents()

        assert content._main_splitter.count() == 2
        assert content._section_splitter.count() == 4
        assert content._mid_splitter.count() == 2
        assert content._hardware_splitter.count() == 2
        assert content._camera_rows_splitter.count() == 1
        assert content._camera_row_splitters[0].count() == 2
        assert all(
            not splitter.childrenCollapsible()
            for splitter in (
                content._main_splitter,
                content._section_splitter,
                content._mid_splitter,
                content._hardware_splitter,
                content._camera_rows_splitter,
                *content._camera_row_splitters,
            )
        )

        content.set_is_editable(True)
        qapp.processEvents()
        assert content._camera_rows_splitter.count() == 1
        assert content._camera_row_splitters[0].count() == 3
        assert tuple(camera.name for camera, _panel in content._reach_camera_contents) == (
            "left",
            "right",
            "stimCam",
        )
        stim_panel = content._reach_camera_content_by_model[app_model.stim_camera]
        assert stim_panel.camera_view._content_stack.currentIndex() == 1
        assert not stim_panel._settings.isCaptureEnabled
        assert app_model.stim_camera._display_update_fcn == stim_panel.refresh_image

        stim_panel._settings.setIsVideoCaptureEnabled(True)
        qapp.processEvents()
        assert app_model.stim_camera.is_enabled
        content.set_is_editable(False)
        qapp.processEvents()
        assert tuple(camera.name for camera, _panel in content._reach_camera_contents) == (
            "left",
            "right",
            "stimCam",
        )

        content.set_is_editable(True)
        qapp.processEvents()
        stim_panel = content._reach_camera_content_by_model[app_model.stim_camera]
        stim_panel._settings.setIsVideoCaptureEnabled(False)
        qapp.processEvents()
        content.set_is_editable(False)
        qapp.processEvents()
        assert tuple(camera.name for camera, _panel in content._reach_camera_contents) == (
            "left",
            "right",
        )
    finally:
        content.close()
        content.deleteLater()
