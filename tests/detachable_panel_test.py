import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QSplitter,
)

from tools.acquisition.view.detachable_panel import (  # noqa: E402
    EXPANDED_WIDTH_FRACTION,
    DetachablePanelHost,
    PanelState,
)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def host(qapp):
    splitter = QSplitter(Qt.Orientation.Horizontal)
    left = QLabel("left")
    panel = QLabel("panel")
    splitter.addWidget(left)
    splitter.addWidget(panel)
    splitter.resize(1000, 600)
    splitter.setSizes([700, 300])
    panel_host = DetachablePanelHost(panel, splitter, 1, title="Panel")
    yield panel_host, splitter, panel
    panel_host.collapse()
    splitter.deleteLater()


def test_expanded_geometry_is_right_aligned_in_the_reference():
    reference = QRect(100, 50, 1000, 600)

    geometry = DetachablePanelHost.expanded_geometry(reference)

    assert geometry.width() == int(1000 * EXPANDED_WIDTH_FRACTION)
    assert geometry.right() == reference.right()
    assert geometry.y() == reference.y()
    assert geometry.height() == reference.height()


def test_expand_moves_the_panel_into_its_own_window(host):
    panel_host, splitter, panel = host

    panel_host.expand()

    assert panel_host.state is PanelState.EXPANDED
    assert splitter.widget(1) is not panel
    assert panel_host.window is not None
    assert panel.window() is panel_host.window


def test_collapse_returns_the_panel_to_its_slot(host):
    panel_host, splitter, panel = host
    panel_host.expand()

    panel_host.collapse()

    assert panel_host.state is PanelState.DOCKED
    assert splitter.widget(1) is panel
    assert panel_host.window is None


def test_collapse_preserves_the_panel_slot_size(host):
    panel_host, splitter, _panel = host
    before = splitter.sizes()

    panel_host.expand()
    panel_host.collapse()

    assert splitter.sizes()[1] == pytest.approx(before[1], abs=2)


def test_detach_gives_the_panel_a_titled_window(host):
    panel_host, splitter, panel = host

    panel_host.detach()

    assert panel_host.state is PanelState.DETACHED
    assert splitter.widget(1) is not panel
    assert panel_host.window.windowTitle() == "Panel"


def test_expand_then_detach_leaves_exactly_one_window(host):
    panel_host, _splitter, panel = host
    panel_host.expand()
    expanded_window = panel_host.window

    panel_host.detach()

    assert panel_host.state is PanelState.DETACHED
    assert panel_host.window is not expanded_window
    assert panel.window() is panel_host.window


def test_detach_ignores_a_geometry_that_is_off_every_screen(host):
    panel_host, _splitter, _panel = host

    panel_host.detach(QRect(-100000, -100000, 400, 300))

    assert panel_host.window.geometry() != QRect(-100000, -100000, 400, 300)


def test_state_changed_reports_each_transition(host):
    panel_host, _splitter, _panel = host
    seen = []
    panel_host.state_changed.connect(seen.append)

    panel_host.expand()
    panel_host.detach()
    panel_host.collapse()

    assert seen == ["expanded", "detached", "docked"]
