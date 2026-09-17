# Detachable Protocol Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the right-side tab widget (Laser Control and Protocol) expand leftward over the other panels and detach into its own window, so the 29-column protocol table is usable.

**Architecture:** One reparenting host with three states. The panel either sits in its splitter slot, or lives in a frameless `Qt.Tool` window sized over the main window, or lives in an ordinary top-level window. Both undocked states are real top-level windows, so the window manager composites them and we never stack a widget over the `QOpenGLWidget` camera views. A placeholder holds the splitter slot while the panel is away.

**Tech Stack:** Python 3.8, PySide6, pytest, Qt offscreen platform for tests.

**Spec:** `docs/superpowers/specs/2026-09-17-protocol-panel-sets-stim-test-design.md`

## Global Constraints

- **Python 3.8.20.** The rig runs `~/anaconda3/envs/reachaq/bin/python`, which is 3.8.20, even though `pyproject.toml` says `requires-python = ">= 3.10"`. Write 3.8-compatible code: `from __future__ import annotations` at the top of every new module, `typing.Optional` / `typing.Tuple` rather than `X | Y` and `tuple[...]`, no `match` statements, no `dict | dict`.
- **All test runs, suites and app launches happen on `christielab10`.** The local Windows checkout is for reading and editing only; it has no PySpin, nidaqmx, torch or hardware. Never run pytest locally.
- Rig SSH: `ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10`. Repo at `~/Documents/reachAQ`.
- **Never kill rig processes by pattern-matching the interpreter path.** List processes and kill by explicit PID.
- Branch `feature-dev`, worked in the worktree `../reachAQ-feature`.
- Test files are `tests/<name>_test.py`. Set `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` **before** any Qt import, and mark the post-`os.environ` imports `# noqa: E402`, matching `tests/signal_stream_ui_test.py`.
- Do not broadly rewrite `autotrainer.core`, `autotrainer.video`, `autotrainer.device`, `autotrainer.inference`, or `autotrainer.behavior`. This plan touches none of them.

---

### Task 1: DetachablePanelHost

The state machine and reparenting, with no dependency on the app model, the main window, or hardware. Everything here is testable offscreen in isolation.

**Files:**
- Create: `tools/acquisition/view/detachable_panel.py`
- Test: `tests/detachable_panel_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PanelState` enum with members `DOCKED`, `EXPANDED`, `DETACHED`, each `.value` being `"docked"`, `"expanded"`, `"detached"`.
  - `DetachablePanelHost(panel: QWidget, splitter: QSplitter, index: int, *, title: str = "Panel", parent: Optional[QObject] = None)`
  - `host.state -> PanelState`, `host.window -> Optional[QWidget]`
  - `host.expand() -> None`, `host.detach(geometry: Optional[QRect] = None) -> None`, `host.collapse() -> None`, `host.reattach() -> None`
  - `host.track(reference: QRect) -> None`
  - `DetachablePanelHost.expanded_geometry(reference: QRect) -> QRect` (static)
  - `host.state_changed` — `Signal(str)`, carrying `PanelState.value`
  - Module constant `EXPANDED_WIDTH_FRACTION = 0.85`

- [ ] **Step 1: Write the failing tests**

Create `tests/detachable_panel_test.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

On the rig:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/detachable_panel_test.py -v'
```

Expected: collection error, `ModuleNotFoundError: No module named 'tools.acquisition.view.detachable_panel'`.

- [ ] **Step 3: Write the implementation**

Create `tools/acquisition/view/detachable_panel.py`:

```python
"""Move one panel between its splitter slot, an overlay, and its own window."""

from __future__ import annotations

import enum
from typing import Optional

from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget


#: Fraction of the reference width an expanded panel covers. The panel is
#: right-aligned inside the reference, so it grows leftward over the panels it
#: covers instead of pushing them aside.
EXPANDED_WIDTH_FRACTION = 0.85


class PanelState(enum.Enum):
    DOCKED = "docked"
    EXPANDED = "expanded"
    DETACHED = "detached"


class DetachablePanelHost(QObject):
    """Own one panel widget's placement across three states.

    Both undocked states are real top-level windows rather than raised child
    widgets. The left side of the main window contains QtGLImageView, a
    QOpenGLWidget, and stacking a sibling above a QOpenGLWidget depends on
    platform compositing. A top-level window is composited by the window
    manager, so the overlay does not depend on widget stacking order at all,
    and detaching reuses the same code path.
    """

    state_changed = Signal(str)

    def __init__(
        self,
        panel: QWidget,
        splitter: QSplitter,
        index: int,
        *,
        title: str = "Panel",
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._panel = panel
        self._splitter = splitter
        self._index = int(index)
        self._title = title
        self._state = PanelState.DOCKED
        self._placeholder: Optional[QWidget] = None
        self._window: Optional[QWidget] = None
        self._reference = QRect()

    @property
    def state(self) -> PanelState:
        return self._state

    @property
    def window(self) -> Optional[QWidget]:
        return self._window

    @staticmethod
    def expanded_geometry(reference: QRect) -> QRect:
        """Right-aligned rectangle covering most of the reference."""
        width = max(1, int(reference.width() * EXPANDED_WIDTH_FRACTION))
        return QRect(
            reference.x() + reference.width() - width,
            reference.y(),
            width,
            reference.height(),
        )

    def expand(self) -> None:
        self._move_to_window(
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint,
            PanelState.EXPANDED,
        )
        self.track(self._reference)

    def detach(self, geometry: Optional[QRect] = None) -> None:
        self._move_to_window(Qt.WindowType.Window, PanelState.DETACHED)
        if geometry is not None and self._is_on_a_screen(geometry):
            self._window.setGeometry(QRect(geometry))

    def collapse(self) -> None:
        """Return the panel to its splitter slot from either undocked state."""
        if self._state is PanelState.DOCKED:
            return
        sizes = self._splitter.sizes()
        self._splitter.replaceWidget(self._index, self._panel)
        if self._placeholder is not None:
            self._placeholder.setParent(None)
            self._placeholder.deleteLater()
            self._placeholder = None
        self._splitter.setSizes(sizes)
        self._discard_window()
        self._set_state(PanelState.DOCKED)

    def reattach(self) -> None:
        """Alias for collapse, for callers that think in terms of detaching."""
        self.collapse()

    def track(self, reference: QRect) -> None:
        """Keep an expanded panel pinned to the right of the reference."""
        if reference is not None and reference.isValid():
            self._reference = QRect(reference)
        if self._state is not PanelState.EXPANDED or self._window is None:
            return
        if not self._reference.isValid():
            return
        self._window.setGeometry(self.expanded_geometry(self._reference))

    def _move_to_window(self, flags, state: PanelState) -> None:
        if self._state is PanelState.DOCKED:
            sizes = self._splitter.sizes()
            placeholder = QWidget()
            placeholder.setSizePolicy(self._panel.sizePolicy())
            self._splitter.replaceWidget(self._index, placeholder)
            self._placeholder = placeholder
            self._splitter.setSizes(sizes)
        window = QWidget(None, flags)
        window.setWindowTitle(self._title)
        layout = QVBoxLayout(window)
        layout.setContentsMargins(0, 0, 0, 0)
        # Reparent the panel into the new window before discarding the old one,
        # so the panel is never briefly owned by a window being destroyed.
        layout.addWidget(self._panel)
        previous = self._window
        self._window = window
        if previous is not None:
            previous.close()
            previous.deleteLater()
        window.show()
        self._set_state(state)

    def _discard_window(self) -> None:
        if self._window is None:
            return
        self._window.close()
        self._window.deleteLater()
        self._window = None

    def _set_state(self, state: PanelState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state.value)

    @staticmethod
    def _is_on_a_screen(geometry: QRect) -> bool:
        if not geometry.isValid() or geometry.width() < 1 or geometry.height() < 1:
            return False
        for screen in QGuiApplication.screens():
            if screen.availableGeometry().intersects(geometry):
                return True
        return False
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/detachable_panel_test.py -v'
```

Expected: 8 passed.

If `test_detach_ignores_a_geometry_that_is_off_every_screen` fails because the offscreen platform reports no screens, `QGuiApplication.screens()` returns an empty list and `_is_on_a_screen` correctly returns `False` — the assertion still holds. If it fails for any other reason, do not weaken the assertion; fix `_is_on_a_screen`.

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/view/detachable_panel.py tests/detachable_panel_test.py
git commit -m "feat(view): host a panel across docked, expanded and detached states"
```

---

### Task 2: Persist the panel's placement

**Files:**
- Modify: `tools/acquisition/model/user_preferences.py` (add accessors beside `window_normal_geometry`, around line 127)
- Test: `tests/user_preferences_panel_test.py`

**Interfaces:**
- Consumes: `PanelState.value` strings from Task 1 (`"docked"`, `"expanded"`, `"detached"`).
- Produces:
  - `UserPreferences.right_panel_state -> str` (property, settable), defaulting to `"docked"`
  - `UserPreferences.right_panel_detached_geometry -> QRect` (property, settable), defaulting to an invalid `QRect()`

- [ ] **Step 1: Write the failing tests**

Create `tests/user_preferences_panel_test.py`:

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.model.user_preferences import UserPreferences  # noqa: E402


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def preferences(qapp, tmp_path):
    return UserPreferences(settings_file_path=tmp_path / "prefs.ini")


def test_right_panel_state_defaults_to_docked(preferences):
    assert preferences.right_panel_state == "docked"


def test_right_panel_state_round_trips(preferences, tmp_path):
    preferences.right_panel_state = "detached"
    preferences.save()

    reloaded = UserPreferences(settings_file_path=tmp_path / "prefs.ini")

    assert reloaded.right_panel_state == "detached"


def test_right_panel_detached_geometry_round_trips(preferences, tmp_path):
    preferences.right_panel_detached_geometry = QRect(20, 30, 800, 600)
    preferences.save()

    reloaded = UserPreferences(settings_file_path=tmp_path / "prefs.ini")

    assert reloaded.right_panel_detached_geometry == QRect(20, 30, 800, 600)


def test_right_panel_detached_geometry_rejects_a_degenerate_rectangle(preferences):
    preferences.right_panel_detached_geometry = QRect(20, 30, 800, 600)

    preferences.right_panel_detached_geometry = QRect(0, 0, 0, 0)

    assert preferences.right_panel_detached_geometry == QRect(20, 30, 800, 600)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/user_preferences_panel_test.py -v'
```

Expected: FAIL with `AttributeError: 'UserPreferences' object has no attribute 'right_panel_state'`.

- [ ] **Step 3: Write the implementation**

In `tools/acquisition/model/user_preferences.py`, add these class constants near the other key strings and the four accessors immediately after the `window_normal_geometry` setter:

```python
    RIGHT_PANEL_STATE_KEY = "ui/panels/right_side_state"
    RIGHT_PANEL_GEOMETRY_KEY = "ui/panels/right_side_detached_geometry"

    @property
    def right_panel_state(self) -> str:
        value = self._settings.value(self.RIGHT_PANEL_STATE_KEY, "docked")
        return str(value) if value else "docked"

    @right_panel_state.setter
    def right_panel_state(self, value: str) -> None:
        self._settings.setValue(self.RIGHT_PANEL_STATE_KEY, str(value))

    @property
    def right_panel_detached_geometry(self) -> QRect:
        value = self._settings.value(self.RIGHT_PANEL_GEOMETRY_KEY, QRect())
        return QRect(value) if isinstance(value, QRect) else QRect()

    @right_panel_detached_geometry.setter
    def right_panel_detached_geometry(self, value: QRect) -> None:
        value = QRect(value)
        # A zero-size or invalid rectangle would restore an unusable window.
        if not value.isValid() or value.width() < 1 or value.height() < 1:
            return
        self._settings.setValue(self.RIGHT_PANEL_GEOMETRY_KEY, value)
```

`QRect` is already imported at the top of this module.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/user_preferences_panel_test.py -v'
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/model/user_preferences.py tests/user_preferences_panel_test.py
git commit -m "feat(preferences): remember where the right-side panel was left"
```

---

### Task 3: Wire the host into the main window

**Files:**
- Modify: `tools/acquisition/view/main_content.py` — imports, `_create_right_side_tabs` (line 404), `__init__` after the `addWidget`/stretch block (around line 199), and `close` (line 488)
- Modify: `tools/acquisition/view/main_window.py` — `moveEvent` (line 967) and `resizeEvent` (line 971)
- Test: `tests/main_content_panel_test.py`

**Interfaces:**
- Consumes: `DetachablePanelHost`, `PanelState` from Task 1; `right_panel_state` and `right_panel_detached_geometry` from Task 2.
- Produces:
  - `MainContent.right_panel_host -> DetachablePanelHost`
  - `MainContent.update_right_panel_tracking() -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/main_content_panel_test.py`. This exercises the host against a real `MainContent` built from the existing `app_model` fixture in `tests/conftest.py`:

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.view.detachable_panel import PanelState  # noqa: E402
from tools.acquisition.view.main_content import MainContent  # noqa: E402


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_content(qapp, app_model):
    content = MainContent(app_model)
    yield content
    content.right_panel_host.collapse()
    content.close()


def test_right_side_tabs_start_docked(main_content):
    assert main_content.right_panel_host.state is PanelState.DOCKED


def test_expanding_takes_the_tabs_out_of_the_splitter(main_content):
    tabs = main_content.right_panel_host
    panel = main_content._right_side_tabs

    tabs.expand()

    assert tabs.state is PanelState.EXPANDED
    assert panel.window() is tabs.window


def test_collapsing_puts_the_tabs_back(main_content):
    tabs = main_content.right_panel_host
    panel = main_content._right_side_tabs
    tabs.expand()

    tabs.collapse()

    assert tabs.state is PanelState.DOCKED
    assert panel.window() is main_content.window()


def test_tracking_repositions_an_expanded_panel(main_content):
    tabs = main_content.right_panel_host
    main_content.resize(1200, 700)
    tabs.expand()

    main_content.update_right_panel_tracking()

    assert tabs.window is not None
    assert tabs.window.width() > 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/main_content_panel_test.py -v'
```

Expected: FAIL with `AttributeError: 'MainContent' object has no attribute 'right_panel_host'`.

- [ ] **Step 3: Add the host and the tracking method to MainContent**

In `tools/acquisition/view/main_content.py`, add to the imports:

```python
from PySide6.QtCore import QRect
from tools.acquisition.view.detachable_panel import DetachablePanelHost, PanelState
```

`QRect` may already be imported from `PySide6.QtCore`; add it to the existing import list rather than duplicating the line.

Immediately after the existing block that ends with

```python
        self._main_splitter.apply_saved_or_default_sizes([1180, 430])
```

insert:

```python
        self._right_panel_host = DetachablePanelHost(
            self._right_side_tabs,
            self._main_splitter,
            1,
            title="Laser Control and Protocol",
            parent=self,
        )
        self._right_panel_host.state_changed.connect(self._remember_right_panel_state)
        self._restore_right_panel_placement()
```

Add these three methods to `MainContent`, next to `protocol_ui_enabled`:

```python
    @property
    def right_panel_host(self) -> DetachablePanelHost:
        return self._right_panel_host

    def update_right_panel_tracking(self) -> None:
        """Keep an expanded panel over this widget as the window moves."""
        self._right_panel_host.track(
            QRect(self.mapToGlobal(self.rect().topLeft()), self.rect().size())
        )

    def _remember_right_panel_state(self, state: str) -> None:
        self._preferences.right_panel_state = state
        if state == PanelState.DETACHED.value:
            window = self._right_panel_host.window
            if window is not None:
                self._preferences.right_panel_detached_geometry = window.geometry()

    def _restore_right_panel_placement(self) -> None:
        """Reopen where the panel was left, falling back to docked."""
        state = self._preferences.right_panel_state
        try:
            if state == PanelState.EXPANDED.value:
                self.update_right_panel_tracking()
                self._right_panel_host.expand()
            elif state == PanelState.DETACHED.value:
                self._right_panel_host.detach(
                    self._preferences.right_panel_detached_geometry
                )
        except Exception:
            logger.exception("Could not restore the right panel placement")
            self._right_panel_host.collapse()
```

In `MainContent.close`, return the panel before the rest of the teardown so the panel is destroyed with its parent rather than orphaned in a live top-level window:

```python
    def close(self):
        self._right_panel_host.collapse()
        self._clear_reach_camera_grid()
        # Ensure the textbox handler is removed from root logger handlers.
        self._diagnostics_content.close()
        super().close()
```

- [ ] **Step 4: Add the expand and detach controls**

In `_create_right_side_tabs`, immediately before `return tabs`, add a corner widget:

```python
        corner = QWidget()
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(0, 0, 4, 0)
        corner_layout.setSpacing(2)
        self._expand_panel_button = QToolButton()
        self._expand_panel_button.setText("Expand")
        self._expand_panel_button.setToolTip(
            "Expand this panel leftward over the other panels"
        )
        self._expand_panel_button.clicked.connect(self._toggle_right_panel_expanded)
        self._detach_panel_button = QToolButton()
        self._detach_panel_button.setText("Detach")
        self._detach_panel_button.setToolTip("Move this panel into its own window")
        self._detach_panel_button.clicked.connect(self._toggle_right_panel_detached)
        corner_layout.addWidget(self._expand_panel_button)
        corner_layout.addWidget(self._detach_panel_button)
        tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
```

Add `QHBoxLayout` and `QToolButton` to the `PySide6.QtWidgets` imports if they are not already there.

Add the two toggles next to the methods from Step 3:

```python
    def _toggle_right_panel_expanded(self) -> None:
        if self._right_panel_host.state is PanelState.EXPANDED:
            self._right_panel_host.collapse()
            return
        self.update_right_panel_tracking()
        self._right_panel_host.expand()

    def _toggle_right_panel_detached(self) -> None:
        if self._right_panel_host.state is PanelState.DETACHED:
            self._right_panel_host.collapse()
            return
        self._right_panel_host.detach(
            self._preferences.right_panel_detached_geometry
        )
```

- [ ] **Step 5: Chain the tracking into the main window's existing handlers**

`MainWindow` already defines both handlers at `main_window.py:967` and `main_window.py:971`. Add one line to each rather than replacing them:

```python
    def moveEvent(self, e):
        self._remember_normal_window_geometry()
        self.main_content.update_right_panel_tracking()
        super(MainWindow, self).moveEvent(e)

    def resizeEvent(self, event):
        self._remember_normal_window_geometry()
        self.main_content.update_right_panel_tracking()
        super().resizeEvent(event)
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/main_content_panel_test.py tests/detachable_panel_test.py tests/user_preferences_panel_test.py -v'
```

Expected: 16 passed.

- [ ] **Step 7: Run the surrounding suites for regressions**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/ -q'
```

Expected: no new failures against the branch baseline. `camera_discovery_test.py` (2), `signal_stream_ui_test.py` (2) and a timing-flaky `video_capture_record_test.py` case are known pre-existing failures recorded in `planning.md`; anything else is a regression from this task.

- [ ] **Step 8: Verify on the rig by eye**

Launch the application on the rig and confirm three things that no offscreen test can prove:

1. The expanded panel draws **above** the OpenGL camera views rather than behind them.
2. Moving and resizing the main window keeps the expanded panel aligned without visible lag or flicker.
3. A detached panel survives an application restart and reopens on the same monitor.

If the `Qt.Tool` window misbehaves under the rig's window manager, the documented fallback is to use `Qt.WindowType.Window` for the expanded state too, accepting a title bar. Do not fall back to a raised child widget.

- [ ] **Step 9: Commit**

```bash
git add tools/acquisition/view/main_content.py tools/acquisition/view/main_window.py tests/main_content_panel_test.py
git commit -m "feat(view): expand and detach the laser and protocol panel"
```

- [ ] **Step 10: Record the decision and cherry-pick**

Add a short entry to `planning.md` covering why both undocked states are top-level windows (the `QOpenGLWidget` stacking constraint), then:

```bash
git add planning.md && git commit -m "docs(planning): record the detachable panel placement decision"
```

Cherry-pick the task's commits onto `demo-mode-and-presentation` per the branch workflow, then run `tests/detachable_panel_test.py` there to confirm the pick is sound.
