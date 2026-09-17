# Laser Stim Bench Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fire a saved laser profile on the bench through the same hardware-triggered path a trial uses — the board emits the STIM3 pulse, which hardware-triggers an already-armed NI analog output — without running a recording session.

**Architecture:** The path already exists and is reused rather than reimplemented. `LaserModel.prepare_pulse_profile` arms a `LaserSynchronizedPulseTrain` with `trigger_source=profile.trigger_terminal`, and `AppModel._trigger_protocol_stim3` fires the board through `hardware.pulse_stim3`. The new work is a bench recipe, a set of refusal guards, timing measurement, and one row of UI per laser channel. The guards are a pure function so they are testable without an armed laser.

**Tech Stack:** Python 3.8, PySide6, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-protocol-panel-sets-stim-test-design.md`

**Depends on:** nothing in the other two plans. Sequenced third by preference, not by necessity.

## Global Constraints

- **Python 3.8.20.** The rig runs `~/anaconda3/envs/reachaq/bin/python`, which is 3.8.20, even though `pyproject.toml` says `requires-python = ">= 3.10"`. Write 3.8-compatible code: `from __future__ import annotations`, `typing.Optional` / `typing.Tuple` rather than `X | Y` and `tuple[...]`, no `match` statements.
- **All test runs, suites and app launches happen on `christielab10`.** Never run pytest locally.
- Rig SSH: `ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10`. Repo at `~/Documents/reachAQ`.
- **Never kill rig processes by pattern-matching the interpreter path.** Kill by explicit PID.
- Branch `feature-dev`, worked in the worktree `../reachAQ-feature`.
- **Only physical STIM3 has a firmware finite pulse.** `CanInterface.pulse_digital_output` rejects anything but `DigitalOutputs.STIMULUS_4`. Do not add a line-selection parameter in this plan; nothing can honour it yet. STIM2 arrives with the firmware follow-up.
- **This test fires a laser.** Every guard in Task 1 is a safety requirement, not a nicety. A refused test must never leave an armed analog output behind.
- Do not modify `autotrainer.device`. This plan calls its existing API only.

---

### Task 1: Bench recipe and refusal guards

Pure logic, no hardware, no Qt. Everything that decides whether a test may run lives here so it can be tested exhaustively.

**Files:**
- Create: `tools/acquisition/model/stim_bench_test.py`
- Test: `tests/stim_bench_test_test.py`

**Interfaces:**
- Consumes: `LaserTriggerRoute` from `trial_protocol_schedule`; `LaserPulseProfile` from `trial_action` (fields used: `profile_id`, `revision`, `channel_id`, `trigger_route`, `trigger_terminal`, `trigger_pulse_us`).
- Produces:
  - `BenchRecipe` — frozen dataclass with `operation_id`, `session_id`, `session_generation`, `protocol_id`, `protocol_revision`, `logical_trial_id`, `attempt_id`; the shape `LaserModel.prepare_pulse_profile` reads for its provenance context.
  - `StimTestResult(profile_id: str, channel_id: int, trigger_terminal: str, trigger_pulse_us: int, arm_to_terminal_ms: Optional[float], detail: str)`
  - `refuse_reason(*, profile, recording_status_value: str, trial_operation_active: bool, laser_backend: str, configured_channel_ids, firmware_capabilities) -> Optional[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/stim_bench_test_test.py`:

```python
import pytest

from tools.acquisition.model.stim_bench_test import (
    BenchRecipe,
    StimTestResult,
    refuse_reason,
)
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


def make_profile(**overrides):
    values = dict(
        profile_id="stim-a",
        revision=1,
        channel_id=1,
        amplitude_volts=2.0,
        pulse_duration_ms=5.0,
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
    )
    values.update(overrides)
    return LaserPulseProfile(**values)


def allowed(**overrides):
    values = dict(
        profile=make_profile(),
        recording_status_value="ready",
        trial_operation_active=False,
        laser_backend="nidaq",
        configured_channel_ids=(1, 2),
        firmware_capabilities=("finite_stim3_pulse",),
    )
    values.update(overrides)
    return values


def test_a_fully_configured_bench_test_is_allowed():
    assert refuse_reason(**allowed()) is None


def test_a_recording_session_refuses_the_test():
    reason = refuse_reason(**allowed(recording_status_value="recording"))

    assert "recording" in reason.lower()


def test_an_active_trial_operation_refuses_the_test():
    reason = refuse_reason(**allowed(trial_operation_active=True))

    assert "trial" in reason.lower()


def test_a_non_nidaq_laser_backend_refuses_the_test():
    reason = refuse_reason(**allowed(laser_backend="null"))

    assert "nidaq" in reason.lower()


def test_an_unconfigured_channel_refuses_the_test():
    reason = refuse_reason(**allowed(configured_channel_ids=(2, 3)))

    assert "channel 1" in reason.lower()


def test_missing_firmware_capability_refuses_the_test():
    reason = refuse_reason(**allowed(firmware_capabilities=()))

    assert "finite_stim3_pulse" in reason


def test_a_direct_software_profile_refuses_the_test():
    reason = refuse_reason(
        **allowed(
            profile=make_profile(
                trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE,
                trigger_terminal="",
            )
        )
    )

    assert "hardware" in reason.lower()


def test_a_missing_profile_refuses_the_test():
    reason = refuse_reason(**allowed(profile=None))

    assert "profile" in reason.lower()


def test_the_bench_recipe_never_claims_a_session_identity():
    recipe = BenchRecipe()

    assert recipe.session_generation == 0
    assert recipe.logical_trial_id == 0
    assert "bench" in recipe.protocol_id


def test_each_bench_recipe_gets_its_own_operation_id():
    assert BenchRecipe().operation_id != BenchRecipe().operation_id


def test_the_result_carries_what_the_operator_needs_to_read():
    result = StimTestResult(
        profile_id="stim-a",
        channel_id=1,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
        arm_to_terminal_ms=3.5,
        detail="completed",
    )

    assert "stim-a" in str(result)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/stim_bench_test_test.py -v'
```

Expected: collection error, `ModuleNotFoundError: No module named 'tools.acquisition.model.stim_bench_test'`.

- [ ] **Step 3: Write the implementation**

Create `tools/acquisition/model/stim_bench_test.py`:

```python
"""Fire one saved laser profile on the bench, off any recording session.

The board emits its firmware-timed STIM3 pulse into an analog output that has
already been armed on that trigger terminal, which is the same route a trial
takes. Nothing here starts a session or touches session evidence.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Iterable, Optional, Sequence

from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


def _bench_operation_id() -> str:
    return "bench-{}".format(uuid.uuid4().hex[:12])


@dataclasses.dataclass(frozen=True)
class BenchRecipe:
    """The shape prepare_pulse_profile reads, identifying itself as a bench run.

    prepare_pulse_profile uses the recipe only to build the operation's
    provenance context, so a recipe that claims no session and no trial is
    honest rather than a stub: this output genuinely belongs to no session.
    """

    operation_id: str = dataclasses.field(default_factory=_bench_operation_id)
    session_id: str = "bench"
    session_generation: int = 0
    protocol_id: str = "stim-bench-test"
    protocol_revision: int = 1
    logical_trial_id: int = 0
    attempt_id: int = 0


@dataclasses.dataclass(frozen=True)
class StimTestResult:
    profile_id: str
    channel_id: int
    trigger_terminal: str
    trigger_pulse_us: int
    arm_to_terminal_ms: Optional[float]
    detail: str = ""

    def __str__(self) -> str:
        if self.arm_to_terminal_ms is None:
            return "Stim test {} on laser {}: {}".format(
                self.profile_id, self.channel_id, self.detail or "completed"
            )
        return (
            "Stim test {} on laser {}: board pulse {} us on {}, "
            "arm to terminal {:.2f} ms".format(
                self.profile_id,
                self.channel_id,
                self.trigger_pulse_us,
                self.trigger_terminal,
                self.arm_to_terminal_ms,
            )
        )


def refuse_reason(
    *,
    profile,
    recording_status_value: str,
    trial_operation_active: bool,
    laser_backend: str,
    configured_channel_ids: Sequence[int],
    firmware_capabilities: Iterable[str],
) -> Optional[str]:
    """Return why this bench test must not run, or None when it may.

    Every branch here is a safety gate. A bench test drives a real laser, so it
    refuses rather than guesses whenever the rig is not in a state where an
    unexpected pulse is harmless.
    """
    if profile is None:
        return "Select a saved laser profile to test."
    if str(recording_status_value).lower() == "recording":
        return "Stim test is refused while a session is recording."
    if trial_operation_active:
        return "Stim test is refused while a trial operation is prepared or active."
    if str(laser_backend) != "nidaq":
        return (
            "Stim test needs the nidaq laser backend; this rig is configured "
            "for {!r}.".format(laser_backend)
        )
    if profile.trigger_route is not LaserTriggerRoute.HARDWARE_STIM3:
        return (
            "Profile {} uses the {} route; the bench test drives the hardware "
            "STIM3 route only.".format(
                profile.profile_id, profile.trigger_route.value
            )
        )
    if not profile.trigger_terminal:
        return (
            "Profile {} has no NI trigger terminal, so the board pulse has "
            "nothing to trigger.".format(profile.profile_id)
        )
    if int(profile.channel_id) not in {int(item) for item in configured_channel_ids}:
        return "Laser channel {} has no hardware mapping.".format(
            int(profile.channel_id)
        )
    if "finite_stim3_pulse" not in set(firmware_capabilities):
        return (
            "The pellet firmware does not report finite_stim3_pulse, so it "
            "cannot emit a timed board trigger."
        )
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/stim_bench_test_test.py -v'
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/model/stim_bench_test.py tests/stim_bench_test_test.py
git commit -m "feat(laser): define the bench stim test recipe and its safety gates"
```

---

### Task 2: Run the test from the app model

**Files:**
- Modify: `tools/acquisition/model/app_model.py` — add the method beside `_trigger_protocol_stim3`
- Test: `tests/stim_bench_run_test.py`

**Interfaces:**
- Consumes: `BenchRecipe`, `StimTestResult`, `refuse_reason` from Task 1; `LaserModel.prepare_pulse_profile`, `LaserModel.release_prepared_profile`, `HardwareModel.pulse_stim3`, `HardwareModel.wait_pending_command_acked` — all existing.
- Produces: `AppModel.run_stim_bench_test(profile_id: str) -> StimTestResult`, raising `RuntimeError` with the refusal text when a guard refuses.

  The channel is taken from the profile rather than passed in, so the caller cannot fire profile A's waveform at channel B.

- [ ] **Step 1: Write the failing test**

Create `tests/stim_bench_run_test.py`. It replaces the laser and hardware collaborators on the shared `app_model` fixture with recorders, following the double-based style of `tests/laser_model_test.py`:

```python
import threading
from types import SimpleNamespace

import pytest

from tools.acquisition.model.stim_bench_test import StimTestResult
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


class FakeOperation:
    def __init__(self):
        self._callbacks = []
        self.cancelled = False

    def add_terminal_callback(self, callback):
        self._callbacks.append(callback)

    def finish(self):
        for callback in tuple(self._callbacks):
            callback(self)

    def cancel(self):
        self.cancelled = True
        return True


class FakeLaser:
    def __init__(self, operation, backend="nidaq", channel_ids=(1,)):
        self.operation = operation
        self.prepared = []
        self.released = []
        self.configuration = SimpleNamespace(
            backend=backend,
            channels=tuple(
                SimpleNamespace(channel_id=item) for item in channel_ids
            ),
        )

    def prepare_pulse_profile(self, profile, recipe):
        self.prepared.append((profile, recipe))
        return self.operation

    def release_prepared_profile(self, operation):
        self.released.append(operation)


class FakeHardware:
    def __init__(self, operation):
        self.operation = operation
        self.pulses = []

    def pulse_stim3(self, duration_us):
        self.pulses.append(duration_us)
        # The board acknowledges, and the armed waveform runs to completion.
        threading.Timer(0.0, self.operation.finish).start()
        return "token"

    def wait_pending_command_acked(self, token, timeout=None):
        return True


def make_profile(**overrides):
    values = dict(
        profile_id="stim-a",
        revision=1,
        channel_id=1,
        amplitude_volts=2.0,
        pulse_duration_ms=5.0,
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
    )
    values.update(overrides)
    return LaserPulseProfile(**values)


@pytest.fixture
def bench(app_model):
    operation = FakeOperation()
    laser = FakeLaser(operation)
    hardware = FakeHardware(operation)
    app_model._laser = laser
    app_model._hardware = hardware
    app_model._laser_profiles = {"stim-a": make_profile()}
    return app_model, laser, hardware, operation


def test_a_bench_test_arms_the_output_before_pulsing_the_board(bench):
    model, laser, hardware, _operation = bench

    result = model.run_stim_bench_test("stim-a")

    assert laser.prepared, "the analog output must be armed first"
    assert hardware.pulses == [1000]
    assert isinstance(result, StimTestResult)
    assert result.arm_to_terminal_ms is not None


def test_the_bench_recipe_claims_no_session(bench):
    model, laser, _hardware, _operation = bench

    model.run_stim_bench_test("stim-a")

    _profile, recipe = laser.prepared[0]
    assert recipe.session_generation == 0
    assert recipe.session_id == "bench"


def test_an_unknown_profile_is_refused(bench):
    model, _laser, hardware, _operation = bench

    with pytest.raises(RuntimeError, match="profile"):
        model.run_stim_bench_test("missing")

    assert hardware.pulses == []


def test_a_board_failure_cancels_the_armed_output(bench):
    model, laser, hardware, operation = bench

    def fail(_duration_us):
        raise RuntimeError("bus down")

    hardware.pulse_stim3 = fail

    with pytest.raises(RuntimeError, match="bus down"):
        model.run_stim_bench_test("stim-a")

    assert operation.cancelled, "a failed test must not leave the output armed"
    assert laser.released == [operation]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/stim_bench_run_test.py -v'
```

Expected: FAIL with `AttributeError: 'AppModel' object has no attribute 'run_stim_bench_test'`.

- [ ] **Step 3: Write the implementation**

In `tools/acquisition/model/app_model.py`, add to the imports:

```python
from tools.acquisition.model.stim_bench_test import (
    BenchRecipe,
    StimTestResult,
    refuse_reason,
)
```

Add this method immediately after `_trigger_protocol_stim3`:

```python
    def run_stim_bench_test(self, profile_id: str) -> StimTestResult:
        """Fire one saved laser profile through the real hardware trigger.

        Arms the analog output on the profile's trigger terminal, then asks the
        board for its firmware-timed STIM3 pulse, which starts the waveform.
        This is the route a trial takes, exercised without a session.
        """
        profile = self._laser_profiles.get(str(profile_id))
        operation = self._trial_action_executor.operation
        refusal = refuse_reason(
            profile=profile,
            recording_status_value=self.session_recording_status.value,
            trial_operation_active=operation is not None,
            laser_backend=self._laser.configuration.backend,
            configured_channel_ids=tuple(
                int(channel.channel_id)
                for channel in self._laser.configuration.channels
            ),
            firmware_capabilities=self._hardware.firmware_compatibility.get(
                "reported_capabilities", ()
            ),
        )
        if refusal is not None:
            raise RuntimeError(refusal)

        finished = threading.Event()
        prepared = self._laser.prepare_pulse_profile(profile, BenchRecipe())
        add_terminal_callback = getattr(prepared, "add_terminal_callback", None)
        if add_terminal_callback is not None:
            add_terminal_callback(lambda _operation: finished.set())
        started = time.perf_counter()
        try:
            token = self._hardware.pulse_stim3(int(profile.trigger_pulse_us))
            if token is None:
                raise RuntimeError("Firmware STIM3 pulse was not queued")
            timeout = max(3.0, profile.trigger_pulse_us / 1e6 + 2.0)
            self._hardware.wait_pending_command_acked(token, timeout=timeout)
            completed = finished.wait(timeout)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        except Exception:
            # Never leave an armed analog output behind on a failed test.
            cancel = getattr(prepared, "cancel", None)
            if cancel is not None:
                cancel()
            raise
        finally:
            self._laser.release_prepared_profile(prepared)

        return StimTestResult(
            profile_id=profile.profile_id,
            channel_id=int(profile.channel_id),
            trigger_terminal=profile.trigger_terminal,
            trigger_pulse_us=int(profile.trigger_pulse_us),
            arm_to_terminal_ms=elapsed_ms if completed else None,
            detail=(
                "completed"
                if completed
                else "board acknowledged but the waveform did not report terminal"
            ),
        )
```

`threading` and `time` are already imported in this module.

- [ ] **Step 4: Confirm the capability source**

No new accessor is needed. `HardwareModel.firmware_compatibility` (`hardware_model.py:295`) already returns `FirmwareCompatibility.to_record()`, a dict whose `reported_capabilities` key holds the capabilities the connected board advertises. The call in Step 3 reads it directly, so there is exactly one source of truth for firmware capability.

Add a test pinning that contract, so a change to the record shape fails here rather than silently disabling the guard. Append to `tests/stim_bench_run_test.py`:

```python
def test_the_capability_guard_reads_the_hardware_model_record(bench):
    model, _laser, hardware, _operation = bench
    hardware.firmware_compatibility = {"reported_capabilities": []}

    with pytest.raises(RuntimeError, match="finite_stim3_pulse"):
        model.run_stim_bench_test("stim-a")
```

and give `FakeHardware` the attribute the real model exposes, in its `__init__`:

```python
        self.firmware_compatibility = {
            "reported_capabilities": ["finite_stim3_pulse"]
        }
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/stim_bench_run_test.py tests/stim_bench_test_test.py -v'
```

Expected: 16 passed.

- [ ] **Step 6: Commit**

```bash
git add tools/acquisition/model/app_model.py tests/stim_bench_run_test.py
git commit -m "feat(laser): run a saved profile through the board trigger on the bench"
```

---

### Task 3: The per-channel test control

**Files:**
- Modify: `tools/acquisition/view/laser_control_content.py` — `_LaserChannelTab.__init__` (line 78) and a new method beside `_run_pulse` (line 746)
- Modify: `todo.md`
- Test: `tests/laser_stim_test_ui_test.py`

**Interfaces:**
- Consumes: `AppModel.run_stim_bench_test`, `StimTestResult` from Task 2; the existing `self._start_operation` and `self._set_parent_status` callbacks already passed into `_LaserChannelTab`.
- Produces: `_LaserChannelTab.stim_profile_selector` (`QComboBox`), `_LaserChannelTab.stim_test_button` (`QPushButton`), `_LaserChannelTab._run_stim_test()`, `_LaserChannelTab.refresh_stim_profiles()`.

- [ ] **Step 1: Write the failing test**

Create `tests/laser_stim_test_ui_test.py`:

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
)
from tools.acquisition.view.laser_control_content import (  # noqa: E402
    _LaserChannelTab,
)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def channel_tab(qapp, app_model):
    channel = LaserChannelConfiguration(
        channel_id=LaserChannelId.LASER_1,
        analog_output="/Dev1/ao0",
        diode_input="/Dev1/ai0",
        shutter_output="/Dev1/port0/line0",
    )
    started = []
    statuses = []
    tab = _LaserChannelTab(
        app_model,
        channel,
        True,
        None,
        lambda status, operation: started.append((status, operation)),
        lambda message, is_error: statuses.append((message, is_error)),
    )
    return tab, started, statuses


def test_the_channel_tab_offers_a_stim_test_control(channel_tab):
    tab, _started, _statuses = channel_tab

    assert tab.stim_test_button.text() == "Test stim (hardware trigger)"


def test_running_a_stim_test_with_no_profile_selected_reports_rather_than_fires(
    channel_tab,
):
    tab, started, statuses = channel_tab
    tab.stim_profile_selector.clear()

    tab._run_stim_test()

    assert started == []
    assert statuses
    assert statuses[-1][1] is True


def test_running_a_stim_test_starts_a_background_operation(channel_tab, app_model):
    tab, started, _statuses = channel_tab
    tab.stim_profile_selector.clear()
    tab.stim_profile_selector.addItem("stim-a", "stim-a")
    tab.stim_profile_selector.setCurrentIndex(0)
    calls = []
    app_model.run_stim_bench_test = lambda profile_id: calls.append(profile_id)

    tab._run_stim_test()

    assert len(started) == 1
    started[0][1]()
    assert calls == ["stim-a"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/laser_stim_test_ui_test.py -v'
```

Expected: FAIL with `AttributeError: '_LaserChannelTab' object has no attribute 'stim_test_button'`.

- [ ] **Step 3: Add the control**

In `_LaserChannelTab.__init__`, in the same area that builds the pulse controls, add:

```python
        stim_row = QHBoxLayout()
        stim_row.addWidget(self._form_label("Stim profile"))
        self.stim_profile_selector = QComboBox()
        self.stim_profile_selector.setToolTip(
            "Saved laser profiles that target this channel"
        )
        stim_row.addWidget(self.stim_profile_selector, stretch=1)
        self.stim_test_button = QPushButton("Test stim (hardware trigger)")
        self.stim_test_button.setToolTip(
            "Arm this channel's analog output on its trigger terminal, then ask "
            "the board for its timed STIM3 pulse to start the waveform"
        )
        self.stim_test_button.clicked.connect(self._run_stim_test)
        stim_row.addWidget(self.stim_test_button)
        self.stim_test_result = QLabel()
        self.stim_test_result.setWordWrap(True)
        self.stim_test_result.setObjectName("LaserPreviewStatus")
```

Add `stim_row` and `self.stim_test_result` to the tab's layout where the pulse controls are assembled, then call `self.refresh_stim_profiles()` at the end of `__init__`.

Add `QComboBox` and `QPushButton` to the `PySide6.QtWidgets` imports if they are not already there.

- [ ] **Step 4: Add the profile refresh and the run handler**

Add both methods beside `_run_pulse`:

```python
    def refresh_stim_profiles(self) -> None:
        """List saved laser profiles that target this channel."""
        previous = self.stim_profile_selector.currentData()
        self.stim_profile_selector.blockSignals(True)
        self.stim_profile_selector.clear()
        state = self._app_model.trial_protocol_state
        for item in state.get("laser_profiles", ()):
            summary = item.get("summary", "")
            if "channel {}".format(self._channel.channel_id.value) not in summary:
                continue
            self.stim_profile_selector.addItem(
                "{} ({})".format(item["profile_id"], summary), item["profile_id"]
            )
        self.stim_profile_selector.blockSignals(False)
        if previous is not None:
            index = self.stim_profile_selector.findData(previous)
            if index >= 0:
                self.stim_profile_selector.setCurrentIndex(index)

    def _run_stim_test(self) -> None:
        profile_id = self.stim_profile_selector.currentData()
        if not profile_id:
            self._set_parent_status(
                "Select a saved laser profile for laser {} first".format(
                    self._channel.channel_id.value
                ),
                True,
            )
            return

        def operation():
            result = self._app_model.run_stim_bench_test(profile_id)
            return str(result)

        self._start_operation(
            "Running stim test {} on laser {}".format(
                profile_id, self._channel.channel_id.value
            ),
            operation,
        )
```

`_start_operation` already moves the work onto a `QThread` and reports through `_operation_finished` / `_operation_failed`, and `set_controls_enabled` already disables the tab's controls during capture, so no new threading or gating is needed.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/laser_stim_test_ui_test.py -v'
```

Expected: 3 passed.

- [ ] **Step 6: Run the laser and full suites for regressions**

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/reachaq_debug christielab10@christielab10 'cd ~/Documents/reachAQ && ~/anaconda3/envs/reachaq/bin/python -m pytest tests/laser_model_test.py tests/signal_stream_ui_test.py tests/ -q'
```

Expected: no new failures against the branch baseline. The known pre-existing failures are the two in `camera_discovery_test.py`, the two in `signal_stream_ui_test.py`, and a timing-flaky case in `video_capture_record_test.py`.

- [ ] **Step 7: Verify on the bench**

This is the step the whole feature exists for, and no test can substitute for it. On the rig, with the laser command line on a scope:

1. Confirm the waveform starts on the **board trigger edge**, not when the button is pressed. Trigger the scope on the STIM3 line and confirm the analog output rises within the expected latency of that edge.
2. Confirm the reported arm-to-terminal time is consistent across repeated presses.
3. Confirm the test refuses while a session is recording, and that refusing leaves no armed output — a subsequent normal trial must still arm and fire correctly.
4. Confirm a profile with `direct_ni_software` is refused with a clear message.

Record the measured latency in `planning.md`.

- [ ] **Step 8: Record the firmware follow-up**

Add to `todo.md`:

```markdown
- Extend the firmware finite stim pulse to a second line so two lasers can be
  driven with board timing in one session. STIM0 and STIM1 are unavailable:
  the pellet board device tree assigns gpiob 11 and gpiob 12 to the tone
  generator as the Tone 1 / Tone 2 TTL confirmations the NI-DAQ records, and
  tone_generator.c drives them directly. The second line is therefore STIM2
  (gpiob 13). Work spans two repositories: in reachAQ-hardware, extend the
  GPIOPulse command beyond STIM3, bump the pellet firmware version and
  advertise a new capability; in reachAQ, add that capability to
  CAPABILITY_BITS in tools/acquisition/model/firmware_compatibility.py, add the
  version to config/pellet-firmware-compatibility.yaml, relax the
  STIMULUS_4-only check in CanInterface.pulse_digital_output and
  emulation_interface.py behind the capability, and add line selection to the
  laser profile and the bench test. Then flash and verify on the rig. The
  physical BNC-to-STIM mapping still needs confirming from the schematic
  before the line is wired to a connector.
```

- [ ] **Step 9: Commit and cherry-pick**

```bash
git add tools/acquisition/view/laser_control_content.py tests/laser_stim_test_ui_test.py todo.md planning.md
git commit -m "feat(laser): add a per-channel hardware-triggered stim test"
```

Cherry-pick this feature's commits onto `demo-mode-and-presentation` and run the three new test files there.
