"""Tests for the Tone 1 to Tone 2 cue pair running inside TrialActionExecutor.

Three behaviours carry the risk:

  * Tone 2 must not fire when the gate is blocked at the deadline. A cue
    delivered into a missing pellet or an ongoing reach is a corrupted trial,
    not a late one.
  * An unlocked trial may wait a block out, but not forever, or a stuck gate
    would hold the pellet cycle open indefinitely.
  * A pending cue must be cancelled when the trial ends, or a timer fires a
    tone into the next trial.

The timer and clock are injected so these run instantly and deterministically.
"""

import pytest

from tools.acquisition.model.cue_timing import CueTimingConfiguration
from tools.acquisition.model.reach_state_source import (
    ReachStateConfiguration,
    ReachStateResolver,
    ReachStateSource,
)
from tools.acquisition.model.trial_action import (
    ToneProfile,
    TrialActionCompiler,
    TrialActionExecutor,
    TrialCompileContext,
)
from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow


class _ManualTimer:
    """A cue timer that fires only when the test says so."""

    fired_deadlines = []

    def __init__(self):
        self._callback = None
        self._deadline = None
        self.cancelled = False

    def schedule(self, deadline_perf_time, callback):
        self._deadline = deadline_perf_time
        self._callback = callback
        type(self).fired_deadlines.append(deadline_perf_time)

    def cancel(self):
        self.cancelled = True
        pending, self._callback = self._callback is not None, None
        return pending

    def join(self, timeout=None):
        return None

    def fire(self, at=None, lateness=0.0):
        callback, self._callback = self._callback, None
        assert callback is not None, "timer was not scheduled"
        callback(self._deadline if at is None else at, lateness)


class _Harness:
    def __init__(self, **executor_kwargs):
        self.tones = []
        _ManualTimer.fired_deadlines = []
        self.timers = []

        def make_timer():
            timer = _ManualTimer()
            self.timers.append(timer)
            return timer

        self.now = 100.0
        self.executor = TrialActionExecutor(
            move_absolute=lambda target: None,
            configure_cover=lambda policy, recipe: None,
            play_tone=lambda profile, phase: self.tones.append((profile.profile_id, phase)),
            prepare_laser=lambda profile, recipe: None,
            cancel_laser=lambda handle: None,
            cue_timer_factory=make_timer,
            clock=lambda: self.now,
            **executor_kwargs,
        )

    @property
    def timer(self):
        return self.timers[-1]


def _recipe(cue_interval_ms=400, cue_tone=True, tone_phase="pellet_presentation"):
    compiler = TrialActionCompiler(
        tone_profiles={
            "tone1": ToneProfile("tone1", 1, 6000, 100),
            "tone2": ToneProfile("tone2", 1, 9000, 100),
        },
        laser_profiles={},
        dcs_to_motor=lambda values: tuple(values),
    )
    updates = {
        "enabled": True,
        "position_mode": "fixed_manual",
        "position_lane": "center",
        "tone_profile_id": "tone1",
        "tone_phase": tone_phase,
    }
    if cue_tone:
        updates["cue_tone_profile_id"] = "tone2"
        updates["cue_interval_fixed_ms"] = cue_interval_ms
    row = TrialProtocolRow(trial_id=1).with_updates(updates)
    context = TrialCompileContext(
        session_id="s1", session_generation=1, protocol_id="p", protocol_revision=1,
        logical_trial_id=1, attempt_id=1, session_seed=7,
        animal_base_dcs=(0.0, 0.0, 0.0),
        lane_offsets_dcs={"center": (0.0, 0.0, 0.0)},
    )
    return compiler.compile(row, context)


SESSION_GENERATION = 1


def _prepare(harness, recipe):
    operation = harness.executor.prepare(recipe)
    harness.executor.bind_send(recipe.operation_id, SESSION_GENERATION, "ctx")
    harness.executor.acknowledge_presentation("ctx")
    return operation


def _details(operation):
    return [item["detail"] for item in operation.to_record()["observations"]]


def test_a_recipe_without_a_cue_pair_arms_nothing():
    harness = _Harness()
    recipe = _recipe(cue_tone=False)
    _prepare(harness, recipe)
    assert harness.timers == []
    assert ("tone2", "tone_2") not in harness.tones


def test_tone_2_fires_at_the_deadline_when_the_gate_is_clear():
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))
    operation = _prepare(harness, _recipe(cue_interval_ms=400))

    assert harness.timer is not None, "the cue pair should have armed"
    harness.now = 100.4
    harness.timer.fire(at=100.4, lateness=0.0004)

    assert ("tone2", "tone_2") in harness.tones
    assert any("Tone 2 acknowledged" in detail for detail in _details(operation))


def test_the_deadline_is_tone_1_plus_the_drawn_interval():
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))
    _prepare(harness, _recipe(cue_interval_ms=500))
    assert _ManualTimer.fired_deadlines[0] == pytest.approx(100.5)


def test_the_achieved_lateness_is_recorded_not_assumed():
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))
    operation = _prepare(harness, _recipe())
    harness.now = 100.4
    harness.timer.fire(at=100.4, lateness=0.0123)

    recorded = [d for d in _details(operation) if "Tone 2 acknowledged" in d]
    assert recorded and "12.300 ms" in recorded[0]


def test_the_transport_cost_is_measured_separately_from_scheduling():
    """The CAN round trip lands after the timer and can dominate the error.

    Reporting only the scheduling lateness would make a 40 ms transport look
    like a sub-microsecond cue.
    """
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))

    def slow_send(profile, phase):
        if phase == "tone_2":
            harness.now += 0.040  # the board acknowledged 40 ms later
        harness.tones.append((profile.profile_id, phase))

    harness.executor._play_tone = slow_send
    operation = _prepare(harness, _recipe())
    harness.now = 100.4
    harness.timer.fire(at=100.4, lateness=0.0005)

    recorded = [d for d in _details(operation) if "Tone 2 acknowledged" in d]
    assert recorded, "the delivery must be recorded"
    assert "transport 40.000 ms" in recorded[0]
    assert "scheduling 0.500 ms" in recorded[0]
    assert "40.500 ms after its deadline" in recorded[0], "the total is what matters"


def test_a_missing_pellet_at_the_deadline_skips_and_resets():
    """A locked trial must not deliver a cue into a missing pellet."""
    harness = _Harness(pellet_presence_provider=lambda: (False, 100.0))
    operation = _prepare(harness, _recipe())
    harness.now = 100.4
    harness.timer.fire(at=100.4)

    assert ("tone2", "tone_2") not in harness.tones
    assert any("skipped and reset" in detail for detail in _details(operation))


def test_an_active_reach_at_the_deadline_skips_and_resets():
    resolver = ReachStateResolver(
        ReachStateConfiguration(source=ReachStateSource.LIVE_TRACKING),
        live_tracking_provider=lambda: (True, 100.39),
    )
    harness = _Harness(
        pellet_presence_provider=lambda: (True, 100.39),
        reach_state_resolver=resolver,
    )
    operation = _prepare(harness, _recipe())
    harness.now = 100.4
    harness.timer.fire(at=100.4)

    assert ("tone2", "tone_2") not in harness.tones
    assert any("skipped and reset" in detail for detail in _details(operation))


def test_stale_evidence_is_judged_by_the_oldest_observation():
    """Fresh reach state must not disguise a stale presence observation."""
    resolver = ReachStateResolver(
        ReachStateConfiguration(source=ReachStateSource.LIVE_TRACKING),
        live_tracking_provider=lambda: (False, 100.39),
    )
    harness = _Harness(
        # Presence is 1.4 s old; the gate allows 500 ms.
        pellet_presence_provider=lambda: (True, 99.0),
        reach_state_resolver=resolver,
        cue_timing_configuration=CueTimingConfiguration(gate_staleness_ms=500),
    )
    operation = _prepare(harness, _recipe())
    harness.now = 100.4
    harness.timer.fire(at=100.4)

    assert ("tone2", "tone_2") not in harness.tones
    assert any("stale" in detail for detail in _details(operation))


def test_a_reach_observation_past_its_own_limit_is_dropped_not_trusted():
    """The resolver filters first, so old reach state never reaches the gate.

    It must not be passed through as "not reaching" either, which would be a
    silent false clear.
    """
    resolver = ReachStateResolver(
        ReachStateConfiguration(source=ReachStateSource.LIVE_TRACKING, max_age_ms=50),
        live_tracking_provider=lambda: (True, 100.0),
    )
    harness = _Harness(
        pellet_presence_provider=lambda: (True, 100.39),
        reach_state_resolver=resolver,
    )
    _prepare(harness, _recipe())
    gate = harness.executor._cue_gate_state(100.4)

    assert gate.reach_active is False
    assert gate.observed_at == pytest.approx(100.39), (
        "a dropped reach observation must not drag the gate age backwards"
    )


def test_an_unlocked_trial_waits_for_the_block_then_fires():
    presence = {"value": (False, 100.39)}
    harness = _Harness(
        pellet_presence_provider=lambda: presence["value"],
        cue_timing_configuration=CueTimingConfiguration(lock_timing=False),
    )
    _prepare(harness, _recipe())

    harness.now = 100.4
    harness.timer.fire(at=100.4)
    assert ("tone2", "tone_2") not in harness.tones, "still blocked"
    assert len(harness.timers) == 2, "the wait should have re-armed"

    presence["value"] = (True, 100.45)
    harness.now = 100.45
    harness.timer.fire(at=100.45)
    assert ("tone2", "tone_2") in harness.tones


def test_an_unlocked_wait_is_bounded():
    """A stuck gate must not hold the pellet cycle open indefinitely."""
    harness = _Harness(
        pellet_presence_provider=lambda: (False, 200.0),
        cue_timing_configuration=CueTimingConfiguration(lock_timing=False),
        cue_max_wait_seconds=1.0,
    )
    operation = _prepare(harness, _recipe())

    harness.now = 200.0
    harness.timer.fire(at=200.0)

    assert ("tone2", "tone_2") not in harness.tones
    assert any("abandoned" in detail for detail in _details(operation))


def test_completing_the_trial_cancels_a_pending_cue():
    """A stray timer must not fire a tone into the next trial."""
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))
    _prepare(harness, _recipe())
    timer = harness.timer

    harness.executor.complete("done")

    assert timer.cancelled is True


def test_cancelling_the_trial_cancels_a_pending_cue():
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))
    operation = _prepare(harness, _recipe())
    timer = harness.timer

    harness.executor.cancel(generation=SESSION_GENERATION, reason="operator stop")

    assert timer.cancelled is True
    assert any("Tone 2 cancelled" in detail for detail in _details(operation))


def test_failing_the_trial_cancels_a_pending_cue():
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))
    _prepare(harness, _recipe())
    timer = harness.timer

    harness.executor.fail(RuntimeError("CAN dropped"))

    assert timer.cancelled is True


def test_an_embedded_sequence_is_not_host_timed():
    """The board owns that sequence; a host timer would describe a cue it does
    not deliver."""
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))
    recipe = _recipe(tone_phase="embedded_in_sequence")
    operation = harness.executor.prepare(recipe)
    harness.executor.bind_send(recipe.operation_id, SESSION_GENERATION, "ctx")
    harness.executor.acknowledge_presentation("ctx")

    assert harness.timers == []


def test_a_failing_tone_send_is_recorded_and_does_not_raise():
    harness = _Harness(pellet_presence_provider=lambda: (True, 100.0))

    def explode(profile, phase):
        if phase == "tone_2":
            raise RuntimeError("CAN timeout")
        harness.tones.append((profile.profile_id, phase))

    harness.executor._play_tone = explode
    operation = _prepare(harness, _recipe())
    harness.now = 100.4
    harness.timer.fire(at=100.4)

    assert any("Tone 2 send failed after" in detail for detail in _details(operation))
