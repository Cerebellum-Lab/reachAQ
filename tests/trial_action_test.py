import dataclasses
from types import SimpleNamespace

import pytest

from autotrainer.device import LaserChannelConfiguration, LaserSystemConfiguration

from tools.acquisition.model.trial_action import (
    LaserPulseProfile,
    PreparedState,
    PreparedTrialOperation,
    ToneProfile,
    TrialActionCompiler,
    TrialActionExecutor,
    TrialCompileContext,
)
from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow


LASERS = LaserSystemConfiguration.from_channels((
    LaserChannelConfiguration(
        channel_id=1,
        analog_output="Dev4/ao0",
        diode_input="Dev4/ai0",
        shutter_output="Dev4/port0/line0",
        trigger_source="/Dev4/PXI_Trig0",
        board_stim_line=3,
    ),
))


def _context(**changes):
    values = dict(
        session_id="session001",
        session_generation=4,
        protocol_id="p",
        protocol_revision=2,
        logical_trial_id=1,
        attempt_id=1,
        session_seed=42,
        animal_base_dcs=(10.0, 20.0, 30.0),
        lane_offsets_dcs={
            "center": (0.0, 0.0, 0.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
        },
    )
    values.update(changes)
    return TrialCompileContext(**values)


def _compiler():
    return TrialActionCompiler(
        tone_profiles={"cue": ToneProfile("cue", 1, 6000, 100)},
        laser_profiles={
            "pulse": LaserPulseProfile("pulse", 3, 2.5, 5.0)
        },
        laser_configuration=LASERS,
        dcs_to_motor=lambda values: tuple(value * 2 for value in values),
    )


def test_compile_fixed_absolute_target_and_profiles():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "position_mode": "fixed_manual",
        "position_lane": "right",
        "shift_x_mm": 0.5,
        "tone_profile_id": "cue",
        "tone_phase": "before_send",
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 1,
        "stimulus_assignment": "always",
        "stimulus_trigger": "tone_1",
    })

    recipe = _compiler().compile(row, _context())

    assert recipe.resolved_dcs_target == (11.5, 20.0, 30.0)
    assert recipe.resolved_motor_target == (23.0, 40.0, 60.0)
    assert recipe.tone_profile.profile_id == "cue"
    assert recipe.laser_profile.profile_id == "pulse"
    assert recipe.stimulus_selected


def test_automatic_warmup_uses_lane_baseline_without_manual_offset():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "position_mode": "reach_derived_automatic",
        "position_lane": "left",
    })

    recipe = _compiler().compile(row, _context())

    assert recipe.resolved_dcs_target == (9.0, 20.0, 30.0)
    assert recipe.position_evidence["automatic_status"] == "insufficient_history"


def test_recommend_only_automatic_shift_is_retained_but_not_applied():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "position_mode": "reach_derived_automatic",
        "position_lane": "left",
    })
    recommendation = {
        "generation": 2,
        "resolved_target_dcs": [14.0, 21.0, 30.0],
        "apply_automatically": False,
    }

    recipe = _compiler().compile(row, _context(
        automatic_target_dcs=None,
        automatic_generation=2,
        automatic_policy={"policy_id": "recommend", "revision": 1},
        automatic_recommendation=recommendation,
    ))

    assert recipe.resolved_dcs_target == (9.0, 20.0, 30.0)
    assert recipe.position_evidence["automatic_status"] == "recommendation_only"
    assert recipe.position_evidence["automatic_recommendation"] == recommendation


def test_retry_repeat_keeps_draw_while_resample_changes_it():
    base = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "stimulus_assignment": "percentage",
        "stimulus_trigger": "tone_1",
        "stimulus_probability_percent": 50,
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 1,
    })

    first = _compiler().compile(base, _context(attempt_id=1))
    repeated = _compiler().compile(base, _context(attempt_id=2))
    resampled = _compiler().compile(
        base.with_updates({"retry_assignment": "resample"}),
        _context(attempt_id=2),
    )

    assert repeated.stimulus_seed == first.stimulus_seed
    assert resampled.stimulus_seed != first.stimulus_seed


def test_compile_rejects_unknown_laser_profile():
    # The row's own trigger route now resolves the firing (a laser row no
    # longer has to agree with the profile's own trigger_route field), so the
    # remaining rejection here is an unknown profile ID.
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "laser_profile_id": "missing",
        "laser_phase": "before_send",
        "laser_trigger_route": "direct_ni_software",
        "laser_channel_id": 1,
        "stimulus_assignment": "always",
        "stimulus_trigger": "first_reach",
    })

    with pytest.raises(ValueError, match="Unknown laser profile"):
        _compiler().compile(row, _context())


def test_compile_rejects_pre_reveal_shorter_than_trigger_pulse():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "cover_policy": "reveal",
        "laser_profile_id": "pulse",
        "laser_phase": "embedded_in_sequence",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 1,
        "stimulus_assignment": "always",
        "stimulus_trigger": "pre_reveal",
        "pre_reveal_ms": 1,
    })

    with pytest.raises(ValueError, match="longer than the STIM3"):
        _compiler().compile(row, _context())


def test_prepared_operation_enforces_generation_and_terminal_state():
    row = TrialProtocolRow(trial_id=1).with_updates({"enabled": True})
    operation = PreparedTrialOperation(_compiler().compile(row, _context()))

    operation.transition(PreparedState.PREPARED)
    operation.transition(PreparedState.SEND_ACCEPTED)
    operation.transition(PreparedState.ACTIVE)
    operation.transition(PreparedState.COMPLETED)

    assert operation.state is PreparedState.COMPLETED
    assert len(operation.to_record()["observations"]) == 5
    with pytest.raises(RuntimeError, match="stale"):
        operation.require_generation(5)
    with pytest.raises(RuntimeError, match="Invalid"):
        operation.transition(PreparedState.FAILED)


def test_executor_prepares_before_send_and_binds_acknowledgement():
    calls = []
    compiler = _compiler()
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "tone_profile_id": "cue",
        "tone_phase": "before_send",
    })
    recipe = compiler.compile(row, _context())
    executor = TrialActionExecutor(
        move_absolute=lambda target: calls.append(("move", target)),
        configure_cover=lambda policy, recipe: calls.append(("cover", policy)),
        play_tone=lambda profile, phase: calls.append(("tone", profile.profile_id, phase)),
        prepare_laser=lambda profile, recipe: calls.append(("laser", profile.profile_id)),
        cancel_laser=lambda handle: calls.append(("cancel_laser", handle)),
    )

    operation = executor.prepare(recipe)
    executor.bind_send(recipe.operation_id, 4, "can-context")
    assert executor.matches_send_context("can-context")
    assert not executor.matches_send_context("unrelated")
    executor.acknowledge_presentation("can-context")
    executor.complete("cycle ended")

    assert calls == [
        ("move", (20.0, 40.0, 60.0)),
        ("cover", "keep_current"),
        ("tone", "cue", "before_send"),
    ]
    assert operation.state is PreparedState.COMPLETED


def test_executor_preparation_failure_cancels_laser_and_creates_no_send():
    calls = []
    laser_row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 1,
        "stimulus_assignment": "always",
        "stimulus_trigger": "tone_1",
    })
    recipe = _compiler().compile(laser_row, _context())
    executor = TrialActionExecutor(
        move_absolute=lambda target: None,
        configure_cover=lambda policy, recipe: None,
        play_tone=lambda profile, phase: None,
        prepare_laser=lambda profile, recipe: (_ for _ in ()).throw(RuntimeError("arm failed")),
        cancel_laser=lambda handle: calls.append(handle),
    )

    with pytest.raises(RuntimeError, match="arm failed"):
        executor.prepare(recipe)

    assert executor.operation.state is PreparedState.FAILED
    with pytest.raises(RuntimeError, match="Prepared state"):
        executor.require_send_permission(recipe.operation_id, 4)


def test_executor_routes_hardware_stimulus_through_firmware_callback():
    calls = []
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 1,
        "stimulus_assignment": "always",
        "stimulus_trigger": "first_reach",
    })
    recipe = _compiler().compile(row, _context())
    executor = TrialActionExecutor(
        move_absolute=lambda target: None,
        configure_cover=lambda policy, recipe: None,
        play_tone=lambda profile, phase: None,
        prepare_laser=lambda profile, recipe: object(),
        cancel_laser=lambda handle: None,
        trigger_hardware_stimulus=lambda profile, recipe, detail: calls.append(
            (recipe.laser_firing.trigger_pulse_us, recipe.operation_id, detail)
        ),
    )

    executor.prepare(recipe)
    executor.trigger_stimulus(recipe.operation_id, 4, detail="frame 7")

    assert calls == [(1000, recipe.operation_id, "frame 7")]


def test_embedded_tone_is_prepared_without_host_playback():
    calls = []
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "tone_profile_id": "cue",
        "tone_phase": "embedded_in_sequence",
    })
    recipe = _compiler().compile(row, _context())
    executor = TrialActionExecutor(
        move_absolute=lambda target: None,
        configure_cover=lambda policy, recipe: None,
        play_tone=lambda profile, phase: calls.append((profile.profile_id, phase)),
        prepare_laser=lambda profile, recipe: None,
        cancel_laser=lambda handle: None,
    )

    executor.prepare(recipe)

    assert calls == [("cue", "embedded_in_sequence")]


def test_executor_persists_laser_result_and_rejects_failed_cycle():
    class FailedLaser:
        state = SimpleNamespace(value="failed")
        error = RuntimeError("AO task failed")

        def to_record(self):
            return {"state": self.state.value, "error": str(self.error)}

        def cancel(self):
            return False

    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 1,
        "stimulus_assignment": "always",
        "stimulus_trigger": "tone_1",
    })
    recipe = _compiler().compile(row, _context())
    executor = TrialActionExecutor(
        move_absolute=lambda target: None,
        configure_cover=lambda policy, recipe: None,
        play_tone=lambda profile, phase: None,
        prepare_laser=lambda profile, recipe: FailedLaser(),
        cancel_laser=lambda handle: handle.cancel(),
    )

    executor.prepare(recipe)
    executor.bind_send(recipe.operation_id, 4, "can-context")
    executor.acknowledge_presentation("can-context")

    with pytest.raises(RuntimeError, match="AO task failed"):
        executor.complete("cycle ended")

    record = executor.operation_record()
    assert record["state"] == "failed"
    assert record["actions"]["laser"]["state"] == "failed"


def test_a_laser_row_compiles_to_the_lasers_own_firing():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 1,
        "stimulus_assignment": "always",
        "stimulus_trigger": "tone_1",
        "tone_profile_id": "cue",
        "tone_phase": "before_send",
    })

    recipe = _compiler().compile(row, _context())

    assert recipe.laser_firing.channel_id == 1
    assert recipe.laser_firing.trigger_terminal == "/Dev4/PXI_Trig0"
    assert recipe.laser_firing.stim_line == 3
    assert recipe.to_record()["laser_firing"]["stim_line"] == 3


def test_a_laser_2_row_fires_on_laser_2s_own_line_and_terminal():
    # christielab10's wiring: laser 1 on STIM3 into PXI_Trig0, laser 2 on
    # STIM2 into PXI_Trig2. The row's route name is hardware_stim3 either way;
    # the board line comes from the laser, not the route.
    lasers = LaserSystemConfiguration.from_channels((
        LASERS.get_channel(1),
        LaserChannelConfiguration(
            channel_id=2,
            analog_output="Dev4/ao1",
            diode_input="Dev4/ai4",
            shutter_output="Dev4/port0/line5",
            trigger_source="/Dev4/PXI_Trig2",
            board_stim_line=2,
        ),
    ))
    compiler = TrialActionCompiler(
        tone_profiles={"cue": ToneProfile("cue", 1, 6000, 100)},
        laser_profiles={"pulse": LaserPulseProfile("pulse", 3, 2.5, 5.0)},
        laser_configuration=lasers,
        dcs_to_motor=lambda values: tuple(value * 2 for value in values),
    )
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 2,
        "stimulus_assignment": "always",
        "stimulus_trigger": "tone_1",
        "tone_profile_id": "cue",
        "tone_phase": "before_send",
    })

    firing = compiler.compile(row, _context()).laser_firing

    assert firing.channel_id == 2
    assert firing.stim_line == 2
    assert firing.trigger_terminal == "/Dev4/PXI_Trig2"


def test_a_laser_row_on_an_unconfigured_laser_does_not_compile():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 2,
        "stimulus_assignment": "always",
        "stimulus_trigger": "tone_1",
        "tone_profile_id": "cue",
        "tone_phase": "before_send",
    })

    with pytest.raises(ValueError, match="Laser 2 is not configured"):
        _compiler().compile(row, _context())
