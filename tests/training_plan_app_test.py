"""The attached training plan's trial actions reach the SEND path."""

import time
from types import SimpleNamespace
from unittest import mock

import pytest

from reachaq.behavior.pellet_trial import TrialOutcome
from reachaq.behavior.training import TrainingPlan, parse_plan
from reachaq.core.animal.animal_subject import AnimalSubject

from reachaq.app.model.app_model import TRAINING_PAUSE_BLOCK_NAME
from reachaq.app.model.app_model_status import SessionRecordingStatus
from reachaq.app.model.trial_protocol_repository import TrialProtocolRepository
from reachaq.app.model.trial_protocol_schedule import ProtocolPatch, TrialProtocolDocument


def _plan(actions, device=None):
    phase = {"phaseId": "app-p0", "trialActions": actions}
    if device is not None:
        phase["device"] = device
    return TrainingPlan(parse_plan({"schemaVersion": 1, "planId": "app-plan", "phases": [phase]}))


def _attach(app_model, plan):
    algorithm = SimpleNamespace(pellet_delivery_enabled=True, pellet_cover_enabled=False,
                                intersession_pellet_shift_enabled=True)
    app_model._protocol_runner.attach(plan, algorithm, object())


def _recording_token(app_model, tmp_path):
    app_model._trial_protocol_repository = TrialProtocolRepository(tmp_path)
    app_model.save_ordered_protocol(TrialProtocolDocument(
        protocol_id="runtime",
        name="Runtime",
        trial_count=1,
        defaults=ProtocolPatch.from_mapping({"enabled": True, "position_mode": "fixed_manual",
                                             "shift_x_mm": 0.5}),
    ))
    app_model._selected_animal = SimpleNamespace(pellet_x=1.0, pellet_y=2.0, pellet_z=3.0)
    _previous, token = app_model._recording_session.begin_record("session001", {})
    app_model._recording_session.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    app_model._protocol_session_seed = 7
    return token


def test_plan_offset_is_compiled_into_the_trial_target(app_model, tmp_path):
    token = _recording_token(app_model, tmp_path)
    without = app_model._compile_protocol_trial(token, 1)
    assert "training_offset_dcs" not in without.position_evidence

    _attach(app_model, _plan([], device={"sendOffset": {"y": -1.0, "z": 0.25}}))
    recipe = app_model._compile_protocol_trial(token, 1)

    assert recipe.resolved_dcs_target == (1.5, 1.0, 3.25)
    assert recipe.position_evidence["training_offset_dcs"] == [0.0, -1.0, 0.25]
    assert recipe.position_evidence["training_directive"]["phaseId"] == "app-p0"


def test_jitter_is_reproducible_for_the_same_trial(app_model, tmp_path):
    token = _recording_token(app_model, tmp_path)
    _attach(app_model, _plan([{"action": "jitterPelletTarget", "x": 0.3}]))
    first = app_model._compile_protocol_trial(token, 1)
    again = app_model._compile_protocol_trial(token, 1)
    assert first.resolved_dcs_target == again.resolved_dcs_target
    assert abs(first.resolved_dcs_target[0] - 1.5) <= 0.3


def test_pause_holds_automatic_sends_until_it_elapses(app_model):
    _attach(app_model, _plan([{"action": "pause", "seconds": 0.2}]))
    assert app_model._protocol_runner.record_trial_outcome("1.1", TrialOutcome.SUCCESS)
    assert app_model._send_block_reasons.holds(TRAINING_PAUSE_BLOCK_NAME)
    deadline = time.monotonic() + 5
    while app_model._send_block_reasons.holds(TRAINING_PAUSE_BLOCK_NAME) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not app_model._send_block_reasons.holds(TRAINING_PAUSE_BLOCK_NAME)


def test_detaching_the_plan_releases_its_pause(app_model):
    plan = _plan([{"action": "pause", "seconds": 60}])
    _attach(app_model, plan)
    app_model._attached_plan = plan
    app_model._attached_animal = AnimalSubject(name="pause-subject")
    plan.property_changed += app_model._on_training_plan_property_changed
    plan.progress_updated += app_model._on_training_plan_progress_updated
    with mock.patch.object(app_model, "_save_animal_metadata"):
        app_model._protocol_runner.record_trial_outcome("1.1", TrialOutcome.FAILURE)
        assert app_model._send_block_reasons.holds(TRAINING_PAUSE_BLOCK_NAME)
        app_model._detach_training_plan()
    assert not app_model._send_block_reasons.holds(TRAINING_PAUSE_BLOCK_NAME)
    assert app_model._training_pause_timer is None


def test_tone_plays_before_send_and_never_blocks_it(app_model):
    _attach(app_model, _plan([{"action": "tone", "frequencyHz": 4000, "durationMs": 200}]))
    with mock.patch.object(app_model._hardware, "play_tone", return_value="token") as play:
        app_model._apply_training_send_actions()
    play.assert_called_once_with(4000, 0.2)
    with mock.patch.object(app_model._hardware, "play_tone", side_effect=RuntimeError("bus")):
        app_model._apply_training_send_actions()  # logged, not raised


def test_no_plan_means_no_send_actions(app_model):
    with mock.patch.object(app_model._hardware, "play_tone") as play:
        app_model._apply_training_send_actions()
    play.assert_not_called()
    assert not app_model._send_block_reasons.holds(TRAINING_PAUSE_BLOCK_NAME)


def test_configuration_brings_its_own_plan_directory(app_model):
    assert app_model.load_configuration() is True
    assert app_model._plan_repo.root == app_model._loaded_config_dir_path / "training_plans"


def test_tone_waits_for_the_send_to_be_allowed(app_model, tmp_path):
    _recording_token(app_model, tmp_path)  # ordered protocol selected, nothing prepared
    _attach(app_model, _plan([{"action": "tone", "frequencyHz": 4000, "durationMs": 200}]))
    with mock.patch.object(app_model._behavior.algorithm.__class__, "is_in_session",
                           new_callable=mock.PropertyMock, return_value=True), \
            mock.patch.object(app_model._hardware, "play_tone") as play:
        with pytest.raises(RuntimeError, match="not been prepared"):
            app_model._guard_protocol_send()
    play.assert_not_called()
