import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from autotrainer.behavior.training import (
    LoadProgressResult,
    PlanFormatError,
    PlanRepository,
    SessionResult,
    TrainingPlan,
    TrainingProgressState,
    TrialKind,
    TrialResult,
    parse_plan,
)
from autotrainer.behavior.training.schema import Tone, describe_predicate

FIXTURES = Path(__file__).parent / "fixtures" / "training_plans"
SUCCESS = TrialResult(TrialKind.SUCCESS, reaches=2, successful_reaches=1, pellets_consumed=1)
FAILURE = TrialResult(TrialKind.FAILURE, reaches=3)


def _document():
    return json.loads((FIXTURES / "reach-shaping.json").read_text(encoding="utf-8"))


@pytest.fixture
def plan():
    return TrainingPlan(parse_plan(_document()))


class FakeAlgorithm:
    def __init__(self):
        self.pellet_delivery_enabled = False
        self.pellet_cover_enabled = False
        self.intersession_pellet_shift_enabled = True


def test_example_plan_parses_with_view_attributes(plan):
    assert plan.plan_id == "reach-shaping"
    assert [p.phase_id for p in plan.phases] == ["shaping-close", "shaping-standard", "shaping-done"]
    first = plan.current_phase
    assert (first.send_x, first.send_y, first.send_z) == (0.0, -1.0, 0.0)
    assert first.is_pellet_delivery_enabled and not first.is_pellet_cover_enabled
    assert len(first.session_actions) == 2
    assert first.advance_predicate is not None and first.fallback_predicate is None
    assert plan.phases[2].send_x is None and plan.phases[2].advance_predicate is None
    assert describe_predicate(first.advance_predicate) == "(trials >= 4 and successRate (last 4) >= 0.75)"


def test_advance_needs_every_condition(plan):
    results = [plan.record_trial(r) for r in (SUCCESS, SUCCESS, FAILURE, SUCCESS)]
    assert results == [SessionResult.CONTINUE] * 3 + [SessionResult.ADVANCE]
    progress = plan.current_phase.progress
    assert (progress.session_count, progress.successes, progress.failures) == (4, 3, 1)
    assert (progress.total_reaches, progress.successful_reaches, progress.pellets_consumed) == (9, 3, 3)
    assert plan.progress_state is TrainingProgressState.InProgress


def test_windowed_rate_waits_for_a_full_window(plan):
    for _ in range(3):
        assert plan.record_trial(SUCCESS) is SessionResult.CONTINUE


def test_success_moves_the_target_within_its_limit(plan):
    for _ in range(6):
        plan.record_trial(SUCCESS)
    assert plan.current_phase.progress.target_offset == (0.0, 1.0, 0.0)
    directive = plan.next_trial_directive("s1:trial-7")
    assert directive.target_offset_dcs == (0.0, 0.0, 0.0)  # sendOffset -1 plus the clamped +1
    assert directive.tone == Tone(4000, 200)
    assert directive.pause_seconds == 0.0


def test_failure_does_not_move_a_success_only_target(plan):
    plan.record_trial(FAILURE)
    assert plan.current_phase.progress.target_offset == (0.0, 0.0, 0.0)


def test_advance_resets_the_visit_and_counts_attempts(plan):
    plan.resume()
    plan.record_trial(SUCCESS)
    changes = []
    plan.property_changed += lambda name, value, old: changes.append((name, value, old))
    updates = []
    plan.progress_updated += lambda: updates.append(1)
    plan.advance()
    standard = plan.current_phase
    assert standard.phase_id == "shaping-standard"
    assert changes[-1][0] == "current_phase" and changes[-1][1] is standard
    assert updates == [1]
    assert standard.progress.session_count == 0 and standard.progress.phase_attempts == 1
    plan.fallback()
    close = plan.current_phase
    assert close.progress.session_count == 0  # per-visit reset
    assert close.progress.lifetime_trials == 1 and close.progress.phase_attempts == 2
    assert plan.total_session_count == 1


def test_fallback_and_last_phase(plan):
    plan.advance()
    for _ in range(2):
        assert plan.record_trial(FAILURE) is SessionResult.CONTINUE
    assert plan.record_trial(TrialResult(TrialKind.NO_REACH)) is SessionResult.FALLBACK
    plan.advance()
    assert not plan.can_advance
    with pytest.raises(RuntimeError):
        plan.advance()
    assert plan.record_trial(SUCCESS) is SessionResult.CONTINUE


def test_jitter_and_pause_are_seeded_and_bounded(plan):
    plan.advance()
    a = plan.next_trial_directive("session-1:trial-3")
    assert a == plan.next_trial_directive("session-1:trial-3")
    draws = [plan.next_trial_directive(f"session-1:trial-{i}") for i in range(200)]
    assert all(abs(d.target_offset_dcs[0]) <= 0.2 and d.target_offset_dcs[1] == 0.0
               and abs(d.target_offset_dcs[2]) <= 0.1 for d in draws)
    assert all(2.0 <= d.pause_seconds <= 4.0 for d in draws)
    assert len({d.target_offset_dcs for d in draws}) > 100
    assert a.tone is None
    assert a.to_record()["phaseId"] == "shaping-standard"


def test_device_settings_apply_per_phase_and_restore_on_detach(plan):
    algorithm = FakeAlgorithm()
    plan.attach(algorithm, object(), None)
    assert algorithm.pellet_delivery_enabled is True
    assert algorithm.intersession_pellet_shift_enabled is False
    plan.advance()
    assert algorithm.pellet_cover_enabled is True
    plan.detach()
    assert (algorithm.pellet_delivery_enabled, algorithm.pellet_cover_enabled,
            algorithm.intersession_pellet_shift_enabled) == (False, False, True)


def test_progress_round_trips(plan):
    plan.resume()
    for result in (SUCCESS, FAILURE, SUCCESS):
        plan.record_trial(result)
    plan.advance()
    plan.record_trial(FAILURE)
    saved = json.loads(json.dumps(plan.serialize_progress()))
    assert saved["plan_id"] == "reach-shaping" and saved["currentPhaseId"] == "shaping-standard"

    restored = TrainingPlan(parse_plan(_document()))
    assert restored.deserialize_progress(saved) is LoadProgressResult.OK
    assert restored.current_phase.phase_id == "shaping-standard"
    assert restored.current_phase.progress.consecutive_failures == 1
    close = restored.phases[0].progress
    assert (close.lifetime_trials, close.target_offset, close.phase_attempts) == (3, (0.0, 0.5, 0.0), 1)
    assert list(close.outcomes) == [True, False, True]
    assert restored.training_start_timestamp == plan.training_start_timestamp
    assert json.loads(json.dumps(restored.serialize_progress())) == saved


def test_changed_phase_is_out_of_date_until_forced(plan):
    plan.record_trial(SUCCESS)
    plan.advance()
    plan.record_trial(SUCCESS)
    saved = plan.serialize_progress()
    document = _document()
    document["phases"][1]["advance"]["value"] = 5
    revised = TrainingPlan(parse_plan(document))
    assert revised.deserialize_progress(copy.deepcopy(saved)) is LoadProgressResult.OUT_OF_DATE
    assert revised.deserialize_progress(copy.deepcopy(saved), force_update=True) is LoadProgressResult.OK
    assert revised.current_phase.phase_id == "shaping-standard"
    assert revised.current_phase.progress.lifetime_trials == 0  # changed phase reset
    assert revised.phases[0].progress.lifetime_trials == 1  # unchanged phase kept


def test_removed_current_phase_restarts_at_the_first(plan):
    plan.advance()
    plan.advance()
    saved = plan.serialize_progress()
    document = _document()
    del document["phases"][2]
    revised = TrainingPlan(parse_plan(document))
    assert revised.deserialize_progress(saved) is LoadProgressResult.OUT_OF_DATE
    assert revised.deserialize_progress(saved, force_update=True) is LoadProgressResult.OK
    assert revised.current_phase.phase_id == "shaping-close"


@pytest.mark.parametrize("record", [
    {"plan_id": "reach-shaping"},
    {"plan_id": "other", "progressVersion": 1},
    {"plan_id": "reach-shaping", "progressVersion": 99, "state": "in_progress",
     "currentPhaseId": None, "trainingStartedAt": None, "phases": {}},
])
def test_unreadable_progress(plan, record):
    assert plan.deserialize_progress(record) is LoadProgressResult.UNKNOWN_ERROR
    assert plan.deserialize_progress(record, force_update=True) is LoadProgressResult.OK
    assert plan.progress_state is TrainingProgressState.NotStarted


def test_progress_survives_the_animal_files_key_rename(plan):
    # AnimalSubject renames session_count <-> trial_count keys on save/load;
    # the progress record uses neither.
    from autotrainer.core.animal.animal_subject import AnimalSubject
    plan.record_trial(SUCCESS)
    saved = plan.serialize_progress()
    persisted = AnimalSubject._rename_progress_count([saved], to_persisted=True)
    assert AnimalSubject._rename_progress_count(persisted, to_persisted=False) == [saved]


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(schemaVersion=2), "schemaVersion"),
    (lambda d: d.update(extra=1), "unknown key"),
    (lambda d: d["phases"].append(copy.deepcopy(d["phases"][0])), "duplicated phaseId"),
    (lambda d: d["phases"][0]["advance"]["all"][0].update(metric="speed"), "metric"),
    (lambda d: d["phases"][0]["advance"]["all"][0].update(window=3), "window"),
    (lambda d: d["phases"][0]["trialActions"].append({"action": "tone", "frequencyHz": 10, "durationMs": 5}),
     "frequencyHz"),
    (lambda d: d["phases"][1]["trialActions"][1].update(minSeconds=9), "pause"),
    (lambda d: d["phases"][0]["trialActions"][0].update(limits={"y": [1, 0]}), "min is above max"),
    (lambda d: d["phases"][0]["device"].update(pelletCover="yes"), "true or false"),
    (lambda d: d.update(phases=[]), "non-empty"),
    (lambda d: d["phases"][0]["trialActions"].append({"action": "laser"}), "action must be one of"),
])
def test_invalid_documents_are_refused(mutate, message):
    document = _document()
    mutate(document)
    with pytest.raises(PlanFormatError, match=message):
        parse_plan(document)


def test_repository_loads_fresh_plans_and_refuses_duplicates(tmp_path):
    shutil.copy(FIXTURES / "reach-shaping.json", tmp_path / "a.json")
    repo = PlanRepository(tmp_path)
    infos = repo.get_plans()
    assert [(i.plan_id, i.name) for i in infos] == [("reach-shaping", "Reach shaping")]
    first, second = repo.get_plan("reach-shaping"), repo.get_plan("reach-shaping")
    assert first is not second and repo.get_plan("missing") is None
    first.record_trial(SUCCESS)
    assert second.current_phase.progress.session_count == 0

    shutil.copy(FIXTURES / "reach-shaping.json", tmp_path / "b.json")
    with pytest.raises(PlanFormatError, match="duplicated planId"):
        repo.get_plans(refresh=True)
    assert [i.plan_id for i in repo.get_plans()] == ["reach-shaping"]  # previous library kept

    document = _document()
    document["planId"] = "other"
    (tmp_path / "b.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PlanFormatError, match="duplicated phaseId"):
        repo.get_plans(refresh=True)


def test_repository_without_a_directory_is_empty(tmp_path):
    assert PlanRepository().get_plans() == []
    assert PlanRepository(tmp_path / "missing").get_plans() == []
