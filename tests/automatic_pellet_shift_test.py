from tools.acquisition.model.automatic_pellet_shift import (
    AutomaticPelletShiftController,
    AutomaticShiftPolicy,
    ReachPositionObservation,
)


def _reach(index, x, outcome="failure"):
    return ReachPositionObservation(f"reach-{index}", outcome, (x, -3.0, 1.0))


def test_legacy_batch_clears_after_window_and_matches_deadband():
    controller = AutomaticPelletShiftController(AutomaticShiftPolicy(window_size=3))

    assert controller.add(_reach(1, 2.5), baseline_dcs=(10, 20, 30)) is None
    assert controller.add(_reach(2, 2.5), baseline_dcs=(10, 20, 30)) is None
    first = controller.add(_reach(3, 2.5), baseline_dcs=(10, 20, 30))
    assert first.recommended_shift_dcs == (1.0, 0.0, 0.0)
    assert first.resolved_target_dcs == (11.0, 20.0, 30.0)
    assert controller.add(_reach(4, 2.5), baseline_dcs=(10, 20, 30)) is None


def test_sliding_window_recalculates_for_every_new_reach():
    policy = AutomaticShiftPolicy(window_method="sliding_last_x", window_size=2)
    controller = AutomaticPelletShiftController(policy)

    controller.add(_reach(1, 2.5), baseline_dcs=(0, 0, 0))
    first = controller.add(_reach(2, 2.5), baseline_dcs=(0, 0, 0))
    controller.accept(first.generation)
    second = controller.add(_reach(3, 0.5), baseline_dcs=(0, 0, 0))

    assert first.reach_ids == ("reach-1", "reach-2")
    assert second.reach_ids == ("reach-2", "reach-3")
    assert second.generation == 2


def test_duplicate_and_ineligible_reaches_do_not_advance_window():
    policy = AutomaticShiftPolicy(window_size=2, eligible_outcomes=frozenset({"failure"}))
    controller = AutomaticPelletShiftController(policy)

    assert controller.add(_reach(1, 2.5, "success"), baseline_dcs=(0, 0, 0)) is None
    assert controller.add(_reach(2, 2.5), baseline_dcs=(0, 0, 0)) is None
    assert controller.add(_reach(2, 3.5), baseline_dcs=(0, 0, 0)) is None
    result = controller.add(_reach(3, 2.5), baseline_dcs=(0, 0, 0))
    assert result.reach_ids == ("reach-2", "reach-3")


def test_accept_is_generation_deduplicated_and_retry_reads_absolute_target():
    controller = AutomaticPelletShiftController(AutomaticShiftPolicy(window_size=1))
    result = controller.add(_reach(1, 3.5), baseline_dcs=(10, 20, 30))

    first = controller.accept(result.generation)
    duplicate = controller.accept(result.generation)

    assert first == result.resolved_target_dcs
    assert duplicate is None
    assert controller.accepted_target == first


def test_recommend_only_policy_does_not_change_accepted_target():
    controller = AutomaticPelletShiftController(AutomaticShiftPolicy(
        window_size=1,
        apply_automatically=False,
    ))
    result = controller.add(_reach(1, 3.5), baseline_dcs=(10, 20, 30))

    assert result.recommended_shift_dcs != (0.0, 0.0, 0.0)
    assert controller.accept(result.generation) is None
    assert controller.accepted_target is None


def test_median_reduction_ignores_one_extreme_reach():
    controller = AutomaticPelletShiftController(AutomaticShiftPolicy(
        window_size=3,
        reduction_method="median",
        deadbands_mm=(0.0, 0.0, 0.0),
        maximum_update_mm=(100.0, 100.0, 100.0),
        maximum_absolute_mm=(100.0, 100.0, 100.0),
    ))

    controller.add(_reach(1, 2.0), baseline_dcs=(0, 0, 0))
    controller.add(_reach(2, 2.0), baseline_dcs=(0, 0, 0))
    result = controller.add(_reach(3, 100.0), baseline_dcs=(0, 0, 0))

    assert result.reduced_reach_offset_dcs[0] == 2.0
