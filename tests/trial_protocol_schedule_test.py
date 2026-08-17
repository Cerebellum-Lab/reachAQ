import pytest

from tools.acquisition.model.trial_protocol_schedule import (
    ActionPhase,
    AutomaticWindowMethod,
    LaserTriggerRoute,
    PelletLane,
    PelletPositionMode,
    ProtocolPatch,
    ProtocolScope,
    StimulusAssignment,
    StimulusTrigger,
    TrialOverride,
    TrialProtocolDocument,
    TrialProtocolRow,
    TrialProtocolSchedule,
)


def test_disabled_default_schedule_is_ordered_safe_and_editable():
    schedule = TrialProtocolSchedule.with_placeholder_rows(3)

    schedule.update(2, "position_mode", PelletPositionMode.FIXED_MANUAL.value)
    updated = schedule.update(2, "shift_y_mm", "1.5")

    assert [row.trial_id for row in schedule.rows] == [1, 2, 3]
    assert not schedule.rows[0].enabled
    assert updated.shift_y_mm == 1.5
    assert schedule.to_records()[1]["position_mode"] == "fixed_manual"
    assert schedule.to_records()[1]["value_sources"]["shift_y_mm"] == "trial:2"


def test_schedule_extends_for_later_trials_with_disabled_row():
    schedule = TrialProtocolSchedule.with_placeholder_rows(2)

    row = schedule.row(5)

    assert row.trial_id == 5
    assert not row.enabled
    assert [item.trial_id for item in schedule.rows] == [1, 2, 5]


def test_protocol_hierarchy_resolves_with_visible_sources():
    document = TrialProtocolDocument(
        protocol_id="complex-a",
        name="Complex A",
        trial_count=6,
        defaults=ProtocolPatch.from_mapping({
            "enabled": True,
            "position_mode": "fixed_manual",
            "position_lane": "center",
        }),
        epochs=(
            ProtocolScope.create(
                "stim-epoch", (2, 4, 6), {"stimulus_category": "stim"}
            ),
        ),
        blocks=(
            ProtocolScope.create(
                "late-block",
                (4, 6),
                {"position_lane": "right"},
                parent_epoch="stim-epoch",
            ),
        ),
        bulk_overrides=(
            ProtocolScope.create(
                "selected-left", (1, 4), {"position_lane": "left"}
            ),
        ),
        trial_overrides=(
            TrialOverride.create(4, {"position_lane": "center"}),
        ),
    )

    resolved = document.resolve()

    assert resolved[0].row.position_lane is PelletLane.LEFT
    assert dict(resolved[0].sources)["position_lane"] == "bulk:selected-left"
    assert resolved[3].row.position_lane is PelletLane.CENTER
    assert dict(resolved[3].sources)["position_lane"] == "trial:4"
    assert resolved[5].row.position_lane is PelletLane.RIGHT
    assert dict(resolved[5].sources)["position_lane"] == "block:late-block"


def test_epochs_may_be_noncontiguous_but_not_overlap():
    first = ProtocolScope.create("first", (1, 3), {})
    second = ProtocolScope.create("second", (2, 3), {})

    with pytest.raises(ValueError, match="overlaps trials"):
        TrialProtocolDocument(
            protocol_id="bad-overlap",
            name="Bad overlap",
            trial_count=3,
            epochs=(first, second),
        )


def test_blocks_must_be_disjoint_subsets_of_parent_epoch():
    epoch = ProtocolScope.create("epoch", (1, 2, 3), {})

    with pytest.raises(ValueError, match="outside parent epoch"):
        TrialProtocolDocument(
            protocol_id="bad-block",
            name="Bad block",
            trial_count=4,
            epochs=(epoch,),
            blocks=(
                ProtocolScope.create(
                    "block", (3, 4), {}, parent_epoch="epoch"
                ),
            ),
        )


def test_document_round_trip_preserves_resolved_rows():
    document = TrialProtocolDocument(
        protocol_id="round-trip",
        name="Round trip",
        trial_count=2,
        defaults=ProtocolPatch.from_mapping({
            "enabled": True,
            "position_mode": "reach_derived_automatic",
            "position_lane": "right",
            "automatic_window_method": "sliding_last_x",
            "automatic_window_size": 8,
            "stimulus_assignment": "always",
            "stimulus_trigger": "first_reach",
            "laser_profile_id": "blue-10hz",
            "laser_phase": "pellet_presentation",
            "laser_trigger_route": "direct_ni_software",
        }),
    )

    loaded = TrialProtocolDocument.from_record(document.to_record())
    row = loaded.resolve()[0].row

    assert loaded.to_record() == document.to_record()
    assert row.automatic_window_method is AutomaticWindowMethod.SLIDING_LAST_X
    assert row.stimulus_assignment is StimulusAssignment.ALWAYS
    assert row.stimulus_trigger is StimulusTrigger.FIRST_REACH
    assert row.laser_trigger_route is LaserTriggerRoute.DIRECT_NI_SOFTWARE
    assert row.laser_phase is ActionPhase.PELLET_PRESENTATION


def test_manual_and_automatic_position_fields_are_distinct():
    with pytest.raises(ValueError, match="fixed_manual"):
        TrialProtocolRow(trial_id=1).with_updates({"shift_x_mm": 1.0})

    automatic = TrialProtocolRow(trial_id=1).with_updates({
        "position_mode": "reach_derived_automatic",
        "position_lane": "left",
        "automatic_window_method": "legacy_batch",
    })

    assert automatic.position_lane is PelletLane.LEFT
    assert automatic.shift_x_mm == 0.0


def test_roi_triggers_are_serializable_but_not_runnable():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "stimulus_assignment": "always",
        "stimulus_trigger": "roi_1",
    })

    with pytest.raises(ValueError, match="future release"):
        row.validate(runnable=True)


def test_pre_reveal_requires_board_route_and_reveal_policy():
    base = {
        "enabled": True,
        "stimulus_assignment": "always",
        "stimulus_trigger": "pre_reveal",
        "pre_reveal_ms": 200,
        "laser_profile_id": "pulse",
        "laser_phase": "embedded_in_sequence",
    }
    with pytest.raises(ValueError, match="Reveal cover policy"):
        TrialProtocolRow(trial_id=1).with_updates({
            **base,
            "laser_trigger_route": "hardware_stim3",
        })
    with pytest.raises(ValueError, match="Hardware STIM3"):
        TrialProtocolRow(trial_id=1).with_updates({
            **base,
            "cover_policy": "reveal",
            "laser_trigger_route": "direct_ni_software",
        })

    row = TrialProtocolRow(trial_id=1).with_updates({
        **base,
        "cover_policy": "reveal",
        "laser_trigger_route": "hardware_stim3",
    })
    assert row.pre_reveal_ms == 200


def test_direct_ni_route_is_limited_to_first_reach():
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "stimulus_assignment": "always",
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "direct_ni_software",
        "stimulus_trigger": "tone_1",
    }, validate=False)

    with pytest.raises(ValueError, match="requires the First Reach"):
        row.validate(runnable=True)


@pytest.mark.parametrize(
    "values, message",
    [
        ({"stimulus_probability_percent": 101}, "between 0 and 100"),
        ({"automatic_window_size": 0}, "between 1 and 10000"),
        ({"pre_reveal_ms": -1}, "between 0 and 60000"),
        ({"shift_z_mm": float("nan")}, "must be finite"),
        ({"unknown": 1}, "Unknown trial protocol field"),
    ],
)
def test_protocol_values_are_strictly_validated(values, message):
    with pytest.raises(ValueError, match=message):
        TrialProtocolRow(trial_id=1).with_updates(values)
