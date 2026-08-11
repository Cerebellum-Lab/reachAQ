from tools.acquisition.model.trial_protocol_schedule import TrialProtocolSchedule


def test_placeholder_schedule_is_ordered_and_editable():
    schedule = TrialProtocolSchedule.with_placeholder_rows(3)

    updated = schedule.update(2, "shift_y_mm", "1.5")

    assert [row.trial_id for row in schedule.rows] == [1, 2, 3]
    assert updated.shift_y_mm == 1.5
    assert schedule.to_records()[1]["shift_y_mm"] == 1.5


def test_schedule_extends_for_later_trials():
    schedule = TrialProtocolSchedule.with_placeholder_rows(2)

    row = schedule.row(5)

    assert row.trial_id == 5
    assert [item.trial_id for item in schedule.rows] == [1, 2, 5]
