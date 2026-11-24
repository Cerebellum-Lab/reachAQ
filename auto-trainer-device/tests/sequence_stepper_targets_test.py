"""sequence_stepper_targets: the stepper targets a compound sequence commands, as
CanDevice._perform_next_compound_step and CanInterface run it."""
import math

from autotrainer.core import Motor
from autotrainer.device import MotorSteps, sequence_stepper_targets
from autotrainer.device.can_device import default_load_pellet, default_move_retract
from autotrainer.device.can_interface import _STEPPER_MAX_POS

NAN3 = (math.nan, math.nan, math.nan)


def _targets(steps, position=NAN3, send_position=NAN3):
    return list(sequence_stepper_targets(MotorSteps.from_raw("t", steps).steps, position, send_position))


def test_absolute_steps_yield_their_value_rates_and_strings_included():
    # christielab10's load_pellet, as move_config.yaml gives it.
    steps = [
        {"type": "y", "value": 0}, {"type": "barrier_arm", "value": 110},
        {"type": "x", "value": 25}, {"type": "load_arm", "value": 5},
        {"type": "z", "value": 22}, {"type": "load_arm", "value": "114,25"},
        {"type": "z", "value": "10,50"}, {"type": "tone", "value": "5000,0.3"},
        {"type": "delay", "value": 1.0},
    ]
    assert _targets(steps) == [("y", 0.0), ("x", 25.0), ("z", 22.0), ("z", 10.0)]


def test_targets_are_clamped_as_can_interface_clamps_them():
    assert _targets([{"type": "x", "value": -3}, {"type": "y", "value": 99}]) == [
        ("x", 0.0), ("y", _STEPPER_MAX_POS)]
    assert _targets([{"type": "y_rel", "value": -25}], position=(1, 10, 1)) == [("y", 0.0)]


def test_relative_steps_follow_the_position_the_steps_before_leave():
    steps = [{"type": "x_rel", "value": 4}, {"type": "x", "value": 20}, {"type": "x_rel", "value": 16}]
    assert _targets(steps, position=(30, 0, 0)) == [("x", 34.0), ("x", 20.0), ("x", 36.0)]


def test_a_relative_step_from_an_unknown_position_is_nan():
    [(axis, target)] = _targets([{"type": "z_rel", "value": 1}])
    assert axis == "z" and math.isnan(target)


def test_send_relative_and_predefined_send_use_the_send_position():
    steps = [{"type": "send_z_rel", "value": 0}, {"type": "send_y_rel", "value": -15},
             {"type": "predefined", "value": "send"}]
    assert _targets(steps, position=(1, 1, 1), send_position=(12, 30, 20)) == [
        ("z", 20.0), ("y", 15.0), ("x", 12.0), ("y", 30.0), ("z", 20.0)]


def test_a_save_as_fixed_step_moves_the_send_position_not_the_stepper():
    steps = [{"x": 33.0, "save_as_fixed": True}, {"x_rel": 1}, {"predefined": "send"}]
    targets = list(sequence_stepper_targets(steps, (5, 5, 5), (10, 10, 10)))
    assert targets == [("x", 33.0), ("x", 6.0), ("x", 33.0), ("y", 10.0), ("z", 10.0)]


def test_home_steps_leave_their_axis_at_zero():
    steps = [{"home": Motor.PELLET_Y_MOTOR}, {"y_rel": 3}, {"predefined": "home"}, {"x_rel": 2}]
    assert list(sequence_stepper_targets(steps, (9, 9, 9), NAN3)) == [("y", 3.0), ("x", 2.0)]


def test_the_built_in_sequences():
    assert _targets_of(default_load_pellet()) == [("x", 0.0), ("z", 5.0), ("z", 0.0)]
    assert _targets_of(default_move_retract(), send=(12, 30, 20)) == [("z", 20.0), ("x", 12.0), ("y", 15.0)]


def _targets_of(steps: MotorSteps, send=NAN3):
    return list(sequence_stepper_targets(steps.steps, NAN3, send))
