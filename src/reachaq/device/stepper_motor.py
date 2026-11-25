"""
Utility functions convert to/from mm and motor turns
"""

# motor data sheet: 48 steps per full turn; 19.7 steps/mm
_MM_PER_TURN = 48.0 / 19.7


def mm_to_turns(mm: float) -> float:
    """
    Return - Number of turns of the stepper motor from a linear distance.
    """
    return mm / _MM_PER_TURN


def turns_to_mm(turns: float) -> float:
    """
    Return - Number of mm in linear distance from number of turns of the stepper motor.
    """
    return turns * _MM_PER_TURN
