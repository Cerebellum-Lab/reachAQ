"""What a board's refusal of a command means, and what the operator is told.

The pellet board acknowledges every command with an int32 error. Zero means
it ran. A negative value is the firmware's errno: the board refused the
command, or a move that was running was stopped. The numbers are picolibc's,
not the host's: ETIMEDOUT is 116 on the board and 110 on Linux, so they are
compared as numbers here, never against the host's ``errno`` module.

What a code means depends on the command: -EPERM is an E-stop for a move but
a line that cannot carry a pulse for GPIO_PULSE, and -EBUSY is "needs homing"
for a stepper move but "still moving" for a servo. So each refusal carries
the class of the command that was refused, and the rules below key on both.
Pellet firmware 2.3.1, docs/releases/v2.3.1.md, "ACK errors".
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Tuple

from autotrainer.core.message import Motor

EPERM = -1
EIO = -5
EAGAIN = -11
EBUSY = -16
ETIMEDOUT = -116

_NAMES = {
    EPERM: "EPERM",
    EIO: "EIO",
    EAGAIN: "EAGAIN",
    EBUSY: "EBUSY",
    ETIMEDOUT: "ETIMEDOUT",
}

_STEPPERS = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)
_AXIS_NAMES = {
    Motor.PELLET_X_MOTOR: "X",
    Motor.PELLET_Y_MOTOR: "Y",
    Motor.PELLET_Z_MOTOR: "Z",
}
_SERVO_NAMES = {
    Motor.PELLET_COVER_SERVO: "cover",
    Motor.PELLET_LOAD_SERVO: "load",
}


def board_error_name(error: int) -> str:
    """The firmware's name for a board error number, or "board error N"."""
    error = int(error)
    return _NAMES.get(error, f"board error {error}")


def is_refusal(error: int) -> bool:
    """Whether an ACK error means the board did not do what it was asked.

    -EAGAIN is not a refusal: a move within 1/microsteps turn of the axis's
    position is answered with it, on a healthy board, because the axis is
    already there.
    """
    return int(error) not in (0, EAGAIN)


class RefusedCommandClass(str, enum.Enum):
    STEPPER_MOVE = "stepper_move"
    STEPPER_HOME = "stepper_home"
    FIXED_XYZ = "fixed_xyz"
    SERVO_MOVE = "servo_move"
    PULSE = "pulse"
    OTHER = "other"


_STEPPER_CLASSES = {
    RefusedCommandClass.STEPPER_MOVE,
    RefusedCommandClass.STEPPER_HOME,
    RefusedCommandClass.FIXED_XYZ,
}
_MOTION_CLASSES = _STEPPER_CLASSES | {RefusedCommandClass.SERVO_MOVE}


def _axes(motors) -> str:
    names = [_AXIS_NAMES[m] for m in motors if m in _AXIS_NAMES]
    return "/".join(names) or "?"


@dataclasses.dataclass(frozen=True)
class CommandRefusal:
    """One command the board refused, as the command loop saw it."""

    token: Any
    error: int
    #: The command the caller sent (SEND_PELLET, SEND_HOME, MOVE_X...).
    command: Any
    #: The frame that was refused: a compound step or the command, for the log.
    step: str
    command_class: RefusedCommandClass
    #: The motors the refused frame drove.
    motors: Tuple[Motor, ...] = ()
    #: The refused frame was a resend after an ACK timeout.
    resent: bool = False
    perf_c: float = math.nan

    @property
    def error_name(self) -> str:
        return board_error_name(self.error)

    @property
    def outcome_unknown(self) -> bool:
        """-EBUSY on a resent stepper command: still moving, or unknown.

        A command whose ACK timed out is re-sent with a new uuid. If the
        first one is still running (a home slower than the timeout), the
        board refuses the resend at once with -EBUSY. That axis is finishing
        a normal move; it does not need a home.
        """
        return (
            self.resent
            and self.error == EBUSY
            and self.command_class in _STEPPER_CLASSES
        )

    @property
    def needs_home(self) -> Tuple[Motor, ...]:
        """The stepper axes the board will refuse to move until each is homed."""
        error, cls = self.error, self.command_class
        if cls not in _MOTION_CLASSES:
            return ()
        if error == EPERM:
            return _STEPPERS  # ESTOP marks every axis
        if cls is RefusedCommandClass.SERVO_MOVE:
            return ()
        if error in (EIO, ETIMEDOUT):
            # A fault or a silence stop: the host cannot tell which axes the
            # board marked, so all of them.
            return _STEPPERS
        if error == EBUSY and not self.outcome_unknown:
            if cls is RefusedCommandClass.FIXED_XYZ:
                return _STEPPERS  # one error for three axes
            return tuple(m for m in self.motors if m in _AXIS_NAMES)
        return ()

    @property
    def message(self) -> str:
        """What the operator is told: what happened, and what to do."""
        return f"{self._advice()} ({self._what()}: {self.error_name} {int(self.error)})"

    def _what(self) -> str:
        command = getattr(self.command, "name", None) or str(self.command)
        if self.step and self.step != command:
            return f"{command}, step {self.step}"
        return command

    def _advice(self) -> str:
        error, cls = self.error, self.command_class
        if cls in _MOTION_CLASSES and error == EPERM:
            return "Pellet motors are in E-stop. Release the E-stop, then home the pellet axes."
        if cls in _STEPPER_CLASSES:
            if self.outcome_unknown:
                return (
                    f"Pellet axis {_axes(self.motors)} was still moving when its command "
                    "was re-sent after an acknowledgement timeout; its final position is "
                    "unknown. Let it stop before the next command."
                )
            if error == EIO and cls is RefusedCommandClass.STEPPER_HOME:
                return (
                    "Pellet stepper drivers did not re-prove during the home. Reboot the "
                    "pellet board and check the 12 V supply."
                )
            if error == EIO:
                return (
                    "Pellet stepper driver fault or drivers not initialised. Home the pellet "
                    "axes to clear it. If the home is refused as well, reboot the pellet "
                    "board and check the 12 V supply."
                )
            if error == ETIMEDOUT:
                return (
                    "Pellet motors switched off after 30 s without host traffic and a move "
                    "was cut off. Home the pellet axes."
                )
            if error == EBUSY and cls is RefusedCommandClass.STEPPER_HOME:
                return (
                    f"Pellet axis {_axes(self.motors)} refused its home: it is busy or "
                    "stuck. Reboot the pellet board."
                )
            if error == EBUSY:
                axes = _axes(self.needs_home)
                return f"Pellet axis {axes} needs homing after a stop. Home the pellet axes."
        if cls is RefusedCommandClass.SERVO_MOVE and error == EBUSY:
            servo = ", ".join(_SERVO_NAMES.get(m, str(m)) for m in self.motors) or "a"
            return f"Pellet {servo} servo was still moving and refused the move."
        if cls is RefusedCommandClass.PULSE and error == EPERM:
            return "The pellet board refused the stimulus pulse: that line cannot carry a pulse."
        if cls is RefusedCommandClass.PULSE and error == EBUSY:
            return "The pellet board refused the stimulus pulse: a pulse is already in flight."
        return f"The pellet board refused the command with {self.error_name}."


class CommandRefusedError(RuntimeError):
    """A command that was waited on was refused by the board; ``refusal`` says how."""

    def __init__(self, refusal: CommandRefusal):
        super().__init__(f"command {refusal.token} was refused by the board: {refusal.message}")
        self.refusal = refusal
