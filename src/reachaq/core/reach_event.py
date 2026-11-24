import dataclasses
import enum

from typing import Optional


class ReachEventMethod:  #  (str, enum.Enum):
        # NB: not enum subclass, as the members are pickled in the saved h5 files,
        # and this requires the readers to have access to the original enum class.
        # which might not always be possible or desirable.

    NONE = "none"
    OTHER = "other"
    RIGHT_HAND = "right_hand"
    LEFT_HAND = "left_hand"
    TONGUE = "tongue"


class ReachEventOutcome:  # (str, enum.Enum):  # see above.

    NONE = "none"
    STALLED = "stalled"
    MISSED = "missed"
    DROPPED = "dropped"
    GRABBED = "grabbed"
    EATEN = "eaten"


@dataclasses.dataclass
class ReachEvent:
    """Represents a general "reach/pellet" event, see ReachEventMethod"""

    # NB: all frame indices: 'init', 'end' and 'max', can be negative
    #  if determined before the start of the corresponding video

    init: int  # frame index, can be negative
    end: Optional[int]  # frame index, can be negative ; None means unknown
    max: Optional[int] = None  # frame index, can be negative ; None means unknown

    # Notice the "str" type hint for both method and outcome,
    #  *not* the 'ReachEventMethod' or 'ReachEventOutcome' type hint.
    method: str = ReachEventMethod.NONE
    outcome: str = ReachEventOutcome.NONE

    delay_since_presented: float = 0  # is basically calculated as `init / fps`
