
import dataclasses
import math
from enum import Enum
from typing import Dict, Optional

from reachaq.core import get_perf_now

_cache_scene_elements: Dict[str, "_BaseSceneElement"] = {}
_cache_scene_elements_str: Dict[str, str] = {}


class _BaseSceneElement(str):  # , Enum):
    """Dedicated str subclass for scene elements,
    main feature is to have the created elements be all cached/singleton,
    allowing use of "is" operator comparison with previous reference to any of them.
    """
    # we might want to use something more flexible, if the model would not use same names than here,
    # like with a mapping dict, and some dynamic attributes + type hints on the class to still help development.

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # set all string values to SceneElement/cls instance,
        # which pre-loads it in _cache_scene_elements:
        for name, value in vars(cls).items():
            if cls.__annotations__.get(name) != 'SceneElement' or not isinstance(value, str):
                continue
            setattr(cls, name, cls(value))

    def __new__(cls, value: str):
        cached = _cache_scene_elements.get(value, None)
        if cached is None:
            tentative = super().__new__(cls, value)
            assert isinstance(tentative, _BaseSceneElement)
            cached = _cache_scene_elements.setdefault(value, tentative)  # threads-safe
        # return the raw str value:
        # otherwise, when used as column(multiindex or not) and in h5 files that will make
        # the column to be saved with pickle,
        # which when unpickled fails if the pickled object type has been moved meanwhile.
        cached_str = _cache_scene_elements_str.setdefault(value, str(cached))
        return cached_str


class SceneElement(_BaseSceneElement):

    Pellet: "SceneElement" = 'Pellet'

    R_Hand: "SceneElement" = 'R_Hand'  # composite element part
    RH_flat: "SceneElement" = 'RH_flat'
    RH_spread: "SceneElement" = 'RH_spread'
    RH_grab: "SceneElement" = 'RH_grab'

    L_Hand: "SceneElement" = 'L_Hand'  # composite element part
    LH_flat: "SceneElement" = 'LH_flat'
    LH_spread: "SceneElement" = 'LH_spread'
    LH_grab: "SceneElement" = 'LH_grab'

    Nose: "SceneElement" = 'Nose'
    Mouth: "SceneElement" = 'Mouth'
    Tongue_mid: "SceneElement" = 'Tongue_mid'
    Tongue_tip: "SceneElement" = 'Tongue_tip'

    Star: "SceneElement" = 'Star'
    Diamond: "SceneElement" = 'Diamond'
    Triangle: "SceneElement" = 'Triangle'

    AnyAnimalPart: "SceneElement" = 'AnyAnimalPart'


AllHandsParts = {
    SceneElement.RH_flat, SceneElement.RH_spread, SceneElement.RH_grab,
    SceneElement.LH_flat, SceneElement.LH_spread, SceneElement.LH_grab,
}


AllAnimalParts = {
    SceneElement.Nose,
    SceneElement.Mouth,
    SceneElement.Tongue_tip,
    SceneElement.Tongue_mid,
} | AllHandsParts


AllNonAnimalParts = {
    SceneElement.Pellet,
    SceneElement.Diamond,
    SceneElement.Triangle,
    SceneElement.Star,
}


AllSceneParts = AllAnimalParts | AllNonAnimalParts


@dataclasses.dataclass
class ScenePartsPresenceContext:
    """Maintain a context of presence/absence status & age of parts/elements"""

    # TODO: for now only considering presence in either of the 2 possible cameras,
    #  but we could handle 3 cases of presence/absence:
    #   1) on any of the possible cams
    #       > can be/is used for checking pellet presence at the end of load sequence
    #   2) on all of the possible cams simultaneously
    #       > should be used to report pellet successfully loaded (more reliably)
    #   3) on all of the possible cams non-simultaneously
    #       > not entirely sure could be useful if we have already 1 + 2

    last_perf_c: float = -math.inf  # last perf counter received, basically the "freshness" of the data

    present_last_perf_c: Dict[str, float] = dataclasses.field(default_factory=dict)
    missing_last_perf_c: Dict[str, float] = dataclasses.field(default_factory=dict)

    def get_part_seen(self, part: str) -> bool:
        """Say if the part was seen on last update"""
        p_present = self.present_last_perf_c.get(part, -math.inf)
        p_missing = self.missing_last_perf_c.get(part, -math.inf)
        if math.isinf(p_present):
            return p_present > 0
        return p_present > p_missing

    def get_presence_age(self, part: str, *, perf_now: Optional[float] = None) -> float:
        """Returns the "presence age" of the part if it's currently present, otherwise -math.inf"""
        if perf_now is None:
            perf_now = self.last_perf_c
        presence_p = self.present_last_perf_c.get(part, None)
        if presence_p is None:
            return -math.inf
        return perf_now - presence_p

    def get_absence_age(self, part: str, *, perf_now: Optional[float] = None) -> float:
        """Returns the "absence age" of the part if it's currently absent, otherwise -math.inf.
        Age can also be math.inf if never seen yet"""
        if perf_now is None:
            perf_now = self.last_perf_c
        presence_p = self.present_last_perf_c.get(part, None)
        absence_p = self.missing_last_perf_c.get(part, None)
        if absence_p is None:
            return math.inf if presence_p is None else -math.inf
        if presence_p is not None and presence_p > absence_p:
            return -math.inf
        return perf_now - absence_p

    def get_recently_seen(self, part: str, missing_delay: float, *, perf_now: Optional[float] = None) -> bool:
        """Returns if part was seen before missing_delay, relatively to perf_now, or self.last_perf_c"""
        if perf_now is None:
            perf_now = self.last_perf_c
        prev_miss = self.missing_last_perf_c.get(part, -math.inf)
        prev_pres = self.present_last_perf_c.get(part, -math.inf)
        return (
            not math.isinf(prev_pres)
            and (prev_pres > prev_miss or perf_now - prev_miss < missing_delay)
        )

    def update_part_seen(self, part: str, seen: bool, *, perf_now: Optional[float] = None):
        if perf_now is None:
            perf_now = get_perf_now()
        prev_seen = self.present_last_perf_c.get(part, -math.inf)
        prev_miss = self.missing_last_perf_c.get(part, -math.inf)
        if seen:
            if prev_seen < prev_miss or math.isinf(prev_seen):
                self.present_last_perf_c[part] = perf_now
                self.missing_last_perf_c.setdefault(part, -math.inf)
        else:
            if prev_miss < prev_seen or math.isinf(prev_miss):
                self.missing_last_perf_c[part] = perf_now
                self.present_last_perf_c.setdefault(part, -math.inf)
        if perf_now > self.last_perf_c:
            self.last_perf_c = perf_now

    def get_animal_absence_age(self, *, perf_now: Optional[float] = None):
        """Returns the animal "absence" age, how old/elapsed time since last absence started,
        can be math.inf if never been absent"""
        if perf_now is None:
            perf_now = get_perf_now()
        max_perf_c = -math.inf
        parts = set(self.missing_last_perf_c) | set(self.present_last_perf_c)
        for part in parts:
            if part in AllAnimalParts:
                perf_c = self.missing_last_perf_c.get(part, -math.inf)
                if perf_c > max_perf_c:
                    max_perf_c = perf_c
        return perf_now - max_perf_c

    def get_animal_presence_age(self, *, perf_now: Optional[float] = None):
        """Returns the animal most recent "presence" age. ie: how old since it has been seen last time"""
        if perf_now is None:
            perf_now = get_perf_now()
        max_perf_c = -math.inf
        parts = set(self.missing_last_perf_c) | set(self.present_last_perf_c)
        for part in parts:
            if part in AllAnimalParts:
                perf_c = self.present_last_perf_c.get(part, -math.inf)
                if perf_c > max_perf_c:
                    max_perf_c = perf_c
        return perf_now - max_perf_c
