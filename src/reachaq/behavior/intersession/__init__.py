from enum import Enum

# must be before following sub-import(s)
class IntersessionState(str, Enum):
    idle = "idle"
    segmentation = "segmentation"
    detection = "detection"


from .intersession_machine import IntersessionMachine
