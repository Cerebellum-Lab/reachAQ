from enum import Enum


class SystemState(str, Enum):
    ready = "ready"
    intersession = "intersession"
