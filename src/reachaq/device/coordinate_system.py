import enum


class CoordinateSystem(str, enum.Enum):
    Motor = "Motor"
    Diamond = "Diamond"


COORDINATE_SYSTEMS = (
    CoordinateSystem.Motor,
    CoordinateSystem.Diamond,
)
