import csv
from dataclasses import dataclass
from typing import List

from typing_extensions import Self

from ..message.measurement_message import MeasurementMessageProtocol


@dataclass
class MeasurementData(MeasurementMessageProtocol):
    """
    Default class that implements the MeasurementMessage protocol.  Actual hardware devices will likely deliver their
    own object that implements the protocol.  However, this class can be used to read back data from the "monitor" CSV
    output files for playback, reprocessing, or other tasks.
    """
    when_val: float
    index_val: int
    pressure_val: int
    temperature_val: float
    humidity_val: float
    head_contact_val: bool

    @property
    def when(self) -> float:
        return self.when_val

    @property
    def timestamp(self) -> int:
        return self.index_val

    @property
    def index(self) -> int:
        return self.index_val

    @property
    def pressure(self) -> int:
        return self.pressure_val

    @property
    def switch(self) -> float:
        return 1.0 if self.head_contact_val is True else 0.0

    @property
    def temperature(self) -> float:
        return self.temperature_val

    @property
    def humidity(self) -> float:
        return self.humidity_val

    @property
    def head_contact(self) -> bool:
        return self.head_contact_val

    @classmethod
    def from_file(cls, path: str) -> List[Self]:
        measurements = []

        with open(str(path), mode="r") as file:
            csv_data = csv.reader(file)

            for line in csv_data:
                if line[0] == "Time":
                    continue

                # Legacy monitor files included a load-cell column between
                # Index and Switch. It is intentionally ignored.
                value_offset = 1 if len(line) >= 7 else 0
                measurement = cls(
                    when_val=float(line[0]),
                    index_val=int(float(line[1])),
                    head_contact_val=int(line[2 + value_offset]) == 1,
                    pressure_val=int(line[3 + value_offset]),
                    temperature_val=float(line[4 + value_offset]),
                    humidity_val=float(line[5 + value_offset]),
                )

                measurements.append(measurement)

        return measurements
