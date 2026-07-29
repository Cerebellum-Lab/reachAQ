from pathlib import Path
from typing import List

from autotrainer.core import MeasurementData
from autotrainer.core import HeadbarPressureMonitor


def _count_detections(detector: HeadbarPressureMonitor, measurements: List[MeasurementData], batch_size: int = 20) -> int:
    detection_count = 0

    measurement_buffer = []

    for measurement in measurements:
        measurement_buffer.append(measurement.pressure)

        if len(measurement_buffer) == batch_size:
            result = detector.update(measurement_buffer)
            if result:
                detection_count += 1
            measurement_buffer.clear()

    return detection_count


def test_headbar_detection():
    path = Path(__file__).parent.joinpath("fixtures").joinpath("pressure_threshold_sample.csv")

    measurements = MeasurementData.from_file(str(path))

    # Default behavior with the known fixture
    detector = HeadbarPressureMonitor()
    detection_count = _count_detections(detector, measurements)
    assert detection_count == 19

    # Lower threshold
    detector.engaged_threshold = 20
    detection_count = _count_detections(detector, measurements)
    assert detection_count == 39

    # Raise threshold
    detector.engaged_threshold = 40
    detection_count = _count_detections(detector, measurements)
    assert detection_count == 12

    # Modify window length
    detector.duration = 0.5
    detection_count = _count_detections(detector, measurements)
    assert detection_count == 24

    # Modify sample rate (window is in seconds - affects buffer/window retain counts)
    detector.sample_rate = 75
    detection_count = _count_detections(detector, measurements)
    assert detection_count == 18


if __name__ == '__main__':
    test_headbar_detection()
