import argparse
import logging
import os
from datetime import datetime
from typing import List

from autotrainer.core import MeasurementData, EventManager
from autotrainer.core import HeadbarPressureMonitor

logging.basicConfig(level=logging.INFO)
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger("autotrainer").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# A work in progress to be able to reprocess data files to assess the headbar pressure monitor behavior.

def _print_detections(detector: HeadbarPressureMonitor, measurements: List[MeasurementData],
                      batch_size: int = 1):
    measurement_buffer = []

    last_time = 0

    for measurement in measurements:
        measurement_buffer.append(measurement.pressure)

        if len(measurement_buffer) == batch_size:
            result = detector.update(measurement_buffer)
            if result:
                if measurement.when - last_time > 1:
                    print(datetime.fromtimestamp(measurement.when).strftime('%Y-%m-%d %H:%M:%S'), result)
                    last_time = measurement.when

            measurement_buffer.clear()


def process_data(file: str):
    measurements = MeasurementData.from_file(file)

    monitor = HeadbarPressureMonitor()
    monitor.engaged_threshold = 10
    _print_detections(monitor, measurements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="the monitor data input file")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error("The data file does not exist")

    EventManager.default()

    process_data(args.file)
