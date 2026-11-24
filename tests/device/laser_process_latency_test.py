"""What the laser process costs a command: in this process against through the laser process.

Both sides drive NidaqLaserController over the fake NI-DAQmx, so the
difference is the process boundary alone; the driver's own time on the rig is
the same either side and is not in these numbers. Measured idle, and with a
thread spinning in pure Python in the calling process, as a busy GUI holds the
GIL. Run with --run-bench (or -m bench); prints p50 / p95 / max in µs.

- set_command_voltage: the call, from entry to return.
- trigger: an armed deferred pulse's trigger(), from the call to the moment
  the start was entered (what NidaqLaserOperation.trigger() returns; in the
  laser process that is its perf_counter, which is system-wide), and to the
  call's return.
"""

import statistics
import threading
import time

import pytest

import laser_process_fake as fake
from reachaq.device import nidaq_laser
from reachaq.device.laser_process import LaserProcessController

COMMANDS = 400
TRIGGERS = 100


def _in_process(tmp_path):
    return fake.open_fake_controller(
        fake.configuration(), log_path=str(tmp_path / "in_process.jsonl"), block_wait=False)


def _through_the_process(tmp_path):
    return LaserProcessController(
        fake.configuration(), factory=fake.FACTORY,
        factory_kwargs={"log_path": str(tmp_path / "process.jsonl"), "block_wait": False})


def _stats(samples):
    samples = sorted(samples)
    return (statistics.median(samples) * 1e6,
            samples[int(0.95 * (len(samples) - 1))] * 1e6, samples[-1] * 1e6)


def _measure(controller):
    commands = []
    for index in range(COMMANDS):
        start = time.perf_counter()
        controller.set_command_voltage(1, 0.5 + (index % 2) * 0.5)
        commands.append(time.perf_counter() - start)
    to_entry, to_return = [], []
    for _ in range(TRIGGERS):
        operation = controller.run_synchronized_pulse_train(fake.armed_pulse(duration_ms=0.2))
        start = time.perf_counter()
        entry = operation.trigger()
        returned = time.perf_counter()
        to_entry.append(entry - start)
        to_return.append(returned - start)
        assert operation.wait_until_finished(5.0)
    return {"set_command_voltage": commands, "trigger to start entry": to_entry,
            "trigger to return": to_return}


class _GilHog:
    def __enter__(self):
        self._stop = threading.Event()

        def spin():
            count = 0
            while not self._stop.is_set():
                count += 1

        self._thread = threading.Thread(target=spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


@pytest.mark.bench
def test_laser_process_latency(tmp_path, capsys):
    rows = []
    for name, opener in (("in process", _in_process),
                         ("laser process", _through_the_process)):
        controller = opener(tmp_path)
        original_load = nidaq_laser._load_nidaqmx
        try:
            for load in ("idle", "GIL hog"):
                if load == "idle":
                    results = _measure(controller)
                else:
                    with _GilHog():
                        results = _measure(controller)
                for measure, samples in results.items():
                    rows.append((name, load, measure) + _stats(samples))
        finally:
            controller.close()
            nidaq_laser._load_nidaqmx = original_load
    with capsys.disabled():
        print()
        print(f"{'path':<14}{'load':<9}{'measure':<26}{'p50 µs':>9}{'p95 µs':>9}{'max µs':>10}")
        for name, load, measure, p50, p95, high in rows:
            print(f"{name:<14}{load:<9}{measure:<26}{p50:>9.1f}{p95:>9.1f}{high:>10.1f}")
