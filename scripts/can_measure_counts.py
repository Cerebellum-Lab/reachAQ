"""Report decoded CAN message rates through reachAQ's selected transport."""

import argparse
import datetime
import logging
import queue
import time
from collections import defaultdict

from autotrainer.device import (
    CanDevice,
    CanTransportConfiguration,
    DeviceConnection,
    Target,
)


logger = logging.getLogger(__name__)


def _print_counts(counts, elapsed):
    total = sum(counts.values())
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"Total: {total / elapsed:.1f} decoded msg/s ;;; {stamp}")
    for kind, count in sorted(counts.items(), key=lambda item: str(item[0])):
        label = getattr(kind, "name", str(kind))
        print(f"{label} -> {count / elapsed:.1f} / s")
    print()


def measure(*, duration_seconds=60.0, refresh_seconds=1.0):
    messages = queue.Queue()
    transport = CanTransportConfiguration.from_environment()
    device = CanDevice(
        can_transport=transport,
        required_targets=(Target.PELLET_DEVICE,),
    )
    connection = DeviceConnection(device, messages, name="can-rate-diagnostic")
    connection.request_connect()
    connection.wait_connected(timeout=5)
    logger.info("Connected using %s on %s", transport.kind.value, transport.channel)

    started = previous = time.perf_counter()
    deadline = started + duration_seconds
    counts = defaultdict(int)
    try:
        while time.perf_counter() < deadline:
            timeout = min(refresh_seconds, max(0.0, deadline - time.perf_counter()))
            try:
                kind, _data = messages.get(timeout=timeout)
            except queue.Empty:
                pass
            else:
                counts[kind] += 1
            now = time.perf_counter()
            if now - previous >= refresh_seconds or now >= deadline:
                _print_counts(counts, max(now - previous, 1e-9))
                counts.clear()
                previous = now
    finally:
        connection.request_disconnect()
        connection.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--refresh", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    measure(duration_seconds=args.duration, refresh_seconds=args.refresh)


if __name__ == "__main__":
    main()
