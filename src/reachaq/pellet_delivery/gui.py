import argparse
import logging
import sys


def main():
    from reachaq.core.logging import setup_logging
    setup_logging("reachaq", logger_level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--allow-can-emulation", help="include CAN emulation as a connection option",
                        default="", type=str)

    args = parser.parse_args()

    # strtobool compatibility is all over the place.
    allow_emulation = args.allow_can_emulation.lower() in ["true", "yes", "1"]

    from reachaq.pellet_delivery.run_pellet_delivery_ui import run_pellet_delivery_ui

    try:
        return run_pellet_delivery_ui(allow_emulation)
    finally:
        from reachaq.core.event import EventManager
        EventManager.try_close_default()


if __name__ == '__main__':
    sys.exit(main())
