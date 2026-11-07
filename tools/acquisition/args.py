
import argparse
from pathlib import Path
from typing import Optional

from tools.acquisition.model.app_model_status import AppModelStatus


def parse_start_mode(value: str):
    # from tools.acquisition.model.app_model_status import AppModelStatus
    try:
        return AppModelStatus(value.lower())  # values are lower, so force it
    except ValueError:
        pass
    try:
        return getattr(AppModelStatus, value.upper())  # names are upper, so force it
    except AttributeError:
        pass
    raise ValueError(f"Unknown AppModelStatus: {value!r}")


class AutoTrainerParsedArgs:
    """For ease of dev / type hints"""

    configuration: Optional[Path] = None
    preferences_file: Optional[Path] = None
    start_mode: AppModelStatus = AppModelStatus.IDLE
    live_inference: Optional[bool] = None
    random_cameras: bool = False
    dev: bool = False
    allow_can_emulation: bool = False


def make_autotrainer_parser(
    *,
    allow_dev_mode: bool = False,
    default_start_mode: AppModelStatus = AppModelStatus.IDLE,
):
    parser = argparse.ArgumentParser(
        prog="Autotrainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--configuration", help="configuration file", default=None, type=Path)
    parser.add_argument("--preferences-file", help="user preference ini file", default=None, type=Path)
    parser.add_argument("--start-mode", help="The desired start system mode",
                        choices=list(v.value for v in AppModelStatus), type=parse_start_mode,
                        default=default_start_mode)
    inference_group = parser.add_mutually_exclusive_group()
    inference_group.add_argument(
        "--live-inference",
        dest="live_inference",
        action="store_true",
        default=None,
        help="enable live inference for this run, overriding the configuration",
    )
    inference_group.add_argument(
        "--no-live-inference",
        dest="live_inference",
        action="store_false",
        help="disable live inference for this run, overriding the configuration",
    )
    parser.add_argument("--random-cameras", help="use in-memory random image cameras instead of physical cameras",
                        action="store_true")
    if allow_dev_mode:
        parser.add_argument("-d", "--dev", help="enable development mode and options", action="store_true")
        parser.add_argument("-e", "--allow-can-emulation", help="include CAN emulation as a connection option",
                            default=False, type=lambda v: v.lower() in {"true", "yes", "1"})
    return parser


def parse_autotrainer_args(*, allow_dev_mode: bool=False) -> AutoTrainerParsedArgs:
    parser = make_autotrainer_parser(allow_dev_mode=allow_dev_mode)
    return parser.parse_args()  # noqa
