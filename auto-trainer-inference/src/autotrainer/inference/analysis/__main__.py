import argparse
import logging
import sys
from pathlib import Path
from pprint import pprint
from datetime import datetime, date

import verboselogs

from autotrainer.behavior.pellet_shift import ShiftXYZBufferHandler
from autotrainer.core import get_verbose_logger, ProjectInfo
from autotrainer.core.config_directory import DEFAULT_CONFIG_DIR
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.configuration.behavior_configuration import ShiftXYZBufferHandlerConfig
from autotrainer.core.logging import setup_logging, get_console_handler

from autotrainer.inference.analysis import intersession_process

logger = get_verbose_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unit")
    parser.add_argument("date", type=lambda v: datetime.strptime(v, "%Y%m%d"))
    parser.add_argument("session", type=int)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, default=None,
                        help="calibration folder to use; by default the session's own is found under"
                             " --calib-root, by the id in its metadata or else by its start time")
    parser.add_argument("--calib-root", type=Path, default=DEFAULT_CONFIG_DIR.expanduser(),
                        help=f"where {DEFAULT_3D_CALIB_DIR_NAME} and its timestamped backups are")
    parser.add_argument("--log-level", type=lambda val: int(val) if val.isdigit() else val,
                        default=logging.WARNING)
    parser.add_argument("--analysis-debug-level", type=int, help="integer", default=0)
    args = parser.parse_args()
    logging.root.setLevel(args.log_level)
    get_console_handler().setLevel(args.log_level)
    # and then execute:
    project_info = ProjectInfo(
        root=args.data_dir,
        device_id=args.unit,
        when=args.date,
        session=args.session,
    )
    result = intersession_process(
        project_info,
        calib_dir=args.calib_dir,
        calib_root=args.calib_root,
        debug_level=args.analysis_debug_level,
    )
    sess_path = project_info.get_session_path()
    print(f"{sess_path.prefix}: {result.humanize()}")
    handler = ShiftXYZBufferHandler(config=ShiftXYZBufferHandlerConfig())
    shift = handler.make_shift_from_rh_list(result.rh_max_vp_list)
    print(f"Shift result: {shift.round(3)}")


if __name__ == "__main__":
    setup_logging("autotrainer")
    sys.exit(main())
