import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy

from autotrainer.core import ProjectInfo
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.pose_elements import SceneElement
from autotrainer.core.reach_event import ReachEvent

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME

from autotrainer.inference.analysis.prepare_jetson_data import process_raw_data
from autotrainer.inference.analysis.parse_pellet_presentations_jetson import segment_reaches

from autotrainer.inference.analysis import IntersessionResponse

logger = get_verbose_logger(__name__)


AvailableShiftXYZ = numpy.array([[-5, 5], [-5, 5], [-5, 5]])


_segment_reach_debug: int = int(os.getenv("AUTOTRAINER_SEGMENT_REACH_DEBUG", 0))


def intersession_process(
    project: ProjectInfo,
    *,
    calib_dir: Optional[Path] = None,
    frame_rate: int = 150,
    debug_level: int = _segment_reach_debug,
) -> IntersessionResponse:
    """
    Called after pose processing for intersession analysis.

    :param project: current project info for finding/defining file names
    :param calib_dir: calibration directory if not default.
    :param debug_level: integer debug level.
    :return: information required to update behavior for future sessions
    """
    location, _, _ = project.get_session_path()
    logger.info("process intersession pose data using %s", location)
    calib_src_dir = (
        Path(f"~/Autotrainer/{DEFAULT_3D_CALIB_DIR_NAME}") if calib_dir is None else calib_dir
    ).expanduser()
    if not calib_src_dir.is_dir():
        logger.warning("calib_src_dir %s is not a directory",  calib_src_dir)
    calib_src_dir = calib_src_dir.as_posix()
    vid_tag = "." + project.video_write_ext
    dlc_seg = "_raw2D"
    center_method = (1, SceneElement.Diamond)
    #
    df_lr, centered_df_3d = process_raw_data(
        project,
        location, vid_tag, dlc_seg, calib_src_dir, center_method,
        frame_rate=frame_rate,
    )
    #
    results_dict = segment_reaches(
        project_info=project,
        session=location,
        center_method=center_method,
        df_lr=df_lr,
        df_3d=centered_df_3d,
        debug=debug_level,
        frame_rate=frame_rate,
    )
    logger.verbose("process intersession pose data complete %s", results_dict)
    # rename:
    results_dict["food_consumed"] = results_dict.pop("pellets_consumed")
    # all others keys are same than IntersessionResponse fields
    # convert to ReachEvent instances:
    results_dict["reach_events"] = [ReachEvent(**d) for d in results_dict["reach_events"]]
    results_dict["other_events"] = [
        # other events are pellet_events not associated with reach (by hand)
        ReachEvent(
            init=d['placed'],
            end=d['lost'],
            max=-1,
            method=d['method'],
            outcome=d['outcome'],
            delay_since_presented=(
                d['placed'] / frame_rate
                - project.get_first_pellet_presentation_offset()
            ),
        ) for d in results_dict["other_events"]
    ]
    return IntersessionResponse(**results_dict)
