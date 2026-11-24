
from typing import List, Sequence

import pandas
from numpy import ndarray

from reachaq.core import ProjectInfo, get_verbose_logger

logger = get_verbose_logger(__name__)


def intersession_inference(
    pose_data: ndarray, part_names: List[str], project: ProjectInfo, camera_names: Sequence[str]
) -> None:
    """
    Called once for pose data from all video frames passed to the pellet model.

    Writes the pair it always has: the first two recording cameras in configuration
    order (project.camera_names), here the first two of those that were posed. Each is
    taken from its own slot in `camera_names`, never from slot positions 0 and 1.

    :param pose_data: interleaved batch pose data from pellet model, one row per camera
        of `camera_names` in turn
    :param part_names: names associated with each pose position
    :param project: current project info for finding/defining file names
    :param camera_names: the cameras pose_data interleaves, in its order
    :return: None
    """
    camera_names = list(camera_names)
    pair = [name for name in project.camera_names if name in camera_names][:2]
    if len(pair) < 2:
        raise ValueError(
            f"intersession inference needs two posed recording cameras; posed {camera_names},"
            f" recording {list(project.camera_names)}")
    axis_labels = ("x", "y", "likelihood")
    columns = pandas.MultiIndex.from_product([part_names, axis_labels], names=["bodyparts", "coords"])

    for name in pair:
        cam_data = pose_data[camera_names.index(name)::len(camera_names), :]
        indices = range(cam_data.shape[0])
        df_xyp = pandas.DataFrame(cam_data, columns=columns, index=indices)
        df_xyp.to_hdf(project.get_intersession_pose_path(name=name), "df_with_missing", format="table",
                      mode="w")
